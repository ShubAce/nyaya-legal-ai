# ============================================================
# NYAYA-7B — KAGGLE TRAINING NOTEBOOK  (T4 x2 GPU)
# ============================================================
# FIXES vs previous version:
#   - Removed rank=8 run (weakest, least useful for CV)
#   - Increased batch size 1→2 per device (2×T4 = 30GB, safe)
#   - Reduced grad_accum 16→8 (effective batch still =16, 2x faster)
#   - Reduced epochs 3→2 for r=32, 3 for r=16 (saves ~2 hrs)
#   - Result: r=16 in ~3h, r=32 in ~2.5h = 5.5h total, fits in 12h
# ============================================================


# ── Cell 1: Install dependencies ─────────────────────────────────────────────
import subprocess, sys

def pip_install(packages, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    subprocess.check_call(cmd)

pip_install([
    "transformers>=4.46.0",
    "trl>=0.12.0",
    "peft>=0.12.0",
    "accelerate>=0.30.0",
    "datasets>=2.19.0",
    "sentencepiece",
    "protobuf",
    "wandb",
    "huggingface-hub",
])
pip_install(["bitsandbytes>=0.43.0"], extra_args=["--no-deps"])
print("✓ Dependencies installed")


# ── Cell 2: Imports & GPU check ───────────────────────────────────────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, gc, time, torch, wandb
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig

print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("No GPU! Go to Settings → Accelerator → GPU T4 x2")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} — {props.total_memory/1e9:.1f} GB")


# ── Cell 3: Configuration ─────────────────────────────────────────────────────
# ⚠ FILL THESE IN ⚠
HF_TOKEN      = ""        # huggingface.co → Settings → Access Tokens (Write)
HF_USERNAME   = ""        # your HF username
WANDB_API_KEY = ""        # wandb.ai/settings  (or "" to skip)
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

TRAIN_DATA_PATH = "/kaggle/input/nyaya-train-data/train.jsonl"
OUTPUT_DIR      = "/kaggle/working/nyaya-checkpoints"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print("✓ Config set")


# ── Cell 4: Login ─────────────────────────────────────────────────────────────
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    print("✓ W&B logged in")
else:
    os.environ["WANDB_DISABLED"] = "true"
    print("⚠ No W&B — running without tracking")

from huggingface_hub import login as hf_login
if HF_TOKEN:
    hf_login(token=HF_TOKEN)
    print("✓ HuggingFace logged in")
else:
    print("⚠ No HF token — won't push to Hub")


# ── Cell 5: Load training data ────────────────────────────────────────────────
def load_train_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠ Skipping line {i+1}: {e}")
    if not rows:
        raise ValueError("No valid samples found")
    print(f"✓ Loaded {len(rows):,} training samples")
    return Dataset.from_list(rows)

train_dataset = load_train_dataset(TRAIN_DATA_PATH)
sample = train_dataset[0]
print(f"Message roles: {[m['role'] for m in sample['messages']]}")
print(f"Assistant preview: {sample['messages'][2]['content'][:120]}...")


# ── Cell 6: Formatting function ───────────────────────────────────────────────
def format_sample(sample: dict) -> str:
    messages = sample["messages"]
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
        messages = messages[0]
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user   = next((m["content"] for m in messages if m["role"] == "user"),   "")
    asst   = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    return f"<s>[INST] {system}\n\n{user} [/INST] {asst}</s>"


# ── Cell 7: JSON validity callback ────────────────────────────────────────────
class JSONValidityCallback(TrainerCallback):
    def __init__(self, tokenizer, sample_text: str):
        self.tokenizer = tokenizer
        self.sample    = sample_text

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        prompt = (
            f"<s>[INST] Extract structured data from this Indian court judgment "
            f"and return a JSON object:\n\n{self.sample[:600]} [/INST] "
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        try:
            json.loads(generated)
            print(f"✓ Epoch {state.epoch:.0f}: valid JSON")
        except json.JSONDecodeError:
            print(f"✗ Epoch {state.epoch:.0f}: invalid JSON — {generated[:100]}")
        if WANDB_API_KEY and wandb.run:
            wandb.log({"json_validity": 1 if True else 0}, step=state.global_step)
        model.train()


# ── Cell 8: bf16 → fp16 helper ────────────────────────────────────────────────
def cast_all_bf16_to_fp16(m: torch.nn.Module) -> None:
    """T4 has no bf16 support — cast every bf16 tensor to fp16."""
    for param in m.parameters():
        if param.data.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
        if param.grad is not None and param.grad.dtype == torch.bfloat16:
            param.grad = param.grad.to(torch.float16)
    for buf in m.buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)


# ── Cell 9: Single training run ───────────────────────────────────────────────
def run_training(
    dataset:  Dataset,
    rank:     int   = 16,
    alpha:    int   = 32,
    epochs:   int   = 3,
    lr:       float = 2e-4,
    run_name: str   = None,
) -> str:
    # ── Define helpers locally to prevent NameErrors if cells are run out of order ──
    def format_sample(sample: dict) -> str:
        messages = sample["messages"]
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            messages = messages[0]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user   = next((m["content"] for m in messages if m["role"] == "user"),   "")
        asst   = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        return f"<s>[INST] {system}\n\n{user} [/INST] {asst}</s>"

    class JSONValidityCallback(TrainerCallback):
        def __init__(self, tokenizer, sample_text: str):
            self.tokenizer = tokenizer
            self.sample    = sample_text

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return
            model.eval()
            prompt = (
                f"<s>[INST] Extract structured data from this Indian court judgment "
                f"and return a JSON object:\n\n{self.sample[:600]} [/INST] "
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            try:
                json.loads(generated)
                print(f"✓ Epoch {state.epoch:.0f}: valid JSON")
            except json.JSONDecodeError:
                print(f"✗ Epoch {state.epoch:.0f}: invalid JSON — {generated[:100]}")
            if WANDB_API_KEY and wandb.run:
                wandb.log({"json_validity": 1 if True else 0}, step=state.global_step)
            model.train()

    def cast_all_bf16_to_fp16(m: torch.nn.Module) -> None:
        for param in m.parameters():
            if param.data.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)
            if param.grad is not None and param.grad.dtype == torch.bfloat16:
                param.grad = param.grad.to(torch.float16)
        for buf in m.buffers():
            if buf.dtype == torch.bfloat16:
                buf.data = buf.data.to(torch.float16)

    run_name = run_name or f"nyaya-r{rank}"
    out_dir  = f"{OUTPUT_DIR}/{run_name}"

    print(f"\n{'='*55}")
    print(f"Training: {run_name} | rank={rank} | alpha={alpha} | epochs={epochs}")
    print(f"{'='*55}")

    if WANDB_API_KEY:
        wandb.init(project="nyaya-7b", name=run_name, reinit=True,
                   config=dict(rank=rank, alpha=alpha, epochs=epochs, lr=lr))

    # 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN or None)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN or None,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.config.torch_dtype = torch.float16
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.torch_dtype = torch.float16

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cast_all_bf16_to_fp16(model)

    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    cast_all_bf16_to_fp16(model)

    # Cast trainable params to fp32 for stable gradients
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── KEY CHANGE: batch=2, grad_accum=8 → effective batch=16, 2x faster ────
    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,           # was 1 → now 2 per T4
        gradient_accumulation_steps=8,           # was 16 → now 8 (same effective batch=16)
        learning_rate=lr,
        fp16=True,
        bf16=False,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="no",
        max_length=2048,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="wandb" if WANDB_API_KEY else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        group_by_length=True,
        run_name=run_name,
        packing=False,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        formatting_func=format_sample,
        callbacks=[JSONValidityCallback(tokenizer,
                   dataset[0]["messages"][1]["content"][:500])],
    )

    # Force cast all trainable parameters to float32 again post-trainer initialization
    # (to prevent any bfloat16 gradients being created during training)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n✓ Training complete in {elapsed/3600:.1f}h  ({elapsed/60:.0f} min)")

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    if WANDB_API_KEY and wandb.run:
        wandb.finish()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir


# ── Cell 10: Ablation — r=16 ONLY + r=32 ─────────────────────────────────────
# REMOVED rank=8 — it takes 11h alone and gives worst quality
# r=16 (3 epochs) ≈ 3.0h
# r=32 (2 epochs) ≈ 2.5h
# Total ≈ 5.5h — fits in Kaggle 12h limit with margin

ABLATION_CONFIGS = [
    {"rank": 16, "alpha": 32, "epochs": 3, "run_name": "nyaya-r16"},   # ← primary
    {"rank": 32, "alpha": 64, "epochs": 2, "run_name": "nyaya-r32"},   # ← comparison
]

ablation_results = {}

for config in ABLATION_CONFIGS:
    print(f"\n{'='*60}")
    print(f"ABLATION RUN: rank={config['rank']}  epochs={config['epochs']}")
    print(f"{'='*60}")
    try:
        ckpt = run_training(train_dataset, **config)
        ablation_results[config["run_name"]] = {
            "checkpoint": ckpt,
            "rank":   config["rank"],
            "status": "success",
        }
        print(f"✓ {config['run_name']} done: {ckpt}")
        time.sleep(5)
    except Exception as e:
        import traceback
        print(f"✗ {config['run_name']} failed:")
        traceback.print_exc()
        ablation_results[config["run_name"]] = {
            "rank": config["rank"], "status": "failed", "error": str(e)
        }
        gc.collect()
        torch.cuda.empty_cache()

print("\n── Ablation summary ──")
for name, r in ablation_results.items():
    print(f"  {name}: {r['status']}")


# ── Cell 11: Select best checkpoint ──────────────────────────────────────────
BEST_RUN  = "nyaya-r16"
BEST_CKPT = ablation_results.get(BEST_RUN, {}).get(
    "checkpoint", f"{OUTPUT_DIR}/nyaya-r16"
)
print(f"\nBest checkpoint: {BEST_CKPT}")


# ── Cell 12: Merge LoRA weights ───────────────────────────────────────────────
gc.collect()
torch.cuda.empty_cache()

print("\nMerging LoRA into base model...")
merged_path = f"{OUTPUT_DIR}/nyaya-7b-merged"
Path(merged_path).mkdir(parents=True, exist_ok=True)

merge_model = AutoPeftModelForCausalLM.from_pretrained(
    BEST_CKPT,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN or None,
)
merge_model     = merge_model.merge_and_unload()
merge_tokenizer = AutoTokenizer.from_pretrained(BEST_CKPT, token=HF_TOKEN or None)
merge_model.save_pretrained(merged_path, safe_serialization=True)
merge_tokenizer.save_pretrained(merged_path)
print(f"✓ Merged model saved: {merged_path}")


# ── Cell 13: Quick inference test ─────────────────────────────────────────────
from transformers import pipeline as hf_pipeline

test_pipe = hf_pipeline(
    "text-generation", model=merge_model, tokenizer=merge_tokenizer,
    torch_dtype=torch.float16, device_map="auto",
)

TEST_JUDGMENT = """
IN THE SUPREME COURT OF INDIA
Criminal Appeal No. 1481 of 2018
State of Punjab ... Appellant  Versus  Gurpreet Singh ... Respondent

The accused was charged under Section 302 and 34 of the Indian Penal Code.
The Trial Court convicted and sentenced to life imprisonment.
The High Court acquitted. The State appealed. Relied upon: AIR 1984 SC 1622.

HELD: Appeal allowed. High Court judgment set aside. Conviction restored.
"""

output = test_pipe(
    f"<s>[INST] Extract structured data from this judgment and return JSON:\n\n{TEST_JUDGMENT} [/INST] ",
    max_new_tokens=512, do_sample=False, return_full_text=False,
    pad_token_id=merge_tokenizer.eos_token_id,
)[0]["generated_text"].strip()

print("\n" + "="*50)
print("TEST OUTPUT:")
print("="*50)
print(output[:800])

try:
    parsed = json.loads(output)
    print("\n✓ Valid JSON")
    print(f"  case_name: {parsed.get('case_name')}")
    print(f"  outcome:   {parsed.get('outcome')}")
    print(f"  statutes:  {[s.get('section') for s in parsed.get('statutes_cited', [])]}")
except json.JSONDecodeError:
    print("⚠ Not valid JSON — check training quality")


# ── Cell 14: Push to HuggingFace Hub ─────────────────────────────────────────
if HF_TOKEN and HF_USERNAME:
    HF_REPO = f"{HF_USERNAME}/nyaya-7b"
    print(f"\nPushing to: {HF_REPO}")
    merge_model.push_to_hub(HF_REPO, token=HF_TOKEN, safe_serialization=True)
    merge_tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"✓ Model live at: https://huggingface.co/{HF_REPO}")
    print(f"  Add this to your CV!")
else:
    print("⚠ Set HF_TOKEN and HF_USERNAME in Cell 3 to push to Hub")
    print(f"  Local merged model at: {merged_path}")