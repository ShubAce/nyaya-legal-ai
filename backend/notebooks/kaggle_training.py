# ============================================================
# NYAYA-7B — KAGGLE TRAINING NOTEBOOK  (T4 x2 GPU)
# ============================================================
# Run this on Kaggle with:
#   Settings → Accelerator → GPU T4 x2
#   Settings → Internet    → On
#
# Steps this notebook performs:
#   1. Install dependencies
#   2. Imports & GPU check
#   3. Configuration
#   4. W&B / HuggingFace login
#   5. Load training data
#   6. Formatting function
#   7. JSON validity callback
#   8. Single training run (run_training)
#   9. LoRA rank ablation (r=8, 16, 32)
#  10. Select best checkpoint
#  11. Merge LoRA weights into full model
#  12. Quick inference test
#  13. Push to HuggingFace Hub
#
# Expected runtime: ~4 hours on T4 x2 for 3 epochs / 1500+ samples
# ============================================================


# ── Cell 1: Install dependencies ─────────────────────────────────────────────
# IMPORTANT: Kaggle pre-installs RAPIDS/CUDA packages (dask-cuda, numba-cuda,
# cuml, cudf, etc.) with tightly pinned versions. Installing packages that pull
# in newer versions of numba/cuda-core will break RAPIDS.
#
# Strategy:
#   • Pin package versions known to work on Kaggle T4 images.
#   • Use --no-deps for bitsandbytes to avoid pulling in conflicting CUDA pkgs.
#   • Never touch numba, cuda-core, dask-cuda, or RAPIDS packages.

import subprocess, sys

def pip_install(packages, extra_args=None):
    """Install packages via pip, optionally with extra args like --no-deps."""
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    subprocess.check_call(cmd)

# Core training stack — these do NOT depend on RAPIDS/numba
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

# bitsandbytes — install with --no-deps to avoid pulling in cuda-core conflicts
pip_install(["bitsandbytes>=0.43.0"], extra_args=["--no-deps"])

print("✓ Dependencies installed")


# ── Cell 2: Imports & GPU check ───────────────────────────────────────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, gc, time, torch, wandb
from typing import List, Optional, Dict, Any
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig

print(f"✓ PyTorch      {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected! Go to Settings → Accelerator → GPU T4 x2"
    )

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB")
    # T4 does NOT support BFloat16 natively — Ampere (SM 8.0+) only
    print(f"        bf16 support: {props.major >= 8}")


# ── Cell 3: Configuration ─────────────────────────────────────────────────────
# ⚠ FILL THESE IN before running ⚠
HF_TOKEN      = ""        # your HuggingFace token (Settings → Access Tokens)
HF_USERNAME   = ""        # e.g. "your-username"
WANDB_API_KEY = ""        # your W&B API key (wandb.ai/settings)
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Kaggle dataset path — upload data/processed/train.jsonl as a Kaggle dataset
TRAIN_DATA_PATH = "/kaggle/input/nyaya-train-data/train.jsonl"

# Output directory
OUTPUT_DIR = "/kaggle/working/nyaya-checkpoints"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("✓ Config set")


# ── Cell 4: W&B and HF login ──────────────────────────────────────────────────
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    print("✓ W&B logged in")
else:
    os.environ["WANDB_DISABLED"] = "true"
    print("⚠ No W&B key — running without experiment tracking")

from huggingface_hub import login as hf_login
if HF_TOKEN:
    hf_login(token=HF_TOKEN)
    print("✓ HuggingFace logged in")
else:
    print("⚠ No HF token — model will not be pushed to Hub")


# ── Cell 5: Load training data ────────────────────────────────────────────────
def load_train_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training data not found at: {path}\n"
            f"Upload your train.jsonl as a Kaggle dataset and check the path."
        )
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠ Skipping malformed line {i+1}: {e}")
    if not rows:
        raise ValueError(f"No valid training samples found in {path}")
    print(f"✓ Loaded {len(rows):,} training samples")
    return Dataset.from_list(rows)

train_dataset = load_train_dataset(TRAIN_DATA_PATH)

# Preview one sample
sample = train_dataset[0]
print(f"\nSample fields : {list(sample.keys())}")
print(f"Message roles : {[m['role'] for m in sample['messages']]}")
print(f"Assist preview: {sample['messages'][2]['content'][:120]}...")


# ── Cell 6: Formatting function ───────────────────────────────────────────────
def format_sample(sample: dict) -> str:
    """
    Convert ChatML messages → Mistral instruct format.
    SFTTrainer calls this per-sample and expects a single str return.
    """
    messages = sample["messages"]

    # Handle both single sample (list of dicts) and batched (list of lists)
    # Modern trl calls this per-sample, so messages should be a list of dicts
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
        # Batched path (legacy): take the first message list
        messages = messages[0]

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user   = next((m["content"] for m in messages if m["role"] == "user"),   "")
    asst   = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    return f"<s>[INST] {system}\n\n{user} [/INST] {asst}</s>"


# ── Cell 7: JSON validity callback ────────────────────────────────────────────
class JSONValidityCallback(TrainerCallback):
    """Check JSON validity after each epoch and log to W&B."""

    def __init__(self, tokenizer, sample_text: str):
        self.tokenizer = tokenizer
        self.sample    = sample_text
        self.results   = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        prompt = (
            f"<s>[INST] Extract structured data from this Indian court judgment "
            f"and return a JSON object:\n\n{self.sample[:800]} [/INST] "
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=256,           # smaller for speed during eval
                do_sample=False,              # greedy — reproducible
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        valid = True
        try:
            json.loads(generated)
            print(f"✓ Epoch {state.epoch:.0f}: valid JSON output")
        except json.JSONDecodeError:
            valid = False
            print(f"✗ Epoch {state.epoch:.0f}: invalid JSON — {generated[:120]}")

        self.results.append(valid)
        if wandb.run and wandb.run.id:
            wandb.log({"json_validity": int(valid)}, step=state.global_step)
        model.train()


# ── Cell 8: Single training run ───────────────────────────────────────────────
def run_training(
    dataset: Dataset,
    rank:     int   = 16,
    alpha:    int   = 32,
    epochs:   int   = 3,
    lr:       float = 2e-4,
    run_name: str   = None,
) -> str:
    """Run one QLoRA training experiment on T4 GPU. Returns checkpoint path."""
    run_name = run_name or f"nyaya-r{rank}"
    out_dir  = f"{OUTPUT_DIR}/{run_name}"

    print(f"\n{'='*55}")
    print(f"Training: {run_name} | rank={rank} | alpha={alpha} | epochs={epochs}")
    print(f"{'='*55}")

    if WANDB_API_KEY:
        wandb.init(
            project="nyaya-7b",
            name=run_name,
            config=dict(base_model=BASE_MODEL_ID, rank=rank, alpha=alpha,
                        epochs=epochs, lr=lr),
            reinit=True,
        )

    # ── 4-bit quantisation (NF4 + fp16 compute — T4 compatible) ──────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # fp16, NOT bf16 (T4 has no bf16)
        bnb_4bit_use_double_quant=True,
    )

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model in 4-bit ───────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN or None,
        torch_dtype=torch.float16,              # fp16 base dtype
    )
    model.config.use_cache = False              # required for grad-checkpointing

    # ── Helper: force-cast every bf16 tensor to fp16 (T4 has no bf16 support) ─
    def cast_all_bf16_to_fp16(m: torch.nn.Module) -> None:
        """
        Cast ALL bfloat16 tensors (parameters AND buffers) to float16.
        Must be called after prepare_model_for_kbit_training AND after
        get_peft_model because both can silently introduce bf16 tensors.
        """
        for param in m.parameters():
            if param.data.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)
            if param.grad is not None and param.grad.dtype == torch.bfloat16:
                param.grad = param.grad.to(torch.float16)
        for buf in m.buffers():
            if buf.dtype == torch.bfloat16:
                buf.data = buf.data.to(torch.float16)

    # ── Prepare for k-bit training ────────────────────────────────────────────
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    cast_all_bf16_to_fp16(model)   # round 1: fix norm layers cast by prepare

    # ── LoRA adapter ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    cast_all_bf16_to_fp16(model)   # round 2: fix any bf16 re-introduced by get_peft_model

    # Diagnostic: confirm zero bf16 tensors remain before training
    bf16_params  = sum(1 for p in model.parameters() if p.dtype == torch.bfloat16)
    bf16_buffers = sum(1 for b in model.buffers()    if b.dtype == torch.bfloat16)
    print(f"bf16 params={bf16_params}  buffers={bf16_buffers}  ← must both be 0")
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Sample text for the JSON-validity callback
    sample_text = dataset[0]["messages"][1]["content"][:600]

    # ── Training config ───────────────────────────────────────────────────────
    # VRAM budget on 2×T4 (30 GB total, model-parallel via device_map="auto"):
    #   4-bit model  ~4.5 GB  |  LoRA + optimizer  ~2 GB
    #   Activations  ~5 GB    (batch=1, seq=2048, grad-checkpointing saves ~40%)
    #   → total ≈12 GB  ✓  comfortable in 30 GB (15 GB per GPU)
    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,           # 1 sample to avoid OOM on T4 (15 GB VRAM)
        gradient_accumulation_steps=16,          # effective batch = 16 (1 * 16)
        learning_rate=lr,
        fp16=True,                               # T4 supports fp16
        bf16=False,                              # T4 does NOT support bf16
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="no",
        max_length=2048,                         # full context — fine with 30 GB
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="wandb" if WANDB_API_KEY else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required for PEFT
        optim="paged_adamw_8bit",                # 8-bit Adam — essential for T4
        group_by_length=True,
        run_name=run_name,
        packing=False,
        dataloader_num_workers=2,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,              # replaces deprecated 'tokenizer' param
        args=training_args,
        train_dataset=dataset,
        formatting_func=format_sample,
        callbacks=[JSONValidityCallback(tokenizer, sample_text)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n── Diagnostic before training ──")
    print(f"model.config.torch_dtype: {getattr(model.config, 'torch_dtype', None)}")
    
    # Force cast all trainable parameters to float32 to prevent BFloat16 gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            
    trainable_dtypes = set(p.dtype for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameter dtypes: {trainable_dtypes}")
    print(f"Trainer args: fp16={trainer.args.fp16}, bf16={trainer.args.bf16}")
    print(f"Accelerator mixed_precision: {trainer.accelerator.state.mixed_precision}")
    
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n✓ Training complete in {elapsed/3600:.1f}h")

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✓ Checkpoint saved: {out_dir}")

    if WANDB_API_KEY and wandb.run:
        wandb.finish()

    # Free GPU memory before next run
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return out_dir


# ── Cell 9: LoRA rank ablation ────────────────────────────────────────────────
# 2×T4 = 30 GB VRAM: all three ranks (8, 16, 32) fit comfortably.
# rank=32 needs ~2 GB more adapters than rank=16 but is well within budget.
# If you want to save ~2 hours, comment out rank=8 and run only rank=16.

ABLATION_CONFIGS = [
    {"rank": 8,  "alpha": 16, "run_name": "nyaya-r8"},
    {"rank": 16, "alpha": 32, "run_name": "nyaya-r16"},   # ← recommended default
    {"rank": 32, "alpha": 64, "run_name": "nyaya-r32"},   # fine on 2×T4 (30 GB)
]

ablation_results = {}

for config in ABLATION_CONFIGS:
    print(f"\n{'='*60}")
    print(f"ABLATION RUN: rank={config['rank']}")
    print(f"{'='*60}")

    try:
        checkpoint_path = run_training(train_dataset, **config)
        ablation_results[config["run_name"]] = {
            "checkpoint": checkpoint_path,
            "rank":       config["rank"],
            "status":     "success",
        }
        print(f"✓ rank={config['rank']} saved: {checkpoint_path}")
        time.sleep(5)

    except Exception as e:
        import traceback
        print(f"✗ rank={config['rank']} failed:")
        traceback.print_exc()
        ablation_results[config["run_name"]] = {"rank": config["rank"], "status": "failed", "error": str(e)}
        gc.collect()
        torch.cuda.empty_cache()

print("\n── Ablation summary ──")
for name, result in ablation_results.items():
    print(f"  {name}: {result['status']}")


# ── Cell 10: Select best checkpoint ──────────────────────────────────────────
# Default: rank=16. Change after reviewing W&B loss curves.
BEST_RUN  = "nyaya-r16"
BEST_CKPT = ablation_results.get(BEST_RUN, {}).get(
    "checkpoint", f"{OUTPUT_DIR}/nyaya-r16"
)
print(f"\nBest checkpoint: {BEST_CKPT}")


# ── Cell 11: Merge LoRA weights into base model ────────────────────────────────
# Free all GPU memory before merging
gc.collect()
torch.cuda.empty_cache()

print("\nMerging LoRA adapter into base model...")

merged_path = f"{OUTPUT_DIR}/nyaya-7b-merged"
Path(merged_path).mkdir(parents=True, exist_ok=True)

merge_model = AutoPeftModelForCausalLM.from_pretrained(
    BEST_CKPT,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN or None,
)
merge_model = merge_model.merge_and_unload()
print("✓ LoRA weights merged")

merge_tokenizer = AutoTokenizer.from_pretrained(BEST_CKPT, token=HF_TOKEN or None)
merge_model.save_pretrained(merged_path, safe_serialization=True)
merge_tokenizer.save_pretrained(merged_path)
print(f"✓ Merged model saved: {merged_path}")


# ── Cell 12: Quick inference test ─────────────────────────────────────────────
from transformers import pipeline as hf_pipeline

test_pipe = hf_pipeline(
    "text-generation",
    model=merge_model,
    tokenizer=merge_tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

TEST_JUDGMENT = """
IN THE SUPREME COURT OF INDIA
Criminal Appeal No. 1481 of 2018
State of Punjab ... Appellant  Versus  Gurpreet Singh ... Respondent

The accused was charged under Section 302 and 34 of the Indian Penal Code.
The Trial Court convicted and sentenced to life imprisonment.
The High Court acquitted. The State appealed.
Relied upon: AIR 1984 SC 1622.

HELD: Appeal allowed. High Court judgment set aside. Conviction restored.
"""

test_output = test_pipe(
    f"<s>[INST] Extract structured data from this judgment and return JSON:\n\n{TEST_JUDGMENT} [/INST] ",
    max_new_tokens=512,
    do_sample=False,                             # greedy for reproducible test
    return_full_text=False,
    pad_token_id=merge_tokenizer.eos_token_id,
)[0]["generated_text"].strip()

print("\n" + "="*50)
print("TEST INFERENCE OUTPUT:")
print("="*50)
print(test_output[:1000])

try:
    parsed = json.loads(test_output)
    print("\n✓ Valid JSON output")
    print(f"  case_name: {parsed.get('case_name')}")
    print(f"  outcome:   {parsed.get('outcome')}")
    print(f"  statutes:  {[s.get('section') for s in parsed.get('statutes_cited', [])]}")
except json.JSONDecodeError:
    print("⚠ Output is not valid JSON — review training quality")


# ── Cell 13: Push to HuggingFace Hub ─────────────────────────────────────────
if HF_TOKEN and HF_USERNAME:
    HF_REPO = f"{HF_USERNAME}/nyaya-7b"
    print(f"\nPushing to HuggingFace Hub: {HF_REPO}")
    merge_model.push_to_hub(HF_REPO, token=HF_TOKEN, safe_serialization=True)
    merge_tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"✓ Model live at: https://huggingface.co/{HF_REPO}")
else:
    print("⚠ Set HF_TOKEN and HF_USERNAME (Cell 3) to push to Hub")
    print(f"  Local merged model at: {merged_path}")
