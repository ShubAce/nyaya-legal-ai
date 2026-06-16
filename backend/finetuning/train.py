"""
train.py — Phase 2: QLoRA finetuning of Mistral-7B on Kaggle T4 x2.

Upload this file + your data/processed/train.jsonl to a Kaggle dataset,
then run this notebook on GPU T4 x2 (Settings → Accelerator → GPU T4 x2).

Usage:
    python finetuning/train.py --rank 16 --epochs 3 --run-name nyaya-r16
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL_ID  = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN       = os.getenv("HF_TOKEN", "")
WANDB_API_KEY  = os.getenv("WANDB_API_KEY", "")
TRAIN_DATA     = Path("data/processed/train.jsonl")
OUTPUT_BASE    = Path("checkpoints")


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Nyaya-7B QLoRA finetuning")
    p.add_argument("--rank",        type=int,   default=16,
                   help="LoRA rank. Ablate with 8, 16, 32.")
    p.add_argument("--alpha",       type=int,   default=None,
                   help="LoRA alpha. Defaults to 2x rank.")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=2)
    p.add_argument("--grad-accum",  type=int,   default=8,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int,   default=2048)
    p.add_argument("--run-name",    type=str,   default=None,
                   help="W&B run name. Defaults to nyaya-r{rank}.")
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push merged model to HuggingFace Hub after training.")
    p.add_argument("--hf-repo",     type=str,   default=None,
                   help="HuggingFace repo name, e.g. username/nyaya-7b")
    return p.parse_args()


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_train_dataset(path: Path) -> Dataset:
    logger.info(f"Loading training data from {path}")
    if not path.exists():
        logger.error(f"Training data not found at {path}. Run prepare_dataset.py first.")
        sys.exit(1)

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    dataset = Dataset.from_list(rows)
    logger.success(f"Loaded {len(dataset)} training samples")
    return dataset


def format_sample(sample: dict) -> list[str] | str:
    """Convert ChatML messages to Mistral instruct format. Handles both single and batched inputs."""
    messages = sample["messages"]
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
        formatted_list = []
        for batch_msgs in messages:
            system_msg = next((m["content"] for m in batch_msgs if m["role"] == "system"), "")
            user_msg   = next((m["content"] for m in batch_msgs if m["role"] == "user"),   "")
            assist_msg = next((m["content"] for m in batch_msgs if m["role"] == "assistant"), "")
            formatted_list.append(f"<s>[INST] {system_msg}\n\n{user_msg} [/INST] {assist_msg}</s>")
        return formatted_list
    else:
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg   = next((m["content"] for m in messages if m["role"] == "user"),   "")
        assist_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        return f"<s>[INST] {system_msg}\n\n{user_msg} [/INST] {assist_msg}</s>"


# ── Callbacks ─────────────────────────────────────────────────────────────────
class JSONValidityCallback(TrainerCallback):
    """
    At the end of each epoch, generate a sample output and check if it's
    valid JSON. Logs the JSON validity rate to W&B.
    """
    def __init__(self, tokenizer, sample_text: str):
        self.tokenizer   = tokenizer
        self.sample_text = sample_text
        self.valid_count = 0
        self.total_count = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        model.eval()
        prompt = (
            f"<s>[INST] Extract structured data from this Indian court judgment "
            f"and return a JSON object:\n\n{self.sample_text[:1000]} [/INST] "
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        self.total_count += 1
        try:
            json.loads(generated.strip())
            self.valid_count += 1
            logger.success(f"Epoch {state.epoch:.0f}: Generated valid JSON ✓")
        except json.JSONDecodeError:
            logger.warning(f"Epoch {state.epoch:.0f}: Generated invalid JSON ✗")
            logger.debug(f"Generated: {generated[:200]}")

        validity_rate = self.valid_count / self.total_count
        if wandb.run:
            wandb.log({"json_validity_rate": validity_rate}, step=state.global_step)

        model.train()


class MemoryLoggerCallback(TrainerCallback):
    """Log GPU VRAM usage to W&B after each logging step."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and wandb.run:
            for i in range(torch.cuda.device_count()):
                used  = torch.cuda.memory_allocated(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                wandb.log({
                    f"gpu_{i}_used_gb":  round(used, 2),
                    f"gpu_{i}_total_gb": round(total, 2),
                    f"gpu_{i}_pct":      round(used / total * 100, 1),
                }, step=state.global_step)


# ── Main training function ────────────────────────────────────────────────────
def train(args):
    lora_alpha = args.alpha or (args.rank * 2)
    run_name   = args.run_name or f"nyaya-r{args.rank}"
    output_dir = OUTPUT_BASE / run_name

    logger.info("=" * 60)
    logger.info(f"Nyaya-7B QLoRA Training — {run_name}")
    logger.info(f"  Base model:  {BASE_MODEL_ID}")
    logger.info(f"  LoRA rank:   {args.rank}")
    logger.info(f"  LoRA alpha:  {lora_alpha}")
    logger.info(f"  Epochs:      {args.epochs}")
    logger.info(f"  Effective batch: {args.batch_size * args.grad_accum}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info("=" * 60)

    # ── W&B init ──────────────────────────────────────────────────────────────
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "nyaya-7b"),
            name=run_name,
            config={
                "base_model":   BASE_MODEL_ID,
                "lora_rank":    args.rank,
                "lora_alpha":   lora_alpha,
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "grad_accum":   args.grad_accum,
                "lr":           args.lr,
                "max_seq_len":  args.max_seq_len,
            }
        )

    # ── 4-bit quantization config ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NF4 is better than FP4 for weights
        bnb_4bit_compute_dtype=torch.float16, # compute in fp16 for speed
        bnb_4bit_use_double_quant=True,       # nested quantization saves ~0.4 bits/param
    )

    # ── Load tokenizer ────────────────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"  # required for SFTTrainer

    # ── Load model in 4-bit ───────────────────────────────────────────────────
    logger.info("Loading model in 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",          # distributes across both T4s automatically
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False           # required for gradient checkpointing
    model.config.pretraining_tp = 1

    # ── Prepare for k-bit training ────────────────────────────────────────────
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────────────────────
    # Target ALL linear projection layers — more expressive than just attention
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP (FFN)
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Log trainable parameter count
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    logger.success(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    if wandb.run:
        wandb.config.update({"trainable_params": trainable, "trainable_pct": pct})

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_train_dataset(TRAIN_DATA)

    # Sample text for JSONValidityCallback
    sample_text = dataset[0]["messages"][1]["content"][:500]

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        bf16=False,                    # T4 doesn't support bf16
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="no",            # no eval set — we use test_set separately
        max_length=args.max_seq_len,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="wandb" if WANDB_API_KEY else "none",
        gradient_checkpointing=True,   # saves ~30% VRAM at ~20% speed cost
        optim="paged_adamw_8bit",      # 8-bit Adam — essential for T4
        dataloader_num_workers=2,
        group_by_length=True,          # batch similar lengths → less padding waste
        run_name=run_name,
        packing=False,                 # don't pack sequences — cleaner for JSON output tasks
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=format_sample,
        callbacks=[
            JSONValidityCallback(tokenizer, sample_text),
            MemoryLoggerCallback(),
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()

    # ── Save final checkpoint ─────────────────────────────────────────────────
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.success(f"Model saved to {final_path}")

    if wandb.run:
        wandb.finish()

    # ── Optionally push to HuggingFace Hub ────────────────────────────────────
    if args.push_to_hub:
        from merge_and_push import merge_and_push
        hf_repo = args.hf_repo or os.getenv("HF_MODEL_REPO", "")
        if not hf_repo:
            logger.error("--hf-repo not specified and HF_MODEL_REPO not set in .env")
        else:
            merge_and_push(str(final_path), hf_repo)

    return str(final_path)


if __name__ == "__main__":
    args = parse_args()

    # Sanity check GPU
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU found. Training will be extremely slow on CPU.")
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB VRAM")

    train(args)
