#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsloth QLoRA fine-tuning of a Qwen coder model on a Next.js / Tailwind dataset.
Target hardware: Kaggle "GPU T4 x2" (2 x 16 GB, compute capability 7.5, no bf16).

Multi-GPU strategy
------------------
Unsloth's Triton kernels (fast cross-entropy, RMSNorm, RoPE, LoRA) require every tensor
to live on the SAME CUDA device. On T4 x2, Trainer/accelerate wraps the model in
DataParallel, or a device_map spreads layers across both GPUs, which raises:
    ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
This script pins the process to ONE GPU with CUDA_VISIBLE_DEVICES *before* torch is
imported, verifies that exactly one device is visible, and verifies that no model
weights ended up off-GPU.

Run it as the FIRST cell of a fresh Kaggle session (or as `python train.py`), because
CUDA_VISIBLE_DEVICES has no effect once torch has initialised CUDA.
"""

# =============================================================================
# 0. GPU PINNING - must run BEFORE torch / unsloth are imported
# =============================================================================
import os

TRAIN_GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = TRAIN_GPU_ID
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gc
import glob
import importlib.util
import inspect
import json
import subprocess
import sys
import time

AUTO_INSTALL = True  # needs Kaggle "Internet: On"; restart the session if pip changes torch


def ensure_dependencies() -> None:
    if not AUTO_INSTALL or importlib.util.find_spec("unsloth") is not None:
        return
    print("[setup] unsloth not found - installing...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "unsloth", "unsloth_zoo"]
    )


ensure_dependencies()

# unsloth must be imported before trl / transformers / peft so its patches apply.
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import TextStreamer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
# Lighter alternatives if you hit CUDA OOM:
#   "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
#   "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"

MAX_SEQ_LENGTH = 4096          # drop to 2048 if you hit OOM
DATASET_PATH = "/kaggle/input/nextjs-tailwind-dataset/dataset.json"  # .json (array) or .jsonl
OUTPUT_DIR = "/kaggle/working/outputs"
LORA_DIR = "/kaggle/working/qwen-nextjs-lora"
MERGED_DIR = "/kaggle/working/qwen-nextjs-merged-16bit"
GGUF_DIR = "/kaggle/working/qwen-nextjs-gguf"

SEED = 3407
LORA_R = 16
LORA_ALPHA = 32

PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8           # effective batch size = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
OPTIM = "adamw_torch"          # "adamw_8bit" saves a little memory; adamw_torch is the safest choice

RUN_SMOKE_TEST = True
SAVE_MERGED_16BIT = False      # ~15 GB for a 7B model; enable only if you have the disk space
SAVE_GGUF = False              # needs Internet + llama.cpp build; enable when you want a q4_k_m export

SYSTEM_PROMPT = (
    "You are an elite full-stack engineer and UI/UX designer. You write production-ready "
    "Next.js (App Router), TypeScript and Tailwind CSS code with accessible semantics, "
    "polished glassmorphism styling, smooth animations, and clean architecture. "
    "Respond with complete, working code."
)


# =============================================================================
# 2. GPU SAFETY CHECKS
# =============================================================================
def verify_single_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Set the Kaggle accelerator to 'GPU T4 x2' and restart the session."
        )

    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(
            f"Expected exactly 1 visible GPU but found {visible}. CUDA_VISIBLE_DEVICES was applied after "
            "CUDA was already initialised (torch was imported earlier in this kernel). Restart the session "
            "and run this script before any other cell that imports torch."
        )

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    print(
        f"[gpu] {props.name} | {props.total_memory / 1024**3:.1f} GB | "
        f"compute capability {props.major}.{props.minor} | physical GPU id {TRAIN_GPU_ID}"
    )
    if props.major < 8:
        print("[gpu] bf16 is not supported on this GPU, training in fp16 as configured.")


def assert_model_on_gpu(model) -> None:
    devices = {str(param.device) for param in model.parameters()}
    off_gpu = sorted(device for device in devices if device != "cuda:0")
    if off_gpu:
        raise RuntimeError(
            f"Model parameters found on {off_gpu}; everything must sit on cuda:0 for Unsloth's Triton "
            "kernels. Reduce MAX_SEQ_LENGTH or choose a smaller MODEL_NAME instead of offloading to CPU."
        )
    print(f"[gpu] all model parameters are on {sorted(devices)}")


# =============================================================================
# 3. DATA
# =============================================================================
def resolve_dataset_path() -> str:
    if os.path.isfile(DATASET_PATH):
        return DATASET_PATH

    patterns = [
        "/kaggle/input/**/*.json",
        "/kaggle/input/**/*.jsonl",
        "/kaggle/working/*.json",
        "/kaggle/working/*.jsonl",
        "./*.json",
        "./*.jsonl",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            print(f"[data] DATASET_PATH not found, falling back to {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        "No dataset file found. Upload your JSON as a Kaggle dataset and set DATASET_PATH accordingly."
    )


def load_records(path: str) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()

    try:
        parsed = json.loads(raw)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]

    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if instruction and output:
            records.append({"instruction": instruction, "output": output})

    print(f"[data] loaded {len(records)} valid records from {path}")
    return records


def build_text_dataset(records: list, tokenizer) -> Dataset:
    texts, lengths, dropped = [], [], 0

    for record in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])

        # Truncated code teaches the model to stop mid-file, so overlong samples are dropped.
        if n_tokens > MAX_SEQ_LENGTH:
            dropped += 1
            continue

        texts.append(text)
        lengths.append(n_tokens)

    if not texts:
        raise ValueError(
            f"Every sample exceeded MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} tokens. Increase it or shorten the outputs."
        )

    print(
        f"[data] kept {len(texts)} samples | dropped {dropped} over {MAX_SEQ_LENGTH} tokens | "
        f"mean {sum(lengths) / len(lengths):.0f} | max {max(lengths)} tokens"
    )
    return Dataset.from_dict({"text": texts}).shuffle(seed=SEED)


# =============================================================================
# 4. TRAINER (version-tolerant across TRL releases)
# =============================================================================
def build_trainer(model, tokenizer, dataset: Dataset):
    config_params = set(inspect.signature(SFTConfig.__init__).parameters)
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)

    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim=OPTIM,
        fp16=True,               # T4 has no bf16
        bf16=False,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer_kwargs = dict(model=model, train_dataset=dataset)

    # SFT-specific options moved between SFTTrainer and SFTConfig across TRL versions.
    sft_options = {
        "dataset_text_field": "text",
        "packing": False,
        "dataset_num_proc": 1,   # multiprocessing map + CUDA is a common source of hangs on Kaggle
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    for key, value in sft_options.items():
        config_key = key
        if key == "max_seq_length" and key not in config_params and "max_length" in config_params:
            config_key = "max_length"
        if config_key in config_params:
            config_kwargs[config_key] = value
        elif key in trainer_params:
            trainer_kwargs[key] = value

    trainer_kwargs["args"] = SFTConfig(**config_kwargs)
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    # Compute loss on the assistant reply only, not on the system prompt or the instruction.
    masking_kwargs = {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    }
    if "num_proc" in inspect.signature(train_on_responses_only).parameters:
        masking_kwargs["num_proc"] = 1
    return train_on_responses_only(trainer, **masking_kwargs)


def preview_masking(trainer, tokenizer) -> None:
    try:
        row = trainer.train_dataset[0]
        trainable = [tok for tok, label in zip(row["input_ids"], row["labels"]) if label != -100]
        print("[check] first trainable tokens:", repr(tokenizer.decode(trainable[:60])))
    except Exception as exc:  # purely diagnostic
        print(f"[check] masking preview skipped: {exc}")


# =============================================================================
# 5. INFERENCE SMOKE TEST
# =============================================================================
def smoke_test(model, tokenizer) -> None:
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Build an accessible glassmorphism pricing card in Next.js (App Router) with Tailwind CSS "
            "and a hover lift effect.",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("\n" + "=" * 30 + " SMOKE TEST " + "=" * 30)
    with torch.no_grad():
        model.generate(
            **inputs,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
            max_new_tokens=768,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            use_cache=True,
        )
    print("\n" + "=" * 72)


# =============================================================================
# 6. MAIN
# =============================================================================
def main() -> None:
    verify_single_gpu()

    print(f"[model] loading {MODEL_NAME} (4-bit, fp16, max_seq_length={MAX_SEQ_LENGTH})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,                       # 0 is the optimised path in Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",  # lowest VRAM usage, enables long contexts on 16 GB
        random_state=SEED,
        use_rslora=False,
    )
    assert_model_on_gpu(model)

    records = load_records(resolve_dataset_path())
    dataset = build_text_dataset(records, tokenizer)

    trainer = build_trainer(model, tokenizer, dataset)
    preview_masking(trainer, tokenizer)

    gc.collect()
    torch.cuda.empty_cache()
    reserved_before = torch.cuda.max_memory_reserved() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[train] GPU memory before training: {reserved_before:.2f} / {total_memory:.2f} GB")

    started = time.time()
    stats = trainer.train()
    elapsed = time.time() - started

    peak = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"[train] done in {elapsed / 60:.1f} min | final loss {stats.training_loss:.4f} | "
        f"peak reserved {peak:.2f} GB ({peak / total_memory * 100:.1f}%)"
    )

    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print(f"[save] LoRA adapters saved to {LORA_DIR}")

    if RUN_SMOKE_TEST:
        smoke_test(model, tokenizer)

    if SAVE_MERGED_16BIT:
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
        print(f"[save] merged 16-bit model saved to {MERGED_DIR}")

    if SAVE_GGUF:
        model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
        print(f"[save] GGUF (q4_k_m) saved to {GGUF_DIR}")


if __name__ == "__main__":
    main()
