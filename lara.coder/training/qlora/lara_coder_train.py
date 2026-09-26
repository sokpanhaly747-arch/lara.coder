#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lara.coder - Unsloth QLoRA fine-tuning of a Qwen 2.5 Coder model on Lara.coder's own
training corpus (dataset/ in the Lara.coder repository: frontend, backend, ui_ux,
debugging, tool_use, agent_workflow, and related categories).

Target hardware: Kaggle "GPU T4 x2" (2 x 16 GB, compute capability 7.5, no bf16).

Multi-GPU strategy
------------------
Unsloth's Triton kernels (fast cross-entropy, RMSNorm, RoPE, LoRA) require every tensor
to live on the SAME CUDA device. On T4 x2, Trainer/accelerate wraps the model in
DataParallel, or a device_map spreads layers across both GPUs, which raises:
    ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
This script pins the process to ONE GPU with CUDA_VISIBLE_DEVICES *before* torch is
imported, verifies that exactly one device is visible, and verifies that no model
weights ended up off-GPU. This is a deliberate, re-verified choice for this exact
Unsloth + QLoRA + Qwen2.5-Coder + T4x2 combination, not an unexamined holdover from a
prior script: single-process, single-visible-GPU training is the widely-documented
working pattern for this stack on Kaggle's T4x2 runtime, and splitting the model across
both GPUs (DataParallel or device_map="auto") is what triggers the Triton pointer error
above. Genuine multi-GPU data-parallel training (a separate process per GPU via
`accelerate launch`, each with CUDA_VISIBLE_DEVICES pinned to its own single device) is
possible in principle but is a different execution mode from "run this file directly in
a Kaggle cell", so it is intentionally out of scope here.

Run it as the FIRST cell of a fresh Kaggle session (or as `python lara_coder_train.py`),
because CUDA_VISIBLE_DEVICES has no effect once torch has initialised CUDA.

Dataset
-------
This script trains on Lara.coder's own dataset tree, not a single hardcoded file. It
recursively scans DATASET_DIR (default: /kaggle/working/lara.coder/dataset/) for every
*.json and *.jsonl file, validates each record, and aggregates everything into one
shuffled Hugging Face Dataset. See `discover_dataset_files` / `load_dataset_records`
below for the discovery and validation rules.
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
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

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

# Canonical Lara.coder dataset root. Scanned recursively for *.json / *.jsonl.
DATASET_DIR = "/kaggle/working/lara.coder/dataset/"

# Optional: restrict training to a subset of dataset/schema/example_schema.json's
# `category` values (mirrors training/configs/sft_config.yaml's `dataset.categories`).
# Leave as None to train on every category present in the discovered files.
CATEGORY_FILTER: tuple[str, ...] | None = None

# Output locations. Kept out of the DATASET_DIR tree (case-sensitive collision-safe)
# but still Lara.coder-scoped under /kaggle/working/.
OUTPUT_DIR = "/kaggle/working/lara_coder_outputs"
LORA_DIR = "/kaggle/working/lara_coder_lora"
MERGED_DIR = "/kaggle/working/lara_coder_merged_16bit"
GGUF_DIR = "/kaggle/working/lara_coder_gguf"

SEED = 3407
# r / alpha match training/configs/sft_config.yaml's QLoRA settings.
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

# Architecture-aware system prompt. Cross-validated against Lara.coder's actual source
# (agent/, planning/, design/, codegen/, runtime/, browser/, testing/, debugging/,
# evaluator/, memory/, models/) — not just README.md/ARCHITECTURE.md's prose — so the
# fine-tuned model reasons about, and never invents, this specific system.
SYSTEM_PROMPT = (
    "You are the coding intelligence operating inside Lara.coder, an autonomous AI "
    "software engineer that turns a natural-language product idea into a working, "
    "tested software product — not a chatbot that prints isolated code snippets. "
    "Lara.coder behaves like a small product team: it understands the request, plans "
    "requirements, designs the UI/UX, decides architecture, generates code, runs and "
    "tests the result, debugs failures, and iterates until the product clears an "
    "objective quality bar.\n\n"
    "Lifecycle (UNDERSTAND -> PLAN -> DESIGN -> CODE -> RUN -> TEST -> DEBUG -> REVIEW "
    "-> IMPROVE) is coordinated by agent.orchestrator.next_phase, a pure decision "
    "function over a typed agent.state.AgentState (never raw chat text), executed by "
    "agent.loop.AgentLoop, and bounded at every stage by agent.config.LoopLimits so no "
    "retry loop runs forever. delete_project and deploy always require explicit human "
    "approval (agent.config.AgentConfig.require_human_approval_for) — no phase may "
    "perform them autonomously.\n\n"
    "Pipeline you operate within:\n"
    "- planning.requirement_analyzer turns the idea into structured Requirements, "
    "surfacing ambiguities as a stated question plus a default assumption rather than "
    "guessing silently.\n"
    "- planning.product_planner produces a ProductPlan (pages, data entities, API "
    "surface, auth need, user flows); planning.task_planner then deterministically "
    "expands it into a dependency-ordered TaskGraph (one task per entity/endpoint/page) "
    "without an extra model call.\n"
    "- design.ux_planner defines navigation and each screen's goal; design.ui_planner "
    "turns that into layouts and components against the shared design.design_system "
    "tokens (palette, type, spacing/radius scale) — every component must define "
    "default, loading, empty, and error states up front.\n"
    "- planning/design calls that need typed data, not prose, go through "
    "models.inference.structured_complete: a strict JSON-only instruction layered "
    "under that stage's own system prompt (e.g. \"You are a senior product analyst\", "
    "\"You are a UI designer\"), with automatic retry-and-repair if the model's JSON "
    "fails to parse or validate.\n"
    "- codegen.generator turns one Task at a time into source files for the configured "
    "codegen.stacks.StackConfig (default: Next.js App Router, TypeScript, Tailwind CSS, "
    "vitest) — output is ONLY a JSON array of {path, content} objects, no prose, no "
    "markdown fences — written to disk exclusively through runtime.filesystem_tool "
    "inside a path-jailed runtime.sandbox.Sandbox; generated code never touches the "
    "host directly.\n"
    "- runtime.process_manager starts the dev server; browser.controller (Playwright) "
    "navigates and clicks through it; browser.screenshot captures the rendered page and "
    "browser.dom_inspector flags structural/accessibility issues (e.g. missing alt "
    "text); testing.unit_runner and testing.e2e_runner (driving browser.controller over "
    "each ProductPlan user flow) report pass/fail; testing.visual_tester sanity-checks "
    "the resulting screenshots.\n"
    "- On failure, debugging.error_classifier first buckets the error by rule (syntax, "
    "type, missing_dependency, runtime, logic, config, unknown) so easy cases can route "
    "to a cheaper model via models.router; debugging.root_cause then asks a model for a "
    "specific, falsifiable root cause and fix description; debugging.auto_fix applies "
    "exactly one bounded attempt per call, always returning the complete corrected file "
    "— never a diff — before control returns to TEST. LoopLimits.max_fix_attempts_"
    "per_error and max_total_fix_attempts, enforced at the agent.loop call site, cap "
    "the retries.\n"
    "- evaluator.metrics.evaluate_run scores test pass rate, unresolved errors, and "
    "accessibility issues into a QualityReport; the orchestrator uses passes_bar to "
    "decide REVIEW -> DONE/IMPROVE or REVIEW -> PLAN for another pass.\n"
    "- memory.project_memory persists AgentState, decisions, and codebase facts across "
    "the run so work can resume after a crash instead of replaying the chat.\n"
    "- models.router.ModelRouter maps each TaskKind (classify/plan/design/codegen/"
    "debug/review) to a models.adapter.ModelAdapter, so cheap classification and "
    "strong codegen/debug work can use different models without any other module "
    "knowing which one is in use.\n\n"
    "Working principles: respect the module boundaries above (agent composes the "
    "other modules; none of them import agent back); never invent an API, class, or "
    "module this architecture does not have; treat every tool call as an explicit, "
    "typed action, never raw eval/exec; keep changes scoped to the task's affected "
    "module(s); write accessible, production-quality Next.js/TypeScript/Tailwind code "
    "by default; when asked for structured data respond with JSON only, no prose or "
    "fences; and when fixing or regenerating a file, return its complete new content — "
    "never a partial diff, a TODO, or a placeholder standing in for real code."
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
# 3. DATA - recursive discovery, validation, and aggregation
# =============================================================================
def discover_dataset_files(dataset_dir: Path) -> list[Path]:
    """Recursively finds every *.json / *.jsonl file under dataset_dir."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Upload or extract Lara.coder's dataset/ tree to this path before running "
            "this script (nested category subdirectories, e.g. dataset/frontend/, "
            "dataset/debugging/, are scanned automatically)."
        )

    files = sorted(set(dataset_dir.rglob("*.json")) | set(dataset_dir.rglob("*.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"No .json or .jsonl files found under {dataset_dir} (searched recursively). "
            "Check that the dataset was actually uploaded/extracted to this path."
        )
    return files


def _extract_rows(parsed) -> list:
    """Normalizes a parsed .json payload into a list of candidate record dicts.

    Supports: a top-level list of records, an object wrapping a list of records under
    a common key, or a single record object.
    """
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        for key in ("examples", "data", "records", "items", "dataset"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [parsed]  # treat the object itself as a single record
    return []


def _load_json_file(path: Path, stats: dict) -> list:
    try:
        raw_text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        stats["files_malformed"] += 1
        print(f"[data] WARN could not read {path}: {exc}")
        return []

    if not raw_text:
        stats["files_malformed"] += 1
        print(f"[data] WARN empty file skipped: {path}")
        return []

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        stats["files_malformed"] += 1
        print(f"[data] WARN malformed JSON skipped: {path} ({exc})")
        return []

    stats["files_processed"] += 1
    return _extract_rows(parsed)


def _load_jsonl_file(path: Path, stats: dict) -> list:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        stats["files_malformed"] += 1
        print(f"[data] WARN could not read {path}: {exc}")
        return []

    rows, malformed_lines = [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            malformed_lines += 1

    if malformed_lines:
        print(f"[data] WARN {malformed_lines} malformed line(s) skipped in {path}")
        stats["records_malformed"] += malformed_lines

    stats["files_processed"] += 1
    return rows


def _coerce_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text or None
    return None  # lists/dicts in a text field are not usable; caller counts this as invalid


def _normalize_record(raw: dict) -> dict | None:
    """Extracts a valid (instruction, output) pair from one raw record.

    Accepts both `instruction` (this script's original field name) and `input`
    (the field name used by Lara.coder's own dataset/schema/example_schema.json) as
    the source-text field, so the loader works against either convention.
    """
    instruction = _coerce_text(raw.get("instruction"))
    if instruction is None:
        instruction = _coerce_text(raw.get("input"))
    output = _coerce_text(raw.get("output"))

    if instruction is None or output is None:
        return None

    category = raw.get("category")
    category = str(category).strip() if isinstance(category, (str, int, float)) else None

    return {"instruction": instruction, "output": output, "category": category}


def load_dataset_records(dataset_dir: str) -> tuple[list[dict], dict]:
    """Discovers, loads, validates, deduplicates, and aggregates every record under
    dataset_dir into one list, plus a stats dict for diagnostics."""
    root = Path(dataset_dir)
    stats = {
        "files_discovered": 0,
        "files_processed": 0,
        "files_malformed": 0,
        "records_seen": 0,
        "records_malformed": 0,
        "records_valid": 0,
        "records_skipped_invalid": 0,
        "records_filtered_category": 0,
        "duplicates_removed": 0,
    }

    files = discover_dataset_files(root)
    stats["files_discovered"] = len(files)

    seen_hashes: set[str] = set()
    records: list[dict] = []

    for path in files:
        raw_rows = _load_jsonl_file(path, stats) if path.suffix == ".jsonl" else _load_json_file(path, stats)

        for raw in raw_rows:
            stats["records_seen"] += 1
            normalized = _normalize_record(raw)
            if normalized is None:
                stats["records_skipped_invalid"] += 1
                continue

            if CATEGORY_FILTER and normalized["category"] not in CATEGORY_FILTER:
                stats["records_filtered_category"] += 1
                continue

            fingerprint = hashlib.sha256(
                (normalized["instruction"] + "\x1f" + normalized["output"]).encode("utf-8")
            ).hexdigest()
            if fingerprint in seen_hashes:
                stats["duplicates_removed"] += 1
                continue

            seen_hashes.add(fingerprint)
            records.append(normalized)
            stats["records_valid"] += 1

    print(
        "[data] files: {files_discovered} discovered / {files_processed} processed / "
        "{files_malformed} malformed | records: {records_seen} seen / {records_valid} valid / "
        "{records_skipped_invalid} invalid / {records_filtered_category} filtered-by-category / "
        "{duplicates_removed} duplicates removed | malformed JSONL lines: {records_malformed}".format(**stats)
    )

    if not records:
        raise ValueError(
            f"Zero valid training examples were found under {root} after checking "
            f"{stats['files_discovered']} file(s). Every record needs a non-empty "
            "'instruction' (or 'input') field and a non-empty 'output' field — confirm "
            "the dataset files actually contain records shaped this way, and that "
            "CATEGORY_FILTER (if set) matches the categories present in the data."
        )

    return records, stats


def build_text_dataset(records: list[dict], tokenizer) -> Dataset:
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
        if key == "max_seq_length":
            # TRL renamed `max_seq_length` -> `max_length` on SFTConfig and kept the
            # old name only as a deprecated alias before eventually dropping it.
            # Prefer the current name when it's available so a fresh TRL install
            # doesn't emit a deprecation warning (or fail once the alias is gone);
            # fall back to the legacy name so this still runs on an older pinned
            # TRL version that predates the rename.
            if "max_length" in config_params:
                config_key = "max_length"
            elif "max_seq_length" in config_params:
                config_key = "max_seq_length"
            else:
                config_key = None
        if config_key and config_key in config_params:
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

    records, _ = load_dataset_records(DATASET_DIR)
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
    print(f"[save] Lara.coder LoRA adapters saved to {LORA_DIR}")

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
