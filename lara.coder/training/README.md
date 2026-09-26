# training/

Experiments to improve the underlying model. Compatible with limited
GPU resources — nothing here assumes large-cluster infrastructure.
The base model is swappable (see `models/adapter.py`); this folder
only ever produces adapters/checkpoints, never hardcodes the target
model architecture beyond config.

- `configs/` — YAML configs per experiment (SFT, DPO, ...).
- `sft/` — supervised fine-tuning / LoRA / QLoRA entrypoints.
- `dpo/` — preference optimization entrypoints.
- `evaluation/` — post-training eval against `evaluator/benchmarks`.
