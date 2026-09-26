"""DPO preference-optimization entrypoint.

Consumes `preference` category examples from `dataset/` (each with an
`output` (chosen) and `preference_rejected_output` (rejected)).
Same interface-stub status as `sft/train.py` at layer-0.
"""
from __future__ import annotations

import argparse

from dataset.loaders.dataset_loader import load_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DPO training job")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()

    examples = [e for e in load_examples(args.dataset_dir) if e.category == "preference"]
    print(f"Loaded {len(examples)} preference pairs")

    raise NotImplementedError("DPO loop not yet implemented — interface stub for a later layer.")


if __name__ == "__main__":
    main()
