"""SFT/LoRA/QLoRA training entrypoint.

Layer-0: CLI skeleton that loads a config and the dataset, and defines
where a real HF `transformers` + `peft` training loop plugs in. Kept
separate from `dataset/` and `models/` so this can be run offline
without pulling training deps into the runtime agent path.
"""
from __future__ import annotations

import argparse

import yaml

from dataset.loaders.dataset_loader import load_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an SFT/LoRA/QLoRA training job")
    parser.add_argument("--config", required=True, help="Path to a training YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    examples = list(load_examples(config["dataset"]["path"]))
    print(f"Loaded {len(examples)} training examples for base model {config['base_model']}")

    # Layer-0 placeholder: wire up `transformers.Trainer` + `peft.LoraConfig`
    # here once training dependencies are added. Left as an explicit
    # interface point rather than a fake implementation.
    raise NotImplementedError(
        "Training loop not yet implemented — this is an interface stub. "
        "Plug in transformers/peft here per configs/sft_config.yaml."
    )


if __name__ == "__main__":
    main()
