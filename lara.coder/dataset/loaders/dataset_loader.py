"""Loads, validates, and splits training examples for `training/`.

Validates each example against `dataset/schema/example_schema.json` so
malformed data fails fast at load time rather than surfacing as a
confusing training-time error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import jsonschema

SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "example_schema.json"


@dataclass
class TrainingExample:
    id: str
    category: str
    input: str
    output: str
    preference_rejected_output: str | None = None
    metadata: dict | None = None


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def load_examples(examples_dir: str | Path) -> Iterator[TrainingExample]:
    schema = _load_schema()
    for path in sorted(Path(examples_dir).glob("*.json")):
        data = json.loads(path.read_text())
        jsonschema.validate(data, schema)
        yield TrainingExample(**data)


def split(examples: list[TrainingExample], train_ratio: float = 0.9):
    cutoff = int(len(examples) * train_ratio)
    return examples[:cutoff], examples[cutoff:]
