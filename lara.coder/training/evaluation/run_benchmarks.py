"""Runs a trained/adapted model against `evaluator/benchmarks/` tasks
end to end (through the real agent loop) and reports
`evaluator.metrics.QualityReport` scores, so model changes can be
compared objectively rather than by spot-checking outputs."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "evaluator" / "benchmarks"


async def run_all(benchmarks_dir: Path = BENCHMARKS_DIR) -> list[dict]:
    results = []
    for path in sorted(benchmarks_dir.glob("*.json")):
        benchmark = json.loads(path.read_text())
        # Layer-0 placeholder: this should build an AgentState from
        # benchmark["prompt"], run agent.loop.AgentLoop to completion, and
        # score the result against benchmark["expected_acceptance_criteria"]
        # via evaluator.metrics.evaluate_run.
        results.append({"id": benchmark["id"], "status": "not_yet_runnable_at_layer_0"})
    return results


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(json.dumps(asyncio.run(run_all()), indent=2))
