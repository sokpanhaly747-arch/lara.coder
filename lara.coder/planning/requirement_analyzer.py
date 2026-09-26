"""Turns a raw user idea into structured Requirements.

Deliberately surfaces ambiguities rather than silently guessing on
anything that materially changes scope (auth? payments? multi-tenant?).
Each ambiguity carries a stated default so the pipeline can proceed
without blocking on the user, while remaining visible for review.
"""
from __future__ import annotations

from models.adapter import ModelAdapter
from models.inference import structured_complete
from planning.schemas import Ambiguity, Requirements

SYSTEM_PROMPT = """You are a senior product analyst. Given a user's product idea,
extract structured requirements. Identify anything genuinely ambiguous
(e.g. "does this need user accounts?") rather than assuming silently,
and propose a sensible default for each so work isn't blocked.
Return JSON: {summary, goals[], target_users[], must_have_features[],
nice_to_have_features[], constraints[], ambiguities[{question, default_assumption}]}"""


def _parse(data: dict) -> Requirements:
    return Requirements(
        summary=data["summary"],
        goals=data.get("goals", []),
        target_users=data.get("target_users", []),
        must_have_features=data.get("must_have_features", []),
        nice_to_have_features=data.get("nice_to_have_features", []),
        constraints=data.get("constraints", []),
        ambiguities=[Ambiguity(**a) for a in data.get("ambiguities", [])],
    )


async def analyze(idea: str, adapter: ModelAdapter) -> Requirements:
    return await structured_complete(adapter, prompt=idea, parse=_parse, system=SYSTEM_PROMPT)
