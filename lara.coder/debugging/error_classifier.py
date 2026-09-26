"""Categorizes a raw error/stack trace before root-causing it.

Cheap, mostly rule-based classification lets the router (models.router)
send easy cases (a missing import, a typo'd identifier) to a fast path
and reserve the strongest model for genuinely confusing failures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(str, Enum):
    SYNTAX = "syntax"
    TYPE = "type"
    MISSING_DEPENDENCY = "missing_dependency"
    RUNTIME = "runtime"
    LOGIC = "logic"
    CONFIG = "config"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    category: ErrorCategory
    message: str
    file_hint: str | None = None
    confidence: float = 0.5


_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"SyntaxError|Unexpected token", re.I), ErrorCategory.SYNTAX),
    (re.compile(r"TypeError|is not a function|is not assignable", re.I), ErrorCategory.TYPE),
    (re.compile(r"Cannot find module|Module not found", re.I), ErrorCategory.MISSING_DEPENDENCY),
    (re.compile(r"ECONNREFUSED|ENOENT|permission denied", re.I), ErrorCategory.CONFIG),
]


def classify(raw_output: str) -> ClassifiedError:
    for pattern, category in _PATTERNS:
        if pattern.search(raw_output):
            file_match = re.search(r"([./\w-]+\.(tsx?|jsx?|json))", raw_output)
            return ClassifiedError(
                category=category,
                message=raw_output[:500],
                file_hint=file_match.group(1) if file_match else None,
                confidence=0.8,
            )
    return ClassifiedError(category=ErrorCategory.UNKNOWN, message=raw_output[:500], confidence=0.2)
