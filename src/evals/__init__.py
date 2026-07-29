from __future__ import annotations

from typing import Any

__all__ = [
    "BaseBenchmarkAdapter",
    "EvalCase",
    "EvalRecord",
    "Evaluator",
    "ResultWriter",
    "build_failure_record",
    "finalize_static_summary",
]


def __getattr__(name: str) -> Any:
    if name in {"BaseBenchmarkAdapter", "EvalCase", "EvalRecord"}:
        from .base import BaseBenchmarkAdapter, EvalCase, EvalRecord

        return {
            "BaseBenchmarkAdapter": BaseBenchmarkAdapter,
            "EvalCase": EvalCase,
            "EvalRecord": EvalRecord,
        }[name]
    if name == "Evaluator":
        from .evaluator import Evaluator

        return Evaluator
    if name == "ResultWriter":
        from .results import ResultWriter

        return ResultWriter
    if name in {"build_failure_record", "finalize_static_summary"}:
        from .artifacts import build_failure_record, finalize_static_summary

        return {
            "build_failure_record": build_failure_record,
            "finalize_static_summary": finalize_static_summary,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
