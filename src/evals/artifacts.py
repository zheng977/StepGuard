from __future__ import annotations

from typing import Any

from ag_types import Judgment
from evals.base import EvalCase, EvalRecord


def build_failure_record(case: EvalCase, *, status: str, error: str) -> EvalRecord:
    return EvalRecord(
        case_id=case.case_id,
        gold_label=int(case.gold_label),
        pred_label=0,
        judgment=Judgment(
            label=0,
            confidence=0.0,
            reason="",
            metadata={"judgment_parse_status": status},
        ),
        metadata=dict(case.metadata),
        status=status,
        error=error,
        counted_in_metrics=False,
    )


def finalize_static_summary(
    summary: dict[str, Any],
    records: list[EvalRecord],
    *,
    attempted_cases: int | None = None,
) -> dict[str, Any]:
    """Attach run-level counters to a benchmark summary.

    Benchmark adapters own metric semantics. This helper only normalizes
    artifact-level counters so every static eval run reports failure handling
    consistently.
    """
    attempted = len(records) if attempted_cases is None else attempted_cases
    completed = sum(1 for record in records if record.is_success)
    failed = sum(1 for record in records if not record.is_success)
    evaluated = sum(1 for record in records if record.counted_in_metrics)

    payload = dict(summary)
    payload["attempted_cases"] = payload.get("attempted_cases", attempted)
    payload["evaluated_cases"] = payload.get("evaluated_cases", evaluated)
    payload["completed_cases"] = completed
    payload["failed_cases"] = failed
    payload["excluded_failed_cases_from_metrics"] = True
    return payload
