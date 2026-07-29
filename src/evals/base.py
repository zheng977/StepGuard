from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ag_types import Action, GuardrailContext, InteractionHistory, Judgment


class EvalCase(BaseModel):
    case_id: str
    history: InteractionHistory
    action: Action
    context: GuardrailContext | None = None
    gold_label: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRecord(BaseModel):
    case_id: str
    gold_label: int
    pred_label: int
    judgment: Judgment
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    error: str | None = None
    counted_in_metrics: bool = True

    @property
    def is_success(self) -> bool:
        return self.status == "success"


class BaseBenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def load_cases(self) -> list[EvalCase]:
        raise NotImplementedError

    def summarize(self, records: list[EvalRecord]) -> dict[str, Any]:
        """Default summarize using binary metrics.

        Subclasses may override for richer reporting.
        """
        from evals.metrics import _compute_binary_metrics

        valid_records = [r for r in records if r.counted_in_metrics]
        summary = _compute_binary_metrics(valid_records)
        summary["benchmark_name"] = self.name
        summary["evaluated_cases"] = len(valid_records)
        summary["excluded_from_metrics_cases"] = len(records) - len(valid_records)
        return summary
