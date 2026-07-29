"""Adapter for evaluating on training data (JSONL with pre-built prompts).

Data format: JSONL with fields:
  - instruction: complete prompt (already includes agentguard template)
  - output: "safe" or "unsafe"
  - id: sample identifier
  - level: "trajectory" or "action"

The instruction is sent as-is via passthrough prompt profile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ag_types import Action, GuardrailContext, InteractionHistory
from evals.base import BaseBenchmarkAdapter, EvalCase, EvalRecord
from evals.metrics import _compute_binary_metrics


class TrainSetBenchAdapter(BaseBenchmarkAdapter):
    """Evaluate model on its own training data to measure RL learning room."""

    name = "trainset"

    def __init__(
        self,
        input_path: str | Path,
        *,
        limit: int | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.limit = limit

    def load_cases(self) -> list[EvalCase]:
        cases: list[EvalCase] = []
        with self.input_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.limit is not None and i >= self.limit:
                    break
                item = json.loads(line)
                instruction = item["instruction"]
                label_str = item["output"].strip().lower()
                gold_label = 1 if label_str == "unsafe" else 0

                history = InteractionHistory(
                    user_request=instruction,
                    steps=[],
                )
                action = Action(
                    tool_name=None, arguments={}, thought="", raw_text="", step=0,
                )
                context = GuardrailContext(
                    memory={"full_prompt": instruction},
                )
                metadata = _case_metadata(item, label_str=label_str)
                cases.append(
                    EvalCase(
                        case_id=item.get("id", f"train_{i:05d}"),
                        history=history,
                        action=action,
                        context=context,
                        gold_label=gold_label,
                        metadata=metadata,
                    )
                )
        return cases

    def summarize(self, records: list[EvalRecord]) -> dict[str, Any]:
        valid = [r for r in records if r.counted_in_metrics]
        overall = _compute_binary_metrics(valid)

        grouped = {
            "by_level": _summarize_by_metadata(valid, "level"),
            "by_split": _summarize_by_metadata(valid, "split"),
            "by_source_kind": _summarize_by_metadata(valid, "source_kind"),
            "by_sampling_supergroup": _summarize_by_metadata(valid, "sampling_supergroup"),
            "by_sampling_family": _summarize_by_metadata(valid, "sampling_family"),
            "by_sample_type": _summarize_by_metadata(valid, "sample_type"),
            "by_risk_source": _summarize_by_metadata(valid, "risk_source"),
            "by_failure_mode": _summarize_by_metadata(valid, "failure_mode"),
            "by_harm_type": _summarize_by_metadata(valid, "harm_type"),
            "by_real_world_harm": _summarize_by_metadata(valid, "real_world_harm"),
        }

        return {
            "benchmark_name": self.name,
            **overall,
            "attempted_cases": len(records),
            "evaluated_cases": len(valid),
            "excluded_from_metrics_cases": len(records) - len(valid),
            **grouped,
        }


def _case_metadata(item: dict[str, Any], *, label_str: str) -> dict[str, Any]:
    meta = item.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    fields = [
        "level",
        "split",
        "source",
        "source_family",
        "source_kind",
        "origin",
        "sample_type",
        "sampling_supergroup",
        "sampling_family",
        "mix_category",
        "composition_group",
        "risk_source",
        "failure_mode",
        "harm_type",
        "real_world_harm",
        "action_kind",
        "pair_id",
        "group_id",
        "label_source",
        "prompt_name",
        "quality_score",
        "quality_training_value",
        "quality_reason",
        "paper_pool_source",
        "paper_original_id",
        "paper_source_id",
        "paper_case_hash",
        "is_hard_case",
        "tool_name",
        "action_idx",
        "turn_idx",
        "raw_char_count",
        "defense_type",
        "safe_branch_mode",
    ]
    metadata: dict[str, Any] = {"gold_label_str": label_str}
    for field in fields:
        value = item.get(field, meta.get(field))
        if value is not None:
            metadata[field] = value
    metadata.setdefault("level", "unknown")
    return metadata


def _summarize_by_metadata(records: list[EvalRecord], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        value = str(record.metadata.get(key, "unknown"))
        grouped.setdefault(value, []).append(record)
    return {
        value: _compute_binary_metrics(group_records)
        for value, group_records in sorted(grouped.items())
    }
