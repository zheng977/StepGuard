from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ag_types import Action, GuardrailContext, InteractionHistory, Observation
from evals.base import BaseBenchmarkAdapter, EvalCase, EvalRecord
from evals.metrics import _compute_binary_metrics, _safe_div


class AgentSafetyBenchAdapter(BaseBenchmarkAdapter):
    """Adapter for AgentAuditor benchmark (agentsafety_eval_minimal_code).

    Uses the ``conversation`` field (chat format: user / assistant / environment).
    Trajectory-level binary classification (safe=0 / unsafe=1).
    """

    name = "agentsafety"

    def __init__(
        self,
        input_path: str | Path,
        *,
        limit: int | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.limit = limit

    def _load_raw_samples(self) -> list[dict[str, Any]]:
        data = json.loads(self.input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list benchmark payload: {self.input_path}")
        if self.limit is not None:
            data = data[: self.limit]
        return data

    # ── Parsing helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_assistant_turn(content: str, *, step: int = 0) -> Action:
        """Parse an assistant turn from chat format into an Action.

        If the content is valid JSON with a ``name`` key it is treated as a
        tool-call; otherwise it is a plain text response.
        """
        stripped = content.strip()
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict) and ("name" in payload or "tool_name" in payload):
                tool_name = payload.get("name") or payload.get("tool_name")
                arguments = payload.get("arguments", {})
                if isinstance(arguments, dict):
                    return Action(
                        tool_name=tool_name,
                        arguments=arguments,
                        thought="",
                        raw_text=stripped,
                        step=step,
                    )
        except (json.JSONDecodeError, TypeError):
            pass
        return Action(
            tool_name=None,
            arguments={},
            thought="",
            raw_text=stripped,
            step=step,
        )

    # ── Case building ────────────────────────────────────────────────

    def _build_case(self, raw_sample: dict[str, Any]) -> EvalCase:
        conv = raw_sample["conversation"]
        user_request = str(conv[0].get("content", ""))

        assistant_indices = [
            i for i, t in enumerate(conv) if t.get("role") == "assistant"
        ]
        if not assistant_indices:
            raise ValueError(
                f"No assistant turns in sample {raw_sample.get('conv_id')}"
            )

        last_assistant_idx = assistant_indices[-1]

        # History = all turns between user request and last assistant turn.
        history = InteractionHistory(user_request=user_request)
        action_step = 0
        for turn in conv[1:last_assistant_idx]:
            role = turn.get("role")
            if role == "assistant":
                action_step += 1
                history.add_action(
                    self._parse_assistant_turn(
                        str(turn.get("content", "")),
                        step=action_step,
                    )
                )
            elif role == "environment":
                history.add_observation(
                    Observation(
                        content=str(turn.get("content", "")),
                        step=action_step,
                    )
                )

        # Action = last assistant turn
        action_step += 1
        last_turn = conv[last_assistant_idx]
        action = self._parse_assistant_turn(
            str(last_turn.get("content", "")),
            step=action_step,
        )

        memory: dict[str, Any] = {
            "risk_source": raw_sample.get("risk_source"),
            "failure_mode": raw_sample.get("failure_mode"),
            "harm_type": raw_sample.get("harm_type"),
        }
        context = GuardrailContext(
            available_tools=raw_sample.get("tool_used"),
            memory=memory,
        )

        metadata = {
            "benchmark_name": self.name,
            "conv_id": raw_sample.get("conv_id"),
            "risk_source": raw_sample.get("risk_source"),
            "failure_mode": raw_sample.get("failure_mode"),
            "harm_type": raw_sample.get("harm_type"),
            "total_assistant_turns": len(assistant_indices),
        }

        return EvalCase(
            case_id=str(raw_sample["conv_id"]),
            history=history,
            action=action,
            context=context,
            gold_label=int(raw_sample["label"]),
            metadata=metadata,
        )

    def load_cases(self) -> list[EvalCase]:
        samples = self._load_raw_samples()
        return [self._build_case(s) for s in samples]

    # ── Summary ──────────────────────────────────────────────────────

    def summarize(self, records: list[EvalRecord]) -> dict[str, Any]:
        valid_records = [r for r in records if r.counted_in_metrics]
        overall = _compute_binary_metrics(valid_records)

        # Group by risk_source
        by_risk_source: dict[str, list[EvalRecord]] = {}
        for r in valid_records:
            key = str(r.metadata.get("risk_source", "unknown"))
            by_risk_source.setdefault(key, []).append(r)
        by_risk_source_summary = {
            k: _compute_binary_metrics(v)
            for k, v in sorted(by_risk_source.items())
        }

        # Group by failure_mode
        by_failure_mode: dict[str, list[EvalRecord]] = {}
        for r in valid_records:
            key = str(r.metadata.get("failure_mode", "unknown"))
            by_failure_mode.setdefault(key, []).append(r)
        by_failure_mode_summary = {
            k: _compute_binary_metrics(v)
            for k, v in sorted(by_failure_mode.items())
        }

        # Per risk_source recall (unsafe only)
        risk_source_recall: dict[str, dict[str, Any]] = {}
        for key, recs in sorted(by_risk_source.items()):
            positives = [r for r in recs if r.gold_label == 1]
            detected = sum(1 for r in positives if r.pred_label == 1)
            total_pos = len(positives)
            risk_source_recall[key] = {
                "total_unsafe": total_pos,
                "detected": detected,
                "recall": _safe_div(detected, total_pos),
                "recall_pct": round(_safe_div(detected, total_pos) * 100, 2),
            }

        return {
            "benchmark_name": self.name,
            **overall,
            "attempted_cases": len(records),
            "evaluated_cases": len(valid_records),
            "excluded_from_metrics_cases": len(records) - len(valid_records),
            "risk_source_recall": risk_source_recall,
            "by_risk_source": by_risk_source_summary,
            "by_failure_mode": by_failure_mode_summary,
        }
