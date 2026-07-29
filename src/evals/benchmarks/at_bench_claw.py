"""ATBench-Claw: OpenClaw-oriented Agent Trajectory Safety Benchmark.

Data format: OpenClaw session format with events (message/toolCall/toolResult).
Labels: labels.is_safe (bool), labels.risk_source, labels.failure_mode, labels.harm_type.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ag_types import Action, GuardrailContext, InteractionHistory, Observation
from evals.base import BaseBenchmarkAdapter, EvalCase, EvalRecord
from evals.metrics import _compute_binary_metrics, _safe_div


class ATBenchClawAdapter(BaseBenchmarkAdapter):
    """Adapter for ATBench-Claw (OpenClaw session format → trajectory-level binary)."""

    name = "at_bench_claw"

    def __init__(
        self,
        input_path: str | Path,
        *,
        limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.input_path = Path(input_path)
        self.limit = limit

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        """Extract text from OpenClaw content (list of parts or plain string)."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        texts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "text":
                texts.append(part.get("text", ""))
            elif ptype == "thinking":
                t = part.get("text", "")
                if t:
                    texts.append(f"[THINKING]: {t}")
        return "\n".join(texts)

    @staticmethod
    def _extract_tool_calls(content: Any) -> list[dict[str, Any]]:
        """Extract tool calls from assistant message content."""
        if not isinstance(content, list):
            return []
        calls: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "toolCall":
                calls.append({
                    "name": part.get("name", ""),
                    "input": part.get("input", {}),
                })
        return calls

    def _build_case(self, raw_sample: dict[str, Any], idx: int) -> EvalCase:
        events = raw_sample["trajectory"]["events"]
        labels = raw_sample["labels"]

        # Parse events into history + final action
        user_request = ""
        history = InteractionHistory(user_request="")
        action_step = 0
        last_action: Action | None = None

        for event in events:
            msg = event.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                # Extract user request (skip long skill context, take actual request)
                if isinstance(content, list):
                    # Last text part is usually the actual request
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    if len(text_parts) > 1:
                        user_request = text_parts[-1]  # actual request
                    elif text_parts:
                        user_request = text_parts[0]
                else:
                    user_request = str(content)
                history.user_request = user_request

            elif role == "assistant":
                tool_calls = self._extract_tool_calls(content)
                text = self._extract_text_from_content(content)

                if tool_calls:
                    for tc in tool_calls:
                        action_step += 1
                        act = Action(
                            tool_name=tc["name"],
                            arguments=tc["input"] if isinstance(tc["input"], dict) else {},
                            thought=text,
                            raw_text=json.dumps(tc, ensure_ascii=False),
                            step=action_step,
                        )
                        # Push previous action to history, keep latest as candidate
                        if last_action is not None:
                            history.add_action(last_action)
                        last_action = act
                elif text.strip():
                    # Assistant text-only response (final summary)
                    action_step += 1
                    act = Action(
                        tool_name=None,
                        arguments={},
                        thought="",
                        raw_text=text.strip(),
                        step=action_step,
                    )
                    if last_action is not None:
                        history.add_action(last_action)
                    last_action = act

            elif role == "toolResult":
                result_text = self._extract_text_from_content(content)
                if result_text:
                    history.add_observation(Observation(
                        content=result_text,
                        step=action_step,
                    ))

        # If no action was extracted, create a placeholder
        if last_action is None:
            last_action = Action(
                tool_name=None, arguments={}, thought="", raw_text="", step=0
            )

        # Gold label: is_safe=True → 0 (safe), is_safe=False → 1 (unsafe)
        gold_label = 0 if labels.get("is_safe", True) else 1

        metadata = {
            "benchmark_name": self.name,
            "sample_idx": idx,
            "risk_source": labels.get("risk_source", ""),
            "failure_mode": labels.get("failure_mode", ""),
            "harm_type": labels.get("harm_type", ""),
            "defense_type": labels.get("defense_type"),
            "reason": raw_sample.get("reason", ""),
        }

        context = GuardrailContext(
            available_tools=[],
            memory={
                "risk_source": labels.get("risk_source", ""),
                "failure_mode": labels.get("failure_mode", ""),
                "harm_type": labels.get("harm_type", ""),
            },
        )

        return EvalCase(
            case_id=f"claw_{idx}",
            history=history,
            action=last_action,
            context=context,
            gold_label=gold_label,
            metadata=metadata,
        )

    def load_cases(self) -> list[EvalCase]:
        raw = self.input_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload: {self.input_path}")
        if self.limit is not None:
            payload = payload[: self.limit]
        return [self._build_case(s, i) for i, s in enumerate(payload)]

    def summarize(self, records: list[EvalRecord]) -> dict[str, Any]:
        valid_records = [r for r in records if r.counted_in_metrics]
        overall = _compute_binary_metrics(valid_records)

        # Group by risk_source
        by_risk: dict[str, list[EvalRecord]] = {}
        for r in valid_records:
            key = str(r.metadata.get("risk_source", "unknown"))
            by_risk.setdefault(key, []).append(r)

        risk_source_recall: dict[str, dict[str, Any]] = {}
        for key, recs in sorted(by_risk.items()):
            positives = [r for r in recs if r.gold_label == 1]
            detected = sum(1 for r in positives if r.pred_label == 1)
            total_pos = len(positives)
            risk_source_recall[key] = {
                "total_unsafe": total_pos,
                "detected": detected,
                "recall": _safe_div(detected, total_pos),
                "recall_pct": round(_safe_div(detected, total_pos) * 100, 2),
            }

        by_risk_summary = {
            k: _compute_binary_metrics(v) for k, v in sorted(by_risk.items())
        }

        return {
            "benchmark_name": self.name,
            **overall,
            "attempted_cases": len(records),
            "evaluated_cases": len(valid_records),
            "excluded_from_metrics_cases": len(records) - len(valid_records),
            "risk_source_recall": risk_source_recall,
            "by_risk_source": by_risk_summary,
        }
