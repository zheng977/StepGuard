"""AgentDyn trajectory-based static benchmark.

Reads a pre-built JSON dataset (benchmarks/agentdyn/test.json) containing
agent trajectories with trajectory-level labels.

Label scheme:
  - 0 (safe): no-attack trace, utility=True and security=True
  - 1 (unsafe): attack trace, security=False (agent was hijacked)

Each case becomes one EvalCase with the full trajectory as history and the
last tool_call as the candidate action.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ag_types import Action, GuardrailContext, InteractionHistory, Observation
from evals.base import BaseBenchmarkAdapter, EvalCase, EvalRecord
from evals.metrics import _compute_binary_metrics, _safe_div

logger = logging.getLogger(__name__)


class AgentDynBenchAdapter(BaseBenchmarkAdapter):
    """Static benchmark adapter for AgentDyn trajectory dataset."""

    name = "agentdyn_bench"

    def __init__(
        self,
        input_path: str | Path,
        *,
        limit: int | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.limit = limit

    def load_cases(self) -> list[EvalCase]:
        with self.input_path.open() as f:
            dataset = json.load(f)

        cases: list[EvalCase] = []
        for sample in dataset:
            case = self._build_case(sample)
            if case is not None:
                cases.append(case)

        logger.info(
            "Loaded %d cases: %d safe, %d unsafe",
            len(cases),
            sum(1 for c in cases if c.gold_label == 0),
            sum(1 for c in cases if c.gold_label == 1),
        )

        if self.limit is not None:
            cases = cases[: self.limit]
        return cases

    def _build_case(self, sample: dict[str, Any]) -> EvalCase | None:
        trajectory = sample.get("trajectory", [])
        if not trajectory:
            return None

        # Build history
        history = InteractionHistory(
            user_request=sample.get("user_request", ""),
            initial_state=sample.get("system_prompt", ""),
        )

        # Collect all actions and observations; last action becomes candidate
        last_action: Action | None = None
        last_action_step = -1
        step = 0

        for entry in trajectory:
            entry_type = entry.get("type")

            if entry_type == "action":
                action = Action(
                    tool_name=entry.get("tool_name", ""),
                    arguments=dict(entry.get("arguments", {})),
                    thought=entry.get("thought", ""),
                    raw_text=f"{entry.get('tool_name', '')}({json.dumps(entry.get('arguments', {}), ensure_ascii=False)})",
                    step=step,
                )
                last_action = action
                last_action_step = step
                step += 1

            elif entry_type == "observation":
                history.add_observation(Observation(
                    content=entry.get("content", ""),
                    step=step,
                ))

            elif entry_type == "thought":
                # Standalone thought without tool_call — add as action with no tool
                history.add_action(Action(
                    tool_name=None,
                    thought=entry.get("content", ""),
                    raw_text=entry.get("content", ""),
                    step=step,
                ))
                step += 1

        if last_action is None:
            return None

        # Add all actions BEFORE the last one to history
        step = 0
        for entry in trajectory:
            if entry.get("type") != "action":
                continue
            if step >= last_action_step:
                break
            history.add_action(Action(
                tool_name=entry.get("tool_name", ""),
                arguments=dict(entry.get("arguments", {})),
                thought=entry.get("thought", ""),
                raw_text=f"{entry.get('tool_name', '')}({json.dumps(entry.get('arguments', {}), ensure_ascii=False)})",
                step=step,
            ))
            step += 1

        metadata = sample.get("metadata", {})
        metadata["benchmark_name"] = self.name

        # Support pre-computed analysis for two-stage pipeline
        analysis = sample.get("analysis", "")
        context = GuardrailContext(
            memory={"stage1_analysis": analysis} if analysis else None,
        )

        return EvalCase(
            case_id=sample.get("case_id", "unknown"),
            history=history,
            action=last_action,
            context=context,
            gold_label=int(sample.get("label", 0)),
            metadata=metadata,
        )

    def summarize(self, records: list[EvalRecord]) -> dict[str, Any]:
        valid = [r for r in records if r.counted_in_metrics]
        overall = _compute_binary_metrics(valid)

        # Per-suite breakdown
        by_suite: dict[str, list[EvalRecord]] = {}
        for r in valid:
            s = str(r.metadata.get("suite", "unknown"))
            by_suite.setdefault(s, []).append(r)

        suite_summary = {}
        for s, recs in sorted(by_suite.items()):
            suite_summary[s] = _compute_binary_metrics(recs)

        return {
            "benchmark_name": self.name,
            **overall,
            "attempted_cases": len(records),
            "evaluated_cases": len(valid),
            "excluded_from_metrics_cases": len(records) - len(valid),
            "by_suite": suite_summary,
        }
