from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from ag_types import Action, GuardrailContext, InteractionHistory, Observation
from evals.base import BaseBenchmarkAdapter, EvalCase, EvalRecord
from evals.metrics import _compute_binary_metrics


# Regex: ToolName{...} or ToolName({...})
_TOOL_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\s*\(?\s*(\{.*\})\s*\)?\s*$", re.DOTALL)

# Regex: ToolName\nAction Input: {...}
_ACTION_INPUT_RE = re.compile(
    r"^([A-Za-z_]\w*)\s*\n\s*Action\s*Input:\s*(.+)",
    re.DOTALL,
)


def _parse_generic_action(action_text: str, *, thought: str = "", step: int = 0) -> Action:
    """Parse diverse action formats found in rjudge / agentharm data.

    Supported formats (in order of precedence):
      1. ``Complete{...}``
      2. ``Final Answer: ...``
      3. Valid JSON object with ``name`` / ``tool_name`` key
      4. ``ToolName\\nAction Input: {json_args}``
      5. ``ToolName{python_dict_args}`` or ``ToolName({python_dict_args})``
      6. Fallback: raw text, no tool_name
    """
    normalized = action_text.strip()

    # 1. Complete
    if normalized.startswith("Complete"):
        return Action(tool_name=None, arguments={}, thought=thought, raw_text=normalized, step=step)

    # 2. Final Answer
    if normalized.lower().startswith("final answer"):
        return Action(tool_name=None, arguments={}, thought=thought, raw_text=normalized, step=step)

    # 3. JSON tool call
    try:
        payload = json.loads(normalized)
        if isinstance(payload, dict):
            tool_name = payload.get("name") or payload.get("tool_name")
            arguments = payload.get("arguments", {})
            if isinstance(arguments, dict):
                return Action(
                    tool_name=tool_name,
                    arguments=arguments,
                    thought=thought,
                    raw_text=normalized,
                    step=step,
                )
    except (json.JSONDecodeError, TypeError):
        pass

    # 4. ToolName\nAction Input: {args}
    m = _ACTION_INPUT_RE.match(normalized)
    if m:
        tool_name = m.group(1)
        args_str = m.group(2).strip()
        arguments = _try_parse_args(args_str)
        return Action(
            tool_name=tool_name,
            arguments=arguments,
            thought=thought,
            raw_text=normalized,
            step=step,
        )

    # 5. ToolName{args} or ToolName({args})
    m = _TOOL_CALL_RE.match(normalized)
    if m:
        tool_name = m.group(1)
        args_str = m.group(2).strip()
        arguments = _try_parse_args(args_str)
        return Action(
            tool_name=tool_name,
            arguments=arguments,
            thought=thought,
            raw_text=normalized,
            step=step,
        )

    # 6. Fallback
    return Action(tool_name=None, arguments={}, thought=thought, raw_text=normalized, step=step)


def _try_parse_args(args_str: str) -> dict[str, Any]:
    """Try to parse arguments from JSON or Python dict literal."""
    # JSON first
    try:
        result = json.loads(args_str)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    # Python dict literal (safe)
    try:
        result = ast.literal_eval(args_str)
        if isinstance(result, dict):
            return {str(k): v for k, v in result.items()}
    except (ValueError, SyntaxError):
        pass
    return {}


class AgentHarmTrajBenchAdapter(BaseBenchmarkAdapter):
    """Adapter for trajectory benchmarks using AT-Bench-like ``contents`` format.

    Works with AgentHarm, R-Judge, and similar datasets that store trajectories
    as ``contents: [[{role, thought, action / content}, ...]]`` with labels.

    Maps ``id`` -> ``conv_id``.  Trajectory-level binary classification.
    """

    name = "agentharm_traj"

    def __init__(
        self,
        input_path: str | Path,
        *,
        limit: int | None = None,
        bench_name: str | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.limit = limit
        if bench_name is not None:
            self.name = bench_name

    def _load_raw_samples(self) -> list[dict[str, Any]]:
        data = json.loads(self.input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list benchmark payload: {self.input_path}")
        if self.limit is not None:
            data = data[: self.limit]
        return data

    # ── Case building ────────────────────────────────────────────────

    def _build_case(self, raw_sample: dict[str, Any]) -> EvalCase:
        conv = raw_sample["contents"][0]
        user_request = str(conv[0].get("content", ""))

        # Find last agent turn
        last_agent_idx = -1
        for i in range(len(conv) - 1, -1, -1):
            if conv[i].get("role") == "agent":
                last_agent_idx = i
                break

        if last_agent_idx < 0:
            raise ValueError(
                f"No agent turns in sample {raw_sample.get('id')}"
            )

        # History = all turns between user and last agent turn
        history = InteractionHistory(user_request=user_request)
        action_step = 0
        for turn in conv[1:last_agent_idx]:
            role = turn.get("role")
            if role == "agent":
                action_step += 1
                history.add_action(
                    _parse_generic_action(
                        str(turn.get("action", "")),
                        thought=str(turn.get("thought", "")),
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

        # Action = last agent turn
        action_step += 1
        last_turn = conv[last_agent_idx]
        action = _parse_generic_action(
            str(last_turn.get("action", "")),
            thought=str(last_turn.get("thought", "")),
            step=action_step,
        )

        sample_id = str(raw_sample.get("id") or raw_sample.get("conv_id", ""))

        context = GuardrailContext(
            available_tools=None,
            memory={
                "risk_description": raw_sample.get("risk_description", ""),
            },
        )

        agent_count = sum(1 for t in conv if t.get("role") == "agent")
        metadata = {
            "benchmark_name": self.name,
            "conv_id": sample_id,
            "risk_description": raw_sample.get("risk_description"),
            "total_agent_turns": agent_count,
        }

        return EvalCase(
            case_id=sample_id,
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
        return {
            "benchmark_name": self.name,
            **overall,
            "attempted_cases": len(records),
            "evaluated_cases": len(valid_records),
            "excluded_from_metrics_cases": len(records) - len(valid_records),
        }
