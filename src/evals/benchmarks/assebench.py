from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ag_types import Action, InteractionHistory, Observation
from evals.base import BaseBenchmarkAdapter, EvalCase


class ASSEBenchAdapter(BaseBenchmarkAdapter):
    """Adapter for ASSEBench (AgentAuditor-ASSEBench).

    Schema (per sample):
      - id: int
      - profile: str (agent system prompt)
      - contents: list[list[message]]  # outer list usually len=1
        message roles:
          user        -> {role, content}
          agent       -> {role, thought, action}
          environment -> {role, content}
      - label: 0=safe, 1=unsafe
      - application_scenario, risk_type, failure_mode, ambiguous,
        risk_description  (metadata)

    Modes:
      - "trajectory" (default): use the last agent turn as the action,
        everything before it as history. Binary classification over
        full dialog.
      - "last_action": same structure (ASSEBench traces don't have an
        explicit Complete terminator, so we pick the last agent turn
        as the candidate action in either mode).
    """

    name = "assebench"

    def __init__(
        self,
        input_path: str | Path,
        *,
        limit: int | None = None,
        mode: str = "trajectory",
        skip_ambiguous: bool = True,
    ) -> None:
        self.input_path = Path(input_path)
        self.limit = limit
        if mode not in ("trajectory", "last_action"):
            raise ValueError(
                f'Unknown ASSEBench mode "{mode}". '
                'Expected "trajectory" or "last_action".'
            )
        self.mode = mode
        self.skip_ambiguous = skip_ambiguous

    def _load_raw_samples(self) -> list[dict[str, Any]]:
        payload = json.loads(self.input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload: {self.input_path}")
        if self.skip_ambiguous:
            payload = [x for x in payload if int(x.get("ambiguous", 0)) == 0]
        if self.limit is not None:
            return payload[: self.limit]
        return payload

    @staticmethod
    def _parse_action(
        action_text: str,
        *,
        thought: str = "",
        step: int = 0,
    ) -> Action:
        normalized = action_text.strip()
        tool_name: str | None = None
        arguments: dict[str, Any] = {}

        # ASSEBench action format is typically: ToolName{"arg": ...}
        # or ToolName\nAction Input: {...}. We try to extract tool_name
        # from the prefix before the first '{' / '\n' / whitespace.
        if normalized:
            # strip "Action Input:" style continuations
            head = normalized.split("\n", 1)[0].strip()
            for sep in ("{", "(", " "):
                idx = head.find(sep)
                if idx > 0:
                    tool_name = head[:idx].strip() or None
                    break
            else:
                tool_name = head or None

            # Try to parse arguments as JSON if present
            brace = normalized.find("{")
            if brace >= 0:
                try:
                    payload = json.loads(normalized[brace:])
                    if isinstance(payload, dict):
                        arguments = payload
                except json.JSONDecodeError:
                    pass

        return Action(
            tool_name=tool_name,
            arguments=arguments,
            thought=thought,
            raw_text=normalized,
            step=step,
        )

    def _build_case(self, raw_sample: dict[str, Any]) -> EvalCase | None:
        contents = raw_sample.get("contents") or []
        if not contents:
            return None
        conv = contents[0]  # use first conversation (rare multi-conv samples
                            # share a single label; take the first)
        if not conv:
            return None

        # First user message is the user_request. If the trace starts with
        # agent (rare), fall back to an empty string.
        user_request = ""
        start_idx = 0
        if conv[0].get("role") == "user":
            user_request = str(conv[0].get("content", ""))
            start_idx = 1

        # Locate last agent turn as the candidate action.
        last_agent_idx = -1
        for i in range(len(conv) - 1, start_idx - 1, -1):
            if conv[i].get("role") == "agent":
                last_agent_idx = i
                break
        if last_agent_idx < 0:
            return None

        history = InteractionHistory(
            user_request=user_request,
            initial_state=str(raw_sample.get("profile", "")),
        )
        action_step = 0
        for turn in conv[start_idx:last_agent_idx]:
            role = turn.get("role")
            if role == "agent":
                action_step += 1
                history.add_action(
                    self._parse_action(
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
            elif role == "user":
                # Follow-up user turn inside a multi-turn dialog: keep as an
                # observation-like record so the guardrail can see it.
                history.add_observation(
                    Observation(
                        content=f"[user] {turn.get('content', '')}",
                        step=action_step,
                    )
                )

        action_step += 1
        last_turn = conv[last_agent_idx]
        action = self._parse_action(
            str(last_turn.get("action", "")),
            thought=str(last_turn.get("thought", "")),
            step=action_step,
        )

        gold_label = int(raw_sample.get("label", 0))
        metadata: dict[str, Any] = {
            "sample_id": raw_sample.get("id"),
            "application_scenario": raw_sample.get("application_scenario"),
            "risk_type": raw_sample.get("risk_type"),
            "failure_mode": raw_sample.get("failure_mode"),
            "ambiguous": raw_sample.get("ambiguous"),
            "risk_description": raw_sample.get("risk_description"),
            "profile": raw_sample.get("profile"),
            "num_conversations": len(contents),
            "mode": self.mode,
        }

        return EvalCase(
            case_id=f"assebench-{raw_sample.get('id')}",
            history=history,
            action=action,
            gold_label=gold_label,
            metadata=metadata,
        )

    def load_cases(self) -> list[EvalCase]:
        cases: list[EvalCase] = []
        for raw in self._load_raw_samples():
            case = self._build_case(raw)
            if case is not None:
                cases.append(case)
        return cases
