from __future__ import annotations

import json
import re
from typing import Any, Optional

from ag_types import Action, InteractionHistory, Observation
from agent.base import BaseAgent
from agent.prompts import REACT_SYSTEM_PROMPT
from infer.base import BaseInferBackend
from infer.factory import InferFactory


class ReactAgent(BaseAgent):
    """ReAct agent backed by the unified infer interface."""

    def __init__(
        self,
        infer_backend: BaseInferBackend,
        system_prompt: str = REACT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ) -> None:
        super().__init__(infer_backend=infer_backend, system_prompt=system_prompt, max_steps=max_steps)
        self.default_temperature = 0.0

    @classmethod
    def from_api(
        cls,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: str = REACT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ) -> "ReactAgent":
        infer_backend = InferFactory.create(
            "api",
            model=model,
            api_key=api_key,
            base_url=api_base,
        )
        agent = cls(infer_backend=infer_backend, system_prompt=system_prompt, max_steps=max_steps)
        agent.default_temperature = temperature
        return agent

    @classmethod
    def from_openai_compatible(
        cls,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: str = REACT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ) -> "ReactAgent":
        return cls.from_api(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    @classmethod
    def from_vllm(
        cls,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        system_prompt: str = REACT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ) -> "ReactAgent":
        infer_backend = InferFactory.create(
            "vllm",
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        agent = cls(infer_backend=infer_backend, system_prompt=system_prompt, max_steps=max_steps)
        agent.default_temperature = temperature
        return agent

    def act(self, history: InteractionHistory) -> Action:
        messages = self._build_messages(history)
        response = self.infer_backend.chat(messages, temperature=getattr(self, "default_temperature", 0.0))
        text = response.text.strip()
        return self._parse_action(text)

    def receive_observation(self, observation: Observation) -> None:
        return None

    def _build_messages(self, history: InteractionHistory) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        msgs.append({"role": "user", "content": f"User Request: {history.user_request}"})
        if history.initial_state:
            msgs.append({"role": "user", "content": f"Initial State: {history.initial_state}"})

        for item in history.steps:
            if isinstance(item, Action):
                msgs.append({"role": "assistant", "content": item.raw_text})
            else:
                msgs.append({"role": "user", "content": f"Observation: {item.content}"})
        return msgs

    def _parse_action(self, text: str) -> Action:
        if text.startswith("Final Answer:"):
            return Action(tool_name=None, arguments={}, thought="", raw_text=text)

        thought = self._extract_block(r"Thought:\s*(.*)", text)
        tool_name = self._extract_block(r"Action:\s*([^\n]+)", text)
        action_input_raw = self._extract_block(r"Action Input:\s*(\{.*\})", text, multiline=True)
        arguments = {}
        if action_input_raw:
            try:
                arguments = json.loads(action_input_raw)
            except json.JSONDecodeError:
                arguments = {}

        if not tool_name:
            return Action(tool_name=None, arguments={}, thought=thought or "", raw_text=text)
        return Action(tool_name=tool_name.strip(), arguments=arguments, thought=thought or "", raw_text=text)

    @staticmethod
    def _extract_block(pattern: str, text: str, multiline: bool = False) -> Optional[str]:
        flags = re.DOTALL if multiline else 0
        match = re.search(pattern, text, flags=flags)
        return match.group(1).strip() if match else None
