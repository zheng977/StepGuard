from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ag_types import Action, Observation


class ToolAdapter(ABC):
    @abstractmethod
    def call(self, arguments: dict[str, Any]) -> str:
        raise NotImplementedError


class ToolExecutor:
    """Execute tool action via registered adapters."""

    def __init__(self) -> None:
        self._registry: dict[str, ToolAdapter] = {}

    def register(self, name: str, adapter: ToolAdapter) -> None:
        self._registry[name] = adapter

    def execute(self, action: Action) -> Observation:
        if action.tool_name is None:
            raise ValueError("Final action cannot be executed as a tool call.")
        if action.tool_name not in self._registry:
            raise KeyError(f"Unknown tool: {action.tool_name}")
        content = self._registry[action.tool_name].call(action.arguments)
        return Observation(content=str(content), step=action.step)
