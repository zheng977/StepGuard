from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ag_types import Action, InteractionHistory, Observation
from infer.base import BaseInferBackend


class BaseAgent(ABC):
    """Base interface for all agents."""

    def __init__(self, infer_backend: Optional[BaseInferBackend], system_prompt: str = "", max_steps: int = 10) -> None:
        self.infer_backend = infer_backend
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    @abstractmethod
    def act(self, history: InteractionHistory) -> Action:
        """Generate next action based on current interaction history."""
        raise NotImplementedError

    def receive_observation(self, observation: Observation) -> None:
        """Hook for stateful agents. Stateless agents can ignore."""
        return None
