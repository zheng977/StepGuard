from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


Paradigm = Literal["proactive", "predictive"]
JudgmentLabel = Literal[0, 1]


class Action(BaseModel):
    """Agent action a_t."""

    tool_name: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    thought: str = ""
    raw_text: str = ""
    step: int = 0

    @property
    def is_final(self) -> bool:
        return self.tool_name is None


class Observation(BaseModel):
    """Environment observation o_t."""

    content: str
    step: int = 0


HistoryStep = Union[Action, Observation]


class InteractionHistory(BaseModel):
    """H_t = (I, o_0, a_1, o_1, ..., o_{t-1})."""

    user_request: str
    initial_state: str = ""
    steps: list[HistoryStep] = Field(default_factory=list)

    def add_action(self, action: Action) -> None:
        self.steps.append(action)

    def add_observation(self, observation: Observation) -> None:
        self.steps.append(observation)


class GuardrailContext(BaseModel):
    """Optional side context C = {P, M, ...}."""

    safety_policy: Optional[str] = None
    memory: Optional[dict[str, Any]] = None
    available_tools: Optional[list[dict[str, Any]]] = None
    tool_schemas: Optional[list[dict[str, Any]]] = None


class Judgment(BaseModel):
    """Binary judgment y in {0, 1}."""

    label: JudgmentLabel
    confidence: float = 1.0
    reason: str = ""
    risk_category: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Single LLM call response."""

    text: str
    usage: dict[str, int] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Full interaction record."""

    history: InteractionHistory
    judgments: list[Judgment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
