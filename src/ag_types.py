from __future__ import annotations

from typing import Any, Iterator, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


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


class UserMessage(BaseModel):
    """A follow-up instruction, kept distinct from environment observations."""

    role: Literal["user"] = "user"
    content: str


HistoryStep = Union[Action, Observation, UserMessage]


class InteractionHistory(BaseModel):
    """Context before the action under review, including follow-up users.

    Completed trajectories can also have post-action events. Only trajectory
    serializers consume those events; proactive action prompts must not.
    """

    user_request: str
    initial_state: str = ""
    steps: list[HistoryStep] = Field(default_factory=list)
    post_action_steps: list[Union[Observation, UserMessage]] = Field(default_factory=list)

    @field_validator("steps", "post_action_steps", mode="before")
    @classmethod
    def _restore_message_types(cls, value: Any) -> Any:
        # Older Pydantic 2.x union matching can accept message dictionaries as
        # Actions with default fields. Restore users and observations explicitly
        # so loading an archived trajectory preserves its roles and content.
        if not isinstance(value, (list, tuple)):
            return value
        messages = []
        for item in value:
            if isinstance(item, dict):
                if item.get("role") == "user":
                    item = UserMessage.model_validate(item)
                elif "content" in item:
                    item = Observation.model_validate(item)
            messages.append(item)
        return messages

    def add_action(self, action: Action) -> None:
        self.steps.append(action)

    def add_observation(self, observation: Observation) -> None:
        self.steps.append(observation)

    def add_user_message(self, message: UserMessage) -> None:
        self.steps.append(message)

    def iter_trajectory(self, action: Action) -> Iterator[HistoryStep]:
        """Yield the complete record with the reviewed action exactly once."""
        yield from self.steps
        yield action
        yield from self.post_action_steps


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
