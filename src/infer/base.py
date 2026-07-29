from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class InferResponse(BaseModel):
    text: str
    model: str
    raw: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str | None = None


class BaseInferBackend(ABC):
    """Unified inference interface shared by agent and guardrail layers."""

    def __init__(self, *, model: str) -> None:
        self.model = model

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: int = 120,
        **kwargs: Any,
    ) -> InferResponse:
        raise NotImplementedError
