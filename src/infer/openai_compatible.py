from __future__ import annotations

import os
from typing import Any

import httpx
from openai import OpenAI

from infer.base import BaseInferBackend, InferResponse


def _extract_text_from_choice(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            if parts:
                return "\n".join(parts)
        refusal = getattr(message, "refusal", None)
        if isinstance(refusal, str) and refusal.strip():
            return refusal.strip()

    text = getattr(choice, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _normalize_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return {str(key): int(value) for key, value in dumped.items() if isinstance(value, int)}
    if isinstance(usage, dict):
        return {str(key): int(value) for key, value in usage.items() if isinstance(value, int)}
    return {}


class OpenAICompatibleInferBackend(BaseInferBackend):
    """Base backend for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        default_timeout: int = 120,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key
        self.base_url = base_url
        self.default_timeout = int(default_timeout)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: int = 120,
        **kwargs: Any,
    ) -> InferResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout or self.default_timeout,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        text = _extract_text_from_choice(choice)
        return InferResponse(
            text=text,
            model=self.model,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            usage=_normalize_usage(getattr(response, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
        )


class APIInferBackend(OpenAICompatibleInferBackend):
    """Backend for remote OpenAI-compatible APIs.

    Automatically falls back to streaming when the server requires it
    (returns "Stream must be set to true").
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: int = 120,
        **kwargs: Any,
    ) -> InferResponse:
        # Always use streaming for API backends — some providers require it.
        # Falls back to non-streaming only if streaming fails.
        try:
            return self._chat_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
        except Exception:
            if os.getenv("AGENTGUARD_API_STREAM_ONLY", "0") == "1":
                raise
            return super().chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )

    def _chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: int = 120,
        **kwargs: Any,
    ) -> InferResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout or self.default_timeout,
            "stream": True,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        request_kwargs.update(kwargs)

        stream = self.client.chat.completions.create(**request_kwargs)
        parts: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, int] = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                parts.append(delta.content)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                usage = _normalize_usage(chunk_usage)

        return InferResponse(
            text="".join(parts).strip(),
            model=self.model,
            raw={},
            usage=usage,
            finish_reason=finish_reason,
        )


class VLLMInferBackend(OpenAICompatibleInferBackend):
    """Backend for vLLM endpoints exposed through an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        default_timeout: int = 120,
        max_connections: int = 1000,
        max_keepalive_connections: int = 100,
    ) -> None:
        # Local vLLM must not inherit ALL_PROXY/socks (requires socksio); skip env proxies.
        # Raise httpx pool limits so high concurrency (N×100+) does not block on
        # the default 100/20 cap; matches AgentDyn's benchmark-time client.
        BaseInferBackend.__init__(self, model=model)
        self.api_key = api_key
        self.base_url = base_url
        self.default_timeout = int(default_timeout)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(
                timeout=httpx.Timeout(self.default_timeout, connect=30.0),
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive_connections,
                ),
                trust_env=False,
            ),
        )
