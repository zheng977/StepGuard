from __future__ import annotations

from typing import Any

from infer.base import BaseInferBackend
from infer.openai_compatible import APIInferBackend, VLLMInferBackend


class InferFactory:
    @staticmethod
    def create(kind: str, **kwargs: Any) -> BaseInferBackend:
        normalized_kind = str(kind).strip().lower()
        if normalized_kind == "api":
            return APIInferBackend(**kwargs)
        if normalized_kind == "vllm":
            return VLLMInferBackend(**kwargs)
        raise ValueError(f'Unknown infer backend "{kind}". Expected one of: api, vllm.')
