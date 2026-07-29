from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ag_types import Action, GuardrailContext, InteractionHistory, Judgment
from infer.base import BaseInferBackend

from .prompts import (
    DEFAULT_PROMPT_NAME,
    PROFILE_REGISTRY,
    _parse_binary_judgment,
    _parse_fg_labels as _extract_fg_labels,
    build_predictive_prompt,
    build_prompt_tag,
    compute_prompt_hash,
    get_predictive_prompt_template,
)

DEFAULT_RESPONSE_PARSER = "stepguard"


class BaseGuardrail(ABC):
    def __init__(self, infer_backend: BaseInferBackend) -> None:
        self.infer_backend = infer_backend

    @abstractmethod
    def judge(
        self,
        action: Action,
        history: InteractionHistory,
        context: GuardrailContext | None = None,
    ) -> Judgment:
        raise NotImplementedError


def _extract_tag_content(raw_text: str, tag_name: str) -> str:
    match = re.search(
        rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>|"
        rf"<{tag_name}>\s*(.*?)\s*<\\{tag_name}>|"
        rf"<{tag_name}>\s*(.*?)\s*(?=<[A-Za-z]+>|$)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return str(match.group(1) or match.group(2) or match.group(3) or "").strip()


def _parse_safe_unsafe_judgment(raw_text: str) -> tuple[int | None, float]:
    normalized = str(raw_text).strip().lower()
    if normalized == "safe":
        return 0, 1.0
    if normalized == "unsafe":
        return 1, 1.0
    return None, 0.0


def _parse_judgment(raw_text: str, *, response_parser: str) -> tuple[int | None, float, str]:
    normalized_parser = str(response_parser).strip().lower() or DEFAULT_RESPONSE_PARSER
    if normalized_parser == "auto":
        label, confidence = _parse_binary_judgment(raw_text)
        if label is not None:
            return label, confidence, "parsed"
        label, confidence = _parse_safe_unsafe_judgment(raw_text)
        if label is not None:
            return label, confidence, "parsed_safe_unsafe"
        return None, 0.0, "parse_failed"
    if normalized_parser == "strict":
        label, confidence = _parse_binary_judgment(raw_text)
        return label, confidence, "parsed" if label is not None else "parse_failed"
    if normalized_parser == "safe_unsafe":
        label, confidence = _parse_safe_unsafe_judgment(raw_text)
        return label, confidence, "parsed_safe_unsafe" if label is not None else "parse_failed"
    # Profile-based parsers
    if normalized_parser in PROFILE_REGISTRY:
        label, confidence = PROFILE_REGISTRY[normalized_parser].parse_response(raw_text)
        # codex_guardian never returns None (fail-closed)
        if label is not None:
            parse_status = "parsed" if confidence > 0.0 else "parse_failed_closed"
            return label, confidence, parse_status
        return None, 0.0, "parse_failed"
    raise ValueError(
        f'Unknown response parser "{response_parser}". '
        f'Expected one of: auto, strict, safe_unsafe, {", ".join(sorted(PROFILE_REGISTRY))}.'
    )


class PredictiveGuardrail(BaseGuardrail):
    def __init__(
        self,
        infer_backend: BaseInferBackend,
        *,
        temperature: float = 0.0,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int = 16384,
        timeout: int = 120,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        prompt_template: str | None = None,
        prompt_file: str | Path | None = None,
        response_parser: str = DEFAULT_RESPONSE_PARSER,
    ) -> None:
        super().__init__(infer_backend=infer_backend)
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.prompt_name = str(prompt_name).strip() or DEFAULT_PROMPT_NAME
        self.prompt_template = prompt_template
        self.prompt_file = str(prompt_file) if prompt_file is not None else None
        self.response_parser = str(response_parser).strip().lower() or DEFAULT_RESPONSE_PARSER

        if self.prompt_template is not None and prompt_file is not None:
            raise ValueError("Pass only one of prompt_template or prompt_file.")
        if prompt_file is not None:
            self.prompt_template = Path(prompt_file).read_text(encoding="utf-8")
        if self.prompt_template is None:
            self.prompt_template = get_predictive_prompt_template(self.prompt_name)
        self.prompt_hash = compute_prompt_hash(self.prompt_template)
        self.prompt_tag = build_prompt_tag(
            prompt_name=self.prompt_name,
            prompt_file=self.prompt_file,
            prompt_template=self.prompt_template,
        )

    def judge(
        self,
        action: Action,
        history: InteractionHistory,
        context: GuardrailContext | None = None,
    ) -> Judgment:
        return self._judge_with_template(
            action=action,
            history=history,
            context=context,
            prompt_template=self.prompt_template,
            prompt_hash=self.prompt_hash,
            prompt_tag=self.prompt_tag,
        )

    def judge_with_prompt_suffix(
        self,
        action: Action,
        history: InteractionHistory,
        context: GuardrailContext | None = None,
        *,
        prompt_suffix: str,
        prompt_tag_suffix: str = "suffix",
    ) -> Judgment:
        """Judge with the same profile plus a temporary prompt suffix.

        This is used by dynamic runtime mechanisms such as second-pass
        reconsideration. It does not mutate the guardrail instance, so static
        evaluation prompt hashes remain unchanged.
        """

        suffix = str(prompt_suffix).strip()
        prompt_template = self.prompt_template
        if suffix:
            prompt_template = f"{prompt_template}\n\n{suffix}"
        prompt_hash = compute_prompt_hash(prompt_template)
        prompt_tag = f"{self.prompt_tag}-{prompt_tag_suffix}_{prompt_hash}"
        return self._judge_with_template(
            action=action,
            history=history,
            context=context,
            prompt_template=prompt_template,
            prompt_hash=prompt_hash,
            prompt_tag=prompt_tag,
        )

    def _judge_with_template(
        self,
        *,
        action: Action,
        history: InteractionHistory,
        context: GuardrailContext | None,
        prompt_template: str,
        prompt_hash: str,
        prompt_tag: str,
    ) -> Judgment:
        prompt = build_predictive_prompt(
            history=history,
            action=action,
            context=context,
            prompt_template=prompt_template,
            prompt_name=self.prompt_name,
        )

        # Check for pre-computed analysis prefix (two-stage pipeline)
        assistant_prefix = ""
        if context and context.memory:
            assistant_prefix = context.memory.get("stage1_analysis", "")

        messages: list[dict[str, str]] = []
        # Add system prompt if the profile defines one
        profile = PROFILE_REGISTRY.get(self.prompt_name)
        if profile and getattr(profile, "system_prompt", None):
            messages.append({"role": "system", "content": profile.system_prompt})
        messages.append({"role": "user", "content": prompt})
        chat_kwargs: dict = {}
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})
            chat_kwargs["extra_body"] = {
                "continue_final_message": True,
                "add_generation_prompt": False,
            }

        if self.top_p is not None:
            chat_kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            chat_kwargs["presence_penalty"] = self.presence_penalty

        inference_started = time.perf_counter()
        response = self.infer_backend.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            **chat_kwargs,
        )
        inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0

        # For parsing, concat prefix + model continuation
        full_text = (assistant_prefix + response.text) if assistant_prefix else response.text
        prediction = _extract_tag_content(full_text, "Prediction")
        analysis = _extract_tag_content(full_text, "Analysis")
        label, confidence, parse_status = _parse_judgment(full_text, response_parser=self.response_parser)

        # A malformed guard response must not silently permit a live tool call.
        # Static evaluation excludes this status from metrics; dynamic adapters
        # treat the fail-closed result as a block and retain the raw response in
        # the trace for diagnosis.
        if label is None:
            label = 1
            confidence = 1.0
            parse_status = "parse_failed_closed"

        # Extract fine-grained labels if present (for fg evaluation)
        fg_labels = _extract_fg_labels(full_text)

        return Judgment(
            label=label,
            confidence=confidence,
            reason=full_text,
            metadata={
                "paradigm": "predictive",
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                # Client-observed duration of one guard generation request.
                # This includes transport and server queue time, matching the
                # latency experienced by the dynamic pre-action hook.
                "inference_latency_ms": inference_latency_ms,
                "raw": response.raw,
                "prompt": prompt,
                "messages": messages,
                "chat_kwargs": chat_kwargs,
                "prompt_name": self.prompt_name,
                "prompt_file": self.prompt_file,
                "prompt_hash": prompt_hash,
                "prompt_tag": prompt_tag,
                "prompt_len": len(prompt),
                "response_parser": self.response_parser,
                "prediction": prediction,
                "analysis": analysis,
                "judgment_parse_status": parse_status,
                **{f"pred_{k}": v for k, v in fg_labels.items()},
            },
        )
