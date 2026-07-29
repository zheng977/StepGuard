from __future__ import annotations

import json
import os
from typing import Any

from ag_types import Action, GuardrailContext, InteractionHistory, Observation


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    return _jsonable(value)


def _action_text(action: Action) -> str:
    if action.raw_text:
        return action.raw_text
    if action.tool_name is None:
        return action.thought
    return f"{action.tool_name}({json.dumps(action.arguments, ensure_ascii=False)})"


def _history_text(history: InteractionHistory | None) -> str:
    if history is None:
        return ""
    parts = []
    if history.initial_state:
        parts.append(f"<<Initial State>>\n{history.initial_state}")
    for idx, step in enumerate(history.steps):
        if isinstance(step, Action):
            parts.append(
                f"[{idx}] ACTION step={step.step}: {_action_text(step)}"
                + (f"\nThought: {step.thought}" if step.thought else "")
            )
        elif isinstance(step, Observation):
            parts.append(f"[{idx}] OBSERVATION step={step.step}:\n{step.content}")
        else:
            parts.append(f"[{idx}] {step}")
    return "\n\n".join(parts)


def _tool_list_text(context: GuardrailContext | None) -> str:
    if context is None:
        return ""
    tools = context.tool_schemas or context.available_tools or []
    return json.dumps(_jsonable(tools), ensure_ascii=False, indent=2)


def should_record_full_guard_context() -> bool:
    return (
        os.environ.get("AGENTGUARD_RECORD_FULL_PROMPT") == "1"
        or os.environ.get("AGENTGUARD_RECORD_FULL_CONTEXT") == "1"
    )


def aggregate_guard_judgments(judgments: list[dict[str, Any]]) -> dict[str, int | float]:
    """Aggregate token usage and observed latency from per-action guard traces.

    Backends expose OpenAI-style ``prompt_tokens`` / ``completion_tokens`` or
    ``input_tokens`` / ``output_tokens``. Keep both forms compatible and do
    not infer missing totals: unknown usage must remain unknown rather than
    being silently counted as zero.
    """

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    usage_calls = 0
    latency_ms_total = 0.0
    latency_calls = 0

    for judgment in judgments:
        usage = judgment.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
            completion = usage.get("completion_tokens", usage.get("output_tokens"))
            total = usage.get("total_tokens")
            if isinstance(prompt, (int, float)):
                prompt_tokens += int(prompt)
            if isinstance(completion, (int, float)):
                completion_tokens += int(completion)
            if isinstance(total, (int, float)):
                total_tokens += int(total)
            elif isinstance(prompt, (int, float)) or isinstance(completion, (int, float)):
                total_tokens += int(prompt or 0) + int(completion or 0)
            usage_calls += 1

        latency = judgment.get("inference_latency_ms")
        if isinstance(latency, (int, float)) and latency >= 0:
            latency_ms_total += float(latency)
            latency_calls += 1

    return {
        "guard_calls": len(judgments),
        "guard_usage_calls": usage_calls,
        "guard_prompt_tokens": prompt_tokens,
        "guard_completion_tokens": completion_tokens,
        "guard_total_tokens": total_tokens,
        "guard_inference_latency_ms_total": latency_ms_total,
        "guard_latency_calls": latency_calls,
    }


def compact_guard_trace(
    *,
    action: Action,
    judgment: Any | None,
    blocked: bool,
    history: InteractionHistory | None = None,
    context: GuardrailContext | None = None,
    turn_index: int | None = None,
    action_index: int | None = None,
    history_len: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Compact per-action guard trace for dynamic eval debugging.

    This trace is written only to eval artifacts. It is not exposed to the
    agent during replanning.
    """

    metadata = judgment.metadata if judgment is not None else {}
    if not isinstance(metadata, dict):
        metadata = {}

    trace: dict[str, Any] = {
        "turn_index": turn_index,
        "action_index": action_index,
        "step": action.step,
        "history_len": history_len,
        "tool_name": action.tool_name,
        "arguments": _jsonable(action.arguments),
        "thought": action.thought,
        "raw_action": action.raw_text,
        "label": int(judgment.label) if judgment is not None else None,
        "confidence": float(judgment.confidence) if judgment is not None else None,
        "blocked": bool(blocked),
        "risk_category": getattr(judgment, "risk_category", None) if judgment is not None else None,
        "analysis": str(metadata.get("analysis") or ""),
        "prediction": str(metadata.get("prediction") or ""),
        "reason": str(getattr(judgment, "reason", "") or "") if judgment is not None else "",
        "judgment_parse_status": metadata.get("judgment_parse_status"),
        "prompt_name": metadata.get("prompt_name"),
        "prompt_hash": metadata.get("prompt_hash"),
        "prompt_tag": metadata.get("prompt_tag"),
        "prompt_len": metadata.get("prompt_len"),
        "model": metadata.get("model"),
        "usage": _jsonable(metadata.get("usage")),
        "finish_reason": metadata.get("finish_reason"),
        "inference_latency_ms": metadata.get("inference_latency_ms"),
    }
    if "reconsideration" in metadata:
        trace["reconsideration"] = _jsonable(metadata.get("reconsideration"))
    if error:
        trace["error"] = error

    # Full prompts are useful for deep forensics but can make dynamic results
    # very large. Enable explicitly when needed.
    if should_record_full_guard_context():
        user_request = history.user_request if history is not None else ""
        initial_state = history.initial_state if history is not None else ""
        history_text = _history_text(history)
        tool_list_text = _tool_list_text(context)
        current_action_text = _action_text(action)
        guard_prompt = str(metadata.get("prompt") or "")
        guard_messages = _jsonable(metadata.get("messages") or [])
        guard_chat_kwargs = _jsonable(metadata.get("chat_kwargs") or {})
        guard_raw_response = str(getattr(judgment, "reason", "") or "") if judgment is not None else ""

        # Top-level fields are intentionally duplicated from full_context so
        # downstream case-study tooling can detect full-context records without
        # knowing the nested schema.
        trace.update(
            {
                "user_request": user_request,
                "initial_state": initial_state,
                "history_text": history_text,
                "tool_list_text": tool_list_text,
                "current_action_text": current_action_text,
                "guard_prompt": guard_prompt,
                "guard_messages": guard_messages,
                "guard_chat_kwargs": guard_chat_kwargs,
                "guard_raw_response": guard_raw_response,
                # Backward-compatible aliases used by older ad-hoc extractors.
                "prompt": guard_prompt,
            }
        )
        trace["full_context"] = {
            "user_request": user_request,
            "initial_state": initial_state,
            "history": _model_dump(history),
            "history_text": history_text,
            "context": _model_dump(context),
            "tool_schemas": _jsonable((context.tool_schemas if context else None) or []),
            "tool_list_text": tool_list_text,
            "current_action": _model_dump(action),
            "current_action_text": current_action_text,
            "guard_prompt": guard_prompt,
            "guard_messages": guard_messages,
            "guard_chat_kwargs": guard_chat_kwargs,
            "guard_raw_response": guard_raw_response,
        }

    return trace
