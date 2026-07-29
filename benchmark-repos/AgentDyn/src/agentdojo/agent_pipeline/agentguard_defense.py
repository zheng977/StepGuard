"""AgentGuard predictive guardrail as an AgentDyn pipeline defense element.

Intercepts assistant tool_calls BEFORE ToolsExecutor runs them, evaluates
each action via AgentGuard's PredictiveGuardrail, and blocks unsafe ones.

This is the GUARD (defense being tested), not the judge.
The judge is AgentDyn's built-in programmatic evaluation:
  - task.utility(): checks if user task was completed (env state diff)
  - task.security(): checks if injection attack was resisted (env state diff)

Requires PYTHONPATH to include AgentGuard/src for imports below.
"""

import json
import logging
import os
import threading
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

# AgentGuard imports (requires AgentGuard/src on PYTHONPATH)
from ag_types import Action, GuardrailContext, InteractionHistory, Observation
from evals.dynamic.feedback import (
    format_replan_feedback,
    validate_blocked_history_mode,
    validate_feedback_mode,
)
from evals.dynamic.guard_trace import compact_guard_trace
from evals.dynamic.reconsideration import is_blocking_judgment, judge_with_reconsideration
from guardrail.guardrail import PredictiveGuardrail
from infer.factory import InferFactory

logger = logging.getLogger(__name__)


# Per-thread (i.e. per concurrent task) judgment counters. Must be reset
# at task boundary via reset_task_stats() and read via get_task_stats()
# after the task finishes. One task runs in one ThreadPoolExecutor worker,
# so thread-local storage gives us per-task counters without mutex overhead.
_task_stats = threading.local()
_task_events = threading.local()


def reset_task_stats() -> None:
    _task_stats.judged = 0
    _task_stats.blocked = 0
    _task_events.events = []


def get_task_stats() -> tuple[int, int]:
    """Return (judged, blocked) for the current task/thread. Returns (0, 0)
    if reset_task_stats() was not called (e.g. task errored before guard ran)."""
    return (
        int(getattr(_task_stats, "judged", 0)),
        int(getattr(_task_stats, "blocked", 0)),
    )


def get_task_events() -> list[dict]:
    """Return per-judge events (tool_name, args, label, blocked, reason) for
    the current task/thread. Reset via reset_task_stats()."""
    return list(getattr(_task_events, "events", []))


class AgentGuardDefense(BasePipelineElement):
    """Pre-action guardrail using AgentGuard's PredictiveGuardrail.

    Placed BEFORE ToolsExecutor in ToolsExecutionLoop. When the LLM
    produces tool_calls, this element judges each one. If any is unsafe,
    the entire batch is blocked before ToolsExecutor runs. The blocked
    assistant tool-call message is removed from persistent history, and the
    LLM receives optional one-shot feedback for replanning.
    """

    name = "agentguard"

    def __init__(
        self,
        guardrail: PredictiveGuardrail,
        blocking_mode: str = "continue",
        confidence_threshold: float = 0.5,
        generic_feedback: bool = False,
        feedback_mode: str | None = None,
        blocked_history_mode: str = "clean",
        max_replans: int | None = 3,
        guard_reconsideration: str = "off",
    ) -> None:
        self.guardrail = guardrail
        self.blocking_mode = blocking_mode
        self.confidence_threshold = confidence_threshold
        self.max_replans = max_replans
        # Feedback mode when soft-blocking:
        #   "detailed"    — minimal actionable replan notice; no guard reasoning is exposed
        #   "generic"     — ToolSafe-style minimal risk rating only
        #   "self_reflect" — ask agent to self-diagnose + propose alternative;
        #                    no guard reasoning is exposed
        #   "silent"      — no safety feedback is exposed; the blocked action is hidden
        # Precedence: explicit feedback_mode > generic_feedback=True >
        #             env AGENTGUARD_FEEDBACK_MODE > legacy AGENTGUARD_GENERIC_FEEDBACK > "detailed"
        if feedback_mode is not None:
            resolved = feedback_mode
        elif generic_feedback:
            resolved = "generic"
        else:
            env_mode = os.environ.get("AGENTGUARD_FEEDBACK_MODE")
            legacy_generic = os.environ.get("AGENTGUARD_GENERIC_FEEDBACK", "0") == "1"
            resolved = env_mode or ("generic" if legacy_generic else "detailed")
        self.feedback_mode = validate_feedback_mode(resolved)
        self.generic_feedback = self.feedback_mode == "generic"
        self.blocked_history_mode = validate_blocked_history_mode(blocked_history_mode)
        self.guard_reconsideration = guard_reconsideration

    @classmethod
    def from_env(cls) -> "AgentGuardDefense":
        """Construct from environment variables.

        Required:
            AGENTGUARD_MODEL: guard model name

        Optional (with defaults):
            AGENTGUARD_BASE_URL: API endpoint (falls back to OPENAI_BASE_URL)
            AGENTGUARD_API_KEY: API key (falls back to OPENAI_API_KEY)
            AGENTGUARD_PROMPT_NAME: prompt profile (default: general)
            AGENTGUARD_RESPONSE_PARSER: parser (default: strict)
            AGENTGUARD_BLOCKING_MODE: continue | abort (default: continue)
            AGENTGUARD_CONFIDENCE_THRESHOLD: float (default: 0.5)
        """
        model = os.environ.get("AGENTGUARD_MODEL")
        if not model:
            raise ValueError("AGENTGUARD_MODEL environment variable is required")

        base_url = os.environ.get("AGENTGUARD_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("AGENTGUARD_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")

        backend = InferFactory.create(
            "api",
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

        prompt_name = os.environ.get("AGENTGUARD_PROMPT_NAME", "general")
        response_parser = os.environ.get("AGENTGUARD_RESPONSE_PARSER", "strict")
        blocking_mode = os.environ.get("AGENTGUARD_BLOCKING_MODE", "continue")
        confidence_threshold = float(os.environ.get("AGENTGUARD_CONFIDENCE_THRESHOLD", "0.5"))

        guardrail = PredictiveGuardrail(
            infer_backend=backend,
            temperature=0.0,
            max_tokens=4096,
            timeout=120,
            prompt_name=prompt_name,
            response_parser=response_parser,
        )

        logger.info(
            "AgentGuardDefense initialized: model=%s, prompt=%s, mode=%s, threshold=%.2f",
            model, prompt_name, blocking_mode, confidence_threshold,
        )

        return cls(
            guardrail=guardrail,
            blocking_mode=blocking_mode,
            confidence_threshold=confidence_threshold,
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # Guard: only act on assistant messages with tool_calls
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if last_msg["tool_calls"] is None or len(last_msg["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args
        messages = (
            *self._drop_ephemeral_feedback_messages(messages[:-1]),
            last_msg,
        )
        if self.blocked_history_mode == "last_blocked" and len(messages) > 1:
            messages = (
                *self._drop_blocked_messages(messages[:-1]),
                last_msg,
            )

        # Build shared context from message history (excluding current assistant msg)
        history = self._build_history(query, messages[:-1])
        context = self._build_context(runtime)
        step = self._count_agent_steps(messages)

        # Get assistant thought (if any text content alongside tool_calls)
        thought = ""
        if last_msg["content"] is not None:
            thought = get_text_content_as_str(last_msg["content"])

        # Judge each tool_call independently
        judgments = []
        turn_traces = []
        any_blocked = False
        for tool_call in last_msg["tool_calls"]:
            action = Action(
                tool_name=tool_call.function,
                arguments=dict(tool_call.args),
                thought=thought,
                raw_text=f"{tool_call.function}({json.dumps(tool_call.args, ensure_ascii=False)})",
                step=step,
            )

            judgment, _ = judge_with_reconsideration(
                guardrail=self.guardrail,
                action=action,
                history=history,
                context=context,
                confidence_threshold=self.confidence_threshold,
                mode=self.guard_reconsideration,
            )
            judgments.append((tool_call, judgment))

            # Per-task counter (thread-local; safe under ThreadPool concurrency)
            _task_stats.judged = getattr(_task_stats, "judged", 0) + 1
            is_unsafe = is_blocking_judgment(judgment, self.confidence_threshold)
            _task_events.events = getattr(_task_events, "events", [])
            trace = compact_guard_trace(
                action=action,
                judgment=judgment,
                blocked=is_unsafe,
                history=history,
                context=context,
                turn_index=step,
                action_index=len(judgments) - 1,
                history_len=len(history.steps),
            )
            _task_events.events.append(trace)
            turn_traces.append(trace)

            # AGENTGUARD_TRACE_LOG=<path> → append every judge() call for case study.
            # Each line: tool_name / args / label / confidence / reason / full prompt
            # The `prompt` field contains the complete text fed to the guard LLM
            # (user_request + serialized interaction history incl. tool outputs +
            # current action + available tools), which is what the guard "sees".
            trace_path = os.environ.get("AGENTGUARD_TRACE_LOG")
            if trace_path:
                try:
                    _guard_prompt = ""
                    if judgment.metadata and isinstance(judgment.metadata, dict):
                        _guard_prompt = judgment.metadata.get("prompt", "") or ""
                    with open(trace_path, "a", encoding="utf-8") as _tf:
                        _tf.write(json.dumps({
                            "tool_name": tool_call.function,
                            "args": dict(tool_call.args),
                            "thought": thought,
                            "label": int(judgment.label),
                            "confidence": float(judgment.confidence),
                            "reason": (judgment.reason or "")[:2000],
                            "step": step,
                            "user_request": history.user_request,
                            "history_len": len(history.steps),
                            "prompt": _guard_prompt[:12000],
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            if is_unsafe:
                any_blocked = True
                logger.warning(
                    "BLOCKED: %s(%s) — confidence=%.2f reason=%s",
                    tool_call.function,
                    json.dumps(tool_call.args, ensure_ascii=False)[:200],
                    judgment.confidence,
                    judgment.reason[:200],
                )
            else:
                logger.debug(
                    "PASSED: %s — label=%d confidence=%.2f",
                    tool_call.function, judgment.label, judgment.confidence,
                )

        # Store judgments for downstream logging
        extra_args = {**extra_args}
        extra_args.setdefault("agentguard_judgments", [])
        extra_args["agentguard_judgments"].append(
            [
                {
                    "function": tc.function,
                    "arguments": dict(tc.args),
                    "label": j.label,
                    "confidence": j.confidence,
                    "blocked": j.label == 1 and j.confidence >= self.confidence_threshold,
                    "analysis": (j.metadata or {}).get("analysis", ""),
                    "reason": j.reason,
                }
                for tc, j in judgments
            ]
        )

        # All safe → pass through unchanged
        if not any_blocked:
            return query, runtime, env, messages, extra_args

        # --- Blocking ---

        if self.blocking_mode == "abort":
            _task_stats.blocked = getattr(_task_stats, "blocked", 0) + sum(
                1
                for _, j in judgments
                if j.label == 1 and j.confidence >= self.confidence_threshold
            )
            reasons = "; ".join(
                f"{tc.function}: {j.reason[:100]}"
                for tc, j in judgments
                if j.label == 1 and j.confidence >= self.confidence_threshold
            )
            raise AbortAgentError(
                f"[AgentGuard] Blocked unsafe action(s): {reasons}",
                list(messages),
                env,
            )

        replan_count = int(extra_args.get("agentguard_replan_count", 0)) + 1
        extra_args["agentguard_replan_count"] = replan_count
        if self.max_replans is not None and self.max_replans >= 0 and replan_count > self.max_replans:
            extra_args["agentguard_max_replans_exceeded"] = True
            for trace in turn_traces:
                if trace.get("blocked"):
                    trace["would_block"] = True
                    trace["blocked"] = False
                    trace["allowed_after_max_replans"] = True
                    trace["max_replans"] = self.max_replans
            for item in extra_args["agentguard_judgments"][-1]:
                if item.get("blocked"):
                    item["would_block"] = True
                    item["blocked"] = False
                    item["allowed_after_max_replans"] = True
                    item["max_replans"] = self.max_replans
            return query, runtime, env, messages, extra_args

        _task_stats.blocked = getattr(_task_stats, "blocked", 0) + sum(
            1
            for _, j in judgments
            if j.label == 1 and j.confidence >= self.confidence_threshold
        )

        if self.blocked_history_mode in {"toolsafe", "last_blocked"}:
            feedback = "" if self.feedback_mode == "silent" else self._format_replan_feedback(judgments, query)
            tool_messages = [
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(feedback)],
                    tool_call_id=tc.id,
                    tool_call=tc,
                    error=None,
                    agentguard_blocked=True,
                )
                for tc, _ in judgments
            ]
            return query, runtime, env, [*messages, *tool_messages], extra_args

        if self.feedback_mode == "silent":
            # Clean replanning: do not expose the blocked tool call or any
            # guard feedback to the agent. The next LLM call sees the same
            # real environment history as before this attempted action.
            return query, runtime, env, list(messages[:-1]), extra_args

        # Clean replanning with feedback: remove the assistant message that
        # contained the blocked tool call(s), then add feedback as a user
        # message. The agent sees guidance, but not the forbidden action it
        # just attempted, and the guard history remains based on real actions.
        feedback = self._format_replan_feedback(judgments, query)
        feedback_msg = ChatUserMessage(
            role="user",
            content=[text_content_block_from_string(feedback)],
        )
        feedback_msg["agentguard_ephemeral_feedback"] = True
        return query, runtime, env, [*messages[:-1], feedback_msg], extra_args

    def _format_replan_feedback(self, judgments, query: str) -> str:
        blocked = [
            j for _, j in judgments
            if j.label == 1 and j.confidence >= self.confidence_threshold
        ]
        max_conf = max((float(j.confidence) for j in blocked), default=0.0)
        return format_replan_feedback(
            self.feedback_mode,
            confidence=max_conf,
            judgments=blocked,
            benchmark="agentdyn",
            user_request=query,
        )

    # ---- Type conversion helpers ----

    def _build_history(self, query: str, messages: Sequence[ChatMessage]) -> InteractionHistory:
        """Convert AgentDyn message sequence to AgentGuard InteractionHistory."""
        history = InteractionHistory(user_request=query)
        step = 0
        blocked_tool_call_ids = {
            msg.get("tool_call_id")
            for msg in messages
            if msg["role"] == "tool" and msg.get("agentguard_blocked")
        }

        for msg in messages:
            role = msg["role"]

            if role == "system":
                history.initial_state = get_text_content_as_str(msg["content"])

            elif role == "user":
                # user_request already set from query param;
                # additional user messages (e.g. repeat_user_prompt) are ignored
                pass

            elif role == "assistant":
                thought = ""
                if msg["content"] is not None:
                    thought = get_text_content_as_str(msg["content"])

                if msg["tool_calls"] is not None and len(msg["tool_calls"]) > 0:
                    for tc in msg["tool_calls"]:
                        if tc.id in blocked_tool_call_ids:
                            continue
                        history.add_action(Action(
                            tool_name=tc.function,
                            arguments=dict(tc.args),
                            thought=thought,
                            raw_text=f"{tc.function}({json.dumps(tc.args, ensure_ascii=False)})",
                            step=step,
                        ))
                        step += 1
                        thought = ""  # only attach thought to first tool_call
                elif thought:
                    history.add_action(Action(
                        tool_name=None,
                        thought=thought,
                        raw_text=thought,
                        step=step,
                    ))
                    step += 1

            elif role == "tool":
                if msg.get("agentguard_blocked"):
                    continue
                content = get_text_content_as_str(msg["content"])
                error = msg.get("error")
                if error:
                    content = f"[ERROR] {error}\n{content}"
                history.add_observation(Observation(content=content, step=step))

        return history

    @staticmethod
    def _drop_blocked_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        blocked_ids = {
            msg.get("tool_call_id")
            for msg in messages
            if msg["role"] == "tool" and msg.get("agentguard_blocked")
        }
        if not blocked_ids:
            return list(messages)

        kept = []
        for msg in messages:
            if msg["role"] == "tool" and msg.get("tool_call_id") in blocked_ids:
                continue
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                remaining_calls = [
                    tool_call for tool_call in msg["tool_calls"]
                    if tool_call.id not in blocked_ids
                ]
                if not remaining_calls:
                    continue
                if len(remaining_calls) != len(msg["tool_calls"]):
                    msg = {**msg, "tool_calls": remaining_calls}
            kept.append(msg)
        return kept

    @staticmethod
    def _drop_ephemeral_feedback_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        return [
            msg for msg in messages
            if not msg.get("agentguard_ephemeral_feedback")
        ]

    def _build_context(self, runtime: FunctionsRuntime) -> GuardrailContext:
        """Extract tool schemas from FunctionsRuntime."""
        tool_schemas = []
        for name, func in runtime.functions.items():
            schema = {
                "name": name,
                "description": func.description,
            }
            try:
                schema["parameters"] = func.parameters.model_json_schema()
            except Exception:
                pass
            tool_schemas.append(schema)

        return GuardrailContext(tool_schemas=tool_schemas)

    def _count_agent_steps(self, messages: Sequence[ChatMessage]) -> int:
        """Count how many assistant tool-call turns have occurred."""
        count = 0
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                count += 1
        return count
