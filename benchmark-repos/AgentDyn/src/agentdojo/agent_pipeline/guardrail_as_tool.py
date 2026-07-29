"""Mode B: AgentGuard as an agent-callable tool.

Exposes the AgentGuard PredictiveGuardrail as a tool named
``safety_guardian`` that the agent can invoke *proactively* to check
whether a planned action is aligned with the user's request — mirroring
ToolSafe's ``tool_safety_guardian`` pattern.

Usage sketch::

    from agentdojo.agent_pipeline.agentguard_defense import AgentGuardDefense
    from agentdojo.agent_pipeline.guardrail_as_tool import (
        register_safety_guardian, GuardContextBinder,
    )

    defense = AgentGuardDefense.from_env()           # builds PredictiveGuardrail
    register_safety_guardian(runtime, defense.guardrail)
    pipeline_elements = [
        SystemMessage(system_message),
        InitQuery(),
        GuardContextBinder(defense.guardrail),        # stash ctx each turn
        llm,
        ToolsExecutionLoop([ToolsExecutor(...), llm]),
    ]

Context binding: AgentDyn's FunctionsRuntime invokes tools with their
declared arguments only — the tool doesn't see ``messages`` or
``query``. ``GuardContextBinder`` stashes the live ``query`` + past
``messages`` into a thread-local so ``safety_guardian`` can reconstruct
the full InteractionHistory on each call.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str

from ag_types import Action, GuardrailContext, InteractionHistory, Observation
from guardrail.guardrail import PredictiveGuardrail
from guardrail.prompts import extract_risk

logger = logging.getLogger(__name__)


# Thread-local bound state for the safety_guardian tool.
#
# A tool function in AgentDyn has no access to the pipeline's `messages`
# or `query`. We stash them here before each LLM turn so the tool can
# rebuild the InteractionHistory when called.
_tls = threading.local()


def _set_context(
    *,
    query: str,
    messages: Sequence[ChatMessage],
    guardrail: PredictiveGuardrail,
    runtime: FunctionsRuntime,
) -> None:
    _tls.query = query
    _tls.messages = list(messages)
    _tls.guardrail = guardrail
    _tls.runtime = runtime


def _clear_context() -> None:
    for attr in ("query", "messages", "guardrail", "runtime"):
        if hasattr(_tls, attr):
            delattr(_tls, attr)


def _build_history_from_messages(
    query: str, messages: Sequence[ChatMessage]
) -> InteractionHistory:
    """Reconstruct InteractionHistory from AgentDyn chat messages."""
    history = InteractionHistory(user_request=query)
    step = 0
    for msg in messages:
        role = msg.get("role")
        if role == "system" and msg.get("content"):
            history.initial_state = get_text_content_as_str(msg["content"])
        elif role == "assistant":
            thought = (
                get_text_content_as_str(msg["content"])
                if msg.get("content") is not None
                else ""
            )
            for tc in msg.get("tool_calls") or []:
                step += 1
                history.add_action(
                    Action(
                        tool_name=tc.function,
                        arguments=dict(tc.args),
                        thought=thought,
                        raw_text=f"{tc.function}({json.dumps(tc.args, ensure_ascii=False)})",
                        step=step,
                    )
                )
        elif role == "tool":
            content = (
                get_text_content_as_str(msg["content"])
                if msg.get("content") is not None
                else ""
            )
            history.add_observation(Observation(content=content, step=step))
    return history


def safety_guardian(
    tool_name: Annotated[str, Field(description="The name of the tool you plan to call next.")],
    arguments_json: Annotated[
        str,
        Field(
            description=(
                "JSON-encoded arguments you plan to pass to the tool. "
                'Example: \'{"to": "alice@example.com", "body": "..."}\''
            )
        ),
    ],
    thought: Annotated[
        str,
        Field(
            default="",
            description="One-sentence rationale for why this call furthers the user's request.",
        ),
    ] = "",
) -> dict:
    """Consult the safety guardian BEFORE executing any external tool call.

    The guardian evaluates whether the planned action is aligned with the
    original user request given the interaction so far. If the risk rating
    is high, reconsider and pick a safer action.

    Returns:
        A dict with:
          - ``risk_rating`` (float, 0.0-1.0): 0 = aligned, 1 = misaligned.
          - ``decision`` (str): ``"allow"`` or ``"block"``.
          - ``risk`` (str): concrete potential harm if the action executes
            (e.g. "leaks private notes to an attacker"); empty if aligned.
          - ``reason`` (str): brief justification from the guardian.
    """
    guardrail: PredictiveGuardrail | None = getattr(_tls, "guardrail", None)
    query: str = getattr(_tls, "query", "")
    messages = getattr(_tls, "messages", [])

    if guardrail is None:
        return {
            "risk_rating": 0.0,
            "decision": "allow",
            "risk": "",
            "reason": "Guardian not configured; proceeding without check.",
        }

    # Parse arguments (tolerate already-parsed dicts or loose JSON)
    parsed_args: dict
    if isinstance(arguments_json, dict):
        parsed_args = dict(arguments_json)
    else:
        try:
            parsed_args = json.loads(arguments_json) if arguments_json else {}
            if not isinstance(parsed_args, dict):
                parsed_args = {"_raw": parsed_args}
        except json.JSONDecodeError:
            parsed_args = {"_raw": str(arguments_json)}

    history = _build_history_from_messages(query, messages)
    action = Action(
        tool_name=tool_name,
        arguments=parsed_args,
        thought=thought,
        raw_text=f"{tool_name}({json.dumps(parsed_args, ensure_ascii=False)})",
        step=len(history.steps) + 1,
    )
    context = GuardrailContext()

    try:
        judgment = guardrail.judge(action=action, history=history, context=context)
    except Exception as exc:
        logger.exception("safety_guardian: guardrail judge failed")
        return {
            "risk_rating": 0.0,
            "decision": "allow",
            "risk": "",
            "reason": f"Guardian error, proceeding without check: {exc}",
        }

    risk_rating = float(judgment.label)  # 0 or 1; confidence is separate
    decision = "block" if judgment.label == 1 else "allow"
    # Do not expose the guardrail's internal reasoning to the agent. This
    # guardian-as-tool path is not used for the main pre-action defense eval,
    # but keep its output sanitized for consistency.
    risk_desc = "Action flagged by safety monitor." if judgment.label == 1 else ""

    logger.info(
        "safety_guardian: tool=%s label=%d confidence=%.2f risk=%s",
        tool_name,
        judgment.label,
        judgment.confidence,
        (risk_desc or "")[:200],
    )

    return {
        "risk_rating": risk_rating,
        "decision": decision,
        "risk": risk_desc,
        "reason": "",
    }


def register_safety_guardian(
    runtime: FunctionsRuntime,
    guardrail: PredictiveGuardrail,
) -> None:
    """Register ``safety_guardian`` into the runtime and seed the guardrail.

    Should be called once per benchmark task, after constructing the
    runtime and before handing it to the pipeline.
    """
    runtime.register_function(safety_guardian)
    # Seed the thread-local guardrail reference so the tool has a fallback
    # even if GuardContextBinder hasn't run yet in this thread.
    _tls.guardrail = guardrail
    _tls.query = ""
    _tls.messages = []
    _tls.runtime = runtime


class GuardContextBinder(BasePipelineElement):
    """Binds per-turn context (query + messages) to the thread-local.

    Place this **before the LLM** in the outer pipeline AND inside the
    tools-execution loop so the safety_guardian tool always sees fresh
    context when invoked.
    """

    name = "guard_context_binder"

    def __init__(self, guardrail: PredictiveGuardrail) -> None:
        self.guardrail = guardrail

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        _set_context(
            query=query,
            messages=messages,
            guardrail=self.guardrail,
            runtime=runtime,
        )
        return query, runtime, env, messages, extra_args
