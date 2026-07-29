"""Original AgentDojo dynamic benchmark adapter.

This adapter targets ``benchmark-repos/agentdojo`` and keeps it separate from
the AgentDyn fork under ``benchmark-repos/AgentDyn``. The metric convention is
the same as the other dynamic adapters:

- utility: the original user task succeeds.
- security: the injected task does not succeed.
- asr: injected task succeeds.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ag_types import Action, GuardrailContext, InteractionHistory, Observation
from evals.dynamic.base import BaseDynamicBenchmark, DynamicEvalResult
from evals.dynamic.feedback import (
    format_replan_feedback,
    validate_blocked_history_mode,
    validate_feedback_mode,
)
from evals.dynamic.guard_trace import aggregate_guard_judgments, compact_guard_trace
from evals.dynamic.reconsideration import is_blocking_judgment, judge_with_reconsideration

logger = logging.getLogger(__name__)

_task_stats = threading.local()
_task_events = threading.local()


def _serialize_message_obj(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _serialize_message_obj(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _serialize_message_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_message_obj(v) for v in value]
    return value


def reset_task_stats() -> None:
    _task_stats.judged = 0
    _task_stats.blocked = 0
    _task_events.events = []


def get_task_stats() -> tuple[int, int]:
    return (
        int(getattr(_task_stats, "judged", 0)),
        int(getattr(_task_stats, "blocked", 0)),
    )


def get_task_events() -> list[dict[str, Any]]:
    return list(getattr(_task_events, "events", []))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _prepare_agentdojo_source(source_root: str | None = None) -> Path:
    """Prefer the original AgentDojo package over the AgentDyn fork."""

    root = Path(source_root or (_repo_root() / "benchmark-repos" / "agentdojo" / "src")).resolve()
    if not (root / "agentdojo").exists():
        raise FileNotFoundError(f"AgentDojo source root not found: {root}")

    existing = sys.modules.get("agentdojo")
    existing_file = Path(getattr(existing, "__file__", "") or "").resolve() if existing else None
    if existing_file and root not in existing_file.parents:
        for name in list(sys.modules):
            if name == "agentdojo" or name.startswith("agentdojo."):
                del sys.modules[name]

    root_str = str(root)
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != root]
    sys.path.insert(0, root_str)
    return root


class AgentDojoBenchmark(BaseDynamicBenchmark):
    """Dynamic benchmark using the original AgentDojo framework."""

    name = "agentdojo"

    def __init__(
        self,
        agent_model: str,
        agent_api_key: str | None = None,
        agent_base_url: str | None = None,
        agent_port: int | None = None,
        agent_no_proxy: bool = False,
        agent_system_suffix: str | None = None,
        proxy: str | None = None,
        suites: list[str] | None = None,
        attack: str | None = None,
        run_benign_with_attack: bool = True,
        skip_injection_precheck: bool = False,
        user_tasks: list[str] | None = None,
        injection_tasks: list[str] | None = None,
        benchmark_version: str = "v1.2.2",
        logdir: str | None = None,
        force_rerun: bool = True,
        concurrency: int = 1,
        source_root: str | None = None,
    ) -> None:
        self.agent_model = agent_model
        self.agent_api_key = agent_api_key
        self.agent_base_url = agent_base_url
        self.agent_port = agent_port
        self.agent_no_proxy = agent_no_proxy
        self.agent_system_suffix = agent_system_suffix
        self.proxy = proxy
        self.suites = suites or ["workspace", "travel", "banking", "slack"]
        self.attack = attack
        self.run_benign_with_attack = run_benign_with_attack
        self.skip_injection_precheck = skip_injection_precheck
        self.user_tasks = user_tasks
        self.injection_tasks = injection_tasks
        self.benchmark_version = benchmark_version
        self.logdir = Path(logdir) if logdir else None
        self.force_rerun = force_rerun
        self.concurrency = max(1, concurrency)
        self.source_root = source_root
        self._task_semaphore = threading.BoundedSemaphore(self.concurrency)

    def _phase_worker_limit(self, n_items: int) -> int:
        """Bound per-phase worker pools so nested pools do not multiply threads."""
        if n_items <= 0:
            return 1
        active_suites = max(1, min(len(self.suites), 3))
        active_phases = 2 if (self.attack is not None and self.run_benign_with_attack) else 1
        denominator = active_suites * active_phases
        per_phase = max(1, (self.concurrency + denominator - 1) // denominator)
        return max(1, min(n_items, per_phase))

    def run(
        self,
        guardrail: Any,
        *,
        blocking_mode: str = "continue",
        confidence_threshold: float = 0.5,
        generic_feedback: bool = False,
        feedback_mode: str = "self_reflect",
        blocked_history_mode: str = "clean",
        max_replans: int | None = 3,
        guard_reconsideration: str = "off",
    ) -> list[DynamicEvalResult]:
        _prepare_agentdojo_source(self.source_root)

        import openai as openai_pkg
        from agentdojo.agent_pipeline.agent_pipeline import (
            AgentPipeline,
            get_llm,
            load_system_message,
        )
        from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
        from agentdojo.agent_pipeline.tool_execution import (
            ToolsExecutionLoop,
            ToolsExecutor,
            tool_result_to_str,
        )
        from agentdojo.attacks.attack_registry import load_attack
        from agentdojo.logging import NullLogger
        from agentdojo.models import MODEL_PROVIDERS, ModelsEnum
        from agentdojo.task_suite.load_suites import get_suite

        if self.agent_base_url or self.agent_port:
            base_url = self.agent_base_url or f"http://localhost:{self.agent_port}/v1"
            client_kwargs: dict[str, Any] = {
                "api_key": self.agent_api_key or "EMPTY",
                "base_url": base_url,
            }
            if self.agent_no_proxy:
                import httpx

                client_kwargs["http_client"] = httpx.Client(
                    trust_env=False,
                    limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
                    timeout=httpx.Timeout(600.0, connect=10.0),
                )
            client = openai_pkg.OpenAI(**client_kwargs)
            llm = OpenAILLM(client, self.agent_model, temperature=0.0)
            # Original AgentDojo attacks inspect pipeline.name and only know a
            # fixed set of model-name keys. Keep "local" in the name for vLLM
            # agents while preserving the served model name for logs/metadata.
            llm_name = f"local-{self.agent_model}"
            logger.info("Agent LLM: %s at %s", self.agent_model, base_url)
        else:
            if self.agent_api_key:
                os.environ["OPENAI_API_KEY"] = self.agent_api_key
            llm_enum = ModelsEnum(self.agent_model)
            llm = get_llm(MODEL_PROVIDERS[llm_enum], self.agent_model, None, "tool")
            llm_name = self.agent_model

        if self.proxy:
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                os.environ[key] = self.proxy
            logger.info("Proxy for guard: %s", self.proxy)

        system_message = load_system_message(None)
        if self.agent_system_suffix:
            system_message = f"{system_message.rstrip()}\n\n{self.agent_system_suffix.strip()}"
        if guardrail is not None:
            guard_element = _AgentDojoGuardDefense(
                guardrail=guardrail,
                blocking_mode=blocking_mode,
                confidence_threshold=confidence_threshold,
                generic_feedback=generic_feedback,
                feedback_mode=feedback_mode,
                blocked_history_mode=blocked_history_mode,
                max_replans=max_replans,
                guard_reconsideration=guard_reconsideration,
            )
            loop_elements = [guard_element, ToolsExecutor(tool_result_to_str), llm]
            pipeline_tag = "agentguard"
        else:
            loop_elements = [ToolsExecutor(tool_result_to_str), llm]
            pipeline_tag = "no_defense"

        tools_loop = ToolsExecutionLoop(loop_elements)
        pipeline = AgentPipeline([SystemMessage(system_message), InitQuery(), llm, tools_loop])
        pipeline.name = f"{llm_name}-{pipeline_tag}"

        def _run_one_suite(suite_name: str):
            logger.info("Suite %s (attack=%s, concurrency=%d)", suite_name, self.attack, self.concurrency)
            suite = get_suite(self.benchmark_version, suite_name)
            t0 = time.time()
            try:
                with NullLogger():
                    suite_results: dict[str, tuple[Any, ...]] = {}
                    phase_workers = 2 if (self.attack is not None and self.run_benign_with_attack) else 1
                    with ThreadPoolExecutor(max_workers=phase_workers) as phase_pool:
                        futures = {}
                        if self.attack is None or self.run_benign_with_attack:
                            futures[phase_pool.submit(self._run_suite_no_attack, suite, pipeline)] = "benign"
                        if self.attack is not None:
                            attacker = load_attack(self.attack, suite, pipeline)
                            futures[phase_pool.submit(
                                self._run_suite_with_attack, suite, pipeline, attacker
                            )] = "attacked"
                        for future in as_completed(futures):
                            suite_results.update(future.result())
            except Exception as exc:
                logger.error("Suite %s failed: %s", suite_name, exc, exc_info=True)
                return suite_name, {"_error": (False, True, self.attack is not None, 0, 0, [], [], 0.0, str(exc))}, time.time() - t0
            return suite_name, suite_results, time.time() - t0

        results: list[DynamicEvalResult] = []
        # Suite and phase pools may run concurrently, but the semaphore above
        # keeps total in-flight benchmark tasks bounded by self.concurrency.
        suite_workers = min(len(self.suites), 3)
        with ThreadPoolExecutor(max_workers=suite_workers) as pool:
            futures = [pool.submit(_run_one_suite, suite_name) for suite_name in self.suites]
            for future in as_completed(futures):
                suite_name, suite_results, duration = future.result()
                n_tasks = len(suite_results)
                for case_id, task_data in suite_results.items():
                    utility, security, has_attack, judged, blocked, events, raw_messages, task_duration, err = task_data
                    metrics = {
                        "task_duration_sec": task_duration,
                        **aggregate_guard_judgments(events),
                    }
                    results.append(
                        DynamicEvalResult(
                            case_id=f"{suite_name}/{case_id}",
                            utility=bool(utility),
                            security=bool(security),
                            has_attack=bool(has_attack),
                            guard_blocked=(blocked > 0),
                            duration=task_duration,
                            error=err,
                            metadata={
                                "suite": suite_name,
                                "attack_type": self.attack,
                                "agent_model": self.agent_model,
                                "judged_actions": judged,
                                "blocked_actions": blocked,
                                "guard_judgments": events,
                                "attack_success": (not security) if has_attack else None,
                                "raw_messages": raw_messages,
                                **metrics,
                            },
                        )
                    )
                logger.info("Suite %s: %d tasks, %.1fs", suite_name, n_tasks, duration)
        return results

    def _run_suite_no_attack(self, suite, pipeline):
        user_tasks = self._resolve_user_tasks(suite)
        task_results: dict[str, tuple[Any, ...]] = {}

        def _run_one(user_task):
            reset_task_stats()
            err: str | None = None
            raw_messages: list[dict[str, Any]] = []
            task_started = time.perf_counter()
            try:
                with self._task_semaphore:
                    utility, security, messages = suite.run_task_with_pipeline(
                        pipeline, user_task, injection_task=None, injections={},
                        return_messages=True,
                    )
                    raw_messages = _serialize_message_obj(messages)
            except Exception as exc:
                logger.error("Task %s error: %s", user_task.ID, exc)
                utility, security = False, True
                err = str(exc)
            judged, blocked = get_task_stats()
            return user_task.ID, utility, security, judged, blocked, get_task_events(), raw_messages, time.perf_counter() - task_started, err

        progress = tqdm(total=len(user_tasks), desc=f"{suite.name} (no attack)", unit="task")
        ok_count = 0
        block_task_count = 0
        if self.concurrency <= 1:
            for user_task in user_tasks:
                task_id, utility, security, judged, blocked, events, raw_messages, task_duration, err = _run_one(user_task)
                task_results[f"{task_id}/none"] = (utility, security, False, judged, blocked, events, raw_messages, task_duration, err)
                ok_count += bool(utility and not err)
                block_task_count += int(blocked > 0 and not err)
                progress.set_postfix(utility=f"{ok_count}/{progress.n + 1}", blocks=f"{block_task_count}/{progress.n + 1}")
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=self._phase_worker_limit(len(user_tasks))) as pool:
                futures = {pool.submit(_run_one, user_task): user_task for user_task in user_tasks}
                for future in as_completed(futures):
                    user_task = futures[future]
                    try:
                        task_id, utility, security, judged, blocked, events, raw_messages, task_duration, err = future.result()
                    except Exception as exc:
                        logger.error("Task %s failed: %s", user_task.ID, exc)
                        task_id, utility, security, judged, blocked, events, raw_messages, task_duration, err = user_task.ID, False, True, 0, 0, [], [], 0.0, str(exc)
                    task_results[f"{task_id}/none"] = (utility, security, False, judged, blocked, events, raw_messages, task_duration, err)
                    ok_count += bool(utility and not err)
                    block_task_count += int(blocked > 0 and not err)
                    progress.set_postfix(utility=f"{ok_count}/{progress.n + 1}", blocks=f"{block_task_count}/{progress.n + 1}")
                    progress.update(1)
        progress.close()
        return task_results

    def _run_suite_with_attack(self, suite, pipeline, attacker):
        user_tasks = self._resolve_user_tasks(suite)
        injection_tasks = self._resolve_injection_tasks(suite)
        task_results: dict[str, tuple[Any, ...]] = {}
        pairs = [(ut, inj_id, inj_task) for ut in user_tasks for inj_id, inj_task in injection_tasks.items()]

        def _run_pair(user_task, injection_id, injection_task):
            reset_task_stats()
            err: str | None = None
            raw_messages: list[dict[str, Any]] = []
            task_started = time.perf_counter()
            try:
                task_injections = attacker.attack(user_task, injection_task)
                with self._task_semaphore:
                    utility, attack_success, messages = suite.run_task_with_pipeline(
                        pipeline, user_task, injection_task, task_injections,
                        return_messages=True,
                    )
                    raw_messages = _serialize_message_obj(messages)
            except Exception as exc:
                logger.error("Task %s/%s error: %s", user_task.ID, injection_id, exc)
                utility, attack_success = False, False
                err = str(exc)
            judged, blocked = get_task_stats()
            security = not attack_success
            return user_task.ID, injection_id, utility, security, judged, blocked, get_task_events(), raw_messages, time.perf_counter() - task_started, err

        progress = tqdm(total=len(pairs), desc=f"{suite.name} (attack={attacker.name})", unit="pair")
        secure_count = 0
        util_count = 0
        block_task_count = 0
        if self.concurrency <= 1:
            for user_task, injection_id, injection_task in pairs:
                task_id, iid, utility, security, judged, blocked, events, raw_messages, task_duration, err = _run_pair(
                    user_task, injection_id, injection_task
                )
                task_results[f"{task_id}/{iid}"] = (utility, security, True, judged, blocked, events, raw_messages, task_duration, err)
                secure_count += bool(security and not err)
                util_count += bool(utility and not err)
                block_task_count += int(blocked > 0 and not err)
                progress.set_postfix(
                    sec=f"{secure_count}/{progress.n + 1}",
                    util=f"{util_count}/{progress.n + 1}",
                    blocks=f"{block_task_count}/{progress.n + 1}",
                )
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=self._phase_worker_limit(len(pairs))) as pool:
                futures = {
                    pool.submit(_run_pair, user_task, injection_id, injection_task): (user_task, injection_id)
                    for user_task, injection_id, injection_task in pairs
                }
                for future in as_completed(futures):
                    user_task, injection_id = futures[future]
                    try:
                        task_id, iid, utility, security, judged, blocked, events, raw_messages, task_duration, err = future.result()
                    except Exception as exc:
                        logger.error("Task %s/%s failed: %s", user_task.ID, injection_id, exc)
                        task_id, iid, utility, security, judged, blocked, events, raw_messages, task_duration, err = (
                            user_task.ID, injection_id, False, True, 0, 0, [], [], 0.0, str(exc)
                        )
                    task_results[f"{task_id}/{iid}"] = (utility, security, True, judged, blocked, events, raw_messages, task_duration, err)
                    secure_count += bool(security and not err)
                    util_count += bool(utility and not err)
                    block_task_count += int(blocked > 0 and not err)
                    progress.set_postfix(
                        sec=f"{secure_count}/{progress.n + 1}",
                        util=f"{util_count}/{progress.n + 1}",
                        blocks=f"{block_task_count}/{progress.n + 1}",
                    )
                    progress.update(1)
        progress.close()
        return task_results

    def _resolve_user_tasks(self, suite):
        if self.user_tasks:
            return [suite.get_user_task_by_id(uid) for uid in self.user_tasks]
        return list(suite.user_tasks.values())

    def _resolve_injection_tasks(self, suite):
        if self.injection_tasks:
            return {iid: suite.get_injection_task_by_id(iid) for iid in self.injection_tasks}
        return dict(suite.injection_tasks)

    @staticmethod
    def _action_block_rate(results: list[DynamicEvalResult]) -> float | None:
        judged = sum(r.metadata.get("judged_actions", 0) for r in results)
        blocked = sum(r.metadata.get("blocked_actions", 0) for r in results)
        return blocked / judged if judged > 0 else None

    @staticmethod
    def _traj_block_rate(results: list[DynamicEvalResult]) -> float | None:
        if not results:
            return None
        return sum(1 for r in results if r.metadata.get("blocked_actions", 0) > 0) / len(results)

    def summarize(self, results: list[DynamicEvalResult]) -> dict[str, Any]:
        valid = [r for r in results if r.error is None]
        benign = [r for r in valid if not r.has_attack]
        attacked = [r for r in valid if r.has_attack]
        summary: dict[str, Any] = {
            "benchmark_name": self.name,
            "agent_model": self.agent_model,
            "attack": self.attack,
            "run_benign_with_attack": self.run_benign_with_attack,
            "concurrency": self.concurrency,
            "total_tasks": len(valid),
            "benign_tasks": len(benign),
            "attacked_tasks": len(attacked),
            "benign_utility": self._safe_rate([r.utility for r in benign]),
            "utility_under_attack": self._safe_rate([r.utility for r in attacked]) if attacked else None,
            "asr": self._safe_rate([not r.security for r in attacked]) if attacked else None,
            "security_rate": self._safe_rate([r.security for r in attacked]) if attacked else None,
            "benign_block_rate_action": self._action_block_rate(benign),
            "benign_block_rate_traj": self._traj_block_rate(benign),
            "attacked_block_rate_action": self._action_block_rate(attacked) if attacked else None,
            "attacked_block_rate_traj": self._traj_block_rate(attacked) if attacked else None,
            "error_count": sum(1 for r in results if r.error is not None),
        }
        suites = sorted({r.metadata.get("suite", "unknown") for r in valid})
        by_suite: dict[str, dict[str, Any]] = {}
        for suite_name in suites:
            s_benign = [r for r in benign if r.metadata.get("suite") == suite_name]
            s_attacked = [r for r in attacked if r.metadata.get("suite") == suite_name]
            by_suite[suite_name] = {
                "benign_tasks": len(s_benign),
                "attacked_tasks": len(s_attacked),
                "benign_utility": self._safe_rate([r.utility for r in s_benign]) if s_benign else None,
                "utility_under_attack": self._safe_rate([r.utility for r in s_attacked]) if s_attacked else None,
                "asr": self._safe_rate([not r.security for r in s_attacked]) if s_attacked else None,
                "security_rate": self._safe_rate([r.security for r in s_attacked]) if s_attacked else None,
                "benign_block_rate_action": self._action_block_rate(s_benign),
                "benign_block_rate_traj": self._traj_block_rate(s_benign),
                "attacked_block_rate_action": self._action_block_rate(s_attacked) if s_attacked else None,
                "attacked_block_rate_traj": self._traj_block_rate(s_attacked) if s_attacked else None,
            }
        summary["by_suite"] = by_suite
        return summary


class _AgentDojoGuardDefense:
    name = "agentguard"

    def __init__(
        self,
        guardrail: Any,
        blocking_mode: str = "continue",
        confidence_threshold: float = 0.5,
        generic_feedback: bool = False,
        feedback_mode: str = "self_reflect",
        blocked_history_mode: str = "clean",
        max_replans: int | None = 3,
        guard_reconsideration: str = "off",
    ) -> None:
        self.guardrail = guardrail
        self.blocking_mode = blocking_mode
        self.confidence_threshold = confidence_threshold
        self.max_replans = max_replans
        if generic_feedback:
            resolved = "generic"
        elif feedback_mode is not None:
            resolved = feedback_mode
        else:
            env_mode = os.environ.get("AGENTGUARD_FEEDBACK_MODE")
            legacy_generic = os.environ.get("AGENTGUARD_GENERIC_FEEDBACK", "0") == "1"
            resolved = env_mode or ("generic" if legacy_generic else "detailed")
        self.feedback_mode = validate_feedback_mode(resolved)
        self.blocked_history_mode = validate_blocked_history_mode(blocked_history_mode)
        self.guard_reconsideration = guard_reconsideration

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        from agentdojo.agent_pipeline.errors import AbortAgentError
        from agentdojo.functions_runtime import EmptyEnv
        from agentdojo.types import ChatToolResultMessage, ChatUserMessage, get_text_content_as_str, text_content_block_from_string

        if env is None:
            env = EmptyEnv()
        if extra_args is None:
            extra_args = {}
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        last_msg = messages[-1]
        if last_msg["role"] != "assistant" or not last_msg["tool_calls"]:
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

        history = self._build_history(query, messages[:-1])
        context = self._build_context(runtime)
        step = self._count_agent_steps(messages)
        thought = get_text_content_as_str(last_msg["content"]) if last_msg["content"] is not None else ""

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

            trace_path = os.environ.get("AGENTGUARD_TRACE_LOG")
            if trace_path:
                try:
                    prompt = ""
                    if judgment.metadata and isinstance(judgment.metadata, dict):
                        prompt = judgment.metadata.get("prompt", "") or ""
                    with open(trace_path, "a", encoding="utf-8") as trace_file:
                        trace_file.write(json.dumps({
                            "tool_name": tool_call.function,
                            "args": dict(tool_call.args),
                            "thought": thought,
                            "label": int(judgment.label),
                            "confidence": float(judgment.confidence),
                            "reason": (judgment.reason or "")[:2000],
                            "step": step,
                            "user_request": history.user_request,
                            "history_len": len(history.steps),
                            "prompt": prompt[:12000],
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            if is_unsafe:
                any_blocked = True

        extra_args = {**extra_args}
        extra_args.setdefault("agentguard_judgments", [])
        extra_args["agentguard_judgments"].append([
            {
                "function": tc.function,
                "arguments": dict(tc.args),
                "label": judgment.label,
                "confidence": judgment.confidence,
                "blocked": judgment.label == 1 and judgment.confidence >= self.confidence_threshold,
                "analysis": (judgment.metadata or {}).get("analysis", ""),
                "reason": judgment.reason,
            }
            for tc, judgment in judgments
        ])

        if not any_blocked:
            return query, runtime, env, messages, extra_args

        if self.blocking_mode == "abort":
            _task_stats.blocked = getattr(_task_stats, "blocked", 0) + sum(
                1
                for _, judgment in judgments
                if judgment.label == 1 and judgment.confidence >= self.confidence_threshold
            )
            reasons = "; ".join(
                f"{tc.function}: {(judgment.reason or '')[:100]}"
                for tc, judgment in judgments
                if judgment.label == 1 and judgment.confidence >= self.confidence_threshold
            )
            raise AbortAgentError(f"[AgentGuard] Blocked unsafe action(s): {reasons}", list(messages), env)

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
            for _, judgment in judgments
            if judgment.label == 1 and judgment.confidence >= self.confidence_threshold
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
            return query, runtime, env, list(messages[:-1]), extra_args

        feedback_msg = ChatUserMessage(
            role="user",
            content=[text_content_block_from_string(self._format_replan_feedback(judgments, query))],
            agentguard_ephemeral_feedback=True,
        )
        return query, runtime, env, [*messages[:-1], feedback_msg], extra_args

    def _format_replan_feedback(self, judgments, query: str) -> str:
        blocked = [
            judgment for _, judgment in judgments
            if judgment.label == 1 and judgment.confidence >= self.confidence_threshold
        ]
        max_conf = max((float(judgment.confidence) for judgment in blocked), default=0.0)
        return format_replan_feedback(
            self.feedback_mode,
            confidence=max_conf,
            judgments=blocked,
            benchmark="agentdojo",
            user_request=query,
        )

    def _build_history(self, query: str, messages) -> InteractionHistory:
        from agentdojo.types import get_text_content_as_str

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
            elif role == "assistant":
                thought = get_text_content_as_str(msg["content"]) if msg["content"] is not None else ""
                if msg["tool_calls"]:
                    for tool_call in msg["tool_calls"]:
                        if tool_call.id in blocked_tool_call_ids:
                            continue
                        history.add_action(Action(
                            tool_name=tool_call.function,
                            arguments=dict(tool_call.args),
                            thought=thought,
                            raw_text=f"{tool_call.function}({json.dumps(tool_call.args, ensure_ascii=False)})",
                            step=step,
                        ))
                        step += 1
                        thought = ""
                elif thought:
                    history.add_action(Action(tool_name=None, thought=thought, raw_text=thought, step=step))
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
    def _drop_blocked_messages(messages):
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
    def _drop_ephemeral_feedback_messages(messages):
        return [
            msg for msg in messages
            if not msg.get("agentguard_ephemeral_feedback")
        ]

    def _build_context(self, runtime) -> GuardrailContext:
        tool_schemas = []
        for name, func in runtime.functions.items():
            schema = {"name": name, "description": func.description}
            try:
                schema["parameters"] = func.parameters.model_json_schema()
            except Exception:
                pass
            tool_schemas.append(schema)
        return GuardrailContext(tool_schemas=tool_schemas)

    def _count_agent_steps(self, messages) -> int:
        return sum(1 for msg in messages if msg["role"] == "assistant" and msg.get("tool_calls"))
