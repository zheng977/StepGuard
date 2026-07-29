"""AgentDyn dynamic benchmark implementation.

Runs AgentDyn's full agent pipeline with AgentGuard as the defense,
collecting utility and security results per task.

Supports task-level concurrency via ThreadPoolExecutor.

Requires:
    - AgentDyn source on PYTHONPATH (benchmark-repos/AgentDyn/src)
    - AgentDyn dependencies installed (openai, pydantic, pyyaml, etc.)
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evals.dynamic.base import BaseDynamicBenchmark, DynamicEvalResult
from evals.dynamic.guard_trace import aggregate_guard_judgments

logger = logging.getLogger(__name__)


def _serialize_message_obj(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _serialize_message_obj(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _serialize_message_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_message_obj(v) for v in value]
    return value


class AgentDynBenchmark(BaseDynamicBenchmark):
    """Dynamic benchmark using the AgentDyn framework.

    Constructs an AgentDyn pipeline with AgentGuardDefense as the pre-action
    guardrail, then runs user tasks with/without injection attacks.
    """

    name = "agentdyn"

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
    ) -> None:
        self.agent_model = agent_model
        self.agent_api_key = agent_api_key
        self.agent_base_url = agent_base_url
        self.agent_port = agent_port
        self.agent_no_proxy = agent_no_proxy
        self.agent_system_suffix = agent_system_suffix
        self.proxy = proxy
        self.suites = suites or ["shopping", "github", "dailylife"]
        self.attack = attack
        self.run_benign_with_attack = run_benign_with_attack
        self.skip_injection_precheck = skip_injection_precheck
        self.user_tasks = user_tasks
        self.injection_tasks = injection_tasks
        self.benchmark_version = benchmark_version
        self.logdir = Path(logdir) if logdir else None
        self.force_rerun = force_rerun
        self.concurrency = max(1, concurrency)
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
        import os

        import openai as openai_pkg
        from agentdojo.agent_pipeline.agent_pipeline import (
            AgentPipeline, get_llm, load_system_message,
        )
        from agentdojo.agent_pipeline.agentguard_defense import AgentGuardDefense
        from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
        from agentdojo.agent_pipeline.tool_execution import (
            ToolsExecutionLoop, ToolsExecutor, tool_result_to_str,
        )
        from agentdojo.attacks.attack_registry import load_attack
        from agentdojo.benchmark import run_task_without_injection_tasks
        from agentdojo.logging import Logger, NullLogger, TraceLogger
        from agentdojo.models import MODEL_PROVIDERS, ModelsEnum
        from agentdojo.task_suite.load_suites import get_suite

        # ── Build agent LLM (before proxy) ──
        if self.agent_base_url or self.agent_port:
            base_url = self.agent_base_url or f"http://localhost:{self.agent_port}/v1"
            client_kwargs = {"api_key": self.agent_api_key or "EMPTY", "base_url": base_url}
            # agent_no_proxy=True → bypass HTTP_PROXY/HTTPS_PROXY env vars.
            # openai.OpenAI uses httpx which auto-reads proxy env; when the
            # agent is a local/LAN endpoint but the env has an unused cluster
            # proxy, this causes "Connection refused" errors at the proxy port.
            if self.agent_no_proxy:
                import httpx
                # Match openai SDK's default pool sizing so concurrency isn't throttled.
                client_kwargs["http_client"] = httpx.Client(
                    trust_env=False,
                    limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
                    timeout=httpx.Timeout(600.0, connect=10.0),
                )
            client = openai_pkg.OpenAI(**client_kwargs)
            llm = OpenAILLM(client, self.agent_model)
            llm_name = self.agent_model
            logger.info("Agent LLM: %s at %s", self.agent_model, base_url)
        else:
            if self.agent_api_key:
                os.environ["OPENAI_API_KEY"] = self.agent_api_key
            llm_enum = ModelsEnum(self.agent_model)
            llm = get_llm(MODEL_PROVIDERS[llm_enum], self.agent_model, None, "tool")
            llm_name = self.agent_model

        # ── Set proxy for guard (after agent client) ──
        if self.proxy:
            for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                os.environ[k] = self.proxy
            logger.info("Proxy for guard: %s", self.proxy)

        # ── Build pipeline ──
        system_message = load_system_message(None)
        if self.agent_system_suffix:
            system_message = f"{system_message.rstrip()}\n\n{self.agent_system_suffix.strip()}"
        if guardrail is not None:
            guard_element = AgentGuardDefense(
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
            # No guard — pure agent baseline
            loop_elements = [ToolsExecutor(tool_result_to_str), llm]
            pipeline_tag = "no_defense"

        tools_loop = ToolsExecutionLoop(loop_elements)
        pipeline = AgentPipeline(
            [SystemMessage(system_message), InitQuery(), llm, tools_loop]
        )
        pipeline.name = f"{llm_name}-{pipeline_tag}"

        # ── Run suites in parallel ──
        # Each suite runs independently. Suite and phase pools may run
        # concurrently, but self._task_semaphore keeps total in-flight
        # benchmark tasks bounded by self.concurrency.
        def _run_one_suite(suite_name: str):
            logger.info("Suite %s (attack=%s, concurrency=%d)", suite_name, self.attack, self.concurrency)
            suite = get_suite(self.benchmark_version, suite_name)
            t0 = time.time()
            try:
                with NullLogger():
                    suite_results: dict[str, tuple[bool, bool, bool, int, int] | tuple[bool, bool, bool, int, int, str]] = {}
                    # Run benign and attacked phases concurrently within this suite when requested.
                    # They share pipeline (stateless) and agent/guard LLM clients.
                    phase_workers = 2 if (self.attack is not None and self.run_benign_with_attack) else 1
                    with ThreadPoolExecutor(max_workers=phase_workers) as phase_pool:
                        phase_futures = {}
                        if self.attack is None or self.run_benign_with_attack:
                            phase_futures[phase_pool.submit(self._run_suite_no_attack, suite, pipeline)] = "benign"
                        if self.attack is not None:
                            attacker = load_attack(self.attack, suite, pipeline)
                            phase_futures[phase_pool.submit(
                                self._run_suite_with_attack, suite, pipeline, attacker
                            )] = "attacked"
                        for pf in as_completed(phase_futures):
                            suite_results.update(pf.result())
            except Exception as exc:
                logger.error("Suite %s failed: %s", suite_name, exc, exc_info=True)
                return suite_name, {"_error": (False, True, self.attack is not None, 0, 0, [], [], str(exc))}, time.time() - t0
            return suite_name, suite_results, time.time() - t0

        results: list[DynamicEvalResult] = []
        suite_workers = min(len(self.suites), 3)
        with ThreadPoolExecutor(max_workers=suite_workers) as pool:
            futures = [pool.submit(_run_one_suite, s) for s in self.suites]
            for f in as_completed(futures):
                suite_name, suite_results, duration = f.result()
                n_tasks = len(suite_results)
                for case_id, task_data in suite_results.items():
                    guard_judgments = []
                    task_duration = 0.0
                    if len(task_data) == 9:
                        utility, security, has_attack, judged, blocked, guard_judgments, raw_messages, task_duration, err = task_data
                    elif len(task_data) == 8:
                        utility, security, has_attack, judged, blocked, guard_judgments, raw_messages, err = task_data
                    elif len(task_data) == 7:
                        utility, security, has_attack, judged, blocked, guard_judgments, err = task_data
                        raw_messages = []
                    elif len(task_data) == 6:  # legacy error tuple
                        utility, security, has_attack, judged, blocked, err = task_data
                        raw_messages = []
                    elif len(task_data) == 5:
                        utility, security, has_attack, judged, blocked = task_data
                        raw_messages = []
                        err = None
                    else:
                        utility, security, has_attack = task_data
                        judged, blocked = 0, 0
                        raw_messages = []
                        err = None

                    metrics = {
                        "task_duration_sec": task_duration,
                        **aggregate_guard_judgments(guard_judgments),
                    }
                    if err:
                        results.append(DynamicEvalResult(
                            case_id=f"{suite_name}/{case_id}",
                            utility=utility, security=security, has_attack=has_attack,
                            guard_blocked=(blocked > 0),
                            error=err,
                            metadata={
                                "suite": suite_name,
                                "attack_type": self.attack,
                                "agent_model": self.agent_model,
                                "judged_actions": judged,
                                "blocked_actions": blocked,
                                "guard_judgments": guard_judgments,
                                "attack_success": (not security) if has_attack else None,
                                "raw_messages": raw_messages,
                                **metrics,
                            },
                        ))
                        continue
                    results.append(DynamicEvalResult(
                        case_id=f"{suite_name}/{case_id}",
                        utility=utility, security=security, has_attack=has_attack,
                        guard_blocked=(blocked > 0),
                        duration=task_duration,
                        metadata={
                            "suite": suite_name,
                            "attack_type": self.attack,
                            "agent_model": self.agent_model,
                            "judged_actions": judged,
                            "blocked_actions": blocked,
                            "guard_judgments": guard_judgments,
                            "attack_success": (not security) if has_attack else None,
                            "raw_messages": raw_messages,
                            **metrics,
                        },
                    ))
                logger.info("Suite %s: %d tasks, %.1fs", suite_name, n_tasks, duration)
        return results

    # ── Internal: parallel task execution ──

    def _run_suite_no_attack(self, suite, pipeline):
        """Run user tasks without injection, return {case_id: (utility, security, has_attack, judged, blocked)}."""
        from agentdojo.agent_pipeline.agentguard_defense import (
            reset_task_stats,
            get_task_stats,
            get_task_events,
        )
        user_tasks = self._resolve_user_tasks(suite)
        task_results: dict[str, tuple[bool, bool, bool, int, int]] = {}

        def _run_one(user_task):
            # Reset per-task thread-local block counters before pipeline runs
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
            events = get_task_events()
            return user_task.ID, utility, security, judged, blocked, events, raw_messages, time.perf_counter() - task_started, err

        progress = tqdm(total=len(user_tasks), desc=f"{suite.name} (no attack)", unit="task")
        ok_count = 0
        # Traj-level block count: increments once per task that had ≥1 block.
        block_task_count = 0

        if self.concurrency <= 1:
            for ut in user_tasks:
                task_id, utility, security, judged, blocked, events, raw_messages, task_duration, err = _run_one(ut)
                if err:
                    task_results[f"{task_id}/none"] = (utility, security, False, judged, blocked, events, raw_messages, task_duration, err)
                else:
                    task_results[f"{task_id}/none"] = (utility, security, False, judged, blocked, events, raw_messages, task_duration, None)
                    ok_count += utility
                if not err and blocked > 0:
                    block_task_count += 1
                progress.set_postfix(
                    utility=f"{ok_count}/{progress.n+1}",
                    blocks=f"{block_task_count}/{progress.n+1}",
                )
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=self._phase_worker_limit(len(user_tasks))) as pool:
                futures = {pool.submit(_run_one, ut): ut for ut in user_tasks}
                for future in as_completed(futures):
                    try:
                        task_id, utility, security, judged, blocked, events, raw_messages, task_duration, err = future.result()
                        if err:
                            task_results[f"{task_id}/none"] = (utility, security, False, judged, blocked, events, raw_messages, task_duration, err)
                        else:
                            task_results[f"{task_id}/none"] = (utility, security, False, judged, blocked, events, raw_messages, task_duration, None)
                            ok_count += utility
                        if not err and blocked > 0:
                            block_task_count += 1
                    except Exception as exc:
                        ut = futures[future]
                        logger.error("Task %s failed: %s", ut.ID, exc)
                        task_results[f"{ut.ID}/none"] = (False, True, False, 0, 0, [], [], 0.0, str(exc))
                    progress.set_postfix(
                        utility=f"{ok_count}/{progress.n+1}",
                        blocks=f"{block_task_count}/{progress.n+1}",
                    )
                    progress.update(1)

        progress.close()
        return task_results

    def _run_suite_with_attack(self, suite, pipeline, attacker):
        """Run user×injection task pairs, return {case_id: (utility, security, has_attack, judged, blocked)}.

        AgentDyn's native second return value is "injection task succeeded".
        Convert it here to AgentGuard's common `security` meaning: the agent
        stayed safe and resisted the injected task.
        """
        import warnings
        from agentdojo.agent_pipeline.agentguard_defense import (
            reset_task_stats,
            get_task_stats,
            get_task_events,
        )

        user_tasks = self._resolve_user_tasks(suite)
        injection_tasks = self._resolve_injection_tasks(suite)
        task_results: dict[str, tuple[bool, bool, bool, int, int]] = {}

        # Pre-check: injection tasks solvable as user tasks (bypass TraceLogger)
        if not self.skip_injection_precheck and not attacker.is_dos_attack:
            for inj_id, inj_task in injection_tasks.items():
                try:
                    with self._task_semaphore:
                        ok, _ = suite.run_task_with_pipeline(
                            pipeline, inj_task, injection_task=None, injections={},
                        )
                except Exception:
                    ok = False
                if not ok:
                    warnings.warn(f"Injection task {inj_id} not solvable as user task.")

        # Build all (user_task, injection_task) pairs
        pairs = []
        for ut in user_tasks:
            for inj_id, inj_task in injection_tasks.items():
                pairs.append((ut, inj_id, inj_task))

        def _run_pair(ut, inj_id, inj_task):
            reset_task_stats()
            err: str | None = None
            raw_messages: list[dict[str, Any]] = []
            task_started = time.perf_counter()
            try:
                task_injections = attacker.attack(ut, inj_task)
                with self._task_semaphore:
                    utility, attack_success, messages = suite.run_task_with_pipeline(
                        pipeline, ut, inj_task, task_injections,
                        return_messages=True,
                    )
                    raw_messages = _serialize_message_obj(messages)
            except Exception as exc:
                logger.error("Task %s/%s error: %s", ut.ID, inj_id, exc)
                utility, attack_success = False, False
                err = str(exc)
            judged, blocked = get_task_stats()
            events = get_task_events()
            security = not attack_success
            return ut.ID, inj_id, utility, security, judged, blocked, events, raw_messages, time.perf_counter() - task_started, err

        progress = tqdm(total=len(pairs), desc=f"{suite.name} (attack={attacker.name})", unit="pair")
        secure_count = 0
        util_count = 0
        block_task_count = 0  # pairs where guard blocked ≥1 action

        if self.concurrency <= 1:
            for ut, inj_id, inj_task in pairs:
                task_id, iid, utility, security, judged, blocked, events, raw_messages, task_duration, err = _run_pair(ut, inj_id, inj_task)
                if err:
                    task_results[f"{task_id}/{iid}"] = (utility, security, True, judged, blocked, events, raw_messages, task_duration, err)
                else:
                    task_results[f"{task_id}/{iid}"] = (utility, security, True, judged, blocked, events, raw_messages, task_duration, None)
                    secure_count += security
                    util_count += utility
                if not err and blocked > 0:
                    block_task_count += 1
                progress.set_postfix(
                    sec=f"{secure_count}/{progress.n+1}",
                    util=f"{util_count}/{progress.n+1}",
                    blocks=f"{block_task_count}/{progress.n+1}",
                )
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=self._phase_worker_limit(len(pairs))) as pool:
                futures = {
                    pool.submit(_run_pair, ut, inj_id, inj_task): (ut, inj_id)
                    for ut, inj_id, inj_task in pairs
                }
                for future in as_completed(futures):
                    ut, inj_id = futures[future]
                    try:
                        task_id, iid, utility, security, judged, blocked, events, raw_messages, task_duration, err = future.result()
                        if err:
                            task_results[f"{task_id}/{iid}"] = (utility, security, True, judged, blocked, events, raw_messages, task_duration, err)
                        else:
                            task_results[f"{task_id}/{iid}"] = (utility, security, True, judged, blocked, events, raw_messages, task_duration, None)
                            secure_count += security
                            util_count += utility
                        if not err and blocked > 0:
                            block_task_count += 1
                    except Exception as exc:
                        logger.error("Task %s/%s failed: %s", ut.ID, inj_id, exc)
                        task_results[f"{ut.ID}/{inj_id}"] = (False, True, True, 0, 0, [], [], 0.0, str(exc))
                    progress.set_postfix(
                        sec=f"{secure_count}/{progress.n+1}",
                        util=f"{util_count}/{progress.n+1}",
                        blocks=f"{block_task_count}/{progress.n+1}",
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

    # ── Metrics ──

    @staticmethod
    def _action_block_rate(results: list[DynamicEvalResult]) -> float | None:
        """Σ blocked_actions / Σ judged_actions across the given results.
        Returns None if no actions were judged (e.g. no-guard baseline).
        """
        judged = sum(r.metadata.get("judged_actions", 0) for r in results)
        blocked = sum(r.metadata.get("blocked_actions", 0) for r in results)
        return blocked / judged if judged > 0 else None

    @staticmethod
    def _traj_block_rate(results: list[DynamicEvalResult]) -> float | None:
        """Fraction of trajectories that had >=1 block. None when no trajs."""
        if not results:
            return None
        with_block = sum(1 for r in results if r.metadata.get("blocked_actions", 0) > 0)
        return with_block / len(results)

    def summarize(self, results: list[DynamicEvalResult]) -> dict[str, Any]:
        valid = [r for r in results if r.error is None]
        benign = [r for r in valid if not r.has_attack]
        attacked = [r for r in valid if r.has_attack]

        summary: dict[str, Any] = {
            "benchmark_name": self.name,
            "agent_model": self.agent_model,
            "attack": self.attack,
            "run_benign_with_attack": self.run_benign_with_attack,
            "skip_injection_precheck": self.skip_injection_precheck,
            "concurrency": self.concurrency,
            "total_tasks": len(valid),
            "benign_tasks": len(benign),
            "attacked_tasks": len(attacked),
            "benign_utility": self._safe_rate([r.utility for r in benign]),
            "utility_under_attack": self._safe_rate([r.utility for r in attacked]) if attacked else None,
            "asr": self._safe_rate([not r.security for r in attacked]) if attacked else None,
            "security_rate": self._safe_rate([r.security for r in attacked]) if attacked else None,
            # Guard behavior metrics (per-action FPR on benign; block coverage on attacked)
            "benign_block_rate_action": self._action_block_rate(benign),
            "benign_block_rate_traj": self._traj_block_rate(benign),
            "attacked_block_rate_action": self._action_block_rate(attacked) if attacked else None,
            "attacked_block_rate_traj": self._traj_block_rate(attacked) if attacked else None,
            "error_count": sum(1 for r in results if r.error is not None),
        }

        suites = sorted(set(r.metadata.get("suite", "unknown") for r in valid))
        by_suite: dict[str, dict[str, Any]] = {}
        for s in suites:
            s_benign = [r for r in benign if r.metadata.get("suite") == s]
            s_attacked = [r for r in attacked if r.metadata.get("suite") == s]
            by_suite[s] = {
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
