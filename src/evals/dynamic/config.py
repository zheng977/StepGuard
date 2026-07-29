from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DynamicEvalConfig:
    benchmark: str
    model: str | None = None
    no_guard: bool = False
    backend: str = "api"
    base_url: str | None = None
    api_key: str | None = None
    prompt_name: str = "stepguard"
    prompt_file: str | None = None
    response_parser: str = "stepguard"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120
    output_root: str = "results"

    blocking_mode: str = "continue"
    confidence_threshold: float = 0.5
    generic_feedback: bool = False
    # Paper setting: give the agent a sanitized replan message after a block.
    # This is independent of guard reconsideration, which remains disabled.
    feedback_mode: str = "self_reflect"
    blocked_history_mode: str = "clean"
    max_replans: int | None = 3
    guard_reconsideration: str = "off"
    record_full_guard_context: bool = False

    agent_model: str = "gpt-4o-2024-08-06"
    agent_api_key: str | None = None
    agent_base_url: str | None = None
    agent_server_json: str | None = None
    agent_port: int | None = None
    agent_no_proxy: bool = False
    agent_system_suffix: str | None = None
    proxy: str | None = None
    concurrency: int = 1
    suites: list[str] | None = None
    attack: str | None = None
    run_benign_with_attack: bool = True
    skip_injection_precheck: bool = False
    user_tasks: list[str] | None = None
    injection_tasks: list[str] | None = None
    benchmark_version: str = "v1.2.2"
    logdir: str | None = None
    agentdojo_source_root: str | None = None

    subset: str | None = None
    max_turns: int = 10
    dataset_path: str | None = None
    agentharm_source_root: str | None = None
    tools_root: str | None = None
    graders_module_path: str | None = None
    n_irrelevant_tools: int = 0
    judge_model: str | None = None
    judge_base_url: str | None = None
    judge_server_json: str | None = None
    judge_api_key: str | None = None
    behavior_ids: list[str] | None = None
    limit: int | None = None

    @classmethod
    def from_namespace(cls, args: Namespace) -> "DynamicEvalConfig":
        values = {field: getattr(args, field) for field in cls.__dataclass_fields__ if hasattr(args, field)}
        return cls(**values)

    def resolved_run_config(self, *, guard_name: str) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "guard_model": guard_name,
            "guard_backend": self.backend,
            "guard_base_url_set": bool(self.base_url),
            "prompt_name": self.prompt_name,
            "prompt_file": self.prompt_file,
            "response_parser": self.response_parser,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "blocking_mode": self.blocking_mode,
            "confidence_threshold": self.confidence_threshold,
            "generic_feedback": self.generic_feedback,
            "feedback_mode": self.feedback_mode,
            "blocked_history_mode": self.blocked_history_mode,
            "max_replans": self.max_replans,
            "guard_reconsideration": self.guard_reconsideration,
            "record_full_guard_context": self.record_full_guard_context,
            "agent_model": self.agent_model,
            "agent_base_url_set": bool(self.agent_base_url),
            "agent_server_json": self.agent_server_json,
            "agent_port": self.agent_port,
            "agent_no_proxy": self.agent_no_proxy,
            "agent_system_suffix_set": bool(self.agent_system_suffix),
            "agent_api_key_set": bool(self.agent_api_key),
            "api_key_set": bool(self.api_key),
            "concurrency": self.concurrency,
            "proxy_set": bool(self.proxy),
            "judge_model": self.judge_model,
            "judge_base_url_set": bool(self.judge_base_url),
            "judge_server_json": self.judge_server_json,
            "judge_api_key_set": bool(self.judge_api_key),
            "suites": self.suites,
            "attack": self.attack,
            "run_benign_with_attack": self.run_benign_with_attack,
            "skip_injection_precheck": self.skip_injection_precheck,
            "user_tasks": self.user_tasks,
            "injection_tasks": self.injection_tasks,
            "benchmark_version": self.benchmark_version,
            "logdir": self.logdir,
            "agentdojo_source_root": self.agentdojo_source_root,
            "subset": self.subset,
            "max_turns": self.max_turns,
            "dataset_path": self.dataset_path,
            "agentharm_source_root": self.agentharm_source_root,
            "tools_root": self.tools_root,
            "graders_module_path": self.graders_module_path,
            "n_irrelevant_tools": self.n_irrelevant_tools,
            "behavior_ids": self.behavior_ids,
            "limit": self.limit,
            "run_started_at": datetime.now(timezone.utc).isoformat(),
        }

    def run_tag(self, *, guard_prompt_tag: str | None) -> str:
        tag = guard_prompt_tag or "no_defense"
        if self.benchmark == "agentharm":
            tag = f"{tag}_{self.subset or 'harmful'}"
        return tag
