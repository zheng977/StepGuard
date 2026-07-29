from __future__ import annotations

from typing import Any

from evals.dynamic import DYNAMIC_BENCHMARK_REGISTRY
from evals.dynamic.base import BaseDynamicBenchmark
from evals.dynamic.config import DynamicEvalConfig
from src.utils.vllm_server import resolve_vllm_base_url


def create_guardrail(config: DynamicEvalConfig) -> tuple[Any | None, str]:
    if config.no_guard or config.model is None:
        return None, "no_defense"
    if not config.api_key and config.backend == "api":
        raise ValueError("Missing API key. Pass --api-key or export OPENAI_API_KEY.")

    from guardrail.guardrail import PredictiveGuardrail
    from infer.factory import InferFactory

    infer_backend = InferFactory.create(
        config.backend,
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key or "EMPTY",
    )
    guardrail = PredictiveGuardrail(
        infer_backend,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        prompt_name=config.prompt_name,
        prompt_file=config.prompt_file,
        response_parser=config.response_parser,
    )
    return guardrail, config.model


def build_benchmark_kwargs(config: DynamicEvalConfig) -> dict[str, Any]:
    agent_base_url = config.agent_base_url
    agent_no_proxy = config.agent_no_proxy
    if config.agent_server_json:
        resolved = resolve_vllm_base_url(config.agent_server_json)
        agent_base_url = resolved.base_url
        agent_no_proxy = True
    judge_base_url = config.judge_base_url
    if config.judge_server_json:
        judge_base_url = resolve_vllm_base_url(config.judge_server_json).base_url

    if config.benchmark in {"agentdyn", "agentdojo"}:
        return {
            "agent_model": config.agent_model,
            "agent_api_key": config.agent_api_key or ("EMPTY" if (config.agent_port or agent_base_url) else config.api_key),
            "agent_base_url": agent_base_url,
            "agent_port": config.agent_port,
            "agent_no_proxy": agent_no_proxy,
            "agent_system_suffix": config.agent_system_suffix,
            "proxy": config.proxy,
            "concurrency": config.concurrency,
            "suites": config.suites,
            "attack": config.attack,
            "run_benign_with_attack": config.run_benign_with_attack,
            "skip_injection_precheck": config.skip_injection_precheck,
            "user_tasks": config.user_tasks,
            "injection_tasks": config.injection_tasks,
            "benchmark_version": config.benchmark_version,
            "logdir": config.logdir,
            **({"source_root": config.agentdojo_source_root} if config.benchmark == "agentdojo" else {}),
        }
    if config.benchmark == "agentharm":
        return {
            "subset": config.subset or "harmful",
            "dataset_path": config.dataset_path,
            "official_source_root": config.agentharm_source_root,
            "tools_root": config.tools_root,
            "graders_module_path": config.graders_module_path,
            "agent_model": config.agent_model,
            "agent_base_url": agent_base_url,
            "agent_api_key": config.agent_api_key or ("EMPTY" if agent_base_url else config.api_key),
            "agent_no_proxy": agent_no_proxy,
            "agent_system_suffix": config.agent_system_suffix,
            "judge_model": config.judge_model,
            "judge_base_url": judge_base_url,
            "judge_api_key": config.judge_api_key,
            "max_turns": config.max_turns,
            "concurrency": config.concurrency,
            "limit": config.limit,
            "behavior_ids": config.behavior_ids,
            "n_irrelevant_tools": config.n_irrelevant_tools,
        }
    return {}


def create_dynamic_benchmark(config: DynamicEvalConfig) -> BaseDynamicBenchmark:
    bench_cls = DYNAMIC_BENCHMARK_REGISTRY[config.benchmark]
    return bench_cls(**build_benchmark_kwargs(config))
