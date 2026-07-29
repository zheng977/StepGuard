"""Dynamic benchmark evaluation framework."""

from __future__ import annotations

from typing import Any

from evals.dynamic.base import BaseDynamicBenchmark, DynamicEvalResult

DYNAMIC_BENCHMARK_REGISTRY: dict[str, type[BaseDynamicBenchmark]] = {}


def _register_agentdyn() -> None:
    from evals.dynamic.agentdyn import AgentDynBenchmark

    DYNAMIC_BENCHMARK_REGISTRY["agentdyn"] = AgentDynBenchmark


def _register_agentdojo() -> None:
    from evals.dynamic.agentdojo import AgentDojoBenchmark

    DYNAMIC_BENCHMARK_REGISTRY["agentdojo"] = AgentDojoBenchmark


def _register_agentharm() -> None:
    from evals.dynamic.agentharm import AgentHarmBenchmark

    DYNAMIC_BENCHMARK_REGISTRY["agentharm"] = AgentHarmBenchmark


# Lazy registration to avoid import errors when the benchmark isn't available.
try:
    _register_agentdyn()
except ImportError:
    pass
try:
    _register_agentdojo()
except ImportError:
    pass
try:
    _register_agentharm()
except ImportError:
    pass
__all__ = [
    "BaseDynamicBenchmark",
    "DYNAMIC_BENCHMARK_REGISTRY",
    "DynamicEvalResult",
]
