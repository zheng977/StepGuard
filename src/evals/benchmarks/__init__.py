from .agentharm_traj_bench import AgentHarmTrajBenchAdapter
from .agentsafety_bench import AgentSafetyBenchAdapter
from .assebench import ASSEBenchAdapter
from .atbench_pro import ATBenchProAdapter
from .ts_bench import TSBenchAdapter


def _rjudge_factory(input_path, **kwargs):
    return AgentHarmTrajBenchAdapter(input_path, bench_name="rjudge", **kwargs)


BENCHMARK_REGISTRY: dict[str, type] = {
    "ts_bench": TSBenchAdapter,
    "agentsafety": AgentSafetyBenchAdapter,
    "rjudge": _rjudge_factory,
    "atbench_pro_traj": lambda input_path, **kw: ATBenchProAdapter(input_path, mode="trajectory", **kw),
    "assebench": ASSEBenchAdapter,
    "assebench_last_action": lambda input_path, **kw: ASSEBenchAdapter(input_path, mode="last_action", **kw),
}

__all__ = [
    "AgentSafetyBenchAdapter",
    "ASSEBenchAdapter",
    "TSBenchAdapter",
    "BENCHMARK_REGISTRY",
]
