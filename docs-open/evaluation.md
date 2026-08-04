# Evaluation

AgentDoG-Step uses a unified evaluator for static and dynamic agent-safety
benchmarks. Static evaluation sends a serialized action or trajectory to a
guard model. Dynamic evaluation runs an agent loop and asks the guard before
each proposed tool action.

## Static Evaluation

Use `scripts/eval/run_eval_suite.py` for one model across several benchmarks,
`scripts/eval/run_batch_eval.py` for a model matrix on one benchmark, and
`scripts/eval/run_eval.py` for a single run.

| Benchmark type | Prompt / parser |
|---|---|
| Action-level (for example, TS-Bench) | `stepguard` |
| Trajectory-level (ATBench-Pro, R-Judge, AgentSafety, ASSE-Bench) | `stepguard_traj` |
| Third-party baseline | Its native prompt and parser |

The canonical multi-benchmark configuration is
[`configs/eval_suites/static_core.example.yaml`](../configs/eval_suites/static_core.example.yaml).
It uses greedy decoding (`temperature: 0.0`). Evaluation outputs contain the
per-case records, `results_summary.json`, and benchmark-level metric tables.

Static benchmark payloads are not redistributed in this repository. Obtain
each dataset from its official source, observe its license and access terms,
and place it at the `input` path declared in the suite configuration. The
vendored `benchmark-repos/` directory contains source dependencies for the
released dynamic adapters; it does not replace dataset-specific downloads.

## Dynamic Evaluation

Use `scripts/eval/run_dynamic_eval.py` for one guarded agent or
`scripts/eval/run_batch_dynamic_eval.py` for a set of guard models. The agent
model and guard model both use OpenAI-compatible APIs.

The paper configuration is `self_reflect` feedback with clean replanning: the
blocked action is not added to the persisted agent history, and guard reasoning
is not exposed to the agent. See [Dynamic Protocol](dynamic_evaluation.md) for
the exact configuration and intervention sequence. The public batch template
is [`configs/dynamic/self_reflect.example.yaml`](../configs/dynamic/self_reflect.example.yaml).

For attacked tasks, `ASR` is the proportion of cases where the attack succeeds;
lower is better. `security_rate` is the complement, and `utility` measures task
completion on benign tasks.

## Reproducibility

Use fixed checkpoints and greedy decoding for deterministic static runs. When
reporting repeated evaluation, distinguish evaluator-repeat variability from
variation across independently trained model seeds.
