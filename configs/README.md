# Configuration Guide

This directory contains only public, parameterized examples. Set model paths,
API endpoints, and credentials through environment variables; do not add
cluster-specific paths, checkpoints, or credentials to YAML files.

| File | Purpose |
|---|---|
| `batch_eval.example.yaml` | Run one static benchmark against one or more guard models. |
| `eval_suites/static_core.example.yaml` | Run the released static benchmark suite. |
| `dynamic/self_reflect.example.yaml` | Run the paper's dynamic self-reflect protocol. |
| `benchmarks/*.example.yaml` | Run one released static benchmark at a time. |

## Static Evaluation

Set `AGENTGUARD_MODEL_PATH` to a local Hugging Face checkpoint, then run:

```bash
python scripts/eval/run_eval_suite.py \
  --config configs/eval_suites/static_core.example.yaml
```

The core suite covers TS-Bench, ATBench-Pro, R-Judge, AgentSafety, and the
ASSEBench safety/security splits. Use `stepguard` for action-level inputs and
`stepguard_traj` for trajectory-level inputs.

Benchmark payloads are intentionally excluded from the repository. Download
them from their official sources and preserve the paths declared in the
example configurations before running the suite.

For one benchmark at a time, use a configuration under `benchmarks/`:

```bash
python scripts/eval/run_batch_eval.py \
  --config configs/benchmarks/atbench_pro.example.yaml
```

## Dynamic Evaluation

Set the endpoint variables described in
`dynamic/self_reflect.example.yaml`, then run:

```bash
python scripts/eval/run_batch_dynamic_eval.py \
  --config configs/dynamic/self_reflect.example.yaml
```

This is the paper setting: `self_reflect` feedback, clean blocked history,
three replans, and no guard reconsideration.

Historical experiment configurations intentionally do not live in this
directory. Keep a one-off experiment configuration outside the release tree or
record it with the corresponding result artifact.
