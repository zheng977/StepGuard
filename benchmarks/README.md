# Benchmark Data

This repository provides the StepGuard evaluation adapters and example
configurations, but does not redistribute static benchmark payloads. Download
each benchmark from its official source and place it at the path specified by
the corresponding file under `configs/`.

The released core suite expects the following local paths:

- `benchmarks/ts_bench/` for TS-Bench;
- `benchmarks/atbench_pro/ATbench-Pro.json` for ATBench-Pro;
- `benchmarks/rjudge/test.json` for R-Judge;
- `benchmarks/agentsafety/test.json` for AgentSafety;
- `benchmark-repos/AgentAuditor-ASSEBench/ASSEBench/dataset/` for ASSEBench.

R-Judge requires the [declared input contract](../docs-open/evaluation.md#r-judge-input-contract).
The adapter reads every conversation group in order and preserves follow-up
user turns and feedback after the final action. Assemble the upstream records
into a JSON array without changing their contents or labels; external
flattening is not required. Record the source revision and input checksum.

Users are responsible for following the upstream licenses, access conditions,
and usage policies for every benchmark.
