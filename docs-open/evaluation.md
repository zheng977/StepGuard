# Evaluation

StepGuard uses a unified evaluator for static and dynamic agent-safety
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

### R-Judge input contract

The `rjudge` registry entry uses `AgentHarmTrajBenchAdapter`. The input file is
a JSON array of records with an identifier (`id`, or `conv_id`), a binary
`label`, and `contents`: a non-empty list of non-empty conversation groups.
Each turn has role `user`, `agent`, or `environment`. The first turn of the
complete record must be a user request and the record must contain an agent
action. Malformed groups and unknown roles raise a sample-specific error
before inference, rather than being silently skipped.

All groups are read in source order. Group boundaries do not create new
evaluation cases: every record retains its original identifier and label.
Follow-up user turns are stored as `UserMessage` entries, distinct from tool
observations. Agent action step numbers increase across group boundaries.
Already flattened records are also supported, provided all original turns
and roles are retained; no external flattening step is required.

The last agent action is the case's `action`. Earlier turns become
`history.steps`; later observations and user messages become
`history.post_action_steps`. The `stepguard_traj` profile and all released
baseline trajectory profiles render these events in order, with the final
action exactly once. Proactive action profiles use only the preceding
history and the action, so feedback produced after an action is not leaked
into its pre-execution assessment.

To assemble `benchmarks/rjudge/test.json` from an upstream checkout, concatenate
the arrays of records from its data JSON files without altering each record's
`contents` or labels. Record the upstream revision, assembly script, and final
input checksum. Do not turn conversation groups into separately labeled cases
or remove samples when reporting a full-benchmark result.

### R-Judge correction and historical results

This corrects the input omission reported in
[issue #1](https://github.com/zheng977/StepGuard/issues/1). An offline check
against R-Judge revision `83ce301da3ad50dd8b397e772863f5411c3d3dc2` preserves all
3,098 turns across 571 records, including the 16 multi-group records. All six
released trajectory profiles were checked for content and ordering.

Relative to StepGuard revision `47c9011ee73c90be403846d35476ec55d5dab63b`,
the `stepguard_traj` inputs change for 157 records: the 16 multi-group records
and 141 single-group records with previously omitted post-action feedback.
The other 414 single-group inputs are byte-identical. This is an input audit,
not a model-score comparison; original labels are unchanged.

The paper's actual assembled `test.json` and its historical preprocessing
have not been verified against these upstream records. Assessing published
score impact requires the original input, generation script, and per-case
prompts/predictions. Re-evaluate affected models using a consistent input
contract and a separate output directory; do not overwrite historical results
or infer a score/ranking change from input counts alone.

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
