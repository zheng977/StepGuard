# Evaluation Entrypoints

All public evaluation runs use a YAML configuration under `configs/`.

| Script | Use |
|---|---|
| `run_eval.py` | One guard model on one static benchmark. |
| `run_batch_eval.py` | Multiple guard models on one static benchmark. |
| `run_eval_suite.py` | A model matrix across the static core suite. |
| `run_dynamic_eval.py` | One guard model on one dynamic benchmark. |
| `run_batch_dynamic_eval.py` | Multiple guard models on one dynamic benchmark. |
| `run_agent.py` | Minimal OpenAI-compatible agent demonstration. |

Examples:

```bash
python scripts/eval/run_batch_eval.py \
  --config configs/benchmarks/atbench_pro.example.yaml

python scripts/eval/run_eval_suite.py \
  --config configs/eval_suites/static_core.example.yaml

python scripts/eval/run_batch_dynamic_eval.py \
  --config configs/dynamic/self_reflect.example.yaml
```

The dynamic paper setting is `self_reflect` feedback with clean blocked
history, three replans, and guard reconsideration disabled. See
`docs-open/dynamic_evaluation.md` for the complete protocol.
