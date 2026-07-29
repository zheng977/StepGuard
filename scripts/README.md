# Scripts

This directory contains only the public evaluation entry points. Training
commands live in `training/release/scripts/`; final datasets and checkpoints
are distributed separately.

| Script | Purpose |
|---|---|
| `eval/run_eval.py` | Evaluate one guard model on one static benchmark. |
| `eval/run_batch_eval.py` | Evaluate multiple guard models on one static benchmark. |
| `eval/run_eval_suite.py` | Evaluate one model matrix across the static core suite. |
| `eval/run_dynamic_eval.py` | Evaluate one guard in one dynamic benchmark. |
| `eval/run_batch_dynamic_eval.py` | Evaluate multiple guards in one dynamic benchmark. |
| `eval/run_agent.py` | Minimal OpenAI-compatible agent demonstration. |
| `results/index_eval_results.py` | Rebuild a CSV/Markdown index from evaluation summaries. |

Use the parameterized examples under `configs/`:

```bash
python scripts/eval/run_eval_suite.py \
  --config configs/eval_suites/static_core.example.yaml

python scripts/eval/run_batch_dynamic_eval.py \
  --config configs/dynamic/self_reflect.example.yaml
```

Historical data pipelines, cluster operations, checkpoint conversion, and
paper-specific result analyses are intentionally excluded from the release.
