# Quickstart

## Install

```bash
git clone <repository-url> AgentGuard
cd AgentGuard
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For local checkpoint inference, install a compatible vLLM release in the same
environment. For remote inference, provide an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

## Run Static Evaluation

Set the local Hugging Face checkpoint path and run the canonical suite:

```bash
export AGENTGUARD_MODEL_PATH=/path/to/stepguard-checkpoint
CUDA_VISIBLE_DEVICES=0,1 \
  python scripts/eval/run_eval_suite.py \
  --config configs/eval_suites/static_core.example.yaml
```

The suite uses `stepguard` for action-level benchmarks and `stepguard_traj`
for trajectory-level benchmarks. To validate the configuration without starting
vLLM, append `--dry-run`.

## Run One Endpoint Evaluation

For an already running OpenAI-compatible endpoint:

```bash
python scripts/eval/run_eval.py \
  --benchmark ts_bench \
  --input benchmarks/ts_bench \
  --ts-subset all \
  --backend api \
  --model stepguard \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --prompt-name stepguard \
  --response-parser stepguard \
  --temperature 0.0 \
  --concurrency 64 \
  --output-root results/ts_bench
```

For a trajectory benchmark, replace the benchmark input and use
`--prompt-name stepguard_traj --response-parser stepguard_traj`.
