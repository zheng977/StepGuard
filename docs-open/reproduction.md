# Reproduction Guide

This guide is the canonical path for reproducing StepGuard. It separates the
three supported goals because they have different hardware and dependency
requirements:

1. **Use a released checkpoint**: run static or dynamic evaluation now.
2. **Reproduce SFT**: fine-tune Qwen3-4B on SFT-3K after the planned data
   release.
3. **Reproduce Balance-GRPO**: continue from an SFT checkpoint on RL-4K after
   the planned data release.

The repository intentionally does not include model weights, training data,
intermediate checkpoints, cluster launchers, or third-party training source.

## 1. Environment

```bash
git clone https://github.com/zheng977/StepGuard.git
cd StepGuard
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For local checkpoint serving, install vLLM in the same environment:

```bash
pip install -e .[serve]
```

Static evaluation of the released 4B checkpoint requires one or more CUDA
GPUs. The example suite uses two GPUs (`tensor_parallel_size: 2`), but it can
run on one GPU after changing that field to `1` in a copied config. Full SFT
and GRPO reproduction use four H200 GPUs in the reported recipe; other
hardware may require smaller batch sizes or sequence lengths.

## 2. Download Released Checkpoints

Download StepGuard from Hugging Face:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download ninty-seven/StepGuard \
  --local-dir "$PWD/artifacts/StepGuard"
```

The following data layout will apply after the planned corpus release:

```text
stepguard-data/
  sft3k/agentguard_sft3k_sharegpt.jsonl
  rl4k/agentguard_rl4k_grpo.jsonl
```

Use an immutable Hugging Face revision when reporting results. Record the
model repository, revision, evaluator commit, config file, and GPU count with
each run.

## 3. Verify the Installation

Run unit tests and validate the canonical static configuration before starting
a model server:

```bash
PYTHONPATH=src python -m unittest -q \
  tests.test_guardrail tests.test_prompt_render tests.test_eval_suite \
  tests.test_dynamic_eval_runner tests.test_run_eval

python scripts/eval/run_eval_suite.py \
  --config configs/eval_suites/static_core.example.yaml \
  --dry-run
```

The dry run checks benchmark paths, prompt/parser selection, and model launch
arguments without allocating a GPU.

## 4. Static Evaluation

Set the checkpoint path and execute the canonical suite:

```bash
export AGENTGUARD_MODEL_PATH="$PWD/artifacts/stepguard-model"
CUDA_VISIBLE_DEVICES=0,1 \
  python scripts/eval/run_eval_suite.py \
  --config configs/eval_suites/static_core.example.yaml
```

The suite uses greedy decoding (`temperature: 0.0`) and evaluates TS-Bench,
ATBench-Pro, R-Judge, AgentSafety, and the two ASSEBench splits. It writes one
directory per benchmark under `results/static_core_example/`; each contains
per-example predictions and a `results_summary.json`.

Action-level inputs use `stepguard`; trajectory inputs use `stepguard_traj`.
Do not substitute historical prompt names. Their released contract is in
[prompts.md](prompts.md).

To evaluate a served OpenAI-compatible guard rather than a local vLLM model,
use `scripts/eval/run_eval.py` and supply `--backend api`, `--base-url`, and
`--api-key`. See [evaluation.md](evaluation.md).

## 5. Dynamic Evaluation

Dynamic evaluation requires two OpenAI-compatible services: an agent endpoint
and a guard endpoint. The paper setting is self-reflect feedback with clean
history and at most three replans.

```bash
export AGENT_MODEL=<AGENT_MODEL_NAME>
export AGENT_BASE_URL=http://127.0.0.1:8000/v1
export GUARD_MODEL=<GUARD_MODEL_NAME>
export GUARD_BASE_URL=http://127.0.0.1:8001/v1
export GUARD_API_KEY=EMPTY

python scripts/eval/run_batch_dynamic_eval.py \
  --config configs/dynamic/self_reflect.example.yaml
```

The public template runs AgentDojo. AgentDyn additionally needs the retained
third-party benchmark checkout and its dependencies. The exact intervention
policy and reported metrics are documented in
[dynamic_evaluation.md](dynamic_evaluation.md).

## 6. Reproduce SFT After Data Release

SFT uses the external LLaMA-Factory CLI. The SFT-3K corpus is not included in
the initial code-and-model release, so this section becomes runnable after the
data release. Install LLaMA-Factory outside this repository, then ensure
`llamafactory-cli` is available on `PATH`.

```bash
export DATA_ROOT="$PWD/artifacts/stepguard-data"
export OUTPUT_DIR="$PWD/artifacts/stepguard-sft-4b"
export MODEL_NAME_OR_PATH=Qwen/Qwen3-4B-Instruct-2507

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash training/release/scripts/run_sft.sh
```

The wrapper creates a local dataset registry inside `OUTPUT_DIR`, performs
full-parameter SFT for two epochs, and retains at most two model checkpoints.
The hyperparameters are fixed in [recipe.yaml](../training/release/recipe.yaml).
The selected output checkpoint, rather than a hard-coded step number, is the
input to the RL stage.

The released wrapper has been validated with the LLaMA-Factory CLI interface,
but LLaMA-Factory is not vendored. The release tag must record the exact
external LLaMA-Factory version used for a fully environment-pinned training
reproduction; do not silently substitute an arbitrary newer version when
comparing paper numbers.

## 7. Reproduce Balance-GRPO After Data Release

This stage additionally requires the planned RL-4K corpus. It uses external
SLIME pinned to commit
`2640e6cd98c864231b570425e0877dcff295984c` and the compatible Megatron-LM
checkout required by that SLIME revision.

```bash
git clone https://github.com/THUDM/slime.git external/slime
git -C external/slime checkout 2640e6cd98c864231b570425e0877dcff295984c
```

Install the dependencies required by that exact SLIME revision, then convert
the SFT checkpoint once:

```bash
export SLIME_DIR="$PWD/external/slime"
export MEGATRON_DIR=/path/to/Megatron-LM
export HF_CHECKPOINT=/path/to/selected-sft-checkpoint
export TORCH_DIST_CHECKPOINT="$PWD/artifacts/stepguard-sft-torch-dist"

bash training/release/scripts/prepare_slime_checkpoint.sh
```

Launch the 100-update Balance-GRPO run:

```bash
export DATA_ROOT="$PWD/artifacts/stepguard-data"
export REF_LOAD="$TORCH_DIST_CHECKPOINT"
export SAVE_DIR="$PWD/artifacts/stepguard-balance-grpo"
export START_RAY=1
export STEPGUARD_TARGET_GAP=0.0
export STEPGUARD_GAP_LAMBDA=2.0

bash training/release/scripts/run_balance_grpo.sh
```

`STEPGUARD_TARGET_GAP=0.0` reproduces the balanced operating point. Positive
values prefer protective behavior; negative values prefer utility. The method
implementation, clipping, smoothing, and deadband are in
[`training/release/balance_grpo.py`](../training/release/balance_grpo.py).

## 8. What to Compare

For a paper-level reproduction, evaluate the selected SFT and Balance-GRPO
checkpoints with the same static config, greedy decoding, and benchmark
versions. Report Acc, F1, Safe Acc, Unsafe Acc, and the absolute Safe/Unsafe
accuracy gap. For dynamic runs, also report attack success rate and utility
under the fixed self-reflect protocol.

Static decoding is deterministic for a fixed model server and configuration.
Repeated runs measure evaluator/server variation, not independent training
seed variation.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `vLLM process exited before becoming ready` | Verify `CUDA_VISIBLE_DEVICES`, a CUDA-enabled vLLM installation, and `tensor_parallel_size` does not exceed visible GPUs. |
| `${VAR}` cannot be parsed in a YAML config | Export every variable referenced by the config, or replace it in a copied config. |
| `llamafactory-cli` not found | Install LLaMA-Factory in the active environment; it is intentionally not vendored. |
| DeepSpeed cannot find `nvcc` | Use a CUDA development environment with `nvcc` available, and set `CUDA_HOME` to that toolkit. |
| SLIME revision check fails | Checkout exactly `2640e6cd98c864231b570425e0877dcff295984c`; do not use a moving branch head. |
| CUDA OOM during SFT | Reduce `cutoff_len`, increase gradient accumulation to preserve effective batch size, or use a larger GPU configuration. |

For benchmark-specific prompt semantics and dynamic feedback details, use the
linked documentation rather than unpublished historical notes in `docs/`.
