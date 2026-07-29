# StepGuard Training Recipe

This directory documents the released StepGuard training recipe without
vendoring a training framework. The data format, hyperparameters, reward, and
Balance-GRPO update are framework-independent.

## Planned Data Release

The final SFT-3K/RL-4K data release and StepGen engine are forthcoming. This
directory publishes the training recipe and reference implementation now so
that the method is auditable.

| Split | Rows | Format | Use |
|---|---:|---|---|
| SFT-3K | 3,000 | ShareGPT JSONL | supervised fine-tuning |
| RL-4K | 4,000 | JSONL | on-policy GRPO |

The manifest files in the dataset record the final safe/unsafe and
action/trajectory composition. Intermediate synthetic pools, teacher outputs,
rollout logs, checkpoints, and ablation data are not released.

## Reproduction Recipe

The exact hyperparameters are in [recipe.yaml](recipe.yaml).

1. Start from `Qwen/Qwen3-4B-Instruct-2507`.
2. Full-parameter SFT on SFT-3K using the ShareGPT conversations for 2 epochs.
3. Continue from the SFT checkpoint with GRPO on RL-4K for 100 rollout updates.
4. For each update, sample 8 responses for each of 64 prompts with temperature
   1.0 and maximum response length 1,024.
5. Apply the format-gated reward below, normalize rewards within each prompt's
   8 sampled responses, and use the clipped GRPO objective.
6. For Balance-GRPO, multiply each normalized advantage by the class-count and
   accuracy-gap weights returned by [balance_grpo.py](balance_grpo.py).

The reference implementation is deliberately small: `balance_grpo.py`
contains the framework-independent math and `stepguard_slime_adapter.py`
binds that math to the pinned SLIME reward-post-process API.

## Exact End-to-End Reproduction

There are two supported reproduction levels:

- **Method reproduction:** use any full-parameter SFT and GRPO implementation
  with `recipe.yaml` and `balance_grpo.py`. No external training repository is
  required by this codebase.
- **Released stack reproduction:** run the scripts in `scripts/` with
  external LLaMA-Factory and the official
  [SLIME](https://github.com/THUDM/slime) implementation pinned at commit
  `2640e6cd98c864231b570425e0877dcff295984c`. The framework source and
  cluster-specific launch environment are intentionally not vendored here.

## Runnable Entry Points After Data Release

After the data release, download the corpus to a local directory and set
`DATA_ROOT` to that directory. Its expected layout will be:

```text
DATA_ROOT/
  sft3k/agentguard_sft3k_sharegpt.jsonl
  rl4k/agentguard_rl4k_grpo.jsonl
```

### 1. SFT

Install LLaMA-Factory externally, then run:

```bash
export DATA_ROOT=/path/to/stepguard-sft3k-rl4k
export OUTPUT_DIR=/path/to/outputs/stepguard-sft-4b
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash training/release/scripts/run_sft.sh
```

The launcher creates a local LLaMA-Factory dataset registry, performs
full-parameter SFT with ZeRO-3, and saves the two-epoch checkpoint. It does
not modify the downloaded data.

### 2. Prepare the SFT checkpoint for SLIME

Clone SLIME and its Megatron dependency outside this repository:

```bash
git clone https://github.com/THUDM/slime.git external/slime
git -C external/slime checkout 2640e6cd98c864231b570425e0877dcff295984c
```

Follow the pinned SLIME commit's installation instructions for its compatible
Megatron-LM environment. Convert the selected SFT checkpoint once:

```bash
export SLIME_DIR=$PWD/external/slime
export MEGATRON_DIR=/path/to/Megatron-LM
export HF_CHECKPOINT=/path/to/outputs/stepguard-sft-4b/checkpoint-376
export TORCH_DIST_CHECKPOINT=/path/to/checkpoints/stepguard-sft-4b_torch_dist
bash training/release/scripts/prepare_slime_checkpoint.sh
```

### 3. Balance-GRPO

```bash
export DATA_ROOT=/path/to/stepguard-sft3k-rl4k
export HF_CHECKPOINT=/path/to/outputs/stepguard-sft-4b/checkpoint-376
export REF_LOAD=/path/to/checkpoints/stepguard-sft-4b_torch_dist
export SAVE_DIR=/path/to/outputs/stepguard-balance-grpo
export SLIME_DIR=$PWD/external/slime
export MEGATRON_DIR=/path/to/Megatron-LM
export START_RAY=1
bash training/release/scripts/run_balance_grpo.sh
```

`run_balance_grpo.sh` loads
`stepguard_slime_adapter.reward_func` and
`stepguard_slime_adapter.post_process_rewards` through SLIME's public plugin
flags. The latter reproduces SLIME's per-prompt reward normalization and then
applies `c_i * omega_i` before samples enter the GRPO policy loss.

## Reward

The required output has exactly one each of `<Analysis>`, `<Judgment>`,
`<RiskSourcePresent>`, and `<RiskSource>`; trajectory inputs additionally use
one `<UnsafeStep>`. Let `j` be correct safe/unsafe judgment, `r` be correct
risk-source diagnosis, and `f` be valid format. The reward is:

$$
R = f\left(0.7j + 0.3jr\right).
$$

Risk-source credit is gated on a correct safe/unsafe judgment. This keeps the
safety decision as the primary optimization target.

## Balance-GRPO

For a rollout batch, let \(N_y\) be the number of samples with gold class
\(y\), and let \(a_s,a_u\) be smoothed safe and unsafe accuracy. The released
implementation uses:

$$
c_y = \operatorname{clip}\left(\frac{N}{2N_y}, 0.5, 2.0\right)
$$

$$
g = \operatorname{clip}(a_u-a_s-g_0, -0.5, 0.5),
\qquad |g| < 0.02 \Rightarrow g=0
$$

$$
\omega_s = \operatorname{clip}(1+2g, 0.5, 2.0),
\qquad
\omega_u = \operatorname{clip}(1-2g, 0.5, 2.0)
$$

$$
A_i^{\mathrm{balanced}} =
\operatorname{clip}(c_{y_i}\omega_{y_i},0.75,1.5)A_i
$$

The default target gap is \(g_0=0\). Positive `target_gap` favors protective
behavior; negative `target_gap` favors utility. This is a reweighting rule,
not a separate training system.

## Scope

This release makes the paper's training procedure auditable and executable
with the documented external dependencies. It intentionally does not vendor a
third-party training framework, cluster launcher, or dependency lockfile.
