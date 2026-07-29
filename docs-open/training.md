# Training

The published training recipe uses full-parameter SFT followed by GRPO on
Qwen/Qwen3-4B-Instruct-2507. Follow the
[Reproduction Guide](reproduction.md) for the supported end-to-end commands,
external dependency setup, and validation steps.

| Stage | Data | Procedure |
|---|---:|---|
| SFT | 3,000 examples | 2 epochs of supervised fine-tuning |
| RL | 4,000 prompts | 100 GRPO rollout updates |

The final SFT-3K/RL-4K data release is forthcoming. Once released, its
expected structure will be:

```text
DATA_ROOT/
  sft3k/agentguard_sft3k_sharegpt.jsonl
  rl4k/agentguard_rl4k_grpo.jsonl
```

## Recipe

The complete framework-independent hyperparameters are in
[`training/release/recipe.yaml`](../training/release/recipe.yaml). The SFT
stage uses an effective batch size of 16, learning rate `2e-5`, cosine schedule,
and maximum sequence length 16,384. The RL stage samples 8 responses for each
of 64 prompts at temperature 1.0, with a maximum response length of 1,024.

## Balance-GRPO

Balance-GRPO reweights a standard per-prompt normalized GRPO advantage using a
class-count term and a safe/unsafe accuracy-gap term. The implementation is
framework-independent in
[`training/release/balance_grpo.py`](../training/release/balance_grpo.py).

For a runnable reference stack, install LLaMA-Factory and SLIME externally,
then use the wrappers in `training/release/scripts/`. SLIME must be pinned to
commit `2640e6cd98c864231b570425e0877dcff295984c`; it is intentionally not
vendored in this repository. The framework-specific commands are in the
[Reproduction Guide](reproduction.md) and
[`training/release/README.md`](../training/release/README.md).
