# StepGuard Documentation

This directory is the public documentation surface for StepGuard. It describes
the released model interface, evaluation entry points, and training recipe.
Internal experiment notes, cluster paths, intermediate checkpoints, and
unreleased data-generation assets are intentionally excluded.

## Start Here

1. [Reproduction Guide](reproduction.md): the complete supported path from a
   fresh clone through static/dynamic evaluation, SFT, and Balance-GRPO.
2. [Quickstart](quickstart.md): install the package and run an action-level or
   trajectory-level guard evaluation.
3. [Prompt Contract](prompts.md): the two modes in the single released
   StepGuard prompt family and their exact output schema.
4. [Evaluation](evaluation.md): static and dynamic evaluation entry points,
   benchmark mapping, and result artifacts.
5. [Dynamic Protocol](dynamic_evaluation.md): the paper's self-reflect
   feedback setting for interactive benchmarks.
6. [Training](training.md): the published SFT/RL recipe and Balance-GRPO
   reference implementation; end-to-end data release is forthcoming.

The released checkpoints are hosted separately on Hugging Face. The final
SFT-3K/RL-4K corpus and StepGen data-generation engine are planned for a
follow-up release.

## Scope

StepGuard is a predictive guardrail for tool-using agents. It classifies either
the current action or a completed action-observation trajectory as `safe` or
`unsafe`, diagnoses the relevant risk source, and, for a trajectory, identifies
the unsafe agent-action step.
