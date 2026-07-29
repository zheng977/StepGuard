# StepGuard Training Data

This repository intentionally contains no training-data payloads.

The final corpus is planned for a separate data release:

| Split | Rows | Format | Use |
|---|---:|---|---|
| SFT-3K | 3,000 | ShareGPT JSONL | supervised fine-tuning |
| RL-4K | 4,000 | JSONL | GRPO post-training |

Only [SFT manifest](manifests/sft3k_manifest.json) and
[RL manifest](manifests/rl4k_manifest.json) are retained here. They record the
planned final data composition without exposing raw examples or internal
provenance.

For the training recipe and the framework-independent Balance-GRPO reference
implementation, see [training/release/README.md](../release/README.md).
