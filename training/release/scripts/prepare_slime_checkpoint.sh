#!/usr/bin/env bash
set -euo pipefail

# Convert the selected SFT Hugging Face checkpoint to the pinned SLIME
# torch-distributed format used by the actor and reference policy.

SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR to external/slime at the pinned commit}"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to the Megatron-LM checkout required by SLIME}"
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to the SFT Hugging Face checkpoint}"
TORCH_DIST_CHECKPOINT="${TORCH_DIST_CHECKPOINT:?Set TORCH_DIST_CHECKPOINT output path}"

if [[ "$(git -C "${SLIME_DIR}" rev-parse HEAD)" != "2640e6cd98c864231b570425e0877dcff295984c" ]]; then
  echo "SLIME_DIR must be checked out at 2640e6cd98c864231b570425e0877dcff295984c" >&2
  exit 1
fi

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"
PYTHONPATH="${MEGATRON_DIR}" python "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --save "${TORCH_DIST_CHECKPOINT}"
