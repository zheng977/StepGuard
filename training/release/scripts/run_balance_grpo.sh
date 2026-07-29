#!/usr/bin/env bash
set -euo pipefail

# Balance-GRPO on the released RL-4K split. This wrapper targets only the
# pinned SLIME commit documented in training/release/README.md.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RELEASE_DIR="${REPO_ROOT}/training/release"
SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR to external/slime at the pinned commit}"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to the Megatron-LM checkout required by SLIME}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the downloaded StepGuard dataset root}"
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to the selected SFT checkpoint}"
REF_LOAD="${REF_LOAD:?Set REF_LOAD to its torch-distributed conversion}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR for SLIME checkpoints}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
STEPGUARD_TARGET_GAP="${STEPGUARD_TARGET_GAP:-0.0}"
STEPGUARD_GAP_LAMBDA="${STEPGUARD_GAP_LAMBDA:-2.0}"

RL_JSONL="${DATA_ROOT}/rl4k/agentguard_rl4k_grpo.jsonl"
[[ -f "${RL_JSONL}" ]] || { echo "Missing RL data: ${RL_JSONL}" >&2; exit 1; }
[[ "$(git -C "${SLIME_DIR}" rev-parse HEAD)" == "2640e6cd98c864231b570425e0877dcff295984c" ]] || {
  echo "SLIME_DIR must be checked out at 2640e6cd98c864231b570425e0877dcff295984c" >&2; exit 1;
}
[[ -d "${MEGATRON_DIR}" ]] || { echo "Missing MEGATRON_DIR: ${MEGATRON_DIR}" >&2; exit 1; }

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"
mkdir -p "${SAVE_DIR}"

if [[ "${START_RAY:-0}" == "1" ]]; then
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats
fi

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_DIR}:${RELEASE_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"STEPGUARD_TARGET_GAP\":\"${STEPGUARD_TARGET_GAP}\",\"STEPGUARD_GAP_LAMBDA\":\"${STEPGUARD_GAP_LAMBDA}\"}}"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_DIR}/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --ref-load "${REF_LOAD}" \
  --save "${SAVE_DIR}" \
  --save-interval 25 \
  --prompt-data "${RL_JSONL}" \
  --input-key instruction \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --custom-rm-path stepguard_slime_adapter.reward_func \
  --custom-reward-post-process-path stepguard_slime_adapter.post_process_rewards \
  --num-rollout 100 \
  --rollout-batch-size 64 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 1024 \
  --rollout-temperature 1.0 \
  --global-batch-size 512 \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-24576}" \
  --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static "${SGLANG_MEMORY_FRACTION:-0.70}" \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.001 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.001 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 5.0e-7 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-backend flash
