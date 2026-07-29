#!/usr/bin/env bash
set -euo pipefail

# Full-parameter SFT for the released SFT-3K split. LLaMA-Factory is external:
# pip install llamafactory, or clone its official repository separately.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RELEASE_DIR="${REPO_ROOT}/training/release"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the downloaded StepGuard dataset root}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the SFT checkpoint}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

SFT_JSONL="${DATA_ROOT}/sft3k/agentguard_sft3k_sharegpt.jsonl"
if [[ ! -f "${SFT_JSONL}" ]]; then
  echo "Missing SFT data: ${SFT_JSONL}" >&2
  exit 1
fi
if ! command -v llamafactory-cli >/dev/null; then
  echo "llamafactory-cli is not installed. Install LLaMA-Factory outside this repository first." >&2
  exit 1
fi

WORK_DIR="${OUTPUT_DIR}/release_input"
mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"
ln -sfn "${SFT_JSONL}" "${WORK_DIR}/stepguard_sft3k.jsonl"

python - "${WORK_DIR}/dataset_info.json" <<'PY'
import json
import sys

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "stepguard_sft3k": {
                "file_name": "stepguard_sft3k.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                },
            }
        },
        handle,
        indent=2,
    )
PY

FORCE_TORCHRUN=1 llamafactory-cli train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --trust_remote_code true \
  --stage sft \
  --do_train true \
  --finetuning_type full \
  --dataset_dir "${WORK_DIR}" \
  --dataset stepguard_sft3k \
  --template qwen3_nothink \
  --cutoff_len 16384 \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 4 \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --save_only_model true \
  --report_to none \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --learning_rate 2.0e-5 \
  --num_train_epochs 2 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --deepspeed "${RELEASE_DIR}/ds_z3_config.json"
