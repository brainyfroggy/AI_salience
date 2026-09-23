#!/usr/bin/env bash
#SBATCH --job-name=vlm-saliency
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --account=mzding
#SBATCH --qos=mzding
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=180gb
#SBATCH --time=12:00:00

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this script as a Slurm array job." >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-/blue/mzding/yujunchen/projects/AI_salience}"
MANIFEST="${MANIFEST:-$REPO_DIR/manifests/emod_complete_hpg.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/output/vlm_prompt_saliency}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-vlm}"
HF_HOME="${HF_HOME:-/blue/mzding/yujunchen/projects/.cache/huggingface}"
MODEL_IDS="${MODEL_IDS:-${MODELS:-meta-llama/Llama-3.2-11B-Vision-Instruct}}"
MODEL_IDS="${MODEL_IDS//,/|}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
LIMIT="${LIMIT:-0}"

cd "$REPO_DIR"
mkdir -p logs "$OUTPUT_DIR" "$HF_HOME"

reject_restricted_models() {
  local model_ids_lower
  model_ids_lower="$(tr '[:upper:]' '[:lower:]' <<< "$MODEL_IDS")"
  local restricted_patterns=(
    "qwen"
    "internvl"
    "opengvlab"
    "deepseek"
    "chatglm"
    "glm-"
    "minicpm"
    "yi-vl"
    "cogvlm"
    "internlm"
    "baichuan"
    "sensetime"
  )
  local pattern
  for pattern in "${restricted_patterns[@]}"; do
    if [[ "$model_ids_lower" == *"$pattern"* ]]; then
      echo "Refusing restricted model identifier in MODEL_IDS: $MODEL_IDS" >&2
      exit 10
    fi
  done
}

reject_restricted_models

IFS='|' read -r -a MODEL_ARRAY <<< "$MODEL_IDS"
MODEL_COUNT="${#MODEL_ARRAY[@]}"
TOTAL_TASKS="$((MODEL_COUNT * NUM_SHARDS))"
TASK_ZERO="$((SLURM_ARRAY_TASK_ID - 1))"
if [[ "$TASK_ZERO" -lt 0 || "$TASK_ZERO" -ge "$TOTAL_TASKS" ]]; then
  echo "Task $SLURM_ARRAY_TASK_ID outside expected 1-$TOTAL_TASKS" >&2
  exit 3
fi

MODEL_INDEX="$((TASK_ZERO / NUM_SHARDS))"
SHARD_INDEX="$((TASK_ZERO % NUM_SHARDS))"
MODEL_ID="${MODEL_ARRAY[$MODEL_INDEX]}"

module load conda || true
module load cuda || true
source "$VENV_DIR/bin/activate"

echo "Job: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID / $TOTAL_TASKS"
echo "Node: $(hostname)"
echo "Model: $MODEL_ID"
echo "Shard: $SHARD_INDEX / $NUM_SHARDS"
echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT_DIR"
echo "Limit per shard: $LIMIT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

EXTRA_ARGS=()
if [[ "$LIMIT" -gt 0 ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

python scripts/run_vlm_prompt_saliency.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --model-id "$MODEL_ID" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${EXTRA_ARGS[@]}"
