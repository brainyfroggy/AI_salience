#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/blue/mzding/yujunchen/projects/AI_salience}"
MANIFEST="${MANIFEST:-$REPO_DIR/manifests/emod_complete_hpg.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/output/vlm_prompt_saliency}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-vlm}"
HF_HOME="${HF_HOME:-/blue/mzding/yujunchen/projects/.cache/huggingface}"
MODEL_IDS="${MODEL_IDS:-${MODELS:-meta-llama/Llama-3.2-11B-Vision-Instruct}}"
MODEL_IDS="${MODEL_IDS//,/|}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
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

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest does not exist: $MANIFEST" >&2
  exit 2
fi

IFS='|' read -r -a MODEL_ARRAY <<< "$MODEL_IDS"
MODEL_COUNT="${#MODEL_ARRAY[@]}"
TASKS="$((MODEL_COUNT * NUM_SHARDS))"

echo "Submitting VLM saliency array."
echo "Repo: $REPO_DIR"
echo "Manifest: $MANIFEST"
echo "Outputs: $OUTPUT_DIR"
echo "Models: $MODEL_IDS"
echo "Shards/model: $NUM_SHARDS"
echo "Array tasks: $TASKS"
echo "Max concurrent tasks: $MAX_CONCURRENT"
echo "GPUs per task: $GPUS_PER_TASK"
echo "Limit per shard: $LIMIT"

SETUP_JOB="$(sbatch \
  --parsable \
  --export=ALL,REPO_DIR="$REPO_DIR",VENV_DIR="$VENV_DIR",HF_HOME="$HF_HOME",MODEL_IDS="$MODEL_IDS" \
  hpg/setup_vlm_env.sbatch)"

echo "Submitted environment setup job: $SETUP_JOB"

sbatch --array="1-${TASKS}%${MAX_CONCURRENT}" \
  --dependency="afterok:${SETUP_JOB}" \
  --gpus="$GPUS_PER_TASK" \
  --export=ALL,REPO_DIR="$REPO_DIR",MANIFEST="$MANIFEST",OUTPUT_DIR="$OUTPUT_DIR",VENV_DIR="$VENV_DIR",HF_HOME="$HF_HOME",MODEL_IDS="$MODEL_IDS",NUM_SHARDS="$NUM_SHARDS",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",LIMIT="$LIMIT" \
  hpg/vlm_prompt_saliency.sbatch
