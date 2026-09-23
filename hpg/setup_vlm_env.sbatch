#!/usr/bin/env bash
#SBATCH --job-name=ai-salience-env
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --account=mzding
#SBATCH --qos=mzding
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=08:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/blue/mzding/yujunchen/projects/AI_salience}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-vlm}"
HF_HOME="${HF_HOME:-/blue/mzding/yujunchen/projects/.cache/huggingface}"
MODEL_IDS="${MODEL_IDS:-${MODELS:-meta-llama/Llama-3.2-11B-Vision-Instruct}}"
MODEL_IDS="${MODEL_IDS//,/|}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"

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

cd "$REPO_DIR"
mkdir -p logs "$HF_HOME"

module load conda || true
module load cuda || true

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --upgrade \
  "torch" "torchvision" \
  "accelerate" "pillow" "numpy" "safetensors" "sentencepiece" "protobuf" \
  "huggingface_hub" "timm" "einops"
python -m pip install --upgrade "git+https://github.com/huggingface/transformers"

python - <<'PY'
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  IFS='|' read -r -a MODEL_ARRAY <<< "$MODEL_IDS"
  for MODEL_ID in "${MODEL_ARRAY[@]}"; do
    echo "Downloading model snapshot: $MODEL_ID"
    if [[ -n "${HF_TOKEN:-}" ]]; then
      hf download "$MODEL_ID" --token "$HF_TOKEN"
    else
      hf download "$MODEL_ID"
    fi
  done
fi
