#!/bin/bash
#SBATCH --job-name=food-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --account=gusr121337

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG_PATH="${1:-experiments/results/food_recognition_resnet34_adamw_v3/config.yaml}"
CHECKPOINT_PATH="${2:-experiments/results/food_recognition_resnet34_adamw_v3/best.pt}"
EVAL_VARIANT="${3:-clean}"

cd "$PROJECT_ROOT"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

source .venv-snellius/bin/activate

python --version
nvidia-smi || true

python evaluate.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --eval-variant "$EVAL_VARIANT"