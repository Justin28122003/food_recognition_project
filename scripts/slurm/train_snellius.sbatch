#!/bin/bash
#SBATCH --job-name=food-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --account=svankuik

set -euo pipefail

# Usage:
#   sbatch scripts/slurm/train_snellius.sbatch
#   sbatch scripts/slurm/train_snellius.sbatch experiments/configs/train_config.yaml
# Optional env vars:
#   CONDA_ENV_NAME=dl2024
#   PROJECT_ROOT=/path/to/food_recognition_project

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG_PATH="${1:-experiments/configs/train_config_stronger.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dl2024}"

cd "$PROJECT_ROOT"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Config: $CONFIG_PATH"

# Load conda shell integration
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda not found. Load your conda module/environment first."
  exit 1
fi

conda activate "$CONDA_ENV_NAME"

python --version
nvidia-smi || true

# Run training
srun python experiments/train.py --config "$CONFIG_PATH"
