#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_oracle
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH --array=0-11%4
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- Oracle 控制實驗:2x2 設計,4 configs x 3 seeds = 12 jobs ----
#
#            相位未知      相位已知
#   bs=0     bs00        oracle_bs00   <- 純上界,應該接近完美
#   bs=3     bs03        oracle_bs03
#
# 直行相減 = 相位遺失的代價
# 橫列相減 = beamstop 低頻缺失的代價
#
# 這是唯一能把「架構不足」與「相位遺失本身太難」分開的實驗。
# **先跑這個,再決定要不要跑完整的 beamstop 掃描。**

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4     # <-- 先用 module avail 確認版本
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/elviss0915/envs/cdi

CONFIGS=(bs00 bs03 oracle_bs00 oracle_bs03)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
