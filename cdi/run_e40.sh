#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_e40
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-11%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 收斂對照 2x2:4 configs x 3 seeds = 12 jobs ----
#
#                 正常路徑            Oracle(給繞射相位)
#   MNIST         e40_mnist          e40_mnist_oracle
#   procedural    e40_proc           e40_proc_oracle
#
# 背景:epochs=12 時各設定均未收斂,且收斂進度不一致
#       (src_proc 末段降幅 0.95%、oracle_proc 1.42%),
#       導致「oracle 表現不如正常路徑」這種資訊論上不可能的觀察 ——
#       oracle 的輸入是正常輸入的嚴格超集,多給資訊不可能使表現變差。
#
# 全部改為 epochs=40。跑完必須先以 check_convergence.py 確認收斂,
# 再進行任何比較。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(e40_mnist e40_mnist_oracle e40_proc e40_proc_oracle)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
