#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_bs
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH --array=0-14%4
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- Beamstop 掃描:5 configs x 3 seeds = 15 jobs ----
# %4 限制同時最多 4 個。一次全開會同時吃 4 張以上的卡,燒錢速度倍增,
# 而且佇列滿的時候也未必排得到。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4     # <-- 先用 module avail 確認版本

# 非互動 shell 沒被 conda init 過會報 CommandNotFoundError,所以先 source
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/elviss0915/envs/cdi

CONFIGS=(bs00 bs01 bs03 bs06 bs10)
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
