#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_rob
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- probe robustness 重新評估:2 configs x 3 seeds = 6 jobs ----
#
# 先前結論「參數隨機化使表現下降」建立在有缺陷的評估方式上:
#   訓練時劑量 100-10000、beamstop 0-8 隨機,
#   但評估只用固定的一組(劑量 1000、beamstop 3)。
#   等同「範圍練得廣、只考其中一題」。
#
# 兩項修正:
#   1. 縮小隨機化範圍(劑量 300-3000、beamstop 1-6)
#   2. 在 3x3 = 9 組條件下評估
#
# 兩組的評估條件完全相同,唯一差異為訓練時是否隨機化。
#
# 此結論影響階段四(已知探針照明)的設計方向 ——
# 若隨機化實際有效,探針隨機化即為自然的下一步。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(rob_fixed rob_rand)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
