#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_rf
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --array=0-8%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 過取樣重做:3 configs x 3 seeds = 9 jobs ----
#
# 原過取樣實驗混入了未控制的變因:畫布放大時感受野涵蓋率
# 由 183% 降至 91%,與 FRC gain 的下降順序完全一致。
#
# 本組以 dilation 對齊感受野涵蓋率(皆約 183-185%),
# 且參數量完全不變(1,928,450),使過取樣率成為唯一變因。
#
#   rf64   canvas 64  dilation 1  過取樣 2.29
#   rf96   canvas 96  dilation 2  過取樣 3.43
#   rf128  canvas 128 dilation 3  過取樣 4.57
#
# 判讀:gain 仍下降 -> 增加空白區確實有害
#       gain 持平或上升 -> 原趨勢為感受野造成,約束本身無害
#
# 此結果直接影響 ptychography 的評估(該路線的邏輯亦為增加約束)。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(rf64 rf96 rf128)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
