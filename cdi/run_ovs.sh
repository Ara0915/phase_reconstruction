#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_ovs
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-14%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 過取樣實驗重新設計:5 configs x 3 seeds = 15 jobs ----
#
# 問題:前次結果顯示過取樣率越高、表現越差(0.1392/0.1247/0.1162),
#       與「增加已知為零的區域應有助益」的直覺相反。
#
# 成因:畫布放大時感受野涵蓋率同時下降(梯度實測 100% -> 88.5% -> 66.4%)。
#       傅立葉轉換為全域算子,涵蓋率不足會直接損害重建。
#       兩個效應方向相反且混在一起,原實驗無法歸因。
#
# 設計:以 dilation 在參數量完全不變(皆 1,928,450)的前提下對齊涵蓋率。
#
#   主實驗(涵蓋率對齊至 99-100%):
#     ovs64   canvas 64  dil 1 -> 100.0%
#     ovs96   canvas 96  dil 2 ->  99.0%
#     ovs128  canvas 128 dil 2 ->  99.2%
#
#   對照組(固定 canvas 64,只改 dilation),隔離 gridding 效應:
#     ovs64 / ctl_d2 / ctl_d3
#
# 判讀:對照組若無顯著差異,主實驗的差異方可歸因於過取樣率。
#       此結果決定 ptychography(其邏輯亦為增加約束)是否值得投入。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(ovs64 ovs96 ovs128 ctl_d2 ctl_d3)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
