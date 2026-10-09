#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_d50k
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 多樣性延伸:2 configs x 3 seeds = 6 jobs ----
#
#   A div50k_u20k  訓練集 50000, 相異 20000
#   B div50k_u50k  訓練集 50000, 相異 50000
#   C div20000     訓練集 20000, 相異 20000(已完成)
#
#   A vs B:訓練集相同,唯一變因為相異數 —— 主要比較
#   A vs C:相異數相同,唯一變因為訓練集大小 —— 驗證訓練量本身無影響
#
# 動機:既有曲線(訓練集皆 20000)未飽和,且後段斜率大於前段:
#   相異數 100/500/2000/20000 -> FRC gain 0.066/0.111/0.132/0.211
#   材料 MAE 0.458/0.453/0.443/0.304(僅最後一段突破,呈門檻效應)
#
# 注意:訓練集由 20000 增至 50000,每 epoch 梯度步數由 156 增至 390,
#       單一 job 時間約為原本的 2.5 倍,故時限設為 8 小時。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(div50k_u20k div50k_u50k)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
