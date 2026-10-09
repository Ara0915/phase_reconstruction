#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_hio
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 階段三:HIO 後處理 —— 2 configs x 3 seeds = 6 jobs ----
#
# 網路單次前向 -> 初始解 -> HIO 迭代強制資料一致性與支撐約束
#
# 依據:跨分布能力僅在繞射相位已知時出現(2.0/4 對 0.0/4),
#       且資料多樣性推進至 50000 種仍無法帶來該能力。
#       瓶頸為物理資訊,而 HIO 每次迭代正在補這個。
#
# 核心驗收:「贏過基準」能否由 0/4 提升。
#
# 掃描 hio_iters = 0, 5, 10, 20, 50, 100, 200
#   0 為對照(等同純網路),另可畫出速度 vs 品質的取捨曲線,
#   對應計畫 benchmark 的 reconstruction speed 軸。
#
# 訓練設定與 v2_proc / v2_mnist 完全相同,唯一差異為 HIO 後處理。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(hio_proc hio_mnist)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
