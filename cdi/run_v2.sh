#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_v2
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-11%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 振幅損失修正後重跑 2x2:4 configs x 3 seeds = 12 jobs ----
#
#                 正常路徑         Oracle(給繞射相位)
#   MNIST         v2_mnist        v2_mnist_oracle
#   procedural    v2_proc         v2_proc_oracle
#
# 修正內容:振幅損失改為「樣品內」與「空白區」分開計算並各自正規化,
#           權重由 w_amp_out 明確指定,不再由面積比例隱含決定。
#
# 實測效果(相同資料/seed/步數,唯一差異為振幅損失正規化):
#   舊版  振幅偏差 -0.3733  MAE 0.3747
#   新版  振幅偏差 -0.0023  MAE 0.0131
#
# 此修正影響所有用到振幅的指標,故既有結果需重新取得。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(v2_mnist v2_mnist_oracle v2_proc v2_proc_oracle)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
