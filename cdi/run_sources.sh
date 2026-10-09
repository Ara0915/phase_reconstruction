#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_src
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0-11%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 訓練資料來源對照:4 configs x 3 seeds = 12 jobs ----
#
# 已確認的病因(研究日誌 §6.2):Oracle 在訓練分布達理論極限
# (2.065 px vs Nyquist 2.0),但在 Fashion-MNIST 上優勢完全消失(-0.64 dB)。
# Oracle 的任務是反傅立葉轉換 —— 與物體外觀無關的通用運算,
# 學會了就該處處通用。它沒有,所以是在背。
#
# 這組實驗只改「訓練資料來源」一個變因:
#   src_mnist      現行基準(PCA 有效維度 69)
#   src_mixed      多加幾種現成資料集夠不夠?(合計約 30 類)
#   src_proc       程序生成,每樣本獨一無二(PCA 227,3.29x)
#   src_proc_rand  再加上劑量與 beamstop 隨機化
#
# 驗收:以 Oracle 重跑泛化測試。若真的學會通用映射,
#       Oracle 在所有測試分布上的表現應趨於一致。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(src_mnist src_mixed src_proc src_proc_rand)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
