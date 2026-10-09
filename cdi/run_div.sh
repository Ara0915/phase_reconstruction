#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_div
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-11%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 多樣性劑量曲線:4 configs x 3 seeds = 12 jobs ----
#
#   div100 / div500 / div2000 / div20000
#   唯一變因為訓練池中「相異物體」的數量。
#
# 池子總數固定 20000(相異物體重複填滿),故梯度步數、batch size、
# 資料總量皆不變,唯一變動的是模型看過幾種不同結構。
#
# 動機:程序生成資料使 Oracle 泛化落差由 MNIST 的 16.39 dB 降至 8.17 dB,
#       但兩點相差 2000 倍,中間行為未知。
#
# 判讀:曲線持續下降 -> 增加資料仍有效,可讀出所需量
#       曲線已飽和   -> 此方法達極限,須改採其他手段

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(div100 div500 div2000 div20000)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
