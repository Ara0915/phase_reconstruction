#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_norm
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 輸入正規化測試:2 configs x 3 seeds = 6 jobs ----
#
# 問題:oracle 的優勢只存在於訓練分布(MNIST +4.5 dB、Fashion −0.6 dB)。
#       oracle 的任務本質是反傅立葉轉換 —— 與物體形狀無關的通用運算,
#       學會了就該處處通用。它沒有,所以看起來是在背。
#
# 但有一個干擾因素:輸入通道做了**逐張標準化**,會把絕對尺度洗掉。
#       所以 oracle 收到的其實是「差一個未知比例的完整資訊」,
#       那個比例仍須靠先驗猜 —— 泛化失敗可能部分來自這裡。
#
# 這組實驗只改 input_norm 一個變因,與 bs03 / oracle_bs03 對照:
#   泛化落差大幅縮小 -> 是正規化的假象,原結論維持
#   落差照舊         -> 確認是記憶,優先換物體產生器

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(bs03_gn oracle_bs03_gn)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
