#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_os
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-8%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 過取樣驗證:3 configs x 3 seeds = 9 jobs ----
#
# 目的:正面驗證「單張繞射圖的量測資訊不足」這個結論。
#
# 目前該結論來自排除法 —— 已排除劑量、輸出頭、目標函數、誤差分配、
# 輸入表示、資料多樣性、以及「L1 把高頻抹平」(FRC 診斷顯示各模型高頻
# 能量比皆為 0.6~0.8,oracle 能量最低但 FRC 最高,故非能量不足問題)。
# 排除法有風險:若還有未想到的原因,結論即錯誤。
# ptychography 是數週的工程,不應押在未經正面驗證的假設上。
#
# 做法:提高過取樣率(canvas 64 -> 96 -> 128,物體維持 28x28)。
#       已知為零的區域變多 = 相位恢復的約束增加 = 資訊增加,
#       且完全不需要新程式碼。
#
#   os64   過取樣 2.29(對照組)
#   os96   過取樣 3.43
#   os128  過取樣 4.57
#
# 判讀:FRC gain 隨過取樣率單調上升 -> 證實資訊不足,ptychography 值得做
#       FRC gain 不動               -> 結論有誤,須重新檢視方向
#
# 註:canvas=128 的像素數為 64 的 4 倍,訓練時間約 4 倍,故時限設 3 小時。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(os64 os96 os128)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
