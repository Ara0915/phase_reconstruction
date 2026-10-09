#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan5c_tr
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-23%8
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段五 5-3:訓練 24 個網路(4 組 × 2 探針 × 3 seeds;同時最多 8 個)----
#   task 0–5 = A、6–11 = B、12–17 = C、18–23 = D;每組內 DEF2 s0–2、DISK s0–2
#   每個 task 先跑自己那組的單元測試,沒過就不訓練;每個 epoch 存檔,重送會自動續跑。
#   額外參數原樣傳給 scan_5c.py(例:停損重訓 --n-train 100000,配合 --array 指定組別)
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_5c.py --train --task ${SLURM_ARRAY_TASK_ID} $* node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_5c.py --train --task "${SLURM_ARRAY_TASK_ID}" "$@"
echo "[$(date)] 完成"
