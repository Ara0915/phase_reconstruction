#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan8_final
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 階段八:彙整(計時、8a + 8b 的網路評估、原本 13 個條件的退步檢查、η = 0、泛化落差、迭代法的關鍵設定、判讀、圖)----
# 等 11 個 job 都成功再跑(--dependency=afterok:<array job id>);協定估計約 2–3 小時,時限 5 小時。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_8.py --final node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_8.py --final
echo "[$(date)] 完成"
