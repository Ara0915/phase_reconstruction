#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_final_agg
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 期末考:彙整(計時、網路(考試 + 驗證)、CRLB、迭代法的關鍵設定、判讀 f1–f5、圖)----
# 等 17 個包絡 job 都成功再跑(--dependency=afterok:<array job id>)。分段暫存(final_cache/):中斷時重送會沿用已完成的段落。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_final.py --run final node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_final.py --run final
echo "[$(date)] 完成"
