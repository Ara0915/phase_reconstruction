#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan7b_final
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 階段七 7b:彙整(計時、網路評估、比較方式、判讀、迭代法 SSIM、圖);
# 等 7 個 job 都成功再跑(sbatch --dependency=afterok:<array job id>)----
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_7b.py --final node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_7b.py --final
echo "[$(date)] 完成"
