#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan8c
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 階段八 8c:診斷(只評估、不訓練;階段八協定 §十七)----
# 要先通過 python scan_8c.py --smoke。估計約 1 小時以內,時限 4 小時;中斷時重送會從已完成的部分續跑。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_8c.py --run node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_8c.py --run
echo "[$(date)] 完成"
