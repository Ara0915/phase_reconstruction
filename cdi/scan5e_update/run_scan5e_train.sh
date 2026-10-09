#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan5e_tr
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-5
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 優化 P7 訓練:task 0–2 = NAF-B(seed 0–2)、3–5 = NAF-B 小;需先完成 prep ----
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_5e.py --train --task ${SLURM_ARRAY_TASK_ID} $* node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_5e.py --train --task ${SLURM_ARRAY_TASK_ID} "$@"
echo "[$(date)] 完成"
