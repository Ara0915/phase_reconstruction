#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan5h_tr
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --array=0-8
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- P5 學生訓練:0–2 = P5A、3–5 = P5Actl、6–8 = P5B;送件時以 afterany 接在 prep 之後 ----
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_5h.py --train --task ${SLURM_ARRAY_TASK_ID} $* node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_5h.py --train --task ${SLURM_ARRAY_TASK_ID} "$@"
echo "[$(date)] 完成"
