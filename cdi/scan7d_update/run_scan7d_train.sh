#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan7d
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段七 7d(探索型支線,階段七協定 §十七):6 個訓練 job;要先通過 --smoke----
# 同時最多 2 個 job(帳號上限)。array 編號:0–2 = P6B4e8 seed 0–2(8 級估計型,約 3–4 小時);3–5 = P6B4c8 seed 0–2(8 級對照組,約 2 小時)
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_7d.py --task $SLURM_ARRAY_TASK_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_7d.py --task "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] 完成"
