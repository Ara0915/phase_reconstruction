#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan7c
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段七 7c(階段七協定 §十六):6 個訓練 job;要先通過 --smoke 與 --feas(go)----
# 同時最多 2 個 job(帳號上限)。array 編號:0–2 = P6B4e seed 0–2(估計型);3–5 = P6B4c seed 0–2(對照組)
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_7c.py --task $SLURM_ARRAY_TASK_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_7c.py --task "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] 完成"
