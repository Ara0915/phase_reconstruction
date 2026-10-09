#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan7a_cond
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --array=0-7%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段七 7a-1:每個條件一個 job(階段七協定 §九);要先在 dev 節點通過 --smoke ----
# 同時最多 2 個 job(帳號上限),%2 讓其餘的排隊。array 編號 → 條件:0 ideal、1 posL、2 posH、3 prbL、4 prbH、5 doseL、6 doseH、7 combo
set -euo pipefail
CONDS=(ideal posL posH prbL prbH doseL doseH combo)
C=${CONDS[$SLURM_ARRAY_TASK_ID]}
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_7a.py --cond $C node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_7a.py --cond "$C"
echo "[$(date)] 完成"
