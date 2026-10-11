#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_final_env
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH --array=0-16%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 期末考(期末考協定 §九):17 個迭代法包絡 job;要先通過 --smoke ----
# 同時最多 2 個 job(帳號上限)。array 編號:
#   0–11  = A / B 的 12 個條件(ideal posL posH prbL prbH doseL doseH combo cohL cohH posX prbX)
#   12–16 = C 的 5 份考卷(L-ideal L-paper L-combo L-coh L-newdef)
# 每個 job 用考試資料(只有 --run 才允許);中斷時重送同一個編號會續跑(每個 seed 的暫存檔;版本不同就拒絕)。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_final.py --run $SLURM_ARRAY_TASK_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_final.py --run "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] 完成"
