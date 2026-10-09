#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan8
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-10%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段八(階段八協定 §十):11 個 job;要先通過 --smoke ----
# 同時最多 2 個 job(帳號上限)。array 編號:
#   0–4  = 5 份考卷的迭代法包絡(L-ideal、L-paper、L-combo、L-coh、L-newdef;各約 1 小時以上)
#   5–7  = P6B4e8-L seed 0–2(晶格 + 原本資料各半;各約 65–75 分鐘)
#   8–10 = P6B4e8-C seed 0–2(只有原本資料;各約 65 分鐘)
# 中斷時重送同一個編號會續跑(考卷:每個 seed 的暫存檔;訓練:每個 epoch 的 checkpoint;版本不同就拒絕)。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_8.py --task $SLURM_ARRAY_TASK_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_8.py --task "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] 完成"
