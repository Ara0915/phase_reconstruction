#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_check42ds
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 階段四 4-2a:展開版重訓跑完後的評分與判讀(check_probe42_ds.py,不帶參數)----
# 以 job 執行,避免互動 session 逾時中斷;結果寫到 /work/elviss0915/runs/probe42_ds.json 與 figs_probe42_ds/。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] check_probe42_ds.py node=$(hostname)"
python check_probe42_ds.py
echo "[$(date)] 完成"
