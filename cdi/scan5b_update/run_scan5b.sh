#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan5b
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 階段五 5-2b:對手定案(起點 × 演算法的時間-誤差包絡;參數原樣傳給 scan_5b.py)----
#   完整量測(會先跑全部檢查,沒過就停):sbatch run_scan5b.sh
#   只跑檢查:                          sbatch run_scan5b.sh --check
# 結果寫到 /work/elviss0915/runs/scan5b.json(量測後先存 scan5b_raw.json)與 ~/cdi/figs_scan5b/。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] scan_5b.py $* node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_5b.py "$@"
echo "[$(date)] 完成"
