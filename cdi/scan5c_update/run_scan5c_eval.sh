#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_scan5c_ev
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 階段五 5-3:評估全部網路(訓練全部完成後;用 --dependency=afterok:<訓練的 job id> 送)----
#   同一個 job 重新計時迭代法與網路,對手 = 5-2b 的包絡在網路時間的值(協定 §12.4–12.5)
#   只跑全部單元測試:sbatch run_scan5c_eval.sh --check
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
MODE="--eval"
for x in "$@"; do [ "$x" = "--check" ] && MODE=""; done
echo "[$(date)] scan_5c.py ${MODE} $* node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scan_5c.py ${MODE} "$@"
echo "[$(date)] 完成"
