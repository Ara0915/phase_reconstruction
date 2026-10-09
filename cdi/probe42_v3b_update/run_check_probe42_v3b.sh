#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_check42v3b
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 階段四 4-2a 展開版 v3b:關卡或完整評分(參數原樣傳給 check_probe42_v3b.py)----
#   關卡:sbatch run_check_probe42_v3b.sh --gate
#   完整:sbatch run_check_probe42_v3b.sh
# 以 job 執行,避免互動 session 逾時中斷;結果寫到 /work/elviss0915/runs/probe42_v3b.json 與 figs_probe42_v3b/。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] check_probe42_v3b.py $* node=$(hostname)"
python check_probe42_v3b.py "$@"
echo "[$(date)] 完成"
