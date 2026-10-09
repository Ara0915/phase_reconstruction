#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_probe42b
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /work/elviss0915/runs/logs/%x_%j.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%j.err
# ---- 階段四 4-2b(HIO 部分):複數探針能否打破翻轉(參數原樣傳給 probe_4_2b.py)----
#   送件前檢查:sbatch run_probe42b.sh --check   (約 1–3 分鐘)
#   完整量測:  sbatch run_probe42b.sh           (約 40–60 分鐘)
# 結果寫到 /work/elviss0915/runs/probe42b.json 與 ~/cdi/figs_probe42b/。
set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
cd /home/elviss0915/cdi
echo "[$(date)] probe_4_2b.py $* node=$(hostname)"
python probe_4_2b.py "$@"
echo "[$(date)] 完成"
