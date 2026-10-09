#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_probe42v3b
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段四 4-2a:展開版 v3b(階段四協定 §8.10、§8.12)----
#
#   = probe_v3_unroll + drop_empty(訓練池的空樣本就地換成非空樣本;協定 §8.12)
#
# 分兩階段(協定 §8.10):
#   關卡    sbatch run_probe42_v3b.sh              → 只訓練 seed 0(預設 --array=0)
#           跑完後 check_probe42_v3b.py --gate 通過,才進下一步
#   其餘    sbatch --array=1-2 run_probe42_v3b.sh  → 訓練 seed 1、2
#
# U-Net、Oracle-psi 沿用首批。輸出到 probe_v3b_unroll_s*,不覆蓋任何既有 run。
# 送件前必須:check_probe42_v3b.py --configs-only 全部通過。
# 執行規則:所有 python 一律在計算節點內執行(本腳本即為 job)。

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi

CFG=probe_v3b_unroll
SEED=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}
echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
