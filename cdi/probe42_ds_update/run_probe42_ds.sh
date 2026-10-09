#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_probe42ds
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-2%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段四 4-2a:只重訓展開版,改用分段反傳(階段四協定 §8.5)----
#
#   task 0–2  probe_ds_unroll  seed 0/1/2
#             = 首批 probe_unroll(c16、每 2 輪修正、T = 50)+ unroll_trunc 10 + grad_clip 10
#
# U-Net、Oracle-psi 沿用首批 probe_unet_s*、probe_unet_oracle_s*(不重訓)。
# 首批 probe_unroll_s* 保留作紀錄,不覆蓋(新 run 輸出到 probe_ds_unroll_s*)。
# 送件前必須:check_probe42_ds.py --configs-only 全部通過。
# 執行規則:所有 python 一律在計算節點內執行(本腳本即為 job)。

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi

CFG=probe_ds_unroll
SEED=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}
echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
