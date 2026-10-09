#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_probe42gc
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-8%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段四 4-2a 重訓:三組一律加上梯度裁切(grad_clip 1.0)與非有限值防護(階段四協定 §3.9)----
#
#   task 0–2  probe_gc_unet     U-Net
#   task 3–5  probe_gc_unroll   物理內嵌 U-Net(展開 HIO;c16、每 2 輪修正、T = 50)
#   task 6–8  probe_gc_oracle   U-Net + 繞射相位(Oracle-psi 上限)
#
# 首批(probe_unet / probe_unroll / probe_unet_oracle)保留作紀錄,不覆蓋。
# 送件前必須:check_probe42_gc.py --configs-only 全部通過。
# 執行規則:所有 python 一律在計算節點內執行(本腳本即為 job)。

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    python check_probe42_gc.py --configs-only
fi

CONFIGS=(probe_gc_unet probe_gc_unroll probe_gc_oracle)
CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}
echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
