#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_probe42
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-8%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 階段四 4-2a:平頂探針(r = 12 px)下訓練網路(階段四實驗設計協定 v2 §三)----
#
#   task 0–2  probe_unet          U-Net
#   task 3–5  probe_unroll        物理內嵌 U-Net(展開 HIO;T 與修正網路由 select_unroll.py 決定)
#   task 6–8  probe_unet_oracle   U-Net + 繞射相位(Oracle-psi 上限)
#
# 送件前必須:select_unroll.py 產生 configs/probe_unroll.yaml;check_probe42.py --configs-only 全部通過。
# 時限 3 小時為保險(展開版輪數較多時較慢);train.py 每個 epoch 存檔,被砍可重送續跑。
# 執行規則:所有 python 一律在計算節點內執行(本腳本即為 job)。

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi

# 紀錄用(只有 task 0 會因不通過而中止;真正的把關是送件前的 --configs-only)
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    python check_probe42.py --configs-only
fi

CONFIGS=(probe_unet probe_unroll probe_unet_oracle)
CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}
echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
