#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_arch2
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 第 2 步(相位未知):arch_fft_amb x 3、arch_unroll_amb x 3 ----
#
# 問題:不給相位時,內建轉換的網路(單次 / 展開 HIO 5 輪)能否完成相位回復?
# 對照組 amb_base(arch=unet,同條件、同 loss)已跑完,不重跑。
#
# 本批首次使用 arch=unroll(src/model.py 新增 UnrollNet)。
# 已測:未訓練時展開 5 輪 = hio.py 的 5 次迭代;真值為展開的固定點。
#
# 執行規則:所有 python 一律在計算節點內執行(本腳本即為 job)。

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi

# 紀錄用(只有 task 0 會因不通過而中止;真正的把關是送件前的 --configs-only)
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    python check_arch2.py --configs-only
fi

CONFIGS=(arch_fft_amb arch_unroll_amb)
CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}
echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
