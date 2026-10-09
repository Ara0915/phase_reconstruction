#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_amb
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --array=0-2%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 平凡歧異性不變的訓練目標:amb_base x 3 seeds ----
#
# 問題:網路的模糊(polygon / bandpass 上 ≈ 平庸基準)是否來自「猜不到位置與方向」?
# amb_base 與 ideal_base 只差 amb_invariant(loss 先把真值對齊到最接近的等價解)。
# 對照組 ideal_base 已跑完,不需重跑。
#
# 本批首次使用修改過的 src/losses.py、src/config.py、train.py:
#   amb_invariant=False(預設)時 loss 與原版逐位元相同(已測);
#   train.py 只多記錄 amb_twin / amb_shift 兩欄。
#
# 執行規則:所有 python 一律在計算節點內執行(本腳本即為 job)。

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi

# ---- 紀錄用:在 GPU 上再驗一次變因控制與 loss 單元測試 ----
# 注意:只有 task 0 會因不通過而中止,其餘 task 照跑;真正的把關是送件前的 --configs-only。
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    python check_amb.py --configs-only
fi

SEED=${SLURM_ARRAY_TASK_ID}
echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=amb_base seed=${SEED} node=$(hostname)"
python train.py --config configs/amb_base.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/amb_base_s${SEED}
echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
