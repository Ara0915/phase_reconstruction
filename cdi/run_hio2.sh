#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_hio2
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 階段三延伸:2 configs x 3 seeds = 6 jobs ----
#
#   hio2_net   網路初始解 + HIO,迭代延長至 1000
#   hio2_rand  隨機初始解 + HIO(純 HIO,傳統方法基準)
#
# 兩者訓練完全相同,唯一差異為 HIO 的初始解來源。
#
# 目的一:確認「贏過基準」能否出現第一個轉正。
#   前次 200 次時 texture 增益 -0.24,差 0.24 即轉正且仍在改善。
#
# 目的二:量化網路初始解在速度軸上的價值。
#   每個迭代數皆記錄 ms/sample,可畫出速度 vs 品質曲線。
#
# 注意:1000 次迭代 x 5 個分布,單一 job 較久,時限設 8 小時。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(hio2_net hio2_rand)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
