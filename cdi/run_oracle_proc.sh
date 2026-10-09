#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_oproc
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- 階段一:程序生成產生器的驗收 —— 2 configs x 3 seeds = 6 jobs ----
#
# 建立產生器的目的是消除「辨識類別後由記憶重繪」的捷徑。
# 驗收標準為以 Oracle 重跑泛化測試:Oracle 的任務是反傅立葉轉換,
# 線性、有封閉解、與物體外觀無關,學會了就該在任何分布上表現一致。
#
# 判讀一(捷徑是否消除):
#   oracle + MNIST       泛化落差 14.83 dB(已有)
#   oracle + procedural  泛化落差 ?
#
# 判讀二(材料反演的可達性):
#   procedural 上目前所有設定的材料 MAE 皆劣於平庸基準,
#   MNIST 上則普遍優於。Oracle 能否解出材料,可區分
#   「模型能力不足」與「指標在該資料集不適用」。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

CONFIGS=(oracle_proc oracle_proc_bs00)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"

cd /home/elviss0915/cdi
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
