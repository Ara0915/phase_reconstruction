#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_ideal
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --array=0-8%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err

# ---- §1b.6 理想條件驗證(4-0a 篩選):3 configs x 3 seeds = 9 jobs ----
#
# ============================================================================
# 這批在問什麼
# ============================================================================
# 研究日誌 §一之二 依指導教授回饋，把核心論述改成:
#   「解在理論上唯一，但在本專案的量測條件下逆映射不穩定。」
# 該論述目前只有間接支持(w_dc、HIO、Oracle)。本批直接檢驗它。
#
#   ideal_base          現況條件(= v2_proc + 交叉評估)          ← 基準 + 重現性檢查
#   ideal_bs0_ph1e5         beamstop=0、劑量 x100                    ← 主臂
#   ideal_bs0_ph1e5_oracle  同上 + 給繞射相位                        ← 該條件下的天花板
#
# ideal_base vs ideal_bs0_ph1e5 只差 beamstop_r 與 photons_per_pix(已用程式驗證)。
# ideal_bs0_ph1e5 vs ideal_bs0_ph1e5_oracle 只差 oracle_phase。
#
# ============================================================================
# 為什麼過取樣不在這一批
# ============================================================================
# 過取樣 2.29 未違反唯一性門檻(門檻為 2)，只是餘裕薄;
# 且 §12.3 實測提高過取樣率會使 FRC gain 變差(0.1400 -> 0.1147)。
# 三項綁一起推可能互相抵消而產生假陰性。過取樣改列獨立第三臂(4-0c)。
#
# ============================================================================
# partition 的選擇
# ============================================================================
# 用 normal2(H200),與既有 run_*.sh 一致。
# normal(H100)排隊通常較嚴重,不使用。

# ---- 執行規則 ----
# 所有 python 一律在計算節點內執行(本腳本即為 job,故其內部皆為計算節點)。
# 登入節點只做 sbatch 送件與 squeue 查狀態。

set -euo pipefail

module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi

# ---- 保險:在計算節點內再驗一次變因控制,不通過就中止 ----
# 僅 task 0 執行,避免 9 個 job 重複輸出。
# 這一步刻意放在 job 內部 —— 登入節點不得執行任何 python。
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    echo "---- config 變因控制驗證 ----"
    python check_ideal_conditions.py --configs-only
    echo "---- 驗證通過 ----"
fi

CONFIGS=(ideal_base ideal_bs0_ph1e5 ideal_bs0_ph1e5_oracle)
SEEDS=(0 1 2)
NSEED=${#SEEDS[@]}

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NSEED ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NSEED ))]}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
