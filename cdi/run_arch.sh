#!/bin/bash
#SBATCH -A mst114378
#SBATCH -J cdi_arch
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0-5%2
#SBATCH -o /work/elviss0915/runs/logs/%x_%A_%a.out
#SBATCH -e /work/elviss0915/runs/logs/%x_%A_%a.err
# ---- 架構比較(oracle,理想條件):arch_fft_oracle x 3、arch_attn_oracle x 3 ----
#
# 問題:U-Net 即使給了繞射相位,在 polygon / bandpass 上仍學不會反傅立葉轉換。
# 換成「內建轉換」或「加 attention」後能否學會?
# 對照組 ideal_bs0_ph1e5_oracle(arch=unet)已跑完,不重跑。
#
# 本批首次使用修改過的 src/model.py(新增 FFTUNet、AttnUNet)與 src/config.py(新增 arch 等欄位)。
# arch=unet(預設)時模型與原版逐位元相同(已測)。
#
# 執行規則:所有 python 一律在計算節點內執行(本腳本即為 job)。

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi

# 紀錄用:在 GPU 上再驗一次(只有 task 0 會因不通過而中止;真正的把關是送件前的 --configs-only)
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    python check_arch.py --configs-only
fi

CONFIGS=(arch_fft_oracle arch_attn_oracle)
CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CFG} seed=${SEED} node=$(hostname)"
python train.py --config configs/${CFG}.yaml --seed ${SEED}
python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s${SEED}
echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} 完成"
