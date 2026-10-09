#!/bin/bash
# 在 dev 佇列跑的縮小版煙霧測試。不要直接送 15 個 job。
#
#   srun -A mst114378 -p dev --gres=gpu:1 -c 8 -t 01:00:00 --pty bash
#   bash smoke_test.sh
#
# 要確認四件事:
#   1. 跑完不報錯
#   2. runs/smoke_s0/ 底下出現 ckpt.pt、final.pt、metrics.json
#   3. 中途 Ctrl+C 再跑一次,日誌出現 [resume] 從 epoch N 續跑
#   4. 記下單一 epoch 的秒數 -> 用來估 -t 和總花費

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi
mkdir -p /work/elviss0915/runs/logs

# 縮小版:2000 張、3 epochs
python - << 'PY'
import yaml, pathlib
c = yaml.safe_load(open("configs/bs03.yaml"))
c.update(subset_n=2000, epochs=3, eval_n=128, gen_n=64)
pathlib.Path("configs/smoke.yaml").write_text(
    yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
print("-> configs/smoke.yaml")
PY

python train.py --config configs/smoke.yaml --seed 0
python eval.py  --run-dir /work/elviss0915/runs/smoke_s0

echo
echo "=== 檢查產出 ==="
ls -la /work/elviss0915/runs/smoke_s0/
echo
echo "=== 單 epoch 秒數(拿來估 -t) ==="
python -c "import json;h=json.load(open('/work/elviss0915/runs/smoke_s0/history.json'));print(f\"{sum(x['sec'] for x in h)/len(h):.1f} s/epoch (subset_n=2000)\")"
echo
echo "正式跑是 subset_n=20000 (10x) x 12 epochs,請據此估算 -t 並乘 1.5。"
