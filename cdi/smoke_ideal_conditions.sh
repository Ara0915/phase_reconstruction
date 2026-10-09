#!/bin/bash
# §1b.6 理想條件驗證 —— 送正式 batch 前的煙霧測試。
#
#   srun -A mst114378 -p dev --gres=gpu:1 -c 8 -t 01:00:00 --pty bash
#   bash smoke_ideal_conditions.sh
#
# ---- 為什麼這批特別需要煙霧測試 ----
# photons_per_pix = 100000 **從未跑過**。既有最高只到 dose_1e4 的 10000。
# 劑量拉高 100 倍會讓中心像素的期望光子數由 7,720 -> 772,000,
# 需確認 torch.poisson 在該量級下正常、無數值問題。
#
# 要確認四件事:
#   1. ph=100000 跑得起來,不報錯、不出現 NaN/Inf
#   2. log_mean 確實隨劑量重新校正(檢查 2),平移約 ln(100)=4.605
#   3. beamstop_r=0 的遮罩正確(blocked 應為 0.00%)
#   4. 記下單 epoch 秒數 -> 用來估正式 batch 的 -t

set -euo pipefail
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

cd /home/elviss0915/cdi
mkdir -p /work/elviss0915/runs/logs

# 縮小版:2000 張、3 epochs,base 與 ideal 各跑一次
python - << 'PY'
import yaml, pathlib
for src, dst in [("ideal_base", "smoke_ideal_base"), ("ideal_bs0_ph1e5", "smoke_ideal_bs0")]:
    c = yaml.safe_load(open(f"configs/{src}.yaml"))
    c.update(subset_n=2000, epochs=3, eval_n=128, gen_n=64)
    pathlib.Path(f"configs/{dst}.yaml").write_text(
        yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    print(f"-> configs/{dst}.yaml")
PY

for CFG in smoke_ideal_base smoke_ideal_bs0; do
    echo
    echo "############ ${CFG} ############"
    python train.py --config configs/${CFG}.yaml --seed 0
    python eval.py  --run-dir /work/elviss0915/runs/${CFG}_s0
done

echo
echo "======================================================================"
echo "檢查 2:正規化常數是否隨劑量重新校正"
echo "======================================================================"
python - << 'PY'
import json, math
b = json.load(open("/work/elviss0915/runs/smoke_ideal_base_s0/config_used.json"))
i = json.load(open("/work/elviss0915/runs/smoke_ideal_bs0_s0/config_used.json"))

k = i["photons_per_pix"] / b["photons_per_pix"]
shift = i["log_mean"] - b["log_mean"]
exp = math.log(k)

print(f"  ideal_base   ph={b['photons_per_pix']:>10.0f}  bs={b['beamstop_r']}  "
      f"log_mean={b['log_mean']:.4f}  log_std={b['log_std']:.4f}")
print(f"  ideal_bs0_ph1e5  ph={i['photons_per_pix']:>10.0f}  bs={i['beamstop_r']}  "
      f"log_mean={i['log_mean']:.4f}  log_std={i['log_std']:.4f}")
print()
print(f"  劑量比 {k:g}x   log_mean 實際平移 {shift:+.4f}   預期 ~{exp:.4f}")

if abs(shift) < 1.0:
    print("\n  ❌ 平移量過小 —— log_mean 很可能沒有重新校正,停下來查 train.py")
    raise SystemExit(1)
if abs(shift - exp) > 0.5:
    print(f"\n  ⚠️  平移量與預期差 {shift-exp:+.3f}(log1p 在低計數區非線性,"
          "小幅偏離屬正常;偏離過大則須檢查)")
else:
    print("\n  ✅ 正規化常數確實隨劑量重新校正")

ref_b, ref_i = b["ref_energy"], i["ref_energy"]
if abs(ref_b - ref_i) > 1e-6 * max(abs(ref_b), abs(ref_i)):
    print(f"  ⚠️  ref_energy 兩組不同({ref_b:.6f} vs {ref_i:.6f})—— "
          "它只由物體決定,理應一致")
else:
    print(f"  ✅ ref_energy 兩組一致({ref_b:.6f})—— 符合「與劑量無關」")
PY

echo
echo "======================================================================"
echo "數值健全性 + 單 epoch 秒數"
echo "======================================================================"
python - << 'PY'
import json, math
for n in ("smoke_ideal_base", "smoke_ideal_bs0"):
    p = f"/work/elviss0915/runs/{n}_s0"
    h = json.load(open(f"{p}/history.json"))
    m = json.load(open(f"{p}/metrics.json"))["test"]
    bad = [k for k, v in m.items()
           if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
    sec = sum(x["sec"] for x in h) / len(h)
    print(f"  {n:<16} {sec:6.1f} s/epoch   FRC gain {m['frc_gain']:+.4f}   "
          f"PSNR {m['amp_psnr']:.2f}" + (f"   ❌ NaN/Inf: {bad}" if bad else "   ✅"))
print()
print("  正式跑 subset_n=20000(10x)x epochs=40,請據此估算 -t 並乘 1.5。")
PY

echo
echo "以上全部 ✅ 才送正式 batch:  sbatch run_ideal_conditions.sh"
