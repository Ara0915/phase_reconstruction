#!/usr/bin/env python
"""§1b.6 理想條件驗證 —— 送出前/後的健全性檢查。

用法:
    # 送出前(只檢查 config,不需要跑完)
    python check_ideal_conditions.py --configs-only

    # 跑完後(檢查 config_used.json 與結果)
    python check_ideal_conditions.py

對應「實驗設計_1b6_理想條件驗證.md」的六項檢查。
每一項都對應研究日誌裡一個真實發生過的錯誤。
"""
import argparse
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Cfg

CFG_DIR = Path("configs")
RUN_ROOT = Path("/work/elviss0915/runs")
NAMES = ["ideal_base", "ideal_bs0_ph1e5", "ideal_bs0_ph1e5_oracle"]
SEEDS = [0, 1, 2]

# 參考值:v2_proc 的 FRC gain。若 ideal_base 重現不出來 -> 環境有變。
#
# 取自 v2_proc 實際的 metrics.json(2026-09-23 於計算節點讀出):
#   s0 0.2150   s1 0.2170   s2 0.1999   -> 平均 0.2106,單 seed 標準差 0.0093
#
# (先前由研究日誌推得 0.2115,與實測差 0.0009,方向正確;
#  但當時把「三組 3-seed 平均值之間的全距 0.0013」誤當成 seed 波動,
#  據此把容忍收緊到 0.01 是錯的,已修正。)
#
# 容忍的依據:比較的是兩個「3 seeds 平均」之差,
#   其標準誤 = 0.0093 / sqrt(3) * sqrt(2) = 0.0076
#   容忍 0.02 ≈ 2.6 倍標準誤 -> 純 seed 波動誤報機率約 1%
#   (若用 0.01 ≈ 1.3 倍,純 seed 波動就有約 19% 機率誤報)
#
# 不採用 rr_os64 的 0.1392:該組 match_support_to = null(v2_proc 為 0.0318),
# 訓練分布不同,不是同一個基準。
V2_PROC_FRC_GAIN = 0.2106
V2_PROC_TOL = 0.02

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    mark = "✅" if passed else "❌"
    print(f"{mark} {label}")
    if detail:
        print(f"     {detail}")
    if not passed:
        ok_all = False
    return passed


def warn(label, detail=""):
    print(f"⚠️  {label}")
    if detail:
        print(f"     {detail}")


# ============================================================================
# 檢查 4:變因控制以程式驗證(送出前)
# ============================================================================
def check_configs():
    print("\n" + "=" * 70)
    print("檢查 4:變因控制(§10.4 紀律)")
    print("=" * 70)

    cfgs = {}
    for n in NAMES:
        p = CFG_DIR / f"{n}.yaml"
        if not p.exists():
            check(f"{n}.yaml 存在", False, f"找不到 {p}")
            return None
        cfgs[n] = Cfg.load(p).to_dict()      # Cfg.load 會擋掉未知欄位
    check("三份 config 皆可載入(未知欄位會被 Cfg.load 擋下)", True)

    def diff(a, b):
        return sorted(k for k in cfgs[a] if cfgs[a][k] != cfgs[b][k])

    d1 = diff("ideal_base", "ideal_bs0_ph1e5")
    check("ideal_base vs ideal_bs0_ph1e5 只差 beamstop_r 與 photons_per_pix",
          d1 == ["beamstop_r", "photons_per_pix"], f"實際差異: {d1}")

    d2 = diff("ideal_bs0_ph1e5", "ideal_bs0_ph1e5_oracle")
    check("ideal_bs0_ph1e5 vs ideal_bs0_ph1e5_oracle 只差 oracle_phase",
          d2 == ["oracle_phase"], f"實際差異: {d2}")

    # 劑量比與預期的 log 平移
    k = cfgs["ideal_bs0_ph1e5"]["photons_per_pix"] / cfgs["ideal_base"]["photons_per_pix"]
    print(f"\n     劑量比 = {k:g}x   預期 log_mean 平移 = ln({k:g}) = {math.log(k):.3f}")
    return cfgs


# ============================================================================
# 檢查 2:正規化常數是否在新條件下重新校正(跑完後)
# ============================================================================
def check_norm(cfgs):
    print("\n" + "=" * 70)
    print("檢查 2:正規化常數重新校正(§12b.4)")
    print("=" * 70)

    used = {}
    for n in NAMES:
        p = RUN_ROOT / f"{n}_s0" / "config_used.json"
        if not p.exists():
            warn(f"{p} 還不存在,跳過", "(跑完再執行本檢查)")
            return
        used[n] = json.load(open(p))

    for n in NAMES:
        u = used[n]
        check(f"{n} 的 log_mean 已寫入 config_used.json",
              u.get("log_mean") is not None,
              f"log_mean={u.get('log_mean')}")

    lb, li = used["ideal_base"]["log_mean"], used["ideal_bs0_ph1e5"]["log_mean"]
    k = used["ideal_bs0_ph1e5"]["photons_per_pix"] / used["ideal_base"]["photons_per_pix"]
    expected = math.log(k)
    actual = li - lb
    # log1p 非純線性,低計數區會偏離,故容忍度放寬
    check("log_mean 隨劑量正確平移(= 有重新校正)",
          abs(actual - expected) < 0.5 and abs(actual) > 1.0,
          f"實際平移 {actual:.3f}  預期 ~{expected:.3f}  "
          f"(兩者相同代表沒重新校正)")

    # ref_energy 與劑量無關,三組應一致
    refs = {n: used[n]["ref_energy"] for n in NAMES}
    vals = list(refs.values())
    check("ref_energy 三組一致(它只由物體決定,與劑量無關)",
          max(vals) - min(vals) < 1e-6 * max(abs(v) for v in vals),
          f"{refs}")


# ============================================================================
# 檢查 3:重現性 —— ideal_base 要對得上 v2_proc
# ============================================================================
def check_repro():
    print("\n" + "=" * 70)
    print("檢查 3:重現性(ideal_base 應重現 v2_proc)")
    print("=" * 70)

    gains = []
    for s in SEEDS:
        p = RUN_ROOT / f"ideal_base_s{s}" / "metrics.json"
        if not p.exists():
            warn(f"{p} 還不存在,跳過")
            return
        gains.append(json.load(open(p))["test"]["frc_gain"])

    mean = sum(gains) / len(gains)
    sd = (sum((g - mean) ** 2 for g in gains) / max(len(gains) - 1, 1)) ** 0.5
    print(f"     ideal_base FRC gain = {mean:.4f} ± {sd:.4f}  (seeds: "
          + ", ".join(f"{g:.4f}" for g in gains) + ")")

    if V2_PROC_FRC_GAIN is None:
        warn("未設定 V2_PROC_FRC_GAIN,跳過重現性比對",
             "請在本檔頂端填入 v2_proc 的實際 FRC gain")
        return
    check(f"ideal_base 重現 v2_proc(參考值 {V2_PROC_FRC_GAIN:.4f})",
          abs(mean - V2_PROC_FRC_GAIN) < V2_PROC_TOL,
          f"差異 {mean - V2_PROC_FRC_GAIN:+.4f}  容忍 ±{V2_PROC_TOL}")


# ============================================================================
# 主結果 + 交叉評估矩陣
# ============================================================================
def report():
    print("\n" + "=" * 70)
    print("主結果")
    print("=" * 70)

    def agg(name, key):
        vals = []
        for s in SEEDS:
            p = RUN_ROOT / f"{name}_s{s}" / "metrics.json"
            if not p.exists():
                return None
            vals.append(json.load(open(p))["test"][key])
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        return m, sd

    keys = [("frc_gain", "FRC gain", "{:+.4f}"),
            ("amp_psnr", "PSNR (dB)", "{:.2f}"),
            ("phase_rmse", "Phase RMSE", "{:.4f}"),
            ("frc_res", "解析度 (px)", "{:.3f}"),
            ("material_mae", "材料 MAE", "{:.4f}"),
            ("r_factor", "R-factor", "{:.4f}")]

    print(f"\n{'指標':<14}" + "".join(f"{n:>22}" for n in NAMES))
    print("-" * (14 + 22 * len(NAMES)))
    for k, label, fmt in keys:
        row = f"{label:<14}"
        for n in NAMES:
            r = agg(n, k)
            row += f"{(fmt.format(r[0]) + ' ±' + f'{r[1]:.4f}') if r else '—':>22}"
        print(row)

    # 主判準
    gb, gi = agg("ideal_base", "frc_gain"), agg("ideal_bs0_ph1e5", "frc_gain")
    go = agg("ideal_bs0_ph1e5_oracle", "frc_gain")
    if gb and gi and go:
        print("\n" + "-" * 70)
        print(f"  正常路徑改善      : {gi[0] - gb[0]:+.4f}  "
              f"({gb[0]:+.4f} -> {gi[0]:+.4f})")
        print(f"  理想條件下的落差  : {go[0] - gi[0]:+.4f}  "
              f"(正常 {gi[0]:+.4f} vs Oracle {go[0]:+.4f})")
        print(f"  現況條件下的落差  : 約 +0.08   "
              f"(v2_proc ~0.21 vs v2_proc_oracle 0.2924,§11.3)")
        print("\n  判讀:落差顯著收斂 -> 核心診斷坐實;落差不變 -> 須重新檢討")

    # 交叉評估矩陣
    print("\n" + "=" * 70)
    print("交叉評估矩陣(FRC gain,seed 0)")
    print("=" * 70)
    print("  ⚠️  離開主場的格子帶有 §12b.4 的正規化假影,不可解讀為物理效應")
    for n in NAMES:
        p = RUN_ROOT / f"{n}_s0" / "metrics.json"
        if not p.exists():
            continue
        rob = json.load(open(p)).get("robustness")
        if not rob:
            warn(f"{n} 沒有 robustness 區塊", "eval_photons/eval_beamstops 沒生效?")
            continue
        print(f"\n  {n}:")
        for cond, r in sorted(rob.items()):
            print(f"      {cond:<20} FRC gain {r['frc_gain']:+.4f}   "
                  f"PSNR {r['amp_psnr']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-only", action="store_true")
    a = ap.parse_args()

    cfgs = check_configs()
    if not a.configs_only and cfgs:
        check_norm(cfgs)
        check_repro()
        report()

    print("\n" + "=" * 70)
    print("✅ 全部通過" if ok_all else "❌ 有項目未通過 —— 先處理再往下走")
    print("=" * 70)
