#!/usr/bin/env python
"""把所有 runs/*/metrics.json 收成一張表,計算跨 seed 的平均與標準差。

這才是能放進報告的東西。在**本機**跑就好,不需要 GPU。

用法:
    python collect.py --runs-root /work/elviss0915/runs
    python collect.py --runs-root ./runs --csv summary.csv
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SUITE_ALL = ["mnist_test", "fashion_mnist", "random_shapes",
             "random_texture", "procedural"]


def unseen_gains(m):
    """模型在「未見分布」上，各自贏過該分布平庸基準多少（dB）。

    為何不用「泛化落差 = 訓練分布 − 未見分布」：
    該式含訓練分布的分數，模型在訓練分布上變強時被減數變大，
    即使未見分布同步進步，落差仍可能上升。實測 div50k_u20k → u50k
    落差由 6.76 升至 7.94（看似變差），但 FRC gain、材料 MAE、
    解析度、相位 RMSE 四項均改善 —— 唯一唱反調的正是唯一
    含訓練分布分數的指標。

    為何要減各分布自己的基準:各分布難度不同，
    絕對 PSNR 不可跨分布加總；扣除各自的平庸基準後，
    「贏多少」在不同難度下才是同一個意思。

    使用時須注意：不宜僅以「最差值」判讀。實測顯示 random_texture
    對所有設定皆為負（-3.09 至 -4.89 dB），包含以 MNIST 訓練者 ——
    純隨機紋理無結構可循，對任何學習方法皆為最難情形，
    故最差值常被此單一分布主導而失去鑑別力。
    應併看「贏過基準的分布數」與「中位增益」。

    回傳 {分布名: 相對增益}，僅含未見分布。
    """
    g = m["generalization"]
    homes = set(m.get("home_keys", []))
    out = {}
    for k in SUITE_ALL:
        if k in g and k not in homes:
            out[k] = g[k]["amp_psnr"] - g[k]["amp_psnr_trivial"]
    return out


KEYS = [
    ("amp_psnr", "amp_psnr_trivial", "↑", "amplitude PSNR (dB)"),
    ("phase_rmse", "phase_rmse_trivial", "↓", "phase RMSE (rad)"),
    ("frc_auc", "frc_auc_trivial", "↑", "FRC AUC"),
    ("frc_gain", None, "↑", "FRC gain over trivial"),
    ("frc_res", "frc_res_trivial", "↓", "FRC res @0.5 (px)"),
    ("r_factor", "r_factor_noise_floor", "↓", "R-factor"),
    ("material_mae", "material_mae_trivial", "↓", "材料 MAE"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", default="/work/elviss0915/runs")
ap.add_argument("--csv", default=None)
a = ap.parse_args()

groups = defaultdict(list)
for f in sorted(Path(a.runs_root).glob("*/metrics.json")):
    m = json.load(open(f))
    name = re.sub(r"_s\d+$", "", m["run"])
    groups[name].append(m)

print("注意:下表的 amplitude PSNR / phase RMSE / FRC / R-factor 一律取自"
      "各模型自己的訓練分布(見 home_keys),不是固定的 MNIST —— 跨來源比較"
      "時這是唯一公平的量法。若模型是用 procedural 訓練的,'test' 欄位就是"
      "它在 procedural 分布上的表現,不是在 MNIST 上的表現。\n")
homes_seen = {}
for name, ms in groups.items():
    hk = ms[0].get("home_keys")
    if hk:
        homes_seen[name] = hk
if homes_seen:
    print("各設定的訓練分布(home):")
    for k, v in homes_seen.items():
        print(f"  {k:<16} {v}")
    print()

if not groups:
    raise SystemExit(f"在 {a.runs_root} 底下找不到任何 metrics.json")

rows = []
print(f"{'setting':<16}{'n':>3}  " + "".join(f"{lab:>26}" for _, _, _, lab in KEYS))
print("-" * (21 + 26 * len(KEYS)))
for name, ms in sorted(groups.items()):
    line = f"{name:<16}{len(ms):>3}  "
    row = {"setting": name, "n_seeds": len(ms)}
    for key, ref, arrow, _ in KEYS:
        v = np.array([m["test"][key] for m in ms])
        row[f"{key}_mean"], row[f"{key}_std"] = v.mean(), v.std()
        cell = f"{v.mean():.4f}±{v.std():.4f}"
        if ref:
            rv = np.mean([m["test"][ref] for m in ms])
            row[f"{key}_ref"] = rv
            cell += f" [{rv:.3f}]"
        line += f"{cell:>26}"
    gap = np.array([m["generalization_gap_db"] for m in ms])
    row["gen_gap_mean"], row["gen_gap_std"] = gap.mean(), gap.std()

    # 未見分布的相對增益：最差值為主指標（通用 = 每一種都要能做），
    # 中位數為整體水準（對離群較平均穩健）。
    worst, med, n_unseen, n_win = [], [], [], []
    for m in ms:
        ug = unseen_gains(m)
        n_unseen.append(len(ug))
        if ug:
            v = sorted(ug.values())
            worst.append(v[0])
            med.append(v[len(v) // 2] if len(v) % 2 else
                       0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2]))
            # 贏過該分布自身平庸基準的未見分布數量：
            # 不受單一極端分布主導，較最差值穩健
            n_win.append(sum(1 for x in v if x > 0))
    row["unseen_worst_mean"] = np.mean(worst) if worst else np.nan
    row["unseen_worst_std"] = np.std(worst) if worst else np.nan
    row["unseen_med_mean"] = np.mean(med) if med else np.nan
    row["n_unseen"] = int(np.mean(n_unseen)) if n_unseen else 0
    row["n_win_mean"] = np.mean(n_win) if n_win else np.nan
    spr = np.array([m.get("gen_spread_db", np.nan) for m in ms])
    row["gen_spread_mean"], row["gen_spread_std"] = np.nanmean(spr), np.nanstd(spr)
    sp = np.array([m["speed_ms_per_sample"]["batch64"] for m in ms])
    row["speed_b64_mean"] = sp.mean()
    rows.append(row)
    print(line)

print("\n[] 內是平庸預測器 / 雜訊底線的參考值")
print(f"\n=== 泛化能力（未見分布上的表現，皆越大越好）===")
print(f"{'setting':<16}{'未見n':>6}{'贏過基準':>10}{'中位增益 (dB)':>16}"
      f"{'最差增益 (dB)':>18}{'舊:落差 (dB)':>16}{'速度 b64':>12}")
print("-" * 96)
for r in rows:
    nw = r.get("n_win_mean", float("nan"))
    print(f"{r['setting']:<16}{r['n_unseen']:>6}"
          f"{nw:>7.1f}/{r['n_unseen']:<2}"
          f"{r['unseen_med_mean']:>16.2f}"
          f"{r['unseen_worst_mean']:>11.2f}±{r['unseen_worst_std']:<5.2f}"
          f"{r['gen_gap_mean']:>10.2f}±{r['gen_gap_std']:<4.2f}"
          f"{r['speed_b64_mean']:>12.4f}")

print("\n三欄皆只計未見分布，完全不含訓練分布的分數；")
print("各分布難度不同，故各自扣除自身的平庸基準後方可並列比較。")
print("  「贏過基準」：有幾個未見分布的增益為正 —— 主指標，不受單一分布主導。")
print("  「中位增益」：整體水準。")
print("  「最差增益」：最壞情形。實測 random_texture 對所有設定皆為負")
print("                （純隨機紋理無結構可循，對任何方法皆最難），")
print("                故此欄常被該單一分布主導，不宜單獨判讀。")
print("\n「舊:落差」= 訓練分布 − 最差未見分布，僅供對照。")
print("  該式含訓練分布分數，模型在訓練分布上變強時落差會被推高，")
print("  即使未見分布同步進步亦然，故不宜作為泛化的主要判準。")
print("  另：home 涵蓋越多分布，未見分布就越少（見「未見n」欄），")
print("  例如 src_mixed 僅剩 1 個未見分布，其數值不可與他者直接相比。")

if len(rows) > 1:
    # 只在 home(訓練分布)相同的設定之間比較。
    # 跨 home 比較是無意義的 —— 例如 bs00 的測試集是 MNIST、
    # src_proc 的是 procedural,兩者測的根本不是同一批樣品,
    # 算出來的 σ 值沒有任何意義。(這是先前版本的錯誤。)
    by_home = defaultdict(list)
    for r in rows:
        by_home[tuple(homes_seen.get(r["setting"], ["<unknown>"]))].append(r)

    for hk, grp in by_home.items():
        if len(grp) < 2:
            if len(by_home) > 1:
                print(f"\n[跳過] home={list(hk)} 只有 {len(grp)} 個設定,無從比較")
            continue
        base = grp[0]
        print(f"\n顯著性(home={list(hk)},對照組 = {base['setting']},門檻 3σ):")
        for r in grp[1:]:
            msgs = []
            for key, _, arrow, lab in KEYS:
                s = np.hypot(base[f"{key}_std"], r[f"{key}_std"])
                if s < 1e-9 or base["n_seeds"] < 2 or r["n_seeds"] < 2:
                    continue
                z = (r[f"{key}_mean"] - base[f"{key}_mean"]) / s
                good = z > 0 if arrow == "↑" else z < 0
                if abs(z) >= 3:
                    msgs.append(f"{lab} {'改善' if good else '惡化'} ({abs(z):.1f}σ)")
            print(f"  {r['setting']:<16} "
                  + ("、".join(msgs) if msgs else "無顯著差異"))
    if any(r["n_seeds"] < 2 for r in rows):
        print("\n  (部分設定只有 1 個 seed,無法判斷顯著性)")

if a.csv:
    import csv
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {a.csv}")
