#!/usr/bin/env python
"""彙整 probe robustness 的多重條件評估結果。

先前結論「參數隨機化使表現下降」建立在有缺陷的評估方式上：
訓練時隨機化劑量與 beamstop，卻只以單一固定條件評估。
本腳本比較兩組設定在**相同的多重條件**下的表現。

用法（需在計算節點執行）：
    python collect_rob.py --runs-root /work/elviss0915/runs
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", default="/work/elviss0915/runs")
ap.add_argument("--key", default="frc_gain")
a = ap.parse_args()

groups = defaultdict(list)
for f in sorted(Path(a.runs_root).glob("*/metrics.json")):
    m = json.load(open(f))
    if "robustness" in m:
        groups[re.sub(r"_s\d+$", "", m["run"])].append(m)

if not groups:
    raise SystemExit("找不到含 robustness 欄位的結果")

conds = sorted(next(iter(groups.values()))[0]["robustness"].keys())

print(f"\n=== 各實驗條件下的 {a.key}（越大越好）===")
print(f"{'setting':<12}" + "".join(f"{c:>14}" for c in conds))
print("-" * (12 + 14 * len(conds)))
summary = {}
for name, ms in sorted(groups.items()):
    row, vals = f"{name:<12}", []
    for c in conds:
        v = float(np.mean([m["robustness"][c][a.key] for m in ms]))
        row += f"{v:>14.4f}"
        vals.append(v)
    print(row)
    summary[name] = vals

print(f"\n{'setting':<12}{'最差':>10}{'中位':>10}{'最佳':>10}{'全距':>10}")
print("-" * 52)
for name, vals in sorted(summary.items()):
    sv = sorted(vals)
    print(f"{name:<12}{sv[0]:>10.4f}{sv[len(sv)//2]:>10.4f}"
          f"{sv[-1]:>10.4f}{sv[-1]-sv[0]:>10.4f}")

print("\n判讀：")
print("  中位數 —— 整體水準。隨機化組較高則先前結論應撤回。")
print("  最差值 —— 最不利條件下的表現，即 probe robustness 的核心。")
print("  全距   —— 對條件變化的敏感度，越小代表越穩健。")
