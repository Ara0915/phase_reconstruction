#!/usr/bin/env python
"""檢查各次訓練是否已收斂 —— 比較結果前的必要步驟。

背景:epochs=12 時各設定均未收斂,且收斂進度不一致
      (src_proc 最後降幅 0.95%、oracle_proc 1.42%),
      導致「oracle 表現不如正常路徑」這種資訊論上不可能的觀察。
      oracle 的輸入是正常輸入的嚴格超集,多給資訊不可能使表現變差;
      該現象純粹來自 oracle 因多出輸入通道而收斂較慢。

判準:最後 5 個 epoch 的平均降幅 < 0.1% of loss 視為已收斂。

用法(需在計算節點執行):
    python check_convergence.py --runs-root /work/elviss0915/runs
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", default="/work/elviss0915/runs")
ap.add_argument("--window", type=int, default=5)
ap.add_argument("--threshold", type=float, default=0.001)
a = ap.parse_args()

groups = defaultdict(list)
for f in sorted(Path(a.runs_root).glob("*/history.json")):
    groups[re.sub(r"_s\d+$", "", f.parent.name)].append(f)

print(f"\n{'setting':<20}{'n':>3}{'epochs':>8}{'最終loss':>11}"
      f"{'末段每步降幅':>14}{'佔loss比':>11}{'判定':>8}")
print("-" * 78)
for name, files in sorted(groups.items()):
    rates, finals, eps = [], [], []
    for f in files:
        h = json.load(open(f))
        if len(h) < a.window + 1:
            continue
        tot = [e["total"] for e in h]
        # 末 window 個 epoch 的平均每步降幅
        rate = (tot[-a.window - 1] - tot[-1]) / a.window
        rates.append(rate); finals.append(tot[-1]); eps.append(len(tot))
    if not rates:
        continue
    r, fin = np.mean(rates), np.mean(finals)
    frac = r / max(fin, 1e-12)
    ok = frac < a.threshold
    print(f"{name:<20}{len(rates):>3}{int(np.mean(eps)):>8}{fin:>11.4f}"
          f"{r:>14.5f}{100*frac:>10.2f}%{'已收斂' if ok else '未收斂':>8}")

print(f"\n判準:最後 {a.window} 個 epoch 的平均每步降幅 < "
      f"{100*a.threshold:.1f}% of loss")
print("未收斂的設定不可用於相互比較 —— 收斂進度不同會使比較失效。")
