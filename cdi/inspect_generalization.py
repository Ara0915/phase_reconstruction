#!/usr/bin/env python
"""逐分布拆解泛化表現，找出「最差增益」是由哪個分布造成的。

背景：改用「未見分布的相對增益」後，幾乎所有設定的最差值皆為負
      （-2.30 至 -5.20 dB），即在最不利的未見分布上比平庸預測器還差。
      需釐清這是模型泛化能力不足，或是某個特定分布過於極端
      而使該指標失去鑑別力。

判別方式：
  1. 若「最差」永遠落在同一個分布 → 該分布可能與訓練分布差異過大，
     指標被單一極端案例主導，應檢視其是否適合作為泛化測試集。
  2. 若「最差」隨訓練來源而變 → 反映的是真實的分布距離，指標有效。

另輸出各分布的平庸基準與模型絕對分數，以區分
「模型差」與「該分布本身難度高」。

用法（需在計算節點執行）：
    python inspect_generalization.py --runs-root /work/elviss0915/runs \
        --settings v2_mnist v2_proc v2_proc_oracle div50k_u50k
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SUITES = ["mnist_test", "fashion_mnist", "random_shapes",
          "random_texture", "procedural"]
SHORT = {"mnist_test": "mnist", "fashion_mnist": "fashion",
         "random_shapes": "shapes", "random_texture": "texture",
         "procedural": "proc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/work/elviss0915/runs")
    ap.add_argument("--settings", nargs="+", required=True)
    a = ap.parse_args()

    groups = defaultdict(list)
    for f in sorted(Path(a.runs_root).glob("*/metrics.json")):
        m = json.load(open(f))
        name = re.sub(r"_s\d+$", "", m["run"])
        if name in a.settings:
            groups[name].append(m)
    if not groups:
        raise SystemExit("找不到指定的設定")

    # ---------- 逐分布的相對增益 ----------
    print("\n=== 各分布的相對增益（模型 PSNR − 該分布平庸基準，dB）===")
    print("（* = 該模型的訓練分布，不計入「最差」）\n")
    print(f"{'setting':<18}" + "".join(f"{SHORT[s]:>12}" for s in SUITES)
          + f"{'最差':>10}{'最差來自':>12}")
    print("-" * (18 + 12 * len(SUITES) + 22))
    worst_src = defaultdict(int)
    for name in a.settings:
        if name not in groups:
            continue
        ms = groups[name]
        homes = set(ms[0].get("home_keys", []))
        row = f"{name:<18}"
        gains, labels = [], []
        for s in SUITES:
            vals = [m["generalization"][s]["amp_psnr"]
                    - m["generalization"][s]["amp_psnr_trivial"]
                    for m in ms if s in m["generalization"]]
            if not vals:
                row += f"{'-':>12}"
                continue
            g = float(np.mean(vals))
            mark = "*" if s in homes else " "
            row += f"{g:>11.2f}{mark}"
            if s not in homes:
                gains.append(g); labels.append(SHORT[s])
        if gains:
            i = int(np.argmin(gains))
            row += f"{gains[i]:>10.2f}{labels[i]:>12}"
            worst_src[labels[i]] += 1
        print(row)

    print(f"\n「最差」由各分布造成的次數：{dict(worst_src)}")
    if len(worst_src) == 1:
        k = list(worst_src)[0]
        print(f"  -> 全部由 {k} 造成。該指標被單一分布主導，")
        print(f"     須檢視 {k} 是否與訓練分布差異過大而不適合作為泛化測試。")
    else:
        print("  -> 隨訓練來源而變，反映真實的分布距離，指標有效。")

    # ---------- 絕對分數與基準 ----------
    print("\n=== 絕對分數 vs 該分布的平庸基準（dB）===")
    print("（用以區分「模型差」與「該分布本身難」）\n")
    print(f"{'setting':<18}" + "".join(f"{SHORT[s]:>18}" for s in SUITES))
    print("-" * (18 + 18 * len(SUITES)))
    for name in a.settings:
        if name not in groups:
            continue
        ms = groups[name]
        row = f"{name:<18}"
        for s in SUITES:
            vals = [m["generalization"][s]["amp_psnr"] for m in ms
                    if s in m["generalization"]]
            refs = [m["generalization"][s]["amp_psnr_trivial"] for m in ms
                    if s in m["generalization"]]
            row += f"{np.mean(vals):>11.2f}/{np.mean(refs):<6.2f}" if vals \
                else f"{'-':>18}"
        print(row)
    print("\n格式為「模型/基準」。基準本身高 = 該分布的平均圖就很接近個別樣本。")

    # ---------- 分布的平庸基準強度 ----------
    print("\n=== 各分布平庸基準的絕對值（dB，跨設定平均）===")
    print("基準越高，代表該分布越「容易被平均圖蒙對」，模型越難贏過它。\n")
    for s in SUITES:
        refs = [m["generalization"][s]["amp_psnr_trivial"]
                for ms in groups.values() for m in ms
                if s in m["generalization"]]
        if refs:
            print(f"  {SHORT[s]:<10}{np.mean(refs):>8.2f}")


if __name__ == "__main__":
    main()
