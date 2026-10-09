#!/usr/bin/env python
"""1a:公平比較 src_mnist 與 src_proc 的相位學習能力。

問題:兩者的 home 不同(mnist_test vs procedural),
      主表的 PSNR / FRC 是在各自的資料集上量的,不可直接比較。

做法:
  1. 印出完整的「模型 × 測試集」矩陣 —— 這些數字已存在於 metrics.json
     的 generalization 欄位,不需重新訓練。
  2. 每格改報「相對於該測試集自身基準的進步幅度」:

         相位進步 = (抄振幅基準 - 模型) / 抄振幅基準

     「抄振幅基準」是由重建出的振幅線性預測相位的最小平方解,
     代表「完全不解相位、只把振幅換算過去」能拿到的成績。
     基準隨測試集自動調整,故此比值可跨資料集比較。

  3. 檢查一個干擾因素:若某資料集的振幅與相位本身就比較不相關,
     其抄振幅基準會較差、進步空間較大,比值就會虛高。
     故一併量測各資料集的振幅-相位內在相關性。

用法(需在計算節點執行):
    python fair_compare.py --runs-root /work/elviss0915/runs
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.config import Cfg
from src.data import generalization_suites

SUITES = ["mnist_test", "fashion_mnist", "random_shapes",
          "random_texture", "procedural"]


def phase_gain(rec):
    """模型比「只從振幅推算相位」好多少(比例)。基準隨測試集調整,可跨集比較。"""
    copy = rec["phase_rmse_copy_amp"]
    return (copy - rec["phase_rmse"]) / max(copy, 1e-12)


def trivial_gain(rec):
    """模型比「輸出常數相位」好多少(比例)。"""
    triv = rec["phase_rmse_trivial"]
    return (triv - rec["phase_rmse"]) / max(triv, 1e-12)


def amp_phase_correlation(cfg, n=512):
    """資料集的振幅-相位內在相關性(support 內,逐樣本後平均)。

    這是干擾因素:相關性越低,「抄振幅」越沒用、基準越差,
    模型的相對進步就越容易看起來很大。
    """
    out = {}
    for name, obj in generalization_suites(cfg, n).items():
        a, p = obj[:, 0], obj[:, 1]
        sup = (a > cfg.amp_floor).float()
        cs = []
        for i in range(len(a)):
            m = sup[i] > 0
            if m.sum() < 10:
                continue
            x, y = a[i][m], p[i][m]
            x = x - x.mean(); y = y - y.mean()
            d = (x.norm() * y.norm()).clamp_min(1e-9)
            cs.append(float((x * y).sum() / d))
        out[name] = float(np.mean(cs)) if cs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/work/elviss0915/runs")
    ap.add_argument("--settings", nargs="+",
                    default=["src_mnist", "src_proc", "src_mixed",
                             "src_proc_rand"])
    args = ap.parse_args()

    groups = defaultdict(list)
    for f in sorted(Path(args.runs_root).glob("*/metrics.json")):
        m = json.load(open(f))
        name = re.sub(r"_s\d+$", "", m["run"])
        if name in args.settings:
            groups[name].append(m)

    if not groups:
        raise SystemExit("找不到指定的設定")

    # ---------- 干擾因素 ----------
    cfg = Cfg.from_dict(groups[list(groups)[0]][0]["config"])
    print("\n=== 干擾因素檢查:各測試集的振幅-相位內在相關性 ===")
    print("(相關性低 -> 抄振幅基準差 -> 模型的相對進步容易虛高)")
    corr = amp_phase_correlation(cfg)
    for k in SUITES:
        if k in corr:
            print(f"  {k:<18}{corr[k]:>8.4f}")

    # ---------- 矩陣 ----------
    for label, fn in [("相位:比『抄振幅』好多少 (%)", phase_gain),
                      ("相位:比『輸出常數』好多少 (%)", trivial_gain)]:
        print(f"\n=== {label} ===")
        print(f"{'模型 \\ 測試集':<18}" + "".join(f"{s:>17}" for s in SUITES))
        print("-" * (18 + 17 * len(SUITES)))
        for name in args.settings:
            if name not in groups:
                continue
            homes = groups[name][0].get("home_keys", [])
            row = f"{name:<18}"
            for s in SUITES:
                vals = [fn(m["generalization"][s]) for m in groups[name]
                        if s in m["generalization"]]
                if not vals:
                    row += f"{'-':>17}"
                    continue
                mark = "*" if s in homes else " "
                row += f"{100*np.mean(vals):>13.1f}±{100*np.std(vals):<3.1f}{mark}"
            print(row)
        print("  * = 該模型自己的訓練分布")

    # ---------- 重點對照 ----------
    print("\n=== 重點對照:各自在自己的 home 上 ===")
    print(f"{'設定':<18}{'home':>16}{'相位RMSE':>11}{'抄振幅基準':>12}"
          f"{'進步':>9}{'該集A-φ相關':>13}")
    print("-" * 80)
    for name in args.settings:
        if name not in groups:
            continue
        homes = groups[name][0].get("home_keys", [])
        if len(homes) != 1:
            print(f"{name:<18}{'(多重 home,略)':>16}")
            continue
        h = homes[0]
        recs = [m["generalization"][h] for m in groups[name]]
        pr = np.mean([r["phase_rmse"] for r in recs])
        cp = np.mean([r["phase_rmse_copy_amp"] for r in recs])
        print(f"{name:<18}{h:>16}{pr:>11.4f}{cp:>12.4f}"
              f"{100*(cp-pr)/cp:>8.1f}%{corr.get(h, float('nan')):>13.4f}")

    print("\n判讀:若 src_proc 的『進步』明顯高於 src_mnist,"
          "\n      且兩者的 A-φ 相關性相近(干擾因素不成立),"
          "\n      則『程序生成資料讓模型真正學會解相位』成立。")


if __name__ == "__main__":
    main()
