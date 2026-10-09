#!/usr/bin/env python
"""彙整 HIO 迭代掃描的結果。

`collect.py` 的主表取自 metrics.json 的 `test` 欄位，那是**模型直接輸出**、
未經 HIO 精煉的結果，故 `hio_proc` 與 `v2_proc` 在該表上幾乎完全相同。
HIO 的實際效果存於 `hio` 欄位，需本腳本彙整。

輸出三部分：

1. **訓練分布上的迭代掃描** —— 品質隨迭代數如何變化，含最佳點
2. **未見分布上的「贏過基準」** —— 本階段的核心驗收：能否由 0/4 提升
3. **R-factor 作為停止準則的有效性** —— R 的最小值是否與 FRC 最佳點吻合。
   R-factor 不需 ground truth，若兩者吻合，真實資料上即可用它決定何時停止。

用法（需在計算節點執行）：
    python collect_hio.py --runs-root /work/elviss0915/runs
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


def load(runs_root):
    groups = defaultdict(list)
    for f in sorted(Path(runs_root).glob("*/metrics.json")):
        m = json.load(open(f))
        if "hio" not in m:
            continue
        groups[re.sub(r"_s\d+$", "", m["run"])].append(m)
    return groups


def mean_over_seeds(ms, suite, n_it, key):
    """跨 seed 取平均。缺欄位時回傳 nan —— 早期的執行沒有計時欄位。"""
    vals = [m["hio"][suite][n_it][key] for m in ms
            if suite in m["hio"] and n_it in m["hio"][suite]
            and key in m["hio"][suite][n_it]]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/work/elviss0915/runs")
    a = ap.parse_args()

    groups = load(a.runs_root)
    if not groups:
        raise SystemExit("找不到含 hio 欄位的結果")

    for name, ms in sorted(groups.items()):
        homes = ms[0].get("home_keys", [])
        iters = sorted(ms[0]["hio"][SUITES[0]].keys(), key=int)
        home = homes[0] if homes else "mnist_test"

        # ---------- 1. 訓練分布上的迭代掃描 ----------
        print(f"\n{'='*74}")
        print(f"{name}   （訓練分布 = {home}，n={len(ms)} seeds）")
        print(f"{'='*74}")
        init_src = ms[0].get("hio_init", "network")
        print(f"  初始解來源：{init_src}"
              + ("（純 HIO，傳統方法基準）" if init_src == "random" else "（網路輸出）"))
        print(f"{'迭代':>6}{'FRC gain':>11}{'材料 MAE':>11}{'相位 RMSE':>12}"
              f"{'PSNR':>9}{'R-factor':>11}{'ms/sample':>12}")
        print("-" * 74)
        best_frc, best_r = None, None
        for n_it in iters:
            g = mean_over_seeds(ms, home, n_it, "frc_gain")
            mat = mean_over_seeds(ms, home, n_it, "material_mae")
            pha = mean_over_seeds(ms, home, n_it, "phase_rmse")
            ps = mean_over_seeds(ms, home, n_it, "amp_psnr")
            r = mean_over_seeds(ms, home, n_it, "r_factor")
            t = mean_over_seeds(ms, home, n_it, "hio_ms_per_sample")
            print(f"{n_it:>6}{g:>11.4f}{mat:>11.4f}{pha:>12.4f}"
                  f"{ps:>9.2f}{r:>11.4f}{t:>12.3f}")
            if best_frc is None or g > best_frc[1]:
                best_frc = (n_it, g)
            if best_r is None or r < best_r[1]:
                best_r = (n_it, r)

        base = mean_over_seeds(ms, home, iters[0], "frc_gain")
        print(f"\n  FRC gain 最佳於 {best_frc[0]} 次迭代（{best_frc[1]:+.4f}，"
              f"對 0 次的 {base:+.4f}，改善 {best_frc[1]-base:+.4f}）")
        print(f"  R-factor 最小於 {best_r[0]} 次迭代")
        if best_frc[0] == best_r[0]:
            print("  -> 兩者吻合：R-factor 可作為無真值的停止準則")
        else:
            print(f"  -> 兩者不吻合（{best_r[0]} vs {best_frc[0]}），"
                  "R-factor 作為停止準則須保留誤差")

        # ---------- 2. 核心驗收：未見分布贏過基準數 ----------
        print(f"\n  未見分布的相對增益（模型 − 該分布平庸基準，dB）")
        unseen = [s for s in SUITES if s not in homes]
        print(f"  {'迭代':>6}" + "".join(f"{SHORT[s]:>10}" for s in unseen)
              + f"{'贏過基準':>10}")
        print("  " + "-" * (6 + 10 * len(unseen) + 10))
        for n_it in iters:
            row, n_win = f"  {n_it:>6}", 0
            for s in unseen:
                v = (mean_over_seeds(ms, s, n_it, "amp_psnr")
                     - mean_over_seeds(ms, s, n_it, "amp_psnr_trivial"))
                row += f"{v:>10.2f}"
                if v > 0:
                    n_win += 1
            print(row + f"{n_win:>7}/{len(unseen)}")

    # ---------- 速度 vs 品質對照 ----------
    nets = {k: v for k, v in groups.items()
            if v[0].get("hio_init", "network") == "network"}
    rands = {k: v for k, v in groups.items()
             if v[0].get("hio_init") == "random"}
    if nets and rands:
        print("\n" + "=" * 74)
        print("速度 vs 品質：網路初始解 對 隨機初始解（純 HIO）")
        print("=" * 74)
        print("問題：網路提供的初始解，在速度軸上值多少？")
        print("若網路 + N 次迭代可達純 HIO 需 M 次（M >> N）才能到的品質，")
        print("則網路初始化有實質貢獻，即使跨分布能力未改善。\n")
        nk, rk = sorted(nets)[0], sorted(rands)[0]
        nms, rms = nets[nk], rands[rk]
        home = (nms[0].get("home_keys") or ["procedural"])[0]
        iters = sorted(nms[0]["hio"][home].keys(), key=int)
        print(f"{'迭代':>6}"
              f"{'網路 FRC':>11}{'網路 ms':>10}"
              f"{'隨機 FRC':>11}{'隨機 ms':>10}")
        print("-" * 48)
        for n_it in iters:
            print(f"{n_it:>6}"
                  f"{mean_over_seeds(nms, home, n_it, 'frc_gain'):>11.4f}"
                  f"{mean_over_seeds(nms, home, n_it, 'hio_ms_per_sample'):>10.2f}"
                  f"{mean_over_seeds(rms, home, n_it, 'frc_gain'):>11.4f}"
                  f"{mean_over_seeds(rms, home, n_it, 'hio_ms_per_sample'):>10.2f}")

    print("\n" + "=" * 74)
    print("核心驗收：「贏過基準」能否由 0/4 提升。")
    print("迭代數同時對應計畫 benchmark 的 reconstruction speed 軸 ——")
    print("純網路 0.029 ms/sample，HIO 每次迭代皆有額外成本。")


if __name__ == "__main__":
    main()
