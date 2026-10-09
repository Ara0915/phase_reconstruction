#!/usr/bin/env python
"""從 runs/*/metrics.json 畫圖。在**本機**跑就好,不需要 GPU。

    python plot.py --runs-root ./runs --out figs

產出:
    frc_curves.png     各設定的 FRC 曲線 + 平庸基準
    beamstop_sweep.png 解析度 vs beamstop 大小、真實增益 vs beamstop 大小
                       ← 這張回答真實工程問題「我的 beamstop 能做多大」
    generalization.png 各設定在四種樣品分布上的 PSNR
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 批次環境沒有螢幕,一律存檔不顯示
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", default="./runs")
ap.add_argument("--out", default="./figs")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

groups = defaultdict(list)
for f in sorted(Path(a.runs_root).glob("*/metrics.json")):
    m = json.load(open(f))
    groups[re.sub(r"_s\d+$", "", m["run"])].append(m)
if not groups:
    raise SystemExit(f"在 {a.runs_root} 找不到 metrics.json")

# ---------------- FRC 曲線 ----------------
plt.figure(figsize=(7.5, 4.8))
triv = None
for name, ms in sorted(groups.items()):
    c = ms[0]["test"]["_curves"]
    f = np.array(c["freq"])
    v = np.mean([m["test"]["_curves"]["model"] for m in ms], axis=0)
    plt.plot(f, v, lw=2, label=name)
    triv = (f, np.array(c["trivial"]))
plt.plot(*triv, lw=2, ls="--", c="k", label="trivial baseline")
plt.axhline(.5, ls=":", c="r", lw=1)
plt.xlabel("spatial frequency (1/pixel)"); plt.ylabel("FRC")
plt.title("FRC (cropped to object)"); plt.ylim(0, 1.02)
plt.legend(fontsize=9); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(out / "frc_curves.png", dpi=150); plt.close()

# ---------------- beamstop 掃描 ----------------
bs = {}
for name, ms in groups.items():
    r = ms[0]["config"]["beamstop_r"]
    if all(m["config"]["beamstop_r"] == r for m in ms):
        bs[r] = ms
if len(bs) > 1:
    radii = sorted(bs)
    res = [np.mean([m["test"]["frc_res"] for m in bs[r]]) for r in radii]
    res_s = [np.std([m["test"]["frc_res"] for m in bs[r]]) for r in radii]
    gain = [np.mean([m["test"]["frc_gain"] for m in bs[r]]) for r in radii]
    gain_s = [np.std([m["test"]["frc_gain"] for m in bs[r]]) for r in radii]
    trv = [np.mean([m["test"]["frc_res_trivial"] for m in bs[r]]) for r in radii]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].errorbar(radii, res, yerr=res_s, fmt="o-", lw=2, capsize=4, label="model")
    ax[0].plot(radii, trv, "s--", lw=2, c="gray", label="trivial baseline")
    ax[0].set_xlabel("beamstop radius (pixel)")
    ax[0].set_ylabel("FRC resolution @0.5 (pixel)")
    ax[0].set_title("Resolution vs beamstop size")
    ax[0].invert_yaxis(); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].errorbar(radii, gain, yerr=gain_s, fmt="o-", lw=2, c="tab:purple", capsize=4)
    ax[1].axhline(0, ls="--", c="r", lw=1)
    ax[1].set_xlabel("beamstop radius (pixel)")
    ax[1].set_ylabel("FRC gain over trivial (AUC)")
    ax[1].set_title("Real gain vs beamstop size")
    ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out / "beamstop_sweep.png", dpi=150); plt.close()
    print(f"-> {out/'beamstop_sweep.png'}")

# ---------------- 無真值解析度曲線(徑向 R-factor) ----------------
if any("_r_radial" in ms[0]["test"] for ms in groups.values()):
    plt.figure(figsize=(7.5, 4.8))
    floor = None
    for name, ms in sorted(groups.items()):
        rr = ms[0]["test"].get("_r_radial")
        if rr is None:
            continue
        f = np.array(rr["freq"])
        v = np.mean([m["test"]["_r_radial"]["model"] for m in ms], axis=0)
        plt.semilogy(f, v, lw=2, label=name)
        floor = (f, np.array(rr["floor"]))
    if floor is not None:
        plt.semilogy(*floor, lw=2, ls="--", c="k", label="Poisson noise floor")
    plt.xlabel("spatial frequency (1/pixel)")
    plt.ylabel("R-factor (per ring)")
    plt.title("Ground-truth-free resolution curve (radial R-factor)")
    plt.legend(fontsize=9); plt.grid(alpha=.3, which="both"); plt.tight_layout()
    plt.savefig(out / "r_radial.png", dpi=150); plt.close()
    print(f"-> {out/'r_radial.png'}")

# ---------------- 泛化 ----------------
suites = ["mnist_test", "fashion_mnist", "random_shapes", "random_texture"]
names = sorted(groups)
w = 0.8 / len(names)
plt.figure(figsize=(9, 4.6))
for i, name in enumerate(names):
    y = [np.mean([m["generalization"][s]["amp_psnr"] for m in groups[name]])
         for s in suites]
    e = [np.std([m["generalization"][s]["amp_psnr"] for m in groups[name]])
         for s in suites]
    plt.bar(np.arange(len(suites)) + i * w, y, w, yerr=e, capsize=3, label=name)
plt.xticks(np.arange(len(suites)) + w * (len(names) - 1) / 2, suites, fontsize=9)
plt.ylabel("amplitude PSNR (dB)")
plt.title("Generalization across sample distributions")
plt.legend(fontsize=9); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
plt.savefig(out / "generalization.png", dpi=150); plt.close()

print(f"-> {out/'frc_curves.png'}")
print(f"-> {out/'generalization.png'}")
