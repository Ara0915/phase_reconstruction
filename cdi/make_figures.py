#!/usr/bin/env python
"""產生報告用的重建結果對照圖。

先前的所有結論皆建立在數值指標上，從未實際檢視重建影像。
本腳本補上視覺證據，並可能揭露數值指標看不出的問題
（系統性偽影、邊界效應、特定物體類型的失敗）。

產出四組圖：

  1. `fig_main.png`      真值 / 純網路 / 網路+HIO / Oracle 的並列對照
  2. `fig_memorize.png`  同一模型在五個分布上的重建 —— 記憶的視覺證據
  3. `fig_error.png`     誤差圖（重建 − 真值），檢視誤差的空間結構
  4. `fig_material.png`  材料分布的反演對照

用法（需在計算節點執行）：
    python make_figures.py --runs-root /work/elviss0915/runs --out figs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import Cfg
from src.data import generalization_suites
from src.hio import hio
from src.metrics import psnr, recover_material
from src.model import build_model
from src.physics import beamstop_mask, build_input, forward_measure

SUITES = ["mnist_test", "fashion_mnist", "random_shapes",
          "random_texture", "procedural"]
SHORT = {"mnist_test": "MNIST", "fashion_mnist": "Fashion",
         "random_shapes": "Shapes", "random_texture": "Texture",
         "procedural": "Procedural"}
HOME_MAP = {"mnist": "mnist_test", "fashion": "fashion_mnist",
            "shapes": "random_shapes", "texture": "random_texture",
            "procedural": "procedural"}


def load_run(runs_root, name):
    run = Path(runs_root) / name
    cfg = Cfg.from_dict(json.load(open(run / "config_used.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(run / "final.pt", map_location=dev))
    model.eval()
    return cfg, model, dev


@torch.no_grad()
def reconstruct(cfg, model, dev, suite, n=4, hio_iters=0):
    """回傳 (真值, 重建)，皆已移至 CPU。"""
    obj = generalization_suites(cfg, max(n, cfg.gen_n))[suite][:n].to(dev)
    bs = beamstop_mask(cfg, device=dev)
    counts = forward_measure(obj, bs, cfg)
    pred = model(build_input(obj, counts, bs, cfg))
    if hio_iters > 0:
        pred = hio(pred, counts, bs, cfg, n_iter=hio_iters, beta=cfg.hio_beta)
    return obj.cpu(), pred.cpu()


def _panel(ax, img, cmap, vmin, vmax, title=None, ylabel=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def fig_main(args, out):
    """真值 / 純網路 / 網路+HIO / Oracle 的並列對照。"""
    cols = []
    cfg_n, m_n, dev = load_run(args.runs_root, f"{args.net}_s0")
    home = HOME_MAP[list(cfg_n.train_sources)[0]]
    t, p_net = reconstruct(cfg_n, m_n, dev, home, n=args.n)
    cols.append(("Ground truth", t))
    cols.append(("Network only", p_net))

    _, p_hio = reconstruct(cfg_n, m_n, dev, home, n=args.n,
                           hio_iters=args.hio_iters)
    cols.append((f"Network + HIO ({args.hio_iters})", p_hio))

    try:
        cfg_o, m_o, _ = load_run(args.runs_root, f"{args.oracle}_s0")
        _, p_or = reconstruct(cfg_o, m_o, dev, home, n=args.n)
        cols.append(("Oracle (phase given)", p_or))
    except FileNotFoundError:
        print(f"[warn] 找不到 {args.oracle}_s0，略過 Oracle 欄")

    nrow = 2 * args.n          # 每個樣本兩列：振幅、相位
    fig, ax = plt.subplots(nrow, len(cols), figsize=(2.1 * len(cols), 2.1 * nrow))
    for c, (name, arr) in enumerate(cols):
        for i in range(args.n):
            ttl = name if i == 0 else None
            _panel(ax[2*i, c], arr[i, 0], "gray", 0, 1, ttl,
                   f"#{i} amplitude" if c == 0 else None)
            _panel(ax[2*i+1, c], arr[i, 1], "twilight", 0, cfg_n.phase_max,
                   None, f"#{i} phase" if c == 0 else None)
    # 於各欄標註 PSNR（相對真值）
    for c, (name, arr) in enumerate(cols[1:], start=1):
        ax[0, c].set_xlabel(f"PSNR {psnr(arr[:,0], t[:,0]):.1f} dB", fontsize=8)
        ax[0, c].xaxis.set_label_position("top")
    fig.suptitle(f"Reconstruction comparison (training distribution: "
                 f"{SHORT.get(home, home)})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out / "fig_main.png", dpi=150); plt.close()
    print(f"-> {out/'fig_main.png'}")


def fig_memorize(args, out):
    """同一模型在五個分布上的重建 —— 記憶的視覺證據。

    以 MNIST 訓練的模型在未見分布上若仍畫出「像數字的東西」，
    即為記憶而非通用反演的直觀證據。
    """
    for tag, run in [("MNIST-trained", args.mnist_net),
                     ("Procedural-trained", args.net)]:
        try:
            cfg, model, dev = load_run(args.runs_root, f"{run}_s0")
        except FileNotFoundError:
            print(f"[warn] 找不到 {run}_s0，略過")
            continue
        home = HOME_MAP[list(cfg.train_sources)[0]]
        fig, ax = plt.subplots(2, len(SUITES),
                               figsize=(2.1 * len(SUITES), 4.4))
        for c, s in enumerate(SUITES):
            t, p = reconstruct(cfg, model, dev, s, n=1)
            mark = " *" if s == home else ""
            _panel(ax[0, c], t[0, 0], "gray", 0, 1, SHORT[s] + mark,
                   "Ground truth" if c == 0 else None)
            _panel(ax[1, c], p[0, 0], "gray", 0, 1, None,
                   "Reconstruction" if c == 0 else None)
        fig.suptitle(f"{tag}: same model across five distributions "
                     f"(* = training distribution)", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fn = out / f"fig_memorize_{run}.png"
        plt.savefig(fn, dpi=150); plt.close()
        print(f"-> {fn}")


def fig_error(args, out):
    """誤差圖：檢視誤差的空間結構（邊緣？整體偏移？特定區域？）。"""
    cfg, model, dev = load_run(args.runs_root, f"{args.net}_s0")
    home = HOME_MAP[list(cfg.train_sources)[0]]
    t, p0 = reconstruct(cfg, model, dev, home, n=args.n)
    _, ph = reconstruct(cfg, model, dev, home, n=args.n,
                        hio_iters=args.hio_iters)

    fig, ax = plt.subplots(args.n, 5, figsize=(11, 2.2 * args.n))
    if args.n == 1:
        ax = ax[None, :]
    for i in range(args.n):
        d0 = (p0[i, 0] - t[i, 0]).numpy()
        dh = (ph[i, 0] - t[i, 0]).numpy()
        v = max(abs(d0).max(), abs(dh).max(), 1e-6)
        _panel(ax[i, 0], t[i, 0], "gray", 0, 1,
               "Ground truth" if i == 0 else None)
        _panel(ax[i, 1], p0[i, 0], "gray", 0, 1,
               "Network" if i == 0 else None)
        _panel(ax[i, 2], d0, "bwr", -v, v,
               "Error (network)" if i == 0 else None)
        _panel(ax[i, 3], ph[i, 0], "gray", 0, 1,
               f"+ HIO {args.hio_iters}" if i == 0 else None)
        _panel(ax[i, 4], dh, "bwr", -v, v,
               "Error (+HIO)" if i == 0 else None)
    fig.suptitle("Error maps (red = over-estimate, blue = under-estimate; "
                 "shared colour scale)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out / "fig_error.png", dpi=150); plt.close()
    print(f"-> {out/'fig_error.png'}")


def fig_material(args, out):
    """材料分布的反演對照。

    材料由 φ/A 反解，要求振幅與相位**同時**準確，
    較「兩張圖各自看起來像」嚴格得多。
    """
    cfg, model, dev = load_run(args.runs_root, f"{args.net}_s0")
    home = HOME_MAP[list(cfg.train_sources)[0]]
    t, p0 = reconstruct(cfg, model, dev, home, n=args.n)
    _, ph = reconstruct(cfg, model, dev, home, n=args.n,
                        hio_iters=args.hio_iters)

    m_t = recover_material(t, cfg)
    m_0 = recover_material(p0, cfg)
    m_h = recover_material(ph, cfg)
    sup = (t[:, 0] > cfg.amp_floor).float()

    fig, ax = plt.subplots(args.n, 4, figsize=(9, 2.2 * args.n))
    if args.n == 1:
        ax = ax[None, :]
    for i in range(args.n):
        _panel(ax[i, 0], t[i, 0], "gray", 0, 1,
               "Amplitude (truth)" if i == 0 else None)
        _panel(ax[i, 1], (m_t[i] * sup[i]), "viridis", 0, 1,
               "Material (truth)" if i == 0 else None)
        _panel(ax[i, 2], (m_0[i] * sup[i]), "viridis", 0, 1,
               "Material (network)" if i == 0 else None)
        _panel(ax[i, 3], (m_h[i] * sup[i]), "viridis", 0, 1,
               f"Material (+HIO)" if i == 0 else None)
    fig.suptitle("Material recovery (from phi/A ratio; shown inside support only)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out / "fig_material.png", dpi=150); plt.close()
    print(f"-> {out/'fig_material.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/work/elviss0915/runs")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--net", default="v2_proc", help="主要模型（程序生成訓練）")
    ap.add_argument("--oracle", default="v2_proc_oracle")
    ap.add_argument("--mnist-net", default="v2_mnist")
    ap.add_argument("--hio-iters", type=int, default=500)
    ap.add_argument("-n", type=int, default=3, help="每張圖顯示幾個樣本")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_main(a, out)
    fig_memorize(a, out)
    fig_error(a, out)
    fig_material(a, out)
    print("\n完成。四組圖皆使用既有的 final.pt，未重新訓練。")


if __name__ == "__main__":
    main()
