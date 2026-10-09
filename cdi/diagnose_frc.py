#!/usr/bin/env python
"""診斷 FRC 為何持續輸給平庸基準。

假設:L1 類的逐像素 loss 在「同一張繞射圖對應多種合理重建」時,
      最優策略是輸出這些解的平均,而平均會抹掉高頻 —— 也就是說
      模型在高頻**根本沒有輸出能量**,不是輸出了錯的東西。

判別方式:比較重建與正確答案的徑向功率譜(能量大小),
         再對照 FRC(相關性)。兩者合起來可以區分三種情況:

  (A) 能量偏低 + FRC 低  -> 模型在高頻幾乎不輸出 = 平均化模糊
                            對症:換 loss 標的 或 增加量測約束
  (B) 能量正常 + FRC 低  -> 有輸出但方向錯 = 學錯了
                            對症:不是 loss 的問題,要找別的原因
  (C) 能量偏高 + FRC 低  -> 輸出了假的高頻(雜訊/假影)

用法(需在計算節點執行):
    python diagnose_frc.py --runs-root /work/elviss0915/runs \\
        --runs src_mnist_s0 src_proc_s0 oracle_bs03_gn_s0 --out figs
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
from src.metrics import frc_cropped
from src.model import build_model
from src.physics import beamstop_mask, build_input, forward_measure

HOME_MAP = {"mnist": "mnist_test", "fashion": "fashion_mnist",
            "shapes": "random_shapes", "texture": "random_texture",
            "procedural": "procedural"}


def radial_power(obj, cfg, n_bins=16):
    """物體複數場的徑向功率譜。

    **不做逐張正規化** —— 我們要比較的正是絕對能量大小,
    正規化會把要診斷的東西洗掉。
    """
    psi = torch.polar(obj[:, 0], obj[:, 1])
    F = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    P = F.abs() ** 2
    H = cfg.canvas
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    r = torch.sqrt((yy - H / 2) ** 2 + (xx - H / 2) ** 2)
    edges = torch.linspace(0, H // 2, n_bins + 1)
    fs, ps = [], []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum() < 2:
            continue
        fs.append(float((edges[i] + edges[i + 1]) / 2 / H))
        ps.append(float(P[:, m].mean()))
    return np.array(fs), np.array(ps)


@torch.no_grad()
def diagnose(run_dir, cfg_override=None):
    run = Path(run_dir)
    cfg = Cfg.from_dict(json.load(open(run / "config_used.json")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "final.pt", map_location=device))
    model.eval()
    bs_mask = beamstop_mask(cfg, device=device)

    homes = [HOME_MAP[s] for s in cfg.train_sources if s in HOME_MAP]
    suites = generalization_suites(cfg, cfg.eval_n)
    parts = [suites[k] for k in homes if k in suites]
    obj = (torch.cat(parts, 0) if len(parts) > 1 else parts[0]).to(device)

    counts = forward_measure(obj, bs_mask, cfg)
    pred = model(build_input(obj, counts, bs_mask, cfg))

    p, t = pred.cpu(), obj.cpu()
    # 平庸預測器:輸出資料集平均,作為「完全不輸出高頻」的參考點
    triv_a = t[:, 0].mean(0, keepdim=True).expand_as(t[:, 0])
    sup = (t[:, 0] > cfg.amp_floor).float()
    const = (t[:, 1] * sup).sum() / sup.sum().clamp_min(1)
    triv = torch.stack([triv_a, const * sup], 1)

    f, P_t = radial_power(t, cfg)
    _, P_p = radial_power(p, cfg)
    _, P_v = radial_power(triv, cfg)
    fc, V_p = frc_cropped(p, t, cfg)
    _, V_v = frc_cropped(triv, t, cfg)

    return dict(run=run.name, home=homes, f=f, P_true=P_t, P_pred=P_p,
                P_triv=P_v, frc_f=fc, frc_pred=V_p, frc_triv=V_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/work/elviss0915/runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = [diagnose(Path(args.runs_root) / r) for r in args.runs]

    # ---------------- 表格 ----------------
    print(f"\n{'run':<20}{'頻段':>10}{'真實能量':>12}{'重建能量':>12}"
          f"{'能量比':>9}{'平庸能量比':>12}{'FRC':>8}")
    print("-" * 86)
    for d in res:
        n = len(d["f"])
        for i in [n // 4, n // 2, 3 * n // 4, n - 1]:
            ratio = d["P_pred"][i] / max(d["P_true"][i], 1e-30)
            tratio = d["P_triv"][i] / max(d["P_true"][i], 1e-30)
            j = int(np.argmin(np.abs(d["frc_f"] - d["f"][i])))
            print(f"{d['run'] if i == n // 4 else '':<20}{d['f'][i]:>10.3f}"
                  f"{d['P_true'][i]:>12.3e}{d['P_pred'][i]:>12.3e}"
                  f"{ratio:>9.3f}{tratio:>12.3f}{d['frc_pred'][j]:>8.3f}")
        print()

    # ---------------- 判定 ----------------
    print("=== 判定(取最高頻段)===")
    for d in res:
        i = len(d["f"]) - 1
        ratio = d["P_pred"][i] / max(d["P_true"][i], 1e-30)
        j = int(np.argmin(np.abs(d["frc_f"] - d["f"][i])))
        frc = d["frc_pred"][j]
        if ratio < 0.5:
            verdict = "(A) 高頻能量嚴重不足 -> 平均化模糊,符合假設"
        elif ratio > 2.0:
            verdict = "(C) 高頻能量過剩 -> 輸出了假影"
        elif frc < 0.3:
            verdict = "(B) 能量正常但相關性低 -> 有輸出、方向錯"
        else:
            verdict = "高頻表現正常"
        print(f"  {d['run']:<20} 能量比={ratio:>6.3f}  FRC={frc:>6.3f}  {verdict}")

    # ---------------- 圖 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for d in res:
        l, = ax[0].semilogy(d["f"], d["P_pred"], lw=2, label=f"{d['run']} (recon)")
        ax[0].semilogy(d["f"], d["P_true"], lw=1.5, ls="--",
                       c=l.get_color(), alpha=.6)
        ax[1].plot(d["frc_f"], d["frc_pred"], lw=2, label=d["run"])
    ax[0].semilogy(res[0]["f"], res[0]["P_triv"], lw=2, ls=":", c="k",
                   label="trivial")
    ax[0].set_xlabel("spatial frequency (1/pixel)")
    ax[0].set_ylabel("radial power (absolute)")
    ax[0].set_title("Power spectrum: solid=recon, dashed=ground truth")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")
    ax[1].plot(res[0]["frc_f"], res[0]["frc_triv"], lw=2, ls="--", c="k",
               label="trivial")
    ax[1].axhline(.5, ls=":", c="r", lw=1)
    ax[1].set_xlabel("spatial frequency (1/pixel)")
    ax[1].set_ylabel("FRC")
    ax[1].set_title("FRC (for reference)")
    ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out / "frc_diagnosis.png", dpi=150)
    plt.close()
    print(f"\n-> {out/'frc_diagnosis.png'}")


if __name__ == "__main__":
    main()
