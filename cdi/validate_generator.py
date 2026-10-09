#!/usr/bin/env python
"""驗證程序生成器是否真的解決了問題。

不能憑感覺說「看起來夠複雜」,必須量化四件事:

  1. PCA 有效維度 —— 要幾個主成分才能解釋 90% 變異。
                     直接量「這個資料集有多少獨立的結構」。小 = 好背。
  2. 最近鄰距離   —— 把資料集切一半,每個測試樣本找最近的訓練樣本。
                     小 = 訓練集裡總有一個長得很像的 = 好背。
  3. support 內平均振幅 —— 對比度。兩個資料集要能比較絕對 PSNR,對比度須相近。
  4. 徑向功率譜   —— 檢查有沒有不小心做出**帶限訊號**
                     (upsample 出來的圖高頻是空的,會讓指標虛假變好)。
  5. Support 佔比 —— 要遠低於 50%,滿足相位恢復的過取樣唯一性條件。

**重要:比較時必須對齊 support 佔比。**
稀疏物體天生就有較高的相對高頻能量(孤立細結構在頻域展得開)
與較低的樣本相關係數(兩張稀疏圖重疊少)。
不對齊的話,指標量到的是稀疏度而不是多樣性 ——
這是第一版驗證犯的錯,MNIST support 0.032 對上程序生成 0.088,
結論完全不可信。

用法:
    python validate_generator.py --data-root /work/elviss0915/data --out figs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import Cfg
from src.data import SampleDataset, load_raw, make_shapes, make_texture, stack_dataset
from src.metrics import psnr
from src.procedural import make_procedural, GENERATORS


def radial_power(obj, cfg, n_bins=16):
    """物體(複數場)的徑向功率譜,正規化成總能量為 1。"""
    psi = torch.polar(obj[:, 0], obj[:, 1])
    F = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    P = (F.abs() ** 2)
    P = P / P.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

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
        ps.append(float(P[:, m].mean()))       # 每像素平均功率
    return np.array(fs), np.array(ps)


def effective_dim(obj, thresh=0.90):
    """PCA:要幾個主成分才能解釋 thresh 比例的變異。

    直接量「這個資料集含有多少獨立的結構」,且**與稀疏度無關**。
    小 = 結構種類少 = 網路可以用少量模板記住整個資料集。
    """
    x = obj[:, 0].reshape(len(obj), -1).double()
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    v = (s ** 2)
    v = v / v.sum().clamp_min(1e-12)
    return int((torch.cumsum(v, 0) < thresh).sum().item()) + 1


def nn_distance(obj):
    """資料集切一半,每個「測試」樣本找最近的「訓練」樣本,回傳平均餘弦距離。

    小 = 訓練集裡總有一個長得很像的 = 記憶策略有效。
    這是「可記憶性」最直接的操作型定義。
    """
    x = obj[:, 0].reshape(len(obj), -1)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
    k = len(x) // 2
    d = 1 - x[k:] @ x[:k].T
    return float(d.min(1).values.mean())


def trivial_psnr(obj, cfg):
    """平庸預測器(輸出資料集平均振幅)能拿到的 PSNR。

    **這不是多樣性指標。** 平庸預測器的 MSE 在數學上就等於資料集的變異數,
    所以此數字純粹是「圖有多亮 / 對比多強」的函數。
    實測:把同一批物體整體乘 0.3,平庸 PSNR 由 25.6 升到 36.1 dB,
    而 PCA 維度與最近鄰距離完全不變。

    仍然報告它,因為它是模型在**該資料集內**必須超越的基準線;
    但不可用來比較不同資料集的多樣性。
    """
    mean = obj[:, 0].mean(0, keepdim=True).expand_as(obj[:, 0])
    return psnr(mean, obj[:, 0])


def mean_amp_in_support(obj, cfg):
    """support 內的平均振幅 —— 對比度指標。

    兩個資料集若對比度差很多,絕對 PSNR 就不可直接比較。
    """
    sup = (obj[:, 0] > cfg.amp_floor).float()
    return float((obj[:, 0] * sup).sum() / sup.sum().clamp_min(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/work/elviss0915/data")
    ap.add_argument("--out", default="figs")
    ap.add_argument("-n", type=int, default=512)
    args = ap.parse_args()

    cfg = Cfg(data_root=args.data_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n = args.n

    suites = {}
    try:
        suites["MNIST"] = stack_dataset(
            SampleDataset(cfg, load_raw(cfg, train=True)), n)
    except RuntimeError as e:
        print(f"[warn] 讀不到 MNIST,跳過對照:{e}")
    # 對齊 support:以 MNIST 為基準,否則指標量到的是稀疏度而非多樣性
    ts = None
    if "MNIST" in suites:
        ts = float((suites["MNIST"][:, 0] > cfg.amp_floor).float().mean())
        print(f"[對齊] 以 MNIST 的 support = {ts:.4f} 作為所有程序生成資料集的目標")
        print("       (此機制以分位數裁切,只能減少 support、無法增加;"
              "程序生成的自然 support 約 0.08)")

    suites["random_shapes"] = make_shapes(cfg, n)
    suites["random_texture"] = make_texture(cfg, n)
    for k in GENERATORS:
        suites[f"proc:{k}"] = GENERATORS[k](n, cfg, seed=0, target_support=ts)
    suites["proc:mixed"] = make_procedural(n, cfg, seed=0, target_support=ts)

    # ---------------- 表格 ----------------
    print(f"\n{'資料集':<18}{'support':>10}{'PCA維度':>10}{'最近鄰距離':>13}"
          f"{'平均振幅':>10}{'平庸PSNR':>11}{'高頻能量比':>13}")
    print("-" * 88)
    rows = {}
    for name, obj in suites.items():
        f, p = radial_power(obj, cfg)
        half = len(f) // 2
        hf = p[half:].sum() / p.sum()          # 後半頻段佔總功率的比例
        rows[name] = dict(
            support=float((obj[:, 0] > cfg.amp_floor).float().mean()),
            dim=effective_dim(obj),
            nn=nn_distance(obj),
            amp=mean_amp_in_support(obj, cfg),
            triv=trivial_psnr(obj, cfg),
            hf=float(hf), f=f, p=p)
        r = rows[name]
        print(f"{name:<18}{r['support']:>10.4f}{r['dim']:>10}{r['nn']:>13.4f}"
              f"{r['amp']:>10.3f}{r['triv']:>11.2f}{r['hf']:>13.4f}")

    if "MNIST" in rows:
        print("\n相對 MNIST:")
        b = rows["MNIST"]
        for name, r in rows.items():
            if name == "MNIST":
                continue
            print(f"  {name:<18} PCA維度 x{r['dim']/max(b['dim'],1):>5.2f}   "
                  f"最近鄰距離 x{r['nn']/max(b['nn'],1e-9):>5.2f}   "
                  f"平均振幅 x{r['amp']/max(b['amp'],1e-9):>5.2f}   "
                  f"高頻 x{r['hf']/max(b['hf'],1e-9):>5.2f}")

    # ---------------- 徑向功率譜 ----------------
    plt.figure(figsize=(7.5, 4.8))
    for name, r in rows.items():
        style = dict(lw=2.5, c="k", ls="--") if name == "MNIST" else dict(lw=2)
        plt.semilogy(r["f"], r["p"], label=name, **style)
    plt.xlabel("spatial frequency (1/pixel)")
    plt.ylabel("mean power per pixel (normalized)")
    plt.title("Radial power spectrum of the object")
    plt.legend(fontsize=8); plt.grid(alpha=.3, which="both"); plt.tight_layout()
    plt.savefig(out / "object_spectrum.png", dpi=150); plt.close()
    print(f"\n-> {out/'object_spectrum.png'}")

    # ---------------- 樣品外觀 ----------------
    show = [k for k in suites if k == "MNIST" or k.startswith("proc:")][:5]
    fig, ax = plt.subplots(2, len(show) * 2, figsize=(3 * len(show) * 2 / 1.4, 5.6))
    for c, name in enumerate(show):
        o = suites[name]
        for row in range(2):
            ax[row, 2 * c].imshow(o[row, 0], cmap="gray", vmin=0, vmax=1)
            ax[row, 2 * c].set_title(f"{name}\namplitude" if row == 0 else "",
                                     fontsize=8)
            ax[row, 2 * c + 1].imshow(o[row, 1], cmap="twilight",
                                      vmin=0, vmax=cfg.phase_max)
            ax[row, 2 * c + 1].set_title("phase" if row == 0 else "", fontsize=8)
            ax[row, 2 * c].axis("off"); ax[row, 2 * c + 1].axis("off")
    plt.tight_layout(); plt.savefig(out / "object_samples.png", dpi=150); plt.close()
    print(f"-> {out/'object_samples.png'}")

    # ---------------- 判定 ----------------
    if "MNIST" in rows:
        b = rows["MNIST"]; p = rows["proc:mixed"]
        print("\n=== 判定 ===")
        ok = True
        for lab, cond, msg in [
            ("PCA 有效維度", p["dim"] > 1.5 * b["dim"],
             f"{p['dim']} vs MNIST {b['dim']}(需 > 1.5x)"),
            ("最近鄰距離", p["nn"] > 1.5 * b["nn"],
             f"{p['nn']:.4f} vs MNIST {b['nn']:.4f}(需 > 1.5x)"),
            ("對比度相近", 0.6 < p["amp"] / max(b["amp"], 1e-9) < 1.6,
             f"support 內平均振幅 {p['amp']:.3f} vs MNIST {b['amp']:.3f}"
             " —— 差太多的話絕對 PSNR 不可跨資料集比較,可用 contrast_gamma 調整"),
            ("非帶限(高頻未塌陷)", p["hf"] > 0.5 * b["hf"],
             f"{p['hf']:.4f} vs MNIST {b['hf']:.4f}(需 > 0.5x)"),
            ("過取樣條件", p["support"] < 0.5,
             f"support {p['support']:.4f}"),
            ("support 已對齊", abs(p["support"] - b["support"]) < 0.3 * b["support"],
             f"{p['support']:.4f} vs MNIST {b['support']:.4f}"
             + ("" if p["support"] <= b["support"] else
                "(MNIST 較密,無法向上對齊 — 需改用密度更高的原型參數)")),
        ]:
            print(f"  {'[通過]' if cond else '[未通過]'} {lab:<12} {msg}")
            ok &= bool(cond)
        print("\n" + ("四項全部通過,可以接進訓練流程。" if ok
                      else "有項目未通過,先調整產生器參數。"))


if __name__ == "__main__":
    main()
