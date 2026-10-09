#!/usr/bin/env python
"""第二次進度報告的圖表與數據(依 claude/圖表需求清單_第二次報告.md)。只讀結果檔、不改任何東西。

輸出到 ~/cdi/report_figs/:
  P2_ideal_bars.png            #1 理想條件三組 FRC gain(逐 seed 誤差棒)
  P3_realign_flip.png          #2 對齊前後並列(HIO 解出「翻轉版」的樣本)
  P3_realign_shift.png         #2 對齊前後並列(HIO 解出「平移版」的樣本)
  P3_ambiguity_demo.png        #3 平凡歧異示意(真實物體:原解 / 平移 / 翻轉,繞射強度相同)
  P4_support_frames.png        #4 三個框(+ r = 12 探針)疊在真實物體上
  P5_probe41_r12.png           #5 4-1 的 HIO 曲線(r = 12 單獨)
  P6_flip_fraction.png         #7 7 種探針的 K-HIO 選翻轉比例(100 次)
  P8_20k_DEF2.png              #9 2 萬個場:網路點 vs 最強迭代法(DEF2,b64 / b512)
  P9_confirm_DEF2.png          #10 新測試場確認(b64 / b512)
  P9_100k_DEF2.png             #11 10 萬個場(舊測試場)【可選】
  data_summary.md              全部數字(逐 seed)+ 與定案數字的核對
  orig/                        國網上現成的原圖(figs_realign、figs_probe_4_1、figs_probe42b、figs_scan5c*、figs_scan5d)

用法(計算節點,約 1–3 分鐘):  python collect_report_figs.py
"""
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5 as s5                                                  # noqa: E402
import scan_5b as s5b                                                # noqa: E402
import scan_5c as s5c                                                # noqa: E402

R = s5.RUN_ROOT
QUICK = s5c.QUICK
OUT = Path("report_figs")
SEEDS = [0, 1, 2]
AMP_MIN = 0.05
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"B": "#2a78d6", "C": "#eb6834", "A": "#9a9994", "D": "#1baf7a"}
Q = "_quick" if QUICK else ""
FILES = {
    "p41": R / f"probe_4_1{Q}.json",
    "p42b": R / f"probe42b_raw{Q}.json",
    "s5b": R / f"scan5b_raw{Q}.json",
    "s5c20": R / ("scan5c_raw_quick.json" if QUICK else "scan5c_raw_n20000.json"),
    "s5c100": R / ("scan5c_raw_quick.json" if QUICK else "scan5c_raw_n100000.json"),
    "s5d": R / f"scan5d_raw{Q}.json",
}
summary = []
status = {}


def log(s=""):
    print(s, flush=True)
    summary.append(s)


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "font.size": 10})
    return plt


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.8, which="both")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=160, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    log(f"  → {p}")


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    log(f"  {'✅' if ok else '⚠️'} {label}:{got:.4f}(定案數字 {want:.4f})")
    return ok


def masked_phase(amp, ph):
    return np.where(amp > AMP_MIN, ph, np.nan)


# ============================================================================
# #1 P2:理想條件三組長條圖
# ============================================================================
def fig_ideal():
    plt = plt_()
    groups = [("ideal_base", "baseline\n(beamstop 3, 1e3 photons)", 0.2099),
              ("ideal_bs0_ph1e5", "ideal\n(no beamstop, 1e5 photons)", 0.2166),
              ("ideal_bs0_ph1e5_oracle", "phase given (control)\n(same ideal conditions)", 0.2834)]
    vals = {}
    log("\n## #1 P2 理想條件三組(FRC gain,未對齊,metrics.json['test']['frc_gain'])")
    for name, _, want in groups:
        v = [json.load(open(R / f"{name}_s{s}" / "metrics.json"))["test"]["frc_gain"] for s in SEEDS]
        vals[name] = v
        log(f"  {name}:逐 seed {' / '.join(f'{x:.4f}' for x in v)};平均 {np.mean(v):.4f};標準差 {np.std(v, ddof=1):.4f}")
        check(f"{name} 平均", float(np.mean(v)), want, 0.0006)
    d1 = np.array(vals["ideal_bs0_ph1e5"]) - np.array(vals["ideal_base"])
    d2 = np.array(vals["ideal_bs0_ph1e5_oracle"]) - np.array(vals["ideal_bs0_ph1e5"])
    for lab, d in [("理想 − 基準", d1), ("給相位 − 理想", d2)]:
        z = d.mean() / (d.std(ddof=1) / np.sqrt(3))
        log(f"  {lab}(逐 seed 配對):{d.mean():+.4f}(z {z:.1f})")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style(ax)
    x = np.arange(3)
    means = [np.mean(vals[g]) for g, _, _ in groups]
    sds = [np.std(vals[g], ddof=1) for g, _, _ in groups]
    cols = ["#9a9994", "#2a78d6", "#c9c7c1"]
    ax.bar(x, means, yerr=sds, capsize=6, color=cols, edgecolor=INK2, width=0.6, error_kw={"ecolor": INK})
    for i, (g, _, _) in enumerate(groups):
        ax.scatter([i] * 3, vals[g], color=INK, s=14, zorder=3)
        ax.text(i, means[i] + sds[i] + 0.006, f"{means[i]:.4f}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in groups], fontsize=9)
    ax.set_ylabel("FRC gain (unaligned, mean ± sd, 3 seeds)")
    ax.set_ylim(0, max(means) * 1.25)
    ax.set_title("Ideal measurement barely helps the network: +0.0067 (z 0.9)", fontsize=10, color=INK)
    save(fig, "P2_ideal_bars.png")


# ============================================================================
# #2 / #3 / #4:重建圖(用 ideal_base_s0,與 realign_eval.py 同一套種子,可逐點重現)
# ============================================================================
def recon_data():
    from ambiguity_check import align, to_c, twin
    from realign_eval import load, net_out, proto_labels
    import probe_4_1 as p41
    from src.hio import hio, random_init
    from src.physics import beamstop_mask, forward_measure
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = 0
    cfg, model = load(R / f"ideal_base_s{s}", dev)
    bs = beamstop_mask(cfg, device=dev)
    home = p41.make_home(cfg, cfg.eval_n).to(dev)                        # = generalization_suites(...)["procedural"](不需 MNIST)
    torch.manual_seed(cfg.test_seed + s)                                   # = realign_eval / hio_ideal_conditions
    counts = forward_measure(home, bs, cfg)
    n_it = 5000 if not QUICK else 100
    with torch.no_grad():
        net = net_out(model, home, counts, bs, cfg)
        init = random_init(counts, cfg, seed=cfg.test_seed + s, device=dev)
        hio_out = hio(init, counts, bs, cfg, n_iter=n_it, beta=cfg.hio_beta)
        t = to_c(home)
        net_al, net_tw, net_sh = align(to_c(net), t)
        hio_al, hio_tw, hio_sh = align(to_c(hio_out), t)
    labels, kinds = proto_labels(cfg.eval_n, cfg, home)
    return dict(cfg=cfg, home=home.cpu(), t=t.cpu(), net=to_c(net).cpu(), hio=to_c(hio_out).cpu(),
                net_al=net_al.cpu(), hio_al=hio_al.cpu(), net_tw=net_tw.cpu(), net_sh=net_sh.cpu(),
                hio_tw=hio_tw.cpu(), hio_sh=hio_sh.cpu(), labels=labels.cpu(), kinds=kinds, n_it=n_it, twin=twin)


def nerr(a, t):
    return float(((a - t).abs() ** 2).sum() / (t.abs() ** 2).sum().clamp_min(1e-12))


def fig_realign(D):
    plt = plt_()
    t, kinds, labels = D["t"], D["kinds"], D["labels"]
    n = t.shape[0]
    e_h = np.array([nerr(D["hio_al"][i], t[i]) for i in range(n)])
    e_n = np.array([nerr(D["net_al"][i], t[i]) for i in range(n)])
    tw = D["hio_tw"].numpy().astype(bool)
    sh = D["hio_sh"].numpy()
    shifted = np.abs(sh).sum(1) > 0
    log(f"\n## #2 P3 對齊前後(ideal_base_s0、純 HIO {D['n_it']} 次;{n} 個測試樣本)")
    log(f"  純 HIO:選翻轉 {100 * tw.mean():.0f}%、有平移 {100 * shifted.mean():.0f}%(日誌 §14.4:49% / 97%)")
    log(f"  逐樣本誤差(對齊後,整張 64×64 的 nerr;描述):HIO 中位數 {np.median(e_h):.3f}、網路中位數 {np.median(e_n):.3f}")
    pri = [kinds.index(k) for k in ("polygon", "bandpass", "lattice") if k in kinds]

    def pick(mask):
        for k in pri:                                                      # 優先 polygon(最好看懂),其次 bandpass、lattice
            idx = np.where(mask & (labels.numpy() == k))[0]
            if len(idx):
                return int(idx[np.argmin(e_h[idx])]), kinds[k]
        idx = np.where(mask)[0]
        return (int(idx[np.argmin(e_h[idx])]), kinds[int(labels[idx[np.argmin(e_h[idx])]])]) if len(idx) else (None, None)

    picks = {"flip": pick(tw & (e_h < 0.3)), "shift": pick(~tw & shifted & (np.abs(sh).max(1) >= 2) & (e_h < 0.3))}
    sl = slice(14, 50)
    for tag, (i, kind) in picks.items():
        if i is None:
            log(f"  ⚠️ 找不到合適的「{tag}」樣本")
            continue
        cols = [("truth", t[i]), ("network\n(raw)", D["net"][i]), ("network\n(aligned)", D["net_al"][i]),
                (f"pure HIO {D['n_it']}\n(raw)", D["hio"][i]), (f"pure HIO {D['n_it']}\n(aligned)", D["hio_al"][i])]
        fig, axs = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 5.0))
        for c, (title, z) in enumerate(cols):
            z = z.numpy()[sl, sl]
            amp, ph = np.abs(z), np.angle(z)
            axs[0, c].imshow(amp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            cm = plt.get_cmap("viridis").copy()
            cm.set_bad("#d9d9d9")
            axs[1, c].imshow(masked_phase(amp, ph), cmap=cm, vmin=0, vmax=np.pi / 2, interpolation="nearest")
            axs[0, c].set_title(title, fontsize=9, color=INK)
            for r in range(2):
                axs[r, c].set_xticks([])
                axs[r, c].set_yticks([])
        axs[0, 0].set_ylabel("amplitude", fontsize=9)
        axs[1, 0].set_ylabel("phase (masked)", fontsize=9)
        how = ("HIO found the flipped (twin) version" if tag == "flip"
               else f"HIO found a shifted version ({int(sh[i, 0])}, {int(sh[i, 1])}) px")
        fig.suptitle(f"{kind} sample #{i}: {how}; aligned nerr: HIO {e_h[i]:.3f}, network {e_n[i]:.3f}", fontsize=10)
        fig.tight_layout()
        save(fig, f"P3_realign_{tag}.png")
        log(f"  {tag}:樣本 #{i}({kind});HIO 選翻轉 {bool(tw[i])}、平移 ({int(sh[i, 0])}, {int(sh[i, 1])});"
            f"對齊後 nerr:HIO {e_h[i]:.3f}、網路 {e_n[i]:.3f};網路本身選翻轉 {bool(D['net_tw'][i])}")
    log("  註:相位只畫振幅 > 0.05 處(灰 = 遮罩);網路輸出可能含與輸入無關的固定圖樣(§15.5),挑圖時請目視確認")
    return picks


def rep_polygon(D):
    """示意圖用的代表性 polygon 樣本:真實 support 面積最接近平均 138 px 者。"""
    lab = D["labels"].numpy()
    idx = np.where(lab == D["kinds"].index("polygon"))[0]
    area = np.array([int((D["t"][i].abs() > 0).sum()) for i in idx])
    return int(idx[np.argmin(np.abs(area - 138))])


def fig_ambiguity(D, picks):
    plt = plt_()
    i = rep_polygon(D)
    o = D["t"][i]
    pm = D["cfg"].phase_max
    sup = torch.zeros_like(o.real, dtype=torch.bool)
    c0 = (D["cfg"].canvas - D["cfg"].digit) // 2 - D["cfg"].support_pad
    c1 = c0 + D["cfg"].digit + 2 * D["cfg"].support_pad
    sup[c0:c1, c0:c1] = True
    shift = None
    for dy, dx in [(3, -2), (2, 2), (-2, 3), (2, -1), (1, 1), (-1, 1)]:
        cand = torch.roll(o, (dy, dx), dims=(0, 1))
        if bool((cand.abs()[~sup] == 0).all()):
            shift, osh = (dy, dx), cand
            break
    otw = D["twin"](o) * np.exp(1j * pm)                                   # 共軛翻轉 × 全域相位 φmax → 相位 φmax − φ(−r)
    vers = [("original", o), (f"shifted {shift}", osh) if shift else ("shifted", o), ("conjugate flip (twin)", otw)]
    F = [torch.fft.fft2(v, norm="ortho").abs() ** 2 for _, v in vers]
    d_sh = float((F[1] - F[0]).abs().max() / F[0].max())
    d_tw = float((F[2] - F[0]).abs().max() / F[0].max())
    in_box = [bool((v.abs()[~sup] == 0).all()) for _, v in vers]
    ph_ok = float(np.angle(otw.numpy())[otw.abs().numpy() > AMP_MIN].min()) >= -1e-5
    log(f"\n## #3 P3 平凡歧異示意(真實樣本 #{i})")
    log(f"  繞射強度最大相對差:平移 {d_sh:.1e}、翻轉 {d_tw:.1e}(= 數值誤差 → 強度完全相同)")
    log(f"  三者都在 32 × 32 方框內:{in_box};翻轉解相位仍在 [0, φmax]:{ph_ok}")
    fig, axs = plt.subplots(3, 3, figsize=(8.4, 8.4))
    sl = slice(12, 52)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#d9d9d9")
    for c, ((title, v), f) in enumerate(zip(vers, F)):
        z = v.numpy()[sl, sl]
        amp = np.abs(z)
        axs[0, c].imshow(amp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axs[0, c].add_patch(plt.Rectangle((c0 - 12 - 0.5, c0 - 12 - 0.5), c1 - c0, c1 - c0, fill=False, ec="#eb6834", lw=1))
        axs[1, c].imshow(masked_phase(amp, np.angle(z)), cmap=cm, vmin=0, vmax=pm, interpolation="nearest")
        axs[2, c].imshow(np.log10(np.fft.fftshift(f.numpy()) + 1e-6), cmap="magma", interpolation="nearest")
        axs[0, c].set_title(title, fontsize=10)
        for r in range(3):
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
    axs[0, 0].set_ylabel("amplitude (orange = 32×32 box)")
    axs[1, 0].set_ylabel("phase (masked)")
    axs[2, 0].set_ylabel("log diffraction intensity")
    fig.suptitle("Trivial ambiguities: same diffraction intensity, all inside the known box", fontsize=11)
    fig.tight_layout()
    save(fig, "P3_ambiguity_demo.png")


def fig_frames(D, picks):
    plt = plt_()
    cfg = D["cfg"]
    kinds, labels = D["kinds"], D["labels"].numpy()
    yy0, xx0 = np.mgrid[0:cfg.canvas, 0:cfg.canvas]

    def areas(j):
        tr = D["t"][j].abs().numpy() > 0
        if not tr.any():
            return None
        ys_, xs_ = np.nonzero(tr)
        r2 = ((ys_ - ys_.mean()) ** 2 + (xs_ - xs_.mean()) ** 2).max()
        return int(tr.sum()), int((((yy0 - ys_.mean()) ** 2 + (xx0 - xs_.mean()) ** 2) <= r2).sum())

    best, i = 1e9, 0                                                       # 代表性樣本:兩個面積都最接近平均(138、847)
    for j in range(D["t"].shape[0]):
        a_ = areas(j)
        if a_ is None:
            continue
        sc = abs(np.log(a_[0] / 138)) + abs(np.log(a_[1] / 847))
        if sc < best:
            best, i = sc, j
    amp = D["t"][i].abs().numpy()
    true = amp > 0
    ys, xs = np.nonzero(true)
    cy, cx = ys.mean(), xs.mean()
    rad = np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max())
    yy, xx = np.mgrid[0:cfg.canvas, 0:cfg.canvas]
    circ = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
    c0 = (cfg.canvas - cfg.digit) // 2 - cfg.support_pad
    c1 = c0 + cfg.digit + 2 * cfg.support_pad
    disk = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 12 ** 2
    log(f"\n## #4 P4 三個框(樣本 #{i},{kinds[labels[i]]})")
    log(f"  這個樣本:方框 {(c1 - c0) ** 2} px、包住物體的圓 {int(circ.sum())} px、真實 support {int(true.sum())} px、"
        f"r = 12 探針圓盤 {int(disk.sum())} px")
    log("  平均(§21.1,校準集 1024 張):方框 1024、包住物體的圓 847、真實 138、r = 12 圓盤 448")
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    sl = slice(8, 56)
    ax.imshow(amp[sl, sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    o = 8
    ax.add_patch(plt.Rectangle((c0 - o - 0.5, c0 - o - 0.5), c1 - c0, c1 - c0, fill=False, ec="#eb6834", lw=2,
                               label="box support: 1024 px"))
    ax.add_patch(plt.Circle((cx - o, cy - o), rad, fill=False, ec="#2a78d6", lw=2,
                            label="circle enclosing the object (large probe): ~847 px"))
    ax.contour(true[sl, sl].astype(float), levels=[0.5], colors="#1baf7a", linewidths=1.2)
    ax.plot([], [], color="#1baf7a", lw=1.5, label="true support: ~138 px")
    ax.add_patch(plt.Circle((31.5 - o, 31.5 - o), 12, fill=False, ec="#d6452a", lw=2, ls="--",
                            label="small probe r = 12: 448 px (lights only a part)"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8.5, ncol=1)
    ax.set_title("How tight is each 'where is the object' constraint?\n(areas = averages over 1024 objects, §21.1)",
                 fontsize=10)
    save(fig, "P4_support_frames.png")


# ============================================================================
# #5 P5:4-1 的 HIO 曲線(r = 12)
# ============================================================================
def fig_probe41():
    plt = plt_()
    d = json.load(open(FILES["p41"]))
    res = d["results"]["12"]
    iters = sorted(int(k) for k in res[0]["rec"]["A"].keys())
    name = {"A": "A: box support", "P": "P: probe footprint (known)", "S": "S*ψ: true support (upper bound)"}
    col = {"A": "#2a78d6", "P": "#eb6834", "S": "#1baf7a"}
    log("\n## #5 P5 4-1 曲線(r = 12,對齊後 FRC gain,目標 ψ;逐 seed 平均)")
    log("  迭代 | " + " | ".join(str(i) for i in iters))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style(ax)
    for m in ["A", "P", "S"]:
        v = np.array([[sd["rec"][m][str(it)]["aligned"]["frc_gain"] for sd in res] for it in iters])
        mu, sdv = v.mean(1), v.std(1, ddof=1)
        ax.plot(iters, mu, "--" if m == "S" else "-", color=col[m], lw=2, marker="o", ms=4, label=name[m])
        ax.fill_between(iters, mu - sdv, mu + sdv, color=col[m], alpha=0.15, lw=0)
        log(f"  {m}: " + " | ".join(f"{x:.3f}" for x in mu))
    for it, want in [(20, 0.377), (50, 0.594), (100, 0.646)]:
        if it in iters:
            got = np.mean([sd["rec"]["P"][str(it)]["aligned"]["frc_gain"] for sd in res])
            check(f"P {it} 次", float(got), want, 0.0015)
    ax.set_xscale("log")
    ax.set_xlabel("HIO iterations")
    ax.set_ylabel("aligned FRC gain on ψ (mean ± sd, 3 seeds)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("4-1 (flat-top probe r = 12 px): the known probe footprint acts as the support", fontsize=10)
    save(fig, "P5_probe41_r12.png")


# ============================================================================
# #7 P6:7 種探針的 K-HIO 選翻轉比例
# ============================================================================
def fig_flip():
    plt = plt_()
    d = json.load(open(FILES["p42b"]))
    res, probes = d["results"], d["probes"]
    want = {"DISK": 0.51, "AMP": 0.49, "DEF": 0.07, "DEF2": 0.02, "SPH": 0.02, "AST": 0.06, "COMA": 0.03}
    log("\n## #7 P6 K-HIO 100 次的選翻轉比例(物體空間判定,twin_obj;逐 seed)")
    m, sdv = [], []
    for p in probes:
        v = [sd["rec"]["K"]["100"]["twin_obj"] for sd in res[p]]
        m.append(np.mean(v))
        sdv.append(np.std(v, ddof=1))
        log(f"  {p}:{' / '.join(f'{x:.3f}' for x in v)} → 平均 {np.mean(v):.3f}")
        check(f"{p}", float(np.mean(v)), want[p], 0.006)
    real = {"DISK", "AMP"}
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    style(ax)
    x = np.arange(len(probes))
    cols = ["#9a9994" if p in real else "#eb6834" for p in probes]
    ax.bar(x, m, yerr=sdv, color=cols, edgecolor=INK2, capsize=4, width=0.62)
    ax.axhline(0.5, color=INK2, ls=":", lw=1)
    ax.text(len(probes) - 0.6, 0.51, "0.5 = cannot tell (coin flip)", ha="right", va="bottom", fontsize=8, color=INK2)
    for i, (v, e) in enumerate(zip(m, sdv)):
        ax.text(i, v + e + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(probes)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("fraction choosing the flipped solution\n(K-HIO, 100 iterations)")
    ax.bar([0], [0], color="#9a9994", label="real probe (no phase)")
    ax.bar([0], [0], color="#eb6834", label="complex probe (known phase)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("A known probe phase breaks the flip ambiguity (AMP = same |P| as DEF, no phase)", fontsize=10)
    save(fig, "P6_flip_fraction.png")


# ============================================================================
# #9 / #10 / #11:網路點 vs 最強迭代法(DEF2)
# ============================================================================
def env_plot(env, nets, tim, groups, title, fname, check_vals):
    plt = plt_()
    P = "DEF2"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, B in zip(axs, ["64", "512"]):
        style(ax)
        tt = sorted({s5b.cfg_time(c, it, tim[B], P) for c in s5b.configs() for it in s5b.stops(c)} - {0.0})
        ax.plot(tt, [s5b.envelope(env, P, T, tim[B])[2].mean() for T in tt], color=INK, lw=2,
                label="strongest iterative method at each time (envelope)")
        for g in groups:
            a = s5c.nvals(nets, P, g)
            if a is None:
                continue
            tn = tim[B][f"net:{g}:{P}"]
            e = s5b.envelope(env, P, tn, tim[B])
            r = s5b.ratio(a, e[2])
            ax.plot([tn], [a.mean()], "o", color=COL[g], ms=8, zorder=4,
                    label=f"net {g}: {a.mean():.3f} at {tn:.3f} ms (ratio to envelope {r[1]:.3f})")
            log(f"  batch {B} {g}:網路 {a.mean():.4f}({' / '.join(f'{x:.4f}' for x in a)});{tn:.4f} ms;"
                f"對手 {s5b.fmt_conf(e[0])} {e[1]} 次 {e[2].mean():.4f} → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}")
            if (B, g) in check_vals:
                check(f"batch {B} {g} 比值", float(r[1]), check_vals[(B, g)], 0.0015)
            if g in ("B", "C"):
                ax.plot([tn, tn], [a.mean(), e[2].mean()], color=COL[g], lw=0.8, ls=":")
                ax.plot([tn], [e[2].mean()], "x", color=COL[g], ms=7)
            if True:                                                       # 網路當起點 + 最佳迭代法的下緣
                pts = []
                for m in s5b.METHODS:
                    for it in s5b.ITERS:
                        try:
                            pts.append((tn + it * tim[B][m], s5c.hvals(nets, P, g, m, it).mean()))
                        except (KeyError, TypeError):
                            pass
                if pts:
                    pts.sort()
                    xs, ys, cur = [], [], float("inf")
                    for t_, y in pts:
                        cur = min(cur, y)
                        xs.append(t_)
                        ys.append(cur)
                    ax.plot(xs, ys, color=COL[g], lw=1, ls="--", alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("time per sample (ms)")
        ax.set_title(f"DEF2, batch {B}", fontsize=10)
    axs[0].set_ylabel("nerr_ph on R0 (global phase only)")
    for ax in axs:
        ax.plot([], [], color=INK2, ls="--", lw=1, label="net as starter + best iterative")
        ax.plot([], [], "x", color=INK2, label="envelope at the net's own time")
    for ax in axs:
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, fname)


def fig_20k():
    d = json.load(open(FILES["s5c20"]))
    env = json.load(open(FILES["s5b"]))["results"]
    log("\n## #9 P8 2 萬個場(舊測試場;scan5c_raw_n20000.json + scan5b_raw.json;timing 為該評估 job 重量)")
    b = s5c.nvals(d["results"], "DEF2", "B")
    if not QUICK:
        check("DEF2 B nerr_ph(確認是 2 萬的版本)", float(b.mean()), 0.490, 0.0015)
    env_plot(env, d["results"], d["timing"], ["A", "B", "C", "D"],
             "20k training fields: network (dot) vs strongest iterative method (line)", "P8_20k_DEF2.png",
             {} if QUICK else {("64", "B"): 0.770, ("64", "D"): 0.756})


def fig_100k():
    d = json.load(open(FILES["s5c100"]))
    env = json.load(open(FILES["s5b"]))["results"]
    log("\n## #11 P9 10 萬個場(舊測試場;scan5c_raw_n100000.json)")
    b = s5c.nvals(d["results"], "DEF2", "B")
    if not QUICK:
        check("DEF2 B nerr_ph(10 萬)", float(b.mean()), 0.248, 0.0015)
    env_plot(env, d["results"], d["timing"], ["B", "C"],
             "100k training fields (old test set): network vs strongest iterative method", "P9_100k_DEF2.png",
             {} if QUICK else {("64", "B"): 0.390, ("512", "B"): 0.700, ("64", "C"): 0.458})


def fig_confirm():
    d = json.load(open(FILES["s5d"]))
    log("\n## #10 P9 新測試場確認(seed 90,000,000;scan5d_raw.json)")
    b = s5c.nvals(d["nets"], "DEF2", "B")
    if not QUICK:
        check("DEF2 B nerr_ph(新測試場)", float(b.mean()), 0.249, 0.0015)
    env_plot(d["envelope"], d["nets"], d["timing"], ["B", "C"],
             "Confirmation on 512 never-used test fields: network vs strongest iterative method",
             "P9_confirm_DEF2.png",
             {} if QUICK else {("64", "B"): 0.396, ("512", "B"): 0.704, ("64", "C"): 0.458})


# ============================================================================
def copy_orig():
    log("\n## 國網上的原圖(複製到 report_figs/orig/,僅供參考)")
    for name in ["figs_realign", "figs_probe_4_1", "figs_probe42b", "figs_scan5c", "figs_scan5c_n20000", "figs_scan5d"]:
        src = Path(name)
        if src.is_dir():
            shutil.copytree(src, OUT / "orig" / name, dirs_exist_ok=True)
            log(f"  ✅ {name}/:{', '.join(sorted(p.name for p in src.iterdir()))}")
        else:
            log(f"  — {name}/ 不存在")


def run(tag, fn, *a):
    try:
        r = fn(*a)
        status[tag] = "✅"
        return r
    except Exception as e:                                                 # noqa: BLE001  (一張失敗不影響其他張)
        status[tag] = f"❌ {type(e).__name__}: {e}"
        log(f"  ❌ {tag} 失敗:{type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def main():
    OUT.mkdir(exist_ok=True)
    log("# 第二次報告:圖表與數據(collect_report_figs.py 產生)")
    log(f"結果目錄:{R}")
    run("#1 P2 理想條件長條圖", fig_ideal)
    D = run("#2–#4 重建資料", recon_data)
    if D is not None:
        picks = run("#2 P3 對齊前後", fig_realign, D) or {}
        run("#3 P3 平凡歧異示意", fig_ambiguity, D, picks)
        run("#4 P4 三個框", fig_frames, D, picks)
    run("#5 P5 4-1 曲線", fig_probe41)
    run("#7 P6 選翻轉比例", fig_flip)
    run("#9 P8 2 萬個場", fig_20k)
    run("#10 P9 新測試場", fig_confirm)
    run("#11 P9 10 萬個場", fig_100k)
    run("原圖複製", copy_orig)
    log("\n## 狀態")
    for k, v in status.items():
        log(f"  {v}  {k}")
    (OUT / "data_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\n完成:{OUT}/(把整個資料夾下載回來)")


if __name__ == "__main__":
    main()
