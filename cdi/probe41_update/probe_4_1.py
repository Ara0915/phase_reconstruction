#!/usr/bin/env python
"""階段四 4-1:小探針能否提供 support?(實驗設計 §二十五,HIO,不需訓練)

背景(§二十四):給 HIO 真實 support,10–20 次即解出;但本專案物體細而分散,
大探針或放寬的 support 幾乎不提供範圍資訊(§24.7)。4D-STEM / ptychography 用
比物體小的聚焦探針,每張繞射圖只照亮一小塊,該塊的範圍 = 已知的探針範圍。

問:知道探針範圍,HIO 會不會像 S* 一樣快?

設計:
  探針   平頂圓盤(實數、內部 1、外部 0),圓心 (31.5, 31.5),半徑 6 / 8 / 12 px
  目標   出射波 psi = P * O(被照亮的那一塊);三種方法比同一個目標
  劑量   每張繞射圖的總光子數與平面波相同(ref_energy 以校準集的能量比縮放)
  A      HIO,support = 32x32 方框(不使用探針資訊)
  P      HIO,support = 探針圓盤(實驗上已知)
  S*psi  HIO,support = psi 的真實範圍(上限)

用法(需在計算節點執行;需先跑過 hio_budget.py,並已安裝 src/hio_sw.py):
    python probe_4_1.py --check   # 檔案 + 單元測試(約 1 分鐘)
    python probe_4_1.py           # 完整量測(約 10–20 分鐘)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Cfg                                          # noqa: E402

RUN_ROOT = Path(os.environ.get("CDI_RUN_ROOT", "/work/elviss0915/runs"))
OUT_JSON = RUN_ROOT / "probe_4_1.json"
BUDGET_JSON = RUN_ROOT / "hio_budget.json"
FIG_DIR = Path("figs_probe_4_1")
BASE = "ideal_base"
SEEDS = [0, 1, 2]
RADII = [6, 8, 12]
METHODS = ["A", "P", "S"]
M_LABEL = {"A": "A 方框", "P": "P 探針範圍", "S": "S*ψ 真實範圍"}
ITERS = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
JUDGE_N = [20, 50, 100]
STRONG_N = 50
PROTO_N = 50
CALIB_SEED, CALIB_N = 2026, 1024
REPRO_TOL = 0.005
N_OBJ = None                                    # None = cfg.eval_n(512)

QUICK = os.environ.get("CDI_QUICK") == "1"      # 只供本地測程式流程
if QUICK:
    RADII, ITERS, JUDGE_N, STRONG_N, PROTO_N = [6, 12], [5, 20, 50], [20, 50], 50, 50
    CALIB_N, N_OBJ = 128, 32
    BUDGET_JSON = RUN_ROOT / "hio_budget_quick.json"

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def make_home(cfg, n, seed=None):
    """seed=None 時與 generalization_suites(cfg, n)["procedural"] 完全相同。"""
    from src.procedural import make_procedural
    return make_procedural(n, cfg, seed=cfg.test_seed if seed is None else seed,
                           kinds=tuple(cfg.proc_kinds), weights=tuple(cfg.proc_weights),
                           target_support=cfg.match_support_to,
                           contrast_gamma=cfg.proc_contrast_gamma)


def disk(r, cfg, device=None):
    """平頂圓盤探針 [H, W]:圓心在畫布中心 ((N-1)/2),與方框 support 同心。"""
    n = cfg.canvas
    ax = torch.arange(n, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    c = (n - 1) / 2.0
    d = ((yy - c) ** 2 + (xx - c) ** 2 <= r * r).float()
    return d.to(device) if device is not None else d


def to_psi(objs, D):
    """出射波 psi = P * O(P 為實數平頂,故圓盤內的振幅與相位即物體本身)。"""
    return torch.stack([objs[:, 0] * D, objs[:, 1] * D], dim=1)


_RATIO = {}


def energy_ratio(cfg, r):
    """校準集(≠ 測試集)上 psi 與物體的平均能量比 —— 用來縮放 ref_energy。"""
    from src.physics import calibrate_flux
    if r not in _RATIO:
        cal = make_home(cfg, CALIB_N, seed=CALIB_SEED)
        _RATIO[r] = calibrate_flux(to_psi(cal, disk(r, cfg))) / calibrate_flux(cal)
    return _RATIO[r]


def psi_cfg(cfg, r):
    """同一份設定,只把 ref_energy 乘上能量比:每張繞射圖的總光子數與平面波相同。"""
    c = Cfg.from_dict(cfg.to_dict())
    c.ref_energy = cfg.ref_energy * energy_ratio(cfg, r)
    return c


def run_method(key, cfg_p, init, counts, bs, n, D, psi):
    from src.hio_sw import hio_ext
    if key == "A":
        sup = None
    elif key == "P":
        sup = D.expand(psi.shape[0], *D.shape)
    elif key == "S":
        sup = (psi[:, 0] > 0).float()
    else:
        raise ValueError(key)
    return hio_ext(init, counts, bs, cfg_p, n, beta=cfg_p.hio_beta, support=sup)


# ============================================================================
# 檔案與單元測試
# ============================================================================
def check_files():
    print("=" * 70)
    print("檔案")
    print("=" * 70)
    need = [BUDGET_JSON, Path("src/hio_sw.py"), Path("realign_eval.py"), Path("ambiguity_check.py")]
    for s in SEEDS:
        d = RUN_ROOT / f"{BASE}_s{s}"
        need += [d / "config_used.json", d / "final.pt"]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing,
          "缺:" + ", ".join(missing) if missing else "")
    return not missing


@torch.no_grad()
def unit_tests():
    from realign_eval import load
    from src.hio import hio, random_init
    from src.hio_sw import hio_ext
    from src.physics import beamstop_mask, forward_measure, support_mask

    print("\n" + "=" * 70)
    print("單元測試")
    print("=" * 70)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, _ = load(RUN_ROOT / f"{BASE}_s0", dev)
    box = support_mask(cfg, device=dev)
    bs = beamstop_mask(cfg, device=dev)

    inside = all(bool(((disk(r, cfg, dev) > 0) & (box <= 0)).sum() == 0) for r in RADII)
    check("探針圓盤皆在方框內", inside,
          "面積 " + " / ".join(f"r{r}: {int(disk(r, cfg).sum())} px" for r in RADII))

    home = make_home(cfg, 64).to(dev)
    r = RADII[len(RADII) // 2]
    D = disk(r, cfg, dev)
    psi = to_psi(home, D)
    cfg_p = psi_cfg(cfg, r)

    # (1) support=None 時 = hio()
    torch.manual_seed(0)
    counts = forward_measure(psi, bs, cfg_p)
    init = random_init(counts, cfg_p, seed=1, device=dev)
    a = hio_ext(init, counts, bs, cfg_p, 50, beta=cfg_p.hio_beta)
    b = hio(init, counts, bs, cfg_p, n_iter=50, beta=cfg_p.hio_beta)
    e1 = float((torch.polar(a[:, 0], a[:, 1]) - torch.polar(b[:, 0], b[:, 1])).abs().max())
    check("hio_ext(support=None)= hio()", e1 < 1e-4, f"最大差 {e1:.1e}")

    # (2) 探針 support 下 psi 真值為固定點(無雜訊)
    #     振幅與相位都移到約束邊界內:約 0.08% 的物體像素振幅恰為 1、相位恰為 phase_max,
    #     float32 誤差會讓 hio.py 把它們判為違反(§21.1 的註記;hio.py 維持原行為不修改)。
    #     相位也離開 0:細線上的像素振幅與相位同樣很小,雜訊會讓相位跨過 0。
    t = psi.clone()
    on = t[:, 0] > 0
    t[:, 0] = t[:, 0] * 0.95
    t[:, 1] = torch.where(on, psi.new_tensor(cfg.phase_max) * 0.05 + 0.9 * t[:, 1], t[:, 1])
    c0 = Cfg.from_dict(cfg_p.to_dict())
    c0.add_poisson = False
    cnt0 = forward_measure(t, bs, c0)
    out = hio_ext(t, cnt0, bs, c0, 100, beta=c0.hio_beta, support=D.expand(len(t), *D.shape))
    e2 = float((torch.polar(out[:, 0], out[:, 1]) - torch.polar(t[:, 0], t[:, 1])).abs().max())
    check(f"探針 support(r={r})下 psi 真值為固定點", e2 < 1e-3, f"最大差 {e2:.1e}")

    # (3) 劑量:psi 的平均總光子數(含被 beamstop 擋掉的)≈ 平面波
    #     設計是「同一道光集中到探針上」,比的是打到偵測器平面的全部光子,故不套 beamstop。
    #     用完整測試集(512 張):單張比值的標準差約 0.15,64 張的平均值會差到數 %。
    #     另列「偵測到的」(套 beamstop)作參考:psi 較集中於低頻,被擋掉的比例與平面波不同。
    full = make_home(cfg, N_OBJ or cfg.eval_n).to(dev)
    bs0 = beamstop_mask(cfg, radius=-1, device=dev)
    ratios, seen = [], []
    for rr in RADII:
        cp = psi_cfg(cfg, rr)
        ps = to_psi(full, disk(rr, cfg, dev))
        for mask, out in [(bs0, ratios), (bs, seen)]:
            torch.manual_seed(0)
            cps = forward_measure(ps, mask, cp)
            torch.manual_seed(0)
            cpl = forward_measure(full, mask, cfg)
            out.append(float(cps.sum((1, 2)).mean() / cpl.sum((1, 2)).mean()))
    check(f"psi 的平均總光子數 / 平面波 在 0.95–1.05(不含 beamstop,{len(full)} 張)",
          all(0.95 <= x <= 1.05 for x in ratios),
          " / ".join(f"r{rr}: {x:.3f}" for rr, x in zip(RADII, ratios)))
    print("     (參考)偵測到的光子數比(套 beamstop):"
          + " / ".join(f"r{rr}: {x:.3f}" for rr, x in zip(RADII, seen)))
    print(f"\n     裝置:{dev}")


# ============================================================================
# 量測
# ============================================================================
@torch.no_grad()
def plane_wave_repro(dev):
    """平面波管線重現 hio_budget.json(seed 0:A 50 次、S* 20 次)。"""
    from realign_eval import load, score_all
    from src.hio import hio, random_init
    from src.hio_sw import hio_ext, true_support
    from src.physics import beamstop_mask, forward_measure
    s = SEEDS[0]
    cfg, model = load(RUN_ROOT / f"{BASE}_s{s}", dev)
    n = N_OBJ or cfg.eval_n
    bs = beamstop_mask(cfg, device=dev)
    home = make_home(cfg, n).to(dev)
    torch.manual_seed(cfg.test_seed + s)
    counts = forward_measure(home, bs, cfg)
    init = random_init(counts, cfg, seed=cfg.test_seed + s, device=dev)
    ref = json.load(open(BUDGET_JSON))["results"]["seeds"][0]["rec"]
    got = {("A", 50): score_all(model, home, bs, cfg,
                                hio(init, counts, bs, cfg, n_iter=50, beta=cfg.hio_beta))[0],
           ("S", 20): score_all(model, home, bs, cfg,
                                hio_ext(init, counts, bs, cfg, 20, beta=cfg.hio_beta,
                                        support=true_support(home)))[0]}
    for (k, it), r in got.items():
        a, b = r["aligned"]["frc_gain"], ref[k][str(it)]["aligned"]["frc_gain"]
        if abs(a - b) > REPRO_TOL:
            raise SystemExit(f"平面波 {k} {it} 次:{a:+.4f} 與 hio_budget.json {b:+.4f} 對不上,停下來查")
    check("平面波管線重現 hio_budget.json(A 50 次、S* 20 次)", True)


@torch.no_grad()
def measure(dev):
    from ambiguity_check import align, to_c
    from realign_eval import load, proto_labels, score, score_all
    from src.hio import random_init
    from src.physics import beamstop_mask, forward_measure

    res = {str(r): [] for r in RADII}
    geo = {}
    kinds = None
    for s in SEEDS:
        cfg, model = load(RUN_ROOT / f"{BASE}_s{s}", dev)
        n = N_OBJ or cfg.eval_n
        bs = beamstop_mask(cfg, device=dev)
        home = make_home(cfg, n).to(dev)
        labels, kinds = proto_labels(n, cfg, home)
        for ri, r in enumerate(RADII):
            D = disk(r, cfg, dev)
            psi = to_psi(home, D)
            cfg_p = psi_cfg(cfg, r)
            torch.manual_seed(cfg.test_seed + s + 1000 * (ri + 1))
            counts = forward_measure(psi, bs, cfg_p)
            init = random_init(counts, cfg_p, seed=cfg.test_seed + s, device=dev)
            if s == SEEDS[0]:
                ill = (psi[:, 0] > 0).float().sum((1, 2))
                tot = (home[:, 0] > 0).float().sum((1, 2))
                geo[str(r)] = {"disk_px": int(D.sum()), "illum_px": float(ill.mean()),
                               "illum_frac": float((ill / tot.clamp_min(1))[tot > 0].mean()),
                               "empty": int((ill == 0).sum())}
            rec = {m: {} for m in METHODS}
            for key in METHODS:
                for it in ITERS:
                    pred = run_method(key, cfg_p, init, counts, bs, it, D, psi)
                    sc, al = score_all(model, psi, bs, cfg_p, pred)
                    _, _, sh = align(to_c(pred), to_c(psi))
                    sc["shift_frac"] = float((sh.abs().sum(1) > 0).float().mean())
                    if it == PROTO_N:
                        sc["proto"] = {}
                        for i, k in enumerate(kinds):
                            m = labels.to(dev) == i
                            p = score(model, psi[m], bs, cfg_p, al[m])
                            sc["proto"][k] = {"frc_gain": p["frc_gain"], "material_mae": p["material_mae"]}
                    rec[key][str(it)] = sc
            res[str(r)].append({"run": f"{BASE}_s{s}", "rec": rec,
                                "ref_energy": cfg_p.ref_energy})
            print(f"  [{BASE}_s{s}] r={r} 完成", flush=True)
    return res, geo, kinds


# ============================================================================
# 彙整與判讀
# ============================================================================
def vals(res, r, key, it, field="frc_gain", kind="aligned"):
    return np.array([sd["rec"][key][str(it)][kind][field] for sd in res[str(r)]], float)


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def zpair(d):
    m, s = ms(d)
    return m, s, (m / (s / math.sqrt(len(d))) if s > 0 else float("inf") * np.sign(m))


def first_reach(res, r, key, q):
    for it in ITERS:
        if vals(res, r, key, it).mean() >= q:
            return it
    return None


def report(res, geo, kinds):
    verdict = {}
    for r in RADII:
        g = geo[str(r)]
        print("\n" + "=" * 96)
        print(f"探針半徑 r = {r} px(圓盤 {g['disk_px']} px;被照亮的物體像素平均 {g['illum_px']:.1f},"
              f"佔物體 {100 * g['illum_frac']:.0f}%;全暗樣本 {g['empty']} 張)")
        print("對齊後 FRC gain(mean ± std,3 seeds);括號為材料 MAE")
        print("=" * 96)
        print(f"  {'迭代':>6}" + "".join(f"{M_LABEL[m]:>26}" for m in METHODS))
        for it in ITERS:
            row = f"  {it:>6}"
            for m in METHODS:
                a = ms(vals(res, r, m, it))
                mat = vals(res, r, m, it, "material_mae").mean()
                row += f"{a[0]:>+13.4f}±{a[1]:.4f}({mat:.3f})"
            print(row)
        print(f"  機率水準(A {ITERS[-1]} 次的錯配對齊):"
              f"{vals(res, r, 'A', ITERS[-1], kind='mismatch').mean():+.4f}")
        print("  首次達標:" + "   ".join(
            f"{m} 0.35:{first_reach(res, r, m, 0.35)} 0.60:{first_reach(res, r, m, 0.60)}"
            for m in METHODS))
        for it in [STRONG_N, ITERS[-1]]:
            print(f"  {it} 次 選翻轉 / 有平移:" + "   ".join(
                f"{m} {np.mean([sd['rec'][m][str(it)]['twin_frac'] for sd in res[str(r)]]):.2f} / "
                f"{np.mean([sd['rec'][m][str(it)]['shift_frac'] for sd in res[str(r)]]):.2f}"
                for m in METHODS))
        print(f"  各原型({PROTO_N} 次,對齊後 FRC gain):" + "   ".join(
            f"{k}: " + " / ".join(f"{m} {np.mean([sd['rec'][m][str(PROTO_N)]['proto'][k]['frc_gain'] for sd in res[str(r)]]):+.3f}"
                                  for m in METHODS) for k in kinds))

        print("  主判準 P − A(逐 seed 配對):")
        big, small = False, True
        for it in JUDGE_N:
            d = zpair(vals(res, r, "P", it) - vals(res, r, "A", it))
            print(f"    {it:>4} 次:{d[0]:+.4f} ± {d[1]:.4f}(z {d[2]:+.1f})")
            big |= d[0] >= 0.10 and d[2] > 2
            small &= abs(d[0]) <= 0.02
        v = "有用" if big else ("無用" if small else "部分")
        fp, fs = vals(res, r, "P", STRONG_N).mean(), vals(res, r, "S", STRONG_N).mean()
        strong = fp >= fs - 0.05
        print(f"  強判準({STRONG_N} 次):P {fp:+.4f} vs S*ψ {fs:+.4f} − 0.05 → {'成立' if strong else '不成立'}")
        print(f"  判定:r = {r} 探針範圍{v}")
        verdict[str(r)] = {"main": v, "strong": bool(strong), "P50": float(fp)}

    useful = [r for r in RADII if verdict[str(r)]["main"] == "有用"]
    print("\n" + "=" * 96)
    if useful:
        best = max(verdict[str(r)]["P50"] for r in useful)
        pick = max(r for r in useful if verdict[str(r)]["P50"] >= best - 0.05)
        print(f"結論:探針範圍有用的半徑 {useful} → **重想階段結案**;4-2 探針半徑取 r = {pick} px"
              f"(有用者中 P({STRONG_N}) 與最佳差 ≤ 0.05 的最大 r)")
    else:
        pick = None
        print("結論:沒有任何半徑「有用」→ 重想階段**不結案**;重啟備案(網路估 support)並檢討階段五的依據")
    print("判讀準則見 實驗設計_1b6 §二十五(結果出來前已寫定)")
    verdict["useful"], verdict["pick"] = useful, pick
    return verdict


def make_figures(res, geo, dev):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ 無 matplotlib,略過圖")
        return
    from realign_eval import load
    FIG_DIR.mkdir(exist_ok=True)
    col = {"A": "#2a78d6", "P": "#eb6834", "S": "#1baf7a"}
    sty = {"A": "-", "P": "-", "S": "--"}
    name = {"A": "A: box support", "P": "P: probe footprint (known)", "S": "S*ψ: true support (diagnostic)"}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"

    fig, axs = plt.subplots(1, len(RADII), figsize=(4.4 * len(RADII), 4.2), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, r in zip(axs, RADII):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=grid, lw=0.8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=ink2)
        for m in METHODS:
            mu = np.array([vals(res, r, m, it).mean() for it in ITERS])
            sd = np.array([vals(res, r, m, it).std(ddof=1) for it in ITERS])
            ax.plot(ITERS, mu, sty[m], color=col[m], lw=2, marker="o", ms=5, label=name[m])
            ax.fill_between(ITERS, mu - sd, mu + sd, color=col[m], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("iterations", color=ink)
        g = geo[str(r)]
        ax.set_title(f"probe r = {r} px ({g['disk_px']} px, {100 * g['illum_frac']:.0f}% of object lit)",
                     color=ink, fontsize=10)
    axs[0].set_ylabel("aligned FRC gain on ψ (mean ± std, 3 seeds)", color=ink)
    axs[0].legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    fig.suptitle("4-1: does a known small probe give HIO the support it needs? (target = illuminated patch ψ)",
                 color=ink, fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "probe_4_1_curves.png"
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")

    # 真值:物體與各半徑下被照亮的部分(不含任何重建)
    cfg, _ = load(RUN_ROOT / f"{BASE}_s0", torch.device("cpu"))
    home = make_home(cfg, N_OBJ or cfg.eval_n)
    idx = [0, 1, 2, 3]
    sl = slice(12, 52)
    fig, axs = plt.subplots(len(idx), 1 + len(RADII), figsize=(2.2 * (1 + len(RADII)), 2.2 * len(idx)))
    for i, j in enumerate(idx):
        imgs = [home[j, 0]] + [to_psi(home[j:j + 1], disk(r, cfg))[0, 0] for r in RADII]
        titles = ["object"] + [f"ψ, r = {r}" for r in RADII]
        for c, (im, tt) in enumerate(zip(imgs, titles)):
            a = axs[i, c]
            a.imshow(im[sl, sl].numpy(), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(tt, fontsize=9, color=ink)
    fig.suptitle("Ground truth only: object amplitude and the part lit by each probe", fontsize=10, color=ink)
    fig.tight_layout()
    p = FIG_DIR / "probe_4_1_truth.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑檔案檢查與單元測試")
    a = ap.parse_args()
    if not check_files():
        print("\n❌ 缺檔案,停下來")
        sys.exit(1)
    unit_tests()
    if a.check or not ok_all:
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過,先不要跑完整量測"))
        sys.exit(0 if ok_all else 1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plane_wave_repro(dev)
    print("\n量測")
    res, geo, kinds = measure(dev)
    verdict = report(res, geo, kinds)
    json.dump({"results": res, "geometry": geo, "verdict": verdict, "radii": RADII,
               "iters": ITERS, "quick": QUICK},
              open(OUT_JSON if not QUICK else RUN_ROOT / "probe_4_1_quick.json", "w"))
    make_figures(res, geo, dev)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))


if __name__ == "__main__":
    main()
