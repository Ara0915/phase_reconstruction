#!/usr/bin/env python
"""第三次報告的圖(第二批):需要迭代法包絡檔的 P3、P4、P5、R5。只讀結果檔(驗證資料),本機執行。
包絡檔(國網 /work/elviss0915/runs/)放在 cdi/claude_stage7_figs/:scan8_env_L-paper.json、scan8_env_L-combo.json、scan7a_cond_combo.json。
時間的算法同 scan_7a.cfg_time7、包絡同 envelope7、達到時間與下界同 scan_7b.t_match_conf / t_match_lo、500 次內未達的下界同 8c 的 lower_bound_time。
用法(repo 的 cdi/ 底下):python claude_final/report_figs/make_report_figs2.py"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter            # noqa: E402

HERE = Path(__file__).resolve().parent
CDI = HERE.parents[1]
D7 = CDI / "claude_stage7_figs"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"], "font.size": 15, "axes.unicode_minus": False, "savefig.dpi": 160})
COL = {"P6B4": "#8c8c8c", "P6B4r": "#e8a33d", "P6B4e8": "#2a78d6", "P6B4e8L": "#0b3d91", "P6B4e8C": "#8fbbe8", "iter": "#222222"}
NAME = {"P6B4": "P6B4", "P6B4r": "P6B4r", "P6B4e8": "P6B4e8", "P6B4e8L": "P6B4e8-L", "P6B4e8C": "P6B4e8-C"}

# ---- scan_7a 的設定、停止點與時間(同一套公式)----
METHODS = ["ePIE-C", "ePIE", "AP-C", "AP"] + [f"{m}:{x}" for m in ("ePIE-C", "AP-C") for x in ("pos", "posA", "prb", "both", "bothA")]
INITS = ["rand", "one", "zero", "const", "khio20", "khio100"]
KHIO_N = {"khio20": 20, "khio100": 100}
START, ANNEAL = 5, 20


def configs():
    return [f"{m}|{i}" for m in METHODS for i in INITS] + ["K-HIO"]


def stops(conf, iters):
    return list(iters) if conf == "K-HIO" else [0] + list(iters)


def cfg_time(conf, it, tB):
    tK, tM = tB["K-HIO49"], tB["merge49"]
    if conf == "K-HIO":
        return it * tK + tM
    meth, kind = conf.split("|")
    t0 = KHIO_N[kind] * tK + tM if kind in KHIO_N else 0.0
    if ":" not in meth:
        return t0 + it * tB[meth]
    t = t0 + min(it, START) * tB[f"{meth}|off"]
    if meth.endswith("A"):
        na = min(max(0, it - START), ANNEAL)
        return t + na * tB[f"{meth}|anneal"] + max(0, it - START - ANNEAL) * tB[f"{meth[:-1]}|on"]
    return t + max(0, it - START) * tB[f"{meth}|on"]


def sv_f1(C):
    C = np.asarray(C, float).reshape(3, 3)
    tp = C[1, 1]
    fp, fn = C[:, 1].sum() - tp, C[1].sum() - tp
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def lat_vals(env, fc):
    """晶格包絡檔 → 每個(設定, 次數)的 nerr_c(3 seeds 平均;失敗 = fc)與 SV-F1(各 seed 的 F1 平均;同 scan_8.finalize_rec)。"""
    out = {}
    for c, cr in env.items():
        for it, rec in cr.items():
            nc = [(cs + nf * fc) / max(nn, 1) for cs, nf, nn in zip(rec["c_sum"], rec["c_fail"], rec["c_n"])]
            sv = [sv_f1(cf) for cf in rec["conf"]]
            out[(c, int(it))] = (float(np.mean(nc)), float(np.mean(sv)))
    return out


def curve(vals, iters, tB, k, better="low"):
    """包絡曲線:各時間點以前的最佳值(同 envelope7)。"""
    pts = sorted((cfg_time(c, it, tB), vals[(c, it)][k]) for c in configs() for it in stops(c, iters))
    ts, ys, best = [], [], (np.inf if better == "low" else -np.inf)
    for t, v in pts:
        best = min(best, v) if better == "low" else max(best, v)
        if t > 0:
            ts.append(t)
            ys.append(best)
    return ts, ys


def reach(vals, iters, tB, Q, k=0):
    """最早達到 Q 的時間(上界)與前一個停止點的時間(下界);500 次內未達 → (inf, 最後停止點時間的最小值)。"""
    best, lo = np.inf, np.inf
    for c in configs():
        prev = None
        for it in stops(c, iters):
            if vals[(c, it)][k] <= Q:
                best = min(best, cfg_time(c, it, tB))
                lo = min(lo, cfg_time(c, prev if prev is not None else (it if it == 0 else 0), tB))
                break
            prev = it
    if not np.isfinite(best):
        lo = min(cfg_time(c, stops(c, iters)[-1], tB) for c in configs())
    return best, lo


def finish(fig, name):
    for ax in fig.axes:
        for axis, sc in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if sc == "log":
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(NullFormatter())
    fig.text(0.995, 0.01, "驗證資料", ha="right", va="bottom", fontsize=12, color="#666666")
    fig.savefig(HERE / name, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  圖:", name)


J8 = json.load(open(CDI / "claude_stage8" / "results" / "scan8.json"))
J8c = json.load(open(CDI / "claude_stage8" / "results_8c" / "scan8c.json"))
J7 = json.load(open(D7 / "scan7d.json"))
T8 = J8["timing"]
fc = J8["verdict"]["fail_c"]
crlb = J8c["report"]["crlb"]["L-paper"]["crlb_c"]
EX = {e: json.load(open(D7 / f"scan8_env_{e}.json")) for e in ("L-paper", "L-combo")}
V = {e: lat_vals(EX[e]["env"], fc[e]) for e in EX}
IT8 = EX["L-paper"]["meta"]["iters"]
netc = {nm: float(np.mean(J8["verdict"]["exam"]["L-paper"]["nets"][nm]["nerr_c"])) for nm in ("P6B4e8", "P6B4e8L")}

# 核對:重算的迭代法 500 次內最佳 = scan8.json 的 iter_best
ib = J8["verdict"]["exam"]["L-paper"]["iter_best"]
mine = min(V["L-paper"].items(), key=lambda kv: kv[1][0])
print(f"  核對 L-paper 迭代法最佳:重算 {mine[0]} {mine[1][0]:.5f} vs scan8.json {ib[0]} ×{ib[1]} {ib[2]:.5f}")

# ---------------- P3:時間 vs nerr_c(L-paper),1.5 × CRLB 與 2 × CRLB ----------------
fig, axs = plt.subplots(1, 2, figsize=(17, 6.2))
for ax, B in zip(axs, ("64", "512")):
    ts, ys = curve(V["L-paper"], IT8, T8[B], 0)
    ax.plot(ts, ys, color=COL["iter"], lw=1.6, drawstyle="steps-post", label="迭代法包絡(500 次內)")
    for nm in ("P6B4e8", "P6B4e8L"):
        ax.plot([T8["net"][nm][B]], [netc[nm]], "o", ms=10, color=COL[nm], label=NAME[nm])
    for k, ls in ((1.5, "--"), (2.0, ":")):
        ax.axhline(k * crlb, color="#d6452a", ls=ls, lw=1.3)
        ax.text(ts[0] * 1.05, k * crlb * 1.04, f"{k:g} × CRLB = {k * crlb:.3f}", color="#d6452a", fontsize=12)
    tn = T8["net"]["P6B4e8L"][B]
    out = []
    for k in (1.5, 2.0):
        hi, lo = reach(V["L-paper"], IT8, T8[B], k * crlb)
        out.append(f"k = {k:g}:" + (f"迭代法 500 次內未達 → 網路 ≥ × {lo / tn:.1f}" if not np.isfinite(hi) else
                                     (f"迭代法快 × {tn / hi:.1f}–{tn / lo:.1f}" if hi < tn else f"網路快 × {lo / tn:.1f}–{hi / tn:.1f}")))
    ax.text(0.98, 0.97, "\n".join(out), transform=ax.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("每張整張影像的時間(ms)")
    ax.set_title(f"L-paper,batch {B}")
    ax.grid(True, which="both", color="#e4e3df", lw=0.5)
axs[0].set_ylabel("nerr_c")
axs[0].legend(fontsize=12, frameon=False, loc="lower left")
fig.suptitle("同品質比時間:結論取決於門檻(k = 1.5 為看過結果後選定;k = 2 為事先的主判)", fontsize=16)
fig.tight_layout()
finish(fig, "P3_time_nerr_c.png")

# ---------------- P4:加速倍數 vs k ----------------
ks = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
fig, ax = plt.subplots(figsize=(10, 5.6))
for B, mk, off in (("64", "s", -0.03), ("512", "o", 0.03)):
    tn = T8["net"]["P6B4e8L"][B]
    for k in ks:
        hi, lo = reach(V["L-paper"], IT8, T8[B], k * crlb)
        reach_net = netc["P6B4e8L"] <= k * crlb
        if not reach_net:
            continue
        if np.isfinite(hi):
            ax.plot([k + off, k + off], [lo / tn, hi / tn], "-", color=COL["P6B4e8L"] if B == "512" else COL["P6B4e8"], lw=3)
        else:
            ax.annotate("", xy=(k + off, lo / tn * 1.8), xytext=(k + off, lo / tn), arrowprops=dict(arrowstyle="->", color="#d6452a", lw=2))
            ax.plot([k + off], [lo / tn], mk, color="#d6452a", ms=8)
    ax.plot([], [], "-", lw=3, color=COL["P6B4e8L"] if B == "512" else COL["P6B4e8"], label=f"b{B}:加速區間 [下界, 上界]")
ax.plot([], [], "s", color="#d6452a", label="b64:迭代法 500 次內未達 → 只有下界")
ax.plot([], [], "o", color="#d6452a", label="b512:同上")
ax.axhline(1, color="#444444", lw=1)
for k in (1.5, 2.0):
    ax.axvline(k, color="#bbbbbb", ls=":", lw=1)
ax.set_yscale("log")
ax.set_xticks(ks)
ax.set_xlabel("門檻 Q = k × CRLB 的 k")
ax.set_ylabel("網路(P6B4e8-L)的加速倍數")
ax.set_title("L-paper:加速倍數 vs 門檻(> 1 = 網路較快)")
ax.legend(fontsize=11, frameon=False, loc="upper right")
ax.grid(True, which="both", color="#e4e3df", lw=0.5)
finish(fig, "P4_speed_vs_k.png")

# ---------------- P5:時間 vs SV-F1(L-paper、L-combo)----------------
fig, axs = plt.subplots(1, 2, figsize=(17, 6.0), sharey=True)
for ax, e in zip(axs, ("L-paper", "L-combo")):
    for B, ls in (("512", "-"), ("64", "--")):
        ts, ys = curve({k: (v[0], -v[1]) for k, v in V[e].items()}, IT8, T8[B], 1)
        ax.plot(ts, [-y for y in ys], ls, color=COL["iter"], lw=1.5, drawstyle="steps-post", label=f"迭代法包絡 b{B}")
    for nm in ("P6B4e8", "P6B4e8L"):
        f1 = J8["verdict"]["exam"][e]["nets"][nm]["svf1"]["all"]
        ax.plot([T8["net"][nm]["512"]], [f1], "o", ms=10, color=COL[nm], label=f"{NAME[nm]}({f1:.3f})")
    ax.axhline(0.9, color="#d6452a", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("每張整張影像的時間(ms)")
    ax.set_title({"L-paper": "L-paper(無誤差)", "L-combo": "L-combo(有位置 / 探針誤差)"}[e])
    ax.grid(True, which="both", color="#e4e3df", lw=0.5)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
axs[0].set_ylabel("S 單空缺的判讀 F1")
fig.suptitle("空缺判讀:無誤差時迭代法在網路的時間也看得出;有誤差時同時間網路較好", fontsize=16)
fig.tight_layout()
finish(fig, "P5_time_svf1.png")

# ---------------- R5:combo 的時間 vs nerr_sr(階段七)----------------
C7 = json.load(open(D7 / "scan7a_cond_combo.json"))
IT7 = C7["meta"]["iters"]
v7 = {(c, int(it)): (float(np.mean(rec["nerr_sr"])), 0.0) for c, cr in C7["env"].items() for it, rec in cr.items() if it != "diag"}
T7 = J7["timing"]
fig, ax = plt.subplots(figsize=(10, 6))
ts, ys = curve(v7, IT7, T7["512"], 0)
ax.plot(ts, ys, color=COL["iter"], lw=1.6, drawstyle="steps-post", label="迭代法包絡(b512)")
for nm in ("P6B4", "P6B4r", "P6B4e8"):
    ax.plot([T7["net"][nm]["512"]], [float(np.mean(J7["nets"]["combo"][nm]["nerr_sr"]))], "o", ms=10, color=COL[nm], label=NAME[nm])
ax.axhline(0.05, color="#d6452a", ls="--", lw=1.2)
s = J7["verdict"]["cond"]["combo"]["P6B4e8"]["512"]["speed"]["0.05"]
ax.axvspan(s["t_iter_lo"], s["t_iter"], color="#d6452a", alpha=0.12)
ax.text(0.98, 0.97, f"Q = 0.05:P6B4e8 穩健加速\nb512 × {s['x_lo']:.2f}–{s['x']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=13,
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("每張整張影像的時間(ms)")
ax.set_ylabel("nerr_sr")
ax.set_title("combo(位置 + 探針 + 劑量):同品質比時間")
ax.legend(fontsize=12, frameon=False, loc="lower left")
ax.grid(True, which="both", color="#e4e3df", lw=0.5)
finish(fig, "R5_combo_time.png")
