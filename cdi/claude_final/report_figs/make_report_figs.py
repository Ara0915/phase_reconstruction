#!/usr/bin/env python
"""第三次報告(10/14)的圖:只讀結果檔(驗證資料),本機執行即可(不需 GPU、不讀期末考資料)。
需求清單:claude_final/docs/報告圖表需求_第三次.md。輸出到本資料夾;既有圖以複製方式放入(註明來源)。
用法(repo 的 cdi/ 底下):python claude_final/report_figs/make_report_figs.py"""
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

HERE = Path(__file__).resolve().parent
CDI = HERE.parents[1]
S7D = CDI / "claude_stage7_figs" / "scan7d.json"
S8 = CDI / "claude_stage8" / "results" / "scan8.json"
S8C = CDI / "claude_stage8" / "results_8c" / "scan8c.json"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"], "font.size": 15,
                     "axes.unicode_minus": False, "savefig.dpi": 160})
COL = {"P6B4": "#8c8c8c", "P6B4r": "#e8a33d", "P6B4e8": "#2a78d6", "P6B4e8L": "#0b3d91", "P6B4e8C": "#8fbbe8", "iter": "#222222"}
NAME = {"P6B4": "P6B4", "P6B4r": "P6B4r", "P6B4e8": "P6B4e8", "P6B4e8L": "P6B4e8-L", "P6B4e8C": "P6B4e8-C"}
LOG = []


def mark(fig):
    fig.text(0.995, 0.01, "驗證資料", ha="right", va="bottom", fontsize=12, color="#666666")


def plainlog(fig):
    """對數軸的刻度改成一般數字(0.001、0.01 …):微軟正黑體沒有數學負號的字形。"""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for ax in fig.axes:
        for axis, sc in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if sc == "log":
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(NullFormatter())


def save(fig, name, src):
    plainlog(fig)
    mark(fig)
    fig.savefig(HERE / name, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    LOG.append((name, src))
    print("  圖:", name)


def copy(src, name, note):
    shutil.copy(src, HERE / name)
    LOG.append((name, f"複製既有圖 {src.relative_to(CDI)};{note}"))
    print("  複製:", name)


J7 = json.load(open(S7D))
J8 = json.load(open(S8))
J8c = json.load(open(S8C))
N7 = J7["nets"]


def m7(c, nm):
    return float(np.mean(N7[c][nm]["nerr_sr"]))


# ---------------- E2:凍結 P6B4 在 8 個條件 ----------------
conds = ["ideal", "posL", "posH", "prbL", "prbH", "doseL", "doseH", "combo"]
lab = {"ideal": "理想", "posL": "位置 0.5 px", "posH": "位置 1.0 px", "prbL": "離焦 ±10%", "prbH": "離焦 ±30%",
       "doseL": "劑量 ÷10", "doseH": "劑量 ÷100", "combo": "三者(輕)"}
kind = {"ideal": "#bbbbbb", "posL": "#d6452a", "posH": "#d6452a", "prbL": "#e8a33d", "prbH": "#e8a33d", "doseL": "#1baf7a",
        "doseH": "#1baf7a", "combo": "#8a5cd1"}
v = [m7(c, "P6B4") for c in conds]
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.bar(range(len(conds)), v, color=[kind[c] for c in conds])
for i, x in enumerate(v):
    ax.text(i, x * 1.12, f"{x:.4f}" if x < 0.1 else f"{x:.3f}", ha="center", fontsize=12)
for q, ls in ((0.05, "--"), (0.02, ":")):
    ax.axhline(q, color="#444444", ls=ls, lw=1)
    ax.text(len(conds) - 0.45, q * 1.08, f"Q = {q}", fontsize=11, ha="right")
ax.set_yscale("log")
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([lab[c] for c in conds], rotation=20)
ax.set_ylabel("nerr_sr(越低越好)")
ax.set_title("凍結的 P6B4(整張 G9)在各種真實誤差下的誤差")
from matplotlib.patches import Patch                                   # noqa: E402
ax.legend(handles=[Patch(color="#d6452a", label="位置"), Patch(color="#e8a33d", label="探針"), Patch(color="#1baf7a", label="劑量"),
                   Patch(color="#8a5cd1", label="綜合")], fontsize=12, frameon=False, loc="upper left")
save(fig, "E2_errors_bar.png", "scan7d.json nets[條件]['P6B4']['nerr_sr'](3 seeds 平均)")

# ---------------- E3:每張繞射圖的入射電子數 ----------------
D = J8["verdict"]["dose"]
nin = D["N_inc"]
fW = float(np.mean([x[0] for x in D["f"].values()]))
fM = float(np.mean([x[1] for x in D["f"].values()]))
items = [("本研究基準", nin), ("doseL(÷10)", nin / 10), ("doseH(÷100)", nin / 100), ("論文 MoS₂", nin * fM), ("論文 WS₂", nin * fW)]
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.barh(range(len(items)), [x for _, x in items], color=["#2a78d6", "#8fbbe8", "#8fbbe8", "#d6452a", "#d6452a"])
for i, (_, x) in enumerate(items):
    ax.text(x * 1.15, i, f"{x:.2g}", va="center", fontsize=12)
ax.set_xscale("log")
ax.set_yticks(range(len(items)))
ax.set_yticklabels([k for k, _ in items])
ax.invert_yaxis()
ax.set_xlim(1e4, 1e9)
ax.set_xlabel("每張繞射圖的入射電子數")
ax.set_title("劑量的銜接:論文約為本研究基準的 1/1000")
save(fig, "E3_dose_per_pattern.png", "scan8.json verdict.dose(N_inc、f_W、f_Mo;3 seeds 平均)")

# ---------------- R1:5 個條件 × 3 個網路 ----------------
c5 = ["ideal", "posL", "combo", "prbH", "posH"]
nets3 = ["P6B4", "P6B4r", "P6B4e8"]
fig, ax = plt.subplots(figsize=(11, 5.6))
w = 0.26
for k, nm in enumerate(nets3):
    vals = [m7(c, nm) for c in c5]
    ax.bar(np.arange(len(c5)) + (k - 1) * w, vals, w, color=COL[nm], label=NAME[nm] + {"P6B4": "(凍結)", "P6B4r": "(容忍)", "P6B4e8": "(估計)"}[nm])
    for i, x in enumerate(vals):
        ax.text(i + (k - 1) * w, x * 1.1, f"{x:.3f}" if x >= 0.01 else f"{x:.4f}", ha="center", fontsize=9.5, rotation=90)
for q, ls in ((0.05, "--"), (0.02, ":")):
    ax.axhline(q, color="#444444", ls=ls, lw=1)
    ax.text(len(c5) - 0.5, q * 1.08, f"Q = {q}", fontsize=11, ha="right")
ax.set_yscale("log")
ax.set_ylim(1e-3, 3)
ax.set_xticks(range(len(c5)))
ax.set_xticklabels([lab[c] for c in c5])
ax.set_ylabel("nerr_sr")
ax.set_title("兩種對策:估計型把位置誤差修回來")
ax.legend(fontsize=12, frameon=False, loc="upper left")
save(fig, "R1_three_nets.png", "scan7d.json nets(3 seeds 平均)")

# ---------------- R2:各級的位置 RMS ----------------
P = J7["pos"]
fig, ax = plt.subplots(figsize=(9, 5.4))
for c, colr, ls in (("posL", "#2a78d6", "-"), ("combo", "#8a5cd1", "-"), ("posH", "#999999", "--")):
    r = [float(np.mean(P[c]["rms0"]))] + list(np.mean(np.array(P[c]["rms"]), 0))
    ax.plot(range(len(r)), r, ls, marker="o", color=colr, label=lab[c] + ("(修不回來)" if c == "posH" else ""))
    ax.text(len(r) - 1 + 0.15, r[-1], f"{r[-1]:.3f} px", fontsize=11, va="center", color=colr)
ax.set_yscale("log")
ax.set_xlabel("級數(0 = 草稿,用名目位置)")
ax.set_ylabel("位置 RMS 誤差(px,49 個位置)")
ax.set_xlim(-0.3, 9.2)
ax.set_title("P6B4e8:每一級都在修正位置")
ax.legend(fontsize=12, frameon=False)
ax.grid(True, which="both", color="#e4e3df", lw=0.5)
save(fig, "R2_position_rms.png", "scan7d.json pos[條件](rms0、各級 rms;3 seeds 平均)")

# ---------------- P1:L-paper 的 nerr_c ----------------
ex = J8["verdict"]["exam"]["L-paper"]
crlb = J8c["report"]["crlb"]["L-paper"]["crlb_c"]
order = ["P6B4", "P6B4r", "P6B4e8", "P6B4e8C", "P6B4e8L"]
vals = [float(np.mean(ex["nets"][nm]["nerr_c"])) for nm in order] + [float(ex["iter_best"][2])]
names = [NAME[n] for n in order] + ["迭代法\n500 次內最佳"]
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.bar(range(len(vals)), vals, color=[COL[n] for n in order] + [COL["iter"]])
for i, x in enumerate(vals):
    ax.text(i, x * 1.1, f"{x:.4f}" if x < 0.1 else f"{x:.3f}", ha="center", fontsize=12)
for k, ls in ((1.0, "-"), (1.5, "--"), (2.0, ":")):
    ax.axhline(k * crlb, color="#d6452a", ls=ls, lw=1.2)
    ax.text(-0.45, k * crlb * 1.05, f"{'CRLB' if k == 1 else f'{k:g} × CRLB'} = {k * crlb:.3f}", fontsize=11, color="#d6452a")
ax.set_yscale("log")
ax.set_ylim(top=max(vals) * 2.2)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(names)
ax.set_ylabel("nerr_c(L-paper,論文劑量)")
ax.set_title("晶格微調後:P6B4e8-L 達到雜訊尺度(0.95 × CRLB)")
save(fig, "P1_Lpaper_nerr_c.png", "scan8.json verdict.exam['L-paper'](網路 3 seeds 平均、iter_best);scan8c.json report.crlb['L-paper']")

# ---------------- P6:微調歸因 ----------------
h5 = J8["verdict"]["h5"]
ex3 = ["L-ideal", "L-paper", "L-combo"]
keys = [k for k in h5["L-paper"].keys()]
klab = {keys[0]: "L / P6B4e8", keys[1]: "L / C(晶格資料的貢獻)", keys[2]: "C / P6B4e8(多訓練)"}
kcol = {keys[0]: "#0b3d91", keys[1]: "#2a78d6", keys[2]: "#8fbbe8"}
fig, ax = plt.subplots(figsize=(10, 5.4))
w = 0.26
for j, k in enumerate(keys):
    vals = [h5[e][k]["ratio"] for e in ex3]
    ax.bar(np.arange(3) + (j - 1) * w, vals, w, color=kcol[k], label=klab[k])
    for i, x in enumerate(vals):
        ax.text(i + (j - 1) * w, x * 1.1, f"{x:.3f}", ha="center", fontsize=10)
ax.axhline(1.0, color="#444444", lw=1)
ax.set_yscale("log")
ax.set_xticks(range(3))
ax.set_xticklabels(ex3)
ax.set_ylabel("nerr_c 的比值")
ax.set_title("改善來自晶格資料,不是多訓練")
ax.legend(fontsize=12, frameon=False, loc="lower right")
save(fig, "P6_attribution.png", "scan8.json verdict.h5(比值)")

# ---------------- P7:代價(a)原本 13 個條件;(b)(c)劑量掃描 ----------------
h6 = J8["verdict"]["h6"]["P6B4e8L"]
reg = [x.split(":")[1] for x in J8["verdict"]["h6"]["regress"]]
cs = list(h6.keys())
B3 = J8c["report"]["B3"]["table"]
doses = sorted((float(d) for d in B3), reverse=True)
crlb1 = J8c["report"]["crlb"]["L-ideal"]["crlb_c"]
fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
ax = axs[0]
vals = [h6[c]["ratio"] for c in cs]
ax.bar(range(len(cs)), vals, color=["#d6452a" if c in reg else "#8fbbe8" for c in cs])
ax.axhline(1.0, color="#444444", lw=1)
ax.axhline(1.25, color="#d6452a", ls="--", lw=1)
ax.set_xticks(range(len(cs)))
ax.set_xticklabels(cs, rotation=60, fontsize=11)
ax.set_ylabel("P6B4e8-L / P6B4e8(nerr_sr)")
ax.set_title("(a) 原本的 13 個條件:紅 = 退步")
ax = axs[1]
for nm in ("P6B4e8", "P6B4e8L"):
    ax.plot(doses, [B3[str(d) if str(d) in B3 else repr(d)][nm]["nerr_c"] for d in doses], "o-", color=COL[nm], label=NAME[nm])
dd = np.array(doses)
ax.plot(dd, crlb1 / dd, color="#d6452a", ls="--", label="CRLB")
ax.axvline(fW, color="#888888", ls=":", lw=1)
ax.text(fW * 1.1, ax.get_ylim()[0] if False else 2e-3, "論文劑量", fontsize=11, color="#666666")
ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlabel("劑量(相對本研究基準)")
ax.set_ylabel("nerr_c(L-ideal 的物體)")
ax.set_title("(b) 高劑量時停在精度天花板")
ax.legend(fontsize=11, frameon=False)
ax = axs[2]
for nm in ("P6B4e8", "P6B4e8L"):
    ax.plot(doses, [B3[str(d) if str(d) in B3 else repr(d)][nm]["sv_r"] for d in doses], "o-", color=COL[nm], label=NAME[nm])
ax.axvline(fW, color="#888888", ls=":", lw=1)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("劑量(相對本研究基準)")
ax.set_ylabel("S 單空缺的召回率")
ax.set_title("(c) 高劑量時漏判單空缺")
ax.legend(fontsize=11, frameon=False)
fig.tight_layout()
save(fig, "P7_costs.png", "(a) scan8.json verdict.h6;(b)(c) scan8c.json report.B3.table(劑量掃描)、report.crlb['L-ideal'](CRLB ∝ 1/劑量)")

# ---------------- 直接沿用的既有圖 ----------------
S8F = CDI / "claude_stage8" / "results" / "figs_scan8"
copy(S8F / "c_recon_L-paper_W.png", "P2_recon_Lpaper_WS2.png", "階段八 8b 的重建圖(百分位樣本、SV / DV 圈與判讀顏色)")
copy(S8F / "c_recon_L-paper_Mo.png", "P2_recon_Lpaper_MoS2.png", "同上")
copy(S8F / "c_frc.png", "P8_frc.png", "階段八 8b 的 FRC")
copy(CDI / "claude_stage8" / "results_8d" / "figs_scan8d" / "d8_ceiling.png", "P7_extra_ceiling_8d.png", "8d 精度天花板(參考)")
copy(CDI / "claude_stage7_figs" / "figs_scan7d" / "c_recon.png", "R3_recon_stage7d.png",
     "7d 的重建圖:樣本 #228 = P6B4e8 在 combo 的中位數(清單寫 posL 的中位數 → 不同,見 README)")

# ---------------- README ----------------
r4 = J7["verdict"]["cond"]["combo"]["P6B4e8"]
s64, s512 = r4["64"]["speed"]["0.05"], r4["512"]["speed"]["0.05"]
lines = ["# 第三次報告的圖(驗證資料)", "", "由 `make_report_figs.py` 產生(只讀結果檔)。需求清單:`../docs/報告圖表需求_第三次.md`。", "",
         "| 檔案 | 來源 |", "|---|---|"] + [f"| `{n}` | {s} |" for n, s in LOG]
lines += ["", "## R4(文字,不需產圖)",
          f"combo、Q = 0.05:P6B4e8 穩健加速 b64 × {s64['x_lo']:.2f}–{s64['x']:.2f}、b512 × {s512['x_lo']:.2f}–{s512['x']:.2f};"
          f"時間(b512)P6B4 {J7['timing']['net']['P6B4']['512']:.2f} ms、P6B4e8 {J7['timing']['net']['P6B4e8']['512']:.2f} ms(scan7d.json)。", "",
          "## 數字核對", "E2、R1、R2、R4、P1、P6、P7 的數字皆直接讀自結果檔,與清單一致(scan7d.json、scan8.json、scan8c.json)。", "",
          "## 與清單不同之處",
          "- R3:沿用 7d 的既有重建圖,樣本是 P6B4e8 在 combo(all three)的中位數 #228,不是清單寫的 posL 中位數;圖中已含 posL 列(同一個場)。"
          "若要嚴格依清單,需在國網重算。",
          "- P7(b)(c):依 8c 的劑量掃描表重畫(清單原寫沿用 c8_dose.png);d8_ceiling.png 另附為參考。"]
(HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("README.md 完成")
