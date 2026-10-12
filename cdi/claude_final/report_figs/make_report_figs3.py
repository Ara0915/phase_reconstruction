#!/usr/bin/env python
"""第三次報告的圖(第三批):示意圖 E1、A1、L1、L2、L4。只用驗證場(晶格 87M、程序生成 80M)與既有的產生器 / 量測函式,CPU 即可。
需要能 import scan_8(與它的相依)與 config_used.json 的環境:
    國網:cd ~/cdi && python <repo>/cdi/claude_final/report_figs/make_report_figs3.py
    本機:CDI_CODE=<放 scan_8.py 等的資料夾> CDI_RUN_ROOT=<有 ideal_base_s0/config_used.json 的資料夾> python make_report_figs3.py
(本機的 config 由 configs/ideal_base.yaml 產生,Δx = 0.3577 Å/px 與國網相同;E1 的計數尺度依 ref_energy,只作示意。)"""
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch                 # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("CDI_CODE", os.getcwd()))
import scan_8 as s8                                                   # noqa: E402

a7, a6, b7, s5 = s8.a7, s8.a6, s8.b7, s8.s5
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"], "font.size": 15, "axes.unicode_minus": False,
                     "savefig.dpi": 160})
dev = torch.device("cpu")


def finish(fig, name):
    fig.text(0.995, 0.01, "驗證資料", ha="right", va="bottom", fontsize=12, color="#666666")
    fig.savefig(HERE / name, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  圖:", name)


# ---------------- L1 / L2:晶格示意(L-paper 與 L-newdef 的驗證場,真值) ----------------
dP = s8.setup_exam(0, dev, "L-paper", 2, smoke=False)
dx = dP["dx"]
U = dP["geo"].U.cpu()
ys, xs = torch.where(U)
y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
fig, axs = plt.subplots(1, 2, figsize=(15, 7.2))
for ax, i, mat in ((axs[0], 0, "WS₂"), (axs[1], 1, "MoS₂")):
    ph = dP["fields"][i, 1].cpu().numpy()
    im = ax.imshow(ph[y0:y1, x0:x1], cmap="viridis", vmin=0, vmax=0.9)
    cols = dP["meta"]["cols"][i]
    for p, nS in zip(cols["pos"], cols["nS"]):
        yy, xx = float(p[0]) - y0, float(p[1]) - x0
        if not (0 <= yy < y1 - y0 and 0 <= xx < x1 - x0) or int(nS) == 2:
            continue
        ax.add_patch(Circle((xx, yy), 2.0 if int(nS) == 1 else 2.8, fill=False, color="#ff5a36" if int(nS) == 1 else "white", lw=1.6,
                            ls="-" if int(nS) == 1 else "--"))
    L = 10.0 / dx
    ax.plot([3, 3 + L], [y1 - y0 - 4] * 2, color="white", lw=4)
    ax.text(3 + L / 2, y1 - y0 - 6, "10 Å", color="white", ha="center", fontsize=13)
    ax.set_title(f"{mat}(單層,沿 [001] 投影;晶格常數 {s8.A_LAT['WS2' if i == 0 else 'MoS2']} Å ≈ {s8.A_LAT['WS2' if i == 0 else 'MoS2'] / dx:.1f} px)", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=list(axs), fraction=0.025, label="相位(rad)")
fig.suptitle("晶格訓練集:亮點 = 金屬柱(W / Mo),較暗 = S₂ 柱;紅圈 = S 單空缺、白虛線圈 = 雙空缺", fontsize=15)
finish(fig, "L1_lattice.png")

dN = s8.setup_exam(0, dev, "L-newdef", 2, smoke=False)
fig, axs = plt.subplots(1, 2, figsize=(15, 7.2))
for ax, i in zip(axs, (0, 1)):
    ph = dN["fields"][i, 1].cpu().numpy()
    im = ax.imshow(ph[y0:y1, x0:x1], cmap="viridis", vmin=0, vmax=0.9)
    cols = dN["meta"]["cols"][i]
    for p, nS, cat in zip(cols["pos"], cols["nS"], cols["cat"]):
        yy, xx = float(p[0]) - y0, float(p[1]) - x0
        if not (0 <= yy < y1 - y0 and 0 <= xx < x1 - x0) or int(nS) == 2:
            continue
        colr = {s8.CAT_LD: "#ff5a36", s8.CAT_MR: "#ffd23f"}.get(int(cat), "white")
        ax.add_patch(Circle((xx, yy), 2.2, fill=False, color=colr, lw=1.4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"例 {i + 1}:{'WS₂' if i == 0 else 'MoS₂'}", fontsize=13)
fig.colorbar(im, ax=list(axs), fraction=0.025, label="相位(rad)")
fig.suptitle("只測試、未訓練的新缺陷(L-newdef):紅圈 = 線缺陷、黃圈 = 缺 S 區域、白圈 = 一般空缺", fontsize=15)
finish(fig, "L2_newdef.png")

# ---------------- L4:為什麼要改評分(真值 vs 空白平面) ----------------
O = dP["O"]
m = dP["geo"].U.to(O.real.dtype)
cs = (O * m).sum((1, 2)) / m.sum()
flat = torch.polar(torch.ones_like(O.real), torch.full_like(O.real, float(torch.angle(cs).mean())))
r_t = s8.lat_metrics(O.clone(), dP)
r_f = s8.lat_metrics(flat, dP)
fig, axs = plt.subplots(1, 2, figsize=(13, 6.4))
for ax, img, ttl in ((axs[0], O[0], f"真值\nnerr_sr = {abs(float(r_t['sr'][0])):.4f}、nerr_c = {abs(float(r_t['c'][0])):.4f}"),
                     (axs[1], flat[0], f"空白平面(振幅 1、相位為常數)\n舊指標 nerr_sr = {float(r_f['sr'][0]):.4f}(< 0.02 → 會「過門檻」)、新指標 nerr_c = {float(r_f['c'][0]):.2f}")):
    ax.imshow(np.angle(img.cpu().numpy())[y0:y1, x0:x1], cmap="viridis", vmin=0, vmax=0.9)
    ax.set_title(ttl, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("晶格幾乎是純相位物體:舊指標以 Σ|真值|² 正規化 → 空白影像也很小;新指標以晶格的對比正規化 → 空白影像 = 1", fontsize=14)
finish(fig, "L4_metric.png")

# ---------------- E1:四種誤差 ----------------
d0 = a7.setup_cond(0, dev, "ideal", 2)
geo, pr, ops = d0["geo"], d0["pr"], d0["ops"]
fig = plt.figure(figsize=(18, 9.5))
gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25)
# (1) 位置
ax = fig.add_subplot(gs[:, 0])
st = np.array(geo.starts, float) + geo.W / 2
z = d0["z"][0].cpu().numpy() * 0.5
X5 = 5.0
ax.scatter(st[:, 1], st[:, 0], s=40, color="#999999", label="名目位置(7 × 7、步距 8 px)")
ax.scatter(st[:, 1] + X5 * z[:, 1], st[:, 0] + X5 * z[:, 0], s=40, color="#d6452a", label=f"真實位置(σ = 0.5 px;放大 {X5:g} 倍)")
for (yy, xx), (dy, dx_) in zip(st, z):
    ax.plot([xx, xx + X5 * dx_], [yy, yy + X5 * dy], color="#d6452a", lw=0.8)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_title("① 掃描位置誤差(posL / posH)", fontsize=14)
ax.legend(fontsize=11, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.05))
ax.set_xticks([])
ax.set_yticks([])
# (2) 探針
pp, pm = a7.true_probe_pair(pr, 0.30, dev)
for j, (P, ttl) in enumerate(((pr["P"], "名目探針"), (pp, "離焦 +30%"))):
    ax = fig.add_subplot(gs[j, 1])
    c = geo.W // 2
    Pz = P[c - 16:c + 16, c - 16:c + 16].cpu()
    rgb = matplotlib.colors.hsv_to_rgb(np.stack([(np.angle(Pz.numpy()) + np.pi) / (2 * np.pi), np.ones(Pz.shape),
                                                 (Pz.abs() / Pz.abs().max()).numpy()], -1))
    ax.imshow(rgb)
    ax.set_title(("② 探針誤差(prbL / prbH)\n" if j == 0 else "") + ttl + "(亮度 = 振幅、色相 = 相位)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
# (3) 劑量
cp0 = d0["cp"]
from src.physics import forward_measure                               # noqa: E402
win = d0["fields"][:1, :, geo.starts[a6.CEN49][0]:geo.starts[a6.CEN49][0] + geo.W, geo.starts[a6.CEN49][1]:geo.starts[a6.CEN49][1] + geo.W]
psi = pr["P"] * torch.polar(win[:, 0], win[:, 1])
I = forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), d0["bs"], cp0, add_poisson=False)[0]
gen = torch.Generator().manual_seed(2026)
for j, (dose, ttl) in enumerate(((1.0, "基準"), (0.01, "÷100"))):
    ax = fig.add_subplot(gs[j, 2])
    cnt = torch.poisson((I * dose).clamp_min(0), generator=gen)
    ax.imshow(np.log10(1 + cnt.numpy()), cmap="magma")
    ax.set_title(("③ 劑量(doseL / doseH)\n" if j == 0 else "") + f"{ttl}(同一張繞射圖;log 顯示)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
# (4) 部分同調
cpn = __import__("copy").deepcopy(cp0)
cpn.add_poisson = False
Ic = b7.measure_coh(d0["fields"][:1], geo, cpn, d0["bs"], 0, 1.0, ops, poisson=False)[a6.CEN49][0]
for j, (img, ttl) in enumerate(((I, "完全同調"), (Ic, "部分同調(cohH,光源 0.36 Å)"))):
    ax = fig.add_subplot(gs[j, 3])
    ax.imshow(np.log10(1 + img.numpy()), cmap="magma")
    ax.set_title(("④ 部分同調(cohL / cohH;只測試)\n" if j == 0 else "") + ttl + ("(繞射圖變糊)" if j == 1 else ""), fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("加入的真實誤差:量測用「真的」條件,重建只知道「名目」條件(驗證場,示意)", fontsize=16)
finish(fig, "E1_error_types.png")

# ---------------- A1:兩種對策的架構示意 ----------------
fig, ax = plt.subplots(figsize=(17, 8))
ax.set_xlim(0, 17)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, w, h, txt, fc, ec="#333333", fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


def arrow(x0_, y0_, x1_, y1_):
    ax.annotate("", xy=(x1_, y1_), xytext=(x0_, y0_), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))


for row, (y, name, K, pos, note) in enumerate(((5.1, "P6B4r(容忍型)", 4, False, "訓練資料加入隨機的位置 / 探針 / 劑量誤差;用名目位置"),
                                               (1.2, "P6B4e8(估計型)", 8, True, "每一級多一個位置步:每個掃描位置解 2 × 2 Gauss–Newton,位置逐級更新"))):
    ax.text(0.2, y + 2.15, name, fontsize=17, fontweight="bold", color="#e8a33d" if row == 0 else "#2a78d6")
    ax.text(0.2, y + 1.75, note, fontsize=12.5, color="#444444")
    box(0.2, y, 2.2, 1.3, "49 張繞射圖\n→ U-Net 草稿\n(9 塊合併)", "#eeeeee")
    arrow(2.5, y + 0.65, 3.1, y + 0.65)
    x = 3.2
    shown = [1, 2, "…", K]
    for k in shown:
        if k == "…":
            ax.text(x + 0.4, y + 0.65, "…", fontsize=20, ha="center", va="center")
            x += 0.9
            continue
        if pos:
            box(x, y + 0.72, 2.5, 0.55, f"第 {k} 級:AP 步 + 小 CNN", "#dbe9fb", fs=10.5)
            box(x, y + 0.02, 2.5, 0.55, "位置步(2×2 Gauss–Newton)", "#fde3cf", ec="#d6452a", fs=10.5)
        else:
            box(x, y + 0.2, 2.5, 0.9, f"第 {k} 級:AP 物理步\n+ 小 CNN(名目位置)", "#dbe9fb", fs=10.5)
        arrow(x + 2.6, y + 0.65, x + 2.95, y + 0.65)
        x += 3.05
    box(x, y + 0.2, 1.6, 0.9, "整張影像", "#eeeeee")
ax.text(8.5, 0.35, "相同:U-Net 草稿 + 物理步的展開架構(結構 10/5 凍結)     新增(紅框):位置步、級數 4 → 8、訓練時的位置監督",
        ha="center", fontsize=13, color="#333333")
finish(fig, "A1_two_strategies.png")
