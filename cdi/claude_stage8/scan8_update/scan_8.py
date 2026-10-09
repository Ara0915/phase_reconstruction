#!/usr/bin/env python
"""階段八:TMD 晶格資料集(對齊 Chen et al. 2024, Sci. Rep. 14, 277 的樣品;階段八實驗設計協定 v1.5)。

結構不變,只換資料、只微調權重(使用者 10/5)。
8a:凍結的 P6B4、P6B4r、P6B4e8 直接考晶格的 5 份考卷(L-ideal、L-paper(主)、L-combo、L-coh、L-newdef)。
8b:P6B4e8 在「晶格 + 原本資料各半」上微調 → P6B4e8-L;對照組 P6B4e8-C 只用原本資料、同樣的步數與設定續訓(成對)。
晶格:單層 MoS₂ / WS₂(1H)沿 [001] 投影;Kirkland 投影位勢(abTEM 的係數)+ Debye–Waller,頻帶限制取樣;
      吸收 a = exp(−η·φ)(η = 0.1);缺陷:S 單 / 雙空缺(訓練 + 測試)、線缺陷與缺 S 區域(只測試)。
劑量:對齊論文「每張繞射圖的入射電子數」(WS₂ 1.143 × 10⁵、MoS₂ 1.613 × 10⁵)。
主指標:對比正規化誤差 nerr_c(對齊同 nerr_sr;最佳複數常數 = 1);任務指標:S 空缺判讀(SV-F1)。

用法(需在計算節點執行):
    python scan_8.py --check          # 內建檢查(含劑量校準;寫出 scan8_dose.json)
    python scan_8.py --smoke          # 迷你全流程(dev 節點);通過才可正式執行
    python scan_8.py --task 0..10     # run_scan8_jobs.sh:0–4 = 5 份考卷的迭代法包絡;5–7 = P6B4e8-L seed 0–2;8–10 = P6B4e8-C seed 0–2
    python scan_8.py --final          # 彙整(run_scan8_final.sh)
"""
import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_7d as d7                                                 # noqa: E402  (7d;不修改。也載入 7a / 7b / 7c 的條件與工具)

c7, b7, a7, a6, h, g = d7.c7, d7.b7, d7.a7, d7.a6, d7.h, d7.g
s5, s5b, s5c = d7.s5, d7.s5b, d7.s5c
RUN_ROOT, QUICK, PROBE, SEEDS, SUF = d7.RUN_ROOT, d7.QUICK, d7.PROBE, d7.SEEDS, d7.SUF
D7_MD5 = "b6e14b742fbf45615f87046f5618bb4c"  # 7d 的 scan_7d.py(P6B4e8 的模型、訓練與 13 個條件的結果)

# ============================================================================
# 物理常數(200 kV;abTEM 1.0.10 的值)與 Kirkland 參數
# ============================================================================
LAMBDA = 0.025079340317328468                # 波長 Å(abtem.core.energy.energy2wavelength(200e3))
SIGMA = 7.288401085927866e-4                 # 交互作用常數 rad/(V·Å)(energy2sigma(200e3))
ALPHA = 18.9e-3                              # 會聚半角(論文)
DX_PROTO = 0.358                             # 協定寫的 Δx(Å/px);程式用 ka /(W·α/λ) 實際算出,檢查差 < 0.5%
KAPPA = 0.0208865737082965                   # abtem.core.constants.kappa(參數換算用)
# Kirkland(2010)的參數(abTEM KirklandParametrization().parameters):列 = a、b、c、d,各 3 項
KIRKLAND = {
    "Mo": [[0.61016012, 1.26544, 1.97428762], [0.0911628054, 0.506776025, 5.89590381],
           [0.648028962, 0.0026038082, 0.113887493], [1.46634108, 0.0078433631, 0.15511434]],
    "W": [[0.924906855, 2.75554557, 3.3044006], [0.128663377, 0.765826479, 13.447117],
          [0.329973862, 1.09916444, 0.0206498883], [0.198218895, 13.5087534, 0.0338918459]],
    "S": [[1.01646916, 0.441766748, 0.121503863], [1.69181965, 0.174180288, 167.011091],
          [0.82796667, 0.0233022533, 1.18302846], [2.3034281, 0.15695415, 5.85782891]]}
# abTEM 算出的投影散射因子(= 投影位勢的傅立葉轉換,V·Å²)參考值,供國網上的 --check 核對(不需安裝 abTEM)
KREF_K = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0]            # Å⁻¹
KREF = {"Mo": [492.637, 458.16, 384.256, 311.259, 252.766, 173.34, 124.448, 92.6416, 71.3257, 56.7334, 42.4469, 33.2764],
        "W": [597.62, 563.889, 485.315, 400.462, 329.879, 235.355, 177.577, 138.811, 111.483, 91.5068, 70.2461, 55.4884],
        "S": [247.628, 236.739, 208.958, 174.161, 140.302, 88.5763, 58.3019, 41.2033, 31.0167, 24.5262, 18.3182, 14.2507]}

# ============================================================================
# 晶格(協定 §二)
# ============================================================================
MATS = ["WS2", "MoS2"]                       # 編號 0 = WS₂、1 = MoS₂
A_LAT = {"MoS2": 3.16, "WS2": 3.15}          # 晶格常數 Å
METAL = {"MoS2": "Mo", "WS2": "W"}
U_RMS = {"Mo": 0.05, "W": 0.05, "S": 0.07}   # 熱振動(Å,假設的量級;只用 Debye–Waller)
ETA = 0.1                                    # 吸收:a = exp(−η·φ)
STRAIN = 0.02                                # ε_xx、ε_yy、ε_xy 各 U(−0.02, 0.02)
ROT_MAX = 120.0                              # 旋轉 U(0°, 120°)(三重對稱)
HALO = 16                                    # 產生時每邊多 16 px 再裁
P_SV_MAX, P_DV_MAX = 0.03, 0.02              # 每個場 p_SV ~ U(0, 0.03)、p_DV ~ U(0, 0.02)
LD_LEN, LD_LINES, LD_MIN_IN = (5, 15), (1, 3), 5
MR_RAD = (4.0, 8.0)                          # 缺 S 區域的半徑(Å)
NEG_TOL, NEG_FRAC = -0.05, 0.01              # 負相位的回報門檻(描述)
FRAC_S2 = (1.0 / 3.0, 1.0 / 3.0)             # S₂ 柱的分率座標(金屬在原點)
HOLLOW = [(1.0 / 3.0, 1.0 / 3.0), (-2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, -2.0 / 3.0)]   # S₂ → 3 個最近的空心位置
LD_DIRS = [(1, 0), (0, 1), (-1, 1)]          # S₂ 子晶格的三個方向(晶格平移)
CAT_INTACT, CAT_ISO, CAT_LD, CAT_MR = 0, 1, 2, 3
# L-newdef 的正規化(審查後加入,使用者 10/9 決定):新缺陷讓約一半的可判讀柱是空缺,所有柱的中位數不再代表完整柱
# (連真值本身都判不對)→ 改用高斯混合模型(StatSTEM 式的數原子法,De Backer / Van Aert):每個場的 S₂ 柱訊號 = 三群
# (0、m/2、m;共同寬度)的混合,以 EM 估出完整柱的尺度 m。不用任何真值;缺陷少時與中位數相同。其他 4 份考卷維持中位數(協定)。
NEWDEF_NORM = "gmm"
GMM_ITERS = 100

# ============================================================================
# seed(協定 §3.3)
# ============================================================================
SEED_TRAIN_OBJ = 86_000_000                  # 晶格訓練場:+ 索引(< 100,000)
SEED_VAL = dict(obj=87_000_000, noise=87_100_000, pos=87_200_000, df=87_300_000, newdef=87_400_000, init=87_500_000)
SEED_ETA0_NOISE = 87_600_000
SEED_DOSE_CAL = 88_000_000
SEED_GAP_NOISE = 88_500_000
SEED_SMOKE = 89_000_000                      # smoke / check:89,000,000 起
SEED_SMOKE_SET = dict(obj=89_000_000, noise=89_100_000, pos=89_200_000, df=89_300_000, newdef=89_400_000, init=89_500_000)
SEED_SMOKE_TRAIN = 89_600_000
SEED_SMOKE_GAP = 89_700_000
SEED_CHECK = 89_800_000
TRAIN_ERR_BASE = 1_300_000_000               # 8b 每步的誤差 / Poisson:+ 1,000,000 × seed + 步數
TRAIN_ORDER_BASE = 1_310_000_000             # 8b 資料順序:+ seed
FINAL_LO, FINAL_HI = 98_000_000, 98_999_999  # 期末考的晶格部分:整個區間保留,一律拒絕
MY_RANGES = [(86_000_000, 86_099_999), (87_000_000, 87_699_999), (88_000_000, 88_000_999), (88_500_000, 88_502_999),
             (89_000_000, 89_999_999), (1_300_000_000, 1_309_999_999), (1_310_000_000, 1_310_000_009)]

# ============================================================================
# 考卷(協定 §四)與劑量(§3.4)
# ============================================================================
EXAMS = ["L-ideal", "L-paper", "L-combo", "L-coh", "L-newdef"]
ESPEC = {"L-ideal": dict(pos=0.0, df=0.0, coh=0.0, dose="one", newdef=False),
         "L-paper": dict(pos=0.0, df=0.0, coh=0.0, dose="paper", newdef=False),
         "L-combo": dict(pos=0.5, df=0.10, coh=0.0, dose="paper", newdef=False),
         "L-coh": dict(pos=0.0, df=0.0, coh=0.5, dose="paper", newdef=False),
         "L-newdef": dict(pos=0.0, df=0.0, coh=0.0, dose="one", newdef=True)}
ELABEL = {"L-ideal": "lattice, ideal", "L-paper": "lattice, paper dose", "L-combo": "lattice, paper dose + pos/probe errors",
          "L-coh": "lattice, paper dose + partial coherence", "L-newdef": "lattice, new defect types"}
MAIN_EXAM = "L-paper"
SEEN = ["L-ideal", "L-paper", "L-combo"]
N_PAPER = {"WS2": 1.27e4 * 3.0 ** 2, "MoS2": 4.48e5 * 0.6 ** 2}       # 論文每張的入射電子數
DOSE_MIN = 0.01                              # 訓練的劑量比下限(同 7d)
STEP_A = 8                                   # 步距 8 px

# ============================================================================
# 指標與判讀(協定 §4.3、§4.4、§六)
# ============================================================================
QS, Q_MAIN, SPEED_MIN = b7.QS, b7.Q_MAIN, b7.SPEED_MIN
R_SAMPLE = 1.5                               # 空缺判讀的取樣圓半徑(px)
RHO_HI, RHO_LO, M_INVALID = 0.75, 0.25, 0.25
SVF1_GOOD, SVF1_OK = 0.9, 0.7
FILL_X, FILL_D, FILL_MIN = 2.0, 0.10, 0.10
ROT_BINS, ROT_X = 6, 2.0
GAP_X = 1.5
FRC_N, FRC_ALPHA, FRC_PMIN = 48, 0.25, 1e-6
BOOT_N, BOOT_SEED = 2000, 2026
FIG_SEED = 2026
FAIL_C_MIN, FAIL_C_X = 10.0, 1.5
FAIL_SR = a7.FAIL_VAL                        # nerr_sr 的失敗值(同階段七)
MCH = a7.MCH

# ============================================================================
# 8b(協定 §五)
# ============================================================================
BASE = d7.NEW_E                              # P6B4e8
NEW_L, NEW_C = "P6B4e8L", "P6B4e8C"
FROZEN = ["P6B4", "P6B4r", BASE]
NETS = FROZEN + [NEW_C, NEW_L]
LR_8B = 2e-4                                 # 所有權重(含 b_k)
LAT_POOL_N = a6.N_TRAIN                      # 10 萬(QUICK:64)
TASKS = [("env", e) for e in EXAMS] + [("L", s) for s in SEEDS] + [("C", s) for s in SEEDS]
SMOKE_DIR = RUN_ROOT / f"scan8_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
DOSE_JSON = RUN_ROOT / f"scan8_dose{SUF}.json"
SMOKE_FIELDS = a7.SMOKE_FIELDS
SMOKE_TRAIN_N = c7.SMOKE_TRAIN_N
SMOKE_EPOCHS = 1
ETA0_N = 4 if QUICK else 128
GAP_N = 4 if QUICK else 512
FIG_DIR = Path("figs_scan8")
COL = {"P6B4": "#9a9994", "P6B4r": "#e8a33d", BASE: "#2a78d6", NEW_C: "#8a5cd1", NEW_L: "#1baf7a", "iter": "#0b0b0b"}
TAG = {NEW_L: "P6B4e8-L", NEW_C: "P6B4e8-C"}

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and d7.all_ok()


md5, dump_json, absdiff = a7.md5, a7.dump_json, a7.absdiff


def me_md5():
    return md5(Path(__file__).resolve())


def tg(nm):
    return TAG.get(nm, nm)


def env_json(e, root=None):
    return (root or RUN_ROOT) / f"scan8_env_{e}{SUF}.json"


def final_json(root=None):
    return (root or RUN_ROOT) / f"scan8{SUF}.json"


def model_dir(kind, s, mroot=None):
    name = NEW_L if kind == "L" else NEW_C
    return (mroot / f"{name}_s{s}") if mroot is not None else s5c.run_dir(name, PROBE, s, a6.N_TRAIN)


def guard(seeds):
    """期末考保護:98,000,000–98,999,999 的任何 seed 一律拒絕(協定 §3.3)。"""
    bad = [int(x) for x in seeds if FINAL_LO <= int(x) <= FINAL_HI]
    if bad:
        raise ValueError(f"seed {bad[:3]} 落在期末考保留的區間 {FINAL_LO:,}–{FINAL_HI:,}:拒絕")


# ============================================================================
# 物理:Δx、投影散射因子
# ============================================================================
def pixel_size(pr, W):
    """Δx = k_a /(W·α/λ)(Å/px;同階段七 §十二 的換算)。"""
    return float(pr["meta"]["ka"]) / (W * ALPHA / LAMBDA)


def kirk_F(el, k2):
    """投影散射因子 F(k)(V·Å²;= 投影位勢的 2D 傅立葉轉換,V_proj(r) = ∫ F(k) e^{2πik·r} d²k)。
    k2:k² [Å⁻²](float64)。同 abTEM 的 Kirkland projected_scattering_factor(係數換算同 scaled_parameters)。"""
    p = torch.tensor(KIRKLAND[el], dtype=torch.float64, device=k2.device)
    a = math.pi * p[0] / KAPPA
    b = 2.0 * math.pi * torch.sqrt(p[1])
    c = math.pi ** 1.5 * p[2] / p[3] ** 1.5 / KAPPA
    d = math.pi ** 2 / p[3]
    f = torch.zeros_like(k2)
    for i in range(3):
        f = f + 4 * math.pi * a[i] / (4 * math.pi ** 2 * k2 + b[i] ** 2) \
            + torch.sqrt(math.pi / d[i]) * c[i] * math.pi / d[i] * torch.exp(-math.pi ** 2 * k2 / d[i])
    return f


def element_tables(G, dx, dev):
    """產生網格(G × G、Δx)上各元素的 F(k)·DW(k)·σ/Δx²(complex64 [G, G])。乘上結構因子再逆 FFT = 相位(rad)。
    Debye–Waller:exp(−2π²u²k²)。頻帶 = 方形 Nyquist(k ≤ 1/(2Δx))。"""
    k = torch.fft.fftfreq(G, d=dx, dtype=torch.float64, device=dev)                   # Å⁻¹
    ky, kx = torch.meshgrid(k, k, indexing="ij")
    k2 = ky ** 2 + kx ** 2
    out = {}
    for el in ("Mo", "W", "S"):
        f = kirk_F(el, k2) * torch.exp(-2 * math.pi ** 2 * U_RMS[el] ** 2 * k2) * (SIGMA / dx ** 2)
        out[el] = f.to(torch.complex64)
    return out


# ============================================================================
# 晶格產生器(協定 §2.1–2.3)
# ============================================================================
R_FIX = 34                                   # 晶格整數座標的固定範圍 [−R_FIX, R_FIX]²:缺陷的亂數與產生區大小無關(同一個 seed 在任何場大小都是同一片晶格)


def index_grid(G, dx):
    """晶格的整數座標 (n1, n2) ∈ [−R_FIX, R_FIX]²(固定;檢查足以涵蓋 G × G 的產生區,含應變)。回傳 [A, 2](float64)與 R_FIX。"""
    a_min = min(A_LAT.values()) / dx * (1 - 2 * STRAIN)
    need = int(math.ceil((G / math.sqrt(2)) / (math.sqrt(3) / 2 * a_min))) + 2
    if need > R_FIX:
        raise ValueError(f"產生區 {G} px 需要 R = {need} > R_FIX = {R_FIX}")
    r = torch.arange(-R_FIX, R_FIX + 1, dtype=torch.float64)
    n1, n2 = torch.meshgrid(r, r, indexing="ij")
    return torch.stack([n1.flatten(), n2.flatten()], 1), R_FIX


def draw_params(seed, nidx, material=None, fixed=None):
    """一個場的亂數(CPU,固定順序):材料、旋轉、平移、應變、p_SV、p_DV、每個 S 原子 / S₂ 柱的空缺。"""
    guard([seed])
    gen = torch.Generator().manual_seed(int(seed))
    if material is None:
        m = int(float(torch.rand(1, generator=gen)) < 0.5)                              # 0 = WS₂、1 = MoS₂
    else:
        m = MATS.index(material)
    th = float(torch.rand(1, generator=gen)) * ROT_MAX
    t = torch.rand(2, generator=gen, dtype=torch.float64)
    e = (torch.rand(3, generator=gen, dtype=torch.float64) * 2 - 1) * STRAIN          # ε_yy、ε_xx、ε_xy
    psv = float(torch.rand(1, generator=gen)) * P_SV_MAX
    pdv = float(torch.rand(1, generator=gen)) * P_DV_MAX
    fixed = fixed or {}
    th = fixed.get("theta", th)
    if "t" in fixed:
        t = torch.tensor(fixed["t"], dtype=torch.float64)
    if "e" in fixed:
        e = torch.tensor(fixed["e"], dtype=torch.float64)
    psv, pdv = fixed.get("psv", psv), fixed.get("pdv", pdv)
    rm = (torch.rand(nidx, 2, generator=gen) < psv).sum(1)
    dv = torch.rand(nidx, generator=gen) < pdv
    nS = 2 - rm
    nS[dv] = 0
    cat = torch.where(nS < 2, CAT_ISO, CAT_INTACT)
    return dict(mat=m, theta=th, t=t, e=e, psv=psv, pdv=pdv, nS=nS.long(), cat=cat.long())


def geom_of(prm, idx, F, dx):
    """金屬柱、S₂ 柱(場座標 (y, x);像素中心為整數,場中心 (F − 1)/2)與每個 S₂ 柱的 3 個空心位置。"""
    a = A_LAT[MATS[prm["mat"]]] / dx
    a1 = torch.tensor([0.0, a], dtype=torch.float64)                                  # (y, x)
    a2 = torch.tensor([a * math.sqrt(3) / 2, a / 2], dtype=torch.float64)
    th = math.radians(prm["theta"])
    Rm = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.float64)
    e = prm["e"]
    E = torch.tensor([[float(e[0]), float(e[2])], [float(e[2]), float(e[1])]], dtype=torch.float64)
    M = (torch.eye(2, dtype=torch.float64) + E) @ Rm
    c = (F - 1) / 2.0
    t = prm["t"]
    L = torch.stack([a1, a2])                                                           # [2, 2]

    def place(fr):
        v = (idx + t[None] + torch.tensor(fr, dtype=torch.float64)[None]) @ L          # [A, 2]
        return v @ M.T + c

    metal = place((0.0, 0.0))
    s2 = place(FRAC_S2)
    hol = torch.stack([(torch.tensor(o, dtype=torch.float64)[None] @ L) @ M.T for o in HOLLOW], 1)   # [1, 3, 2]
    return metal, s2, s2[:, None, :] + hol, M


def _in_u(p, U):
    """位置(場座標)四捨五入後落在 U 內。p:[N, 2];U:[F, F] bool(CPU)。"""
    F = U.shape[-1]
    yy, xx = torch.round(p[:, 0]).long(), torch.round(p[:, 1]).long()
    ok = (yy >= 0) & (yy < F) & (xx >= 0) & (xx < F)
    out = torch.zeros(p.shape[0], dtype=torch.bool)
    out[ok] = U[yy[ok], xx[ok]]
    return out


def add_newdef(prm, s2, idx, R, U, seed, dx):
    """只測試的新缺陷(L-newdef;協定 §2.3):線缺陷 1–3 條(沿 S₂ 子晶格的三個方向之一,長 5–15 個位置,每個位置少 1 或 2 個 S,
    至少 5 個位置在 U 內)與缺 S 區域 1 塊(半徑 4–8 Å,中心在 U 內,區內 S 全部移除)。優先順序:缺 S 區域 > 線缺陷 > DV > SV。"""
    guard([seed])
    gen = torch.Generator().manual_seed(int(seed))
    nS, cat = prm["nS"].clone(), prm["cat"].clone()
    side = 2 * R + 1
    inU = _in_u(s2, U)
    cand = torch.where(inU)[0]
    n_lines = int(torch.randint(LD_LINES[0], LD_LINES[1] + 1, (1,), generator=gen))
    lines = []
    for _ in range(n_lines):
        for _try in range(500):
            st = int(cand[int(torch.randint(0, len(cand), (1,), generator=gen))])
            dvec = LD_DIRS[int(torch.randint(0, 3, (1,), generator=gen))]
            ln = int(torch.randint(LD_LEN[0], LD_LEN[1] + 1, (1,), generator=gen))
            i1, i2 = st // side, st % side
            sites = [(i1 + k * dvec[0], i2 + k * dvec[1]) for k in range(ln)]
            sites = [p1 * side + p2 for p1, p2 in sites if 0 <= p1 < side and 0 <= p2 < side]
            if len(sites) == ln and int(inU[sites].sum()) >= LD_MIN_IN:
                break
        else:
            raise RuntimeError("線缺陷:500 次都找不到至少 5 個位置在 U 內的線")
        miss = torch.randint(1, 3, (ln,), generator=gen)
        nS[sites] = 2 - miss
        cat[sites] = CAT_LD
        lines.append({"sites": sites, "dir": list(dvec), "len": ln, "in_u": int(inU[sites].sum())})
    upix = torch.nonzero(U)
    p0 = upix[int(torch.randint(0, len(upix), (1,), generator=gen))].double() + (torch.rand(2, generator=gen, dtype=torch.float64) - 0.5)
    rad = (MR_RAD[0] + (MR_RAD[1] - MR_RAD[0]) * float(torch.rand(1, generator=gen))) / dx
    inside = ((s2 - p0[None]) ** 2).sum(1).sqrt() <= rad
    nS[inside] = 0
    cat[inside] = CAT_MR
    return nS, cat, {"lines": lines, "mr_center": p0.tolist(), "mr_radius_px": rad, "mr_n": int(inside.sum())}


@torch.no_grad()
def make_lattice(seeds, F, dev, materials=None, newdef_seeds=None, U=None, eta=ETA, info=False, fixed=None, dx=None, chunk=None):
    """晶格場 [n, 2, F, F](振幅、相位;float32、dev)與描述。相位 = σ·V_proj(頻帶限制;頻率空間的結構因子 → 逆 FFT),
    產生區 = F + 2·HALO,再裁中央;相位截在 ≥ 0(真空 = 0,同現有物體的慣例),振幅 = exp(−η·φ)。
    info=True 另回傳每個場的 S₂ 柱(位置、S 數、類別)與空心位置(場座標)。"""
    guard(seeds)
    if newdef_seeds is not None:
        guard(newdef_seeds)
    if dx is None:
        raise ValueError("make_lattice 需要 dx(用 pixel_size(pr, W))")
    n = len(seeds)
    G = F + 2 * HALO
    idx, R = index_grid(G, dx)
    A = idx.shape[0]
    tabs = element_tables(G, dx, dev)
    kp = torch.fft.fftfreq(G, dtype=torch.float64, device=dev)                         # 週期 / px
    chunk = chunk or (32 if dev.type == "cpu" else 128)
    out = torch.empty(n, 2, F, F, dtype=torch.float32, device=dev)
    meta = {"mats": [], "theta": [], "psv": [], "pdv": [], "cols": [] if info else None, "newdef": [] if newdef_seeds is not None else None,
            "neg_min": float("inf"), "neg_frac": [], "phase_max": 0.0, "n_cols_u": 0}
    for i0 in range(0, n, chunk):
        ii = list(range(i0, min(n, i0 + chunk)))
        Ym, Ys, Wm, Ws, mats = [], [], [], [], []
        for i in ii:
            prm = draw_params(seeds[i], A, None if materials is None else materials[i], fixed)
            metal, s2, hol, M = geom_of(prm, idx, F, dx)
            nS, cat = prm["nS"], prm["cat"]
            if newdef_seeds is not None:
                nS, cat, nd = add_newdef(prm, s2, idx, R, U, newdef_seeds[i], dx)
                meta["newdef"].append(nd)
            gm, gs = metal + HALO, s2 + HALO
            inm = ((gm >= 0) & (gm < G)).all(1)
            ins = ((gs >= 0) & (gs < G)).all(1) & (nS > 0)
            Ym.append(gm[inm])                                                      # 只留產生區內的原子(結構因子的計算量)
            Ys.append(gs[ins])
            Wm.append(torch.ones(int(inm.sum()), dtype=torch.float64))
            Ws.append(nS[ins].double())
            mats.append(prm["mat"])
            meta["mats"].append(prm["mat"])
            meta["theta"].append(prm["theta"])
            meta["psv"].append(prm["psv"])
            meta["pdv"].append(prm["pdv"])
            if info:
                keep = ((s2 >= -0.5) & (s2 < F - 0.5)).all(1)
                meta["cols"].append({"pos": s2[keep].float(), "hol": hol[keep].float(), "nS": nS[keep], "cat": cat[keep]})

        def sfac(P, w):
            na = max(1, max(len(x) for x in P))                                         # 補齊到同一個原子數(權重 0)
            P = torch.stack([torch.cat([x, torch.zeros(na - len(x), 2, dtype=x.dtype)]) for x in P]).to(dev)       # [b, A', 2]
            w = torch.stack([torch.cat([x, torch.zeros(na - len(x), dtype=x.dtype)]) for x in w]).to(dev)         # [b, A']
            ay = (-2 * math.pi) * torch.remainder(P[..., 0, None] * kp, 1.0)            # [b, A, G](float64 → 角度)
            ax = (-2 * math.pi) * torch.remainder(P[..., 1, None] * kp, 1.0)
            Ey = torch.polar(w[..., None].float().expand(ay.shape).contiguous(), ay.float())
            Ex = torch.polar(torch.ones_like(ax, dtype=torch.float32), ax.float())
            return torch.bmm(Ey.transpose(1, 2), Ex)                                    # [b, G(ky), G(kx)]

        Sm, Ss = sfac(Ym, Wm), sfac(Ys, Ws)
        mt = torch.tensor(mats, device=dev)
        Fm = torch.where(mt[:, None, None] == 0, tabs["W"][None], tabs["Mo"][None])
        ph = torch.fft.ifft2(Fm * Sm + tabs["S"][None] * Ss).real[:, HALO:HALO + F, HALO:HALO + F]
        meta["neg_min"] = min(meta["neg_min"], float(ph.min()))
        meta["neg_frac"] += [float(v) for v in (ph < NEG_TOL).float().mean((1, 2))]
        ph = ph.clamp_min(0.0)
        meta["phase_max"] = max(meta["phase_max"], float(ph.max()))
        out[ii[0]:ii[-1] + 1, 1] = ph
        out[ii[0]:ii[-1] + 1, 0] = torch.exp(-eta * ph)
    return out, meta


# ============================================================================
# 量測(協定 §三;同階段七的前向模型,劑量可逐場不同)
# ============================================================================
def lattice_seeds(s, smoke=False):
    base = SEED_SMOKE_SET if smoke else SEED_VAL
    return {k: (v if k in ("obj", "newdef") else v + 1000 * s) for k, v in base.items()}


def draw_errors_L(sd, n, dev):
    """位置誤差的標準常態亂數 z [n, 49, 2](截在 ± 3)與離焦符號 [n](± 1);同 scan_7a.draw_errors 的抽法,seed 為晶格考卷的。"""
    gz = torch.Generator().manual_seed(sd["pos"])
    z = torch.randn(n, a6.NPOS * a6.NPOS, 2, generator=gz).clamp(-a7.Z_CLIP, a7.Z_CLIP)
    gd = torch.Generator().manual_seed(sd["df"])
    sign = torch.where(torch.rand(n, generator=gd) < 0.5, -1.0, 1.0)
    return z.to(dev), sign.to(dev)


@torch.no_grad()
def measure_L(fields, geo, cp, bs, noise_seed, spec, z, sign, ops, dose, poisson=True):
    """49 張繞射圖。真的條件:探針 = 名目(或離焦 ± df)、平移 pos·z(頻率空間)、部分同調 = 3 × 3 Gauss–Hermite 模式的非同調疊加;
    期望強度同 src.physics.forward_measure(含 beamstop);計數 = Poisson(劑量比 × 期望)/ 劑量比(劑量已知,逐場)。
    雜訊用專用的 generator(seed 固定、依位置順序)。poisson=False → 回傳期望強度。"""
    from src.physics import forward_measure
    W, P, dev = geo.W, geo.pr["P"], fields.device
    n = fields.shape[0]
    if spec["df"] > 0:
        pp, pm = a7.true_probe_pair(geo.pr, spec["df"], dev)
        Pt = torch.where(sign[:, None, None] > 0, pp, pm)
    else:
        Pt = P.expand(n, W, W)
    if spec["coh"] > 0:
        if spec["pos"] or spec["df"]:
            raise ValueError("部分同調只與理想的位置 / 探針組合(同 7b)")
        modes = [(ops.shift(P, torch.tensor(dy, device=dev), torch.tensor(dx_, device=dev)).expand(n, W, W), w)
                 for dy, dx_, w in b7.coh_modes(spec["coh"])]
    else:
        modes = [(Pt, 1.0)]
    cp0 = copy.deepcopy(cp)
    cp0.add_poisson = False
    gen = torch.Generator(device=dev).manual_seed(int(noise_seed))
    dd = dose.to(dev).float()[:, None, None]
    counts = []
    for j, (y0, x0) in enumerate(geo.starts):
        Ow = torch.polar(fields[:, 0, y0:y0 + W, x0:x0 + W], fields[:, 1, y0:y0 + W, x0:x0 + W])
        I = None
        for Pm, w in modes:
            Pj = ops.shift(Pm, spec["pos"] * z[:, j, 0], spec["pos"] * z[:, j, 1]) if spec["pos"] > 0 else Pm
            psi = Pj * Ow
            v = w * forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), bs, cp0) if len(modes) > 1 else \
                forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), bs, cp0)
            I = v if I is None else I + v
        if poisson:
            I = torch.poisson((I * dd).clamp_min(0), generator=gen) / dd * bs
        counts.append(I)
    return counts


# ============================================================================
# 劑量校準(協定 §3.4)
# ============================================================================
def dose_of_seed(s, dev):
    """N_inc = photons_per_pix × W² / ref_energy × Σ|P|²(透明物體、beamstop 之前的總計數期望;用量測實際使用的 cp)。"""
    import probe_4_2b as pb
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[PROBE]
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS["3x3s8"][2])
    W = cfg.canvas
    n_inc = cp.photons_per_pix * W * W / cp.ref_energy * float((pr["P"].abs() ** 2).sum())
    return {"N_inc": n_inc, "f_W": N_PAPER["WS2"] / n_inc, "f_Mo": N_PAPER["MoS2"] / n_inc, "ref_energy": cp.ref_energy,
            "photons_per_pix": cp.photons_per_pix}


def dose_all(dev):
    D = {str(s): dose_of_seed(s, dev) for s in SEEDS}
    fmin = min(min(v["f_W"], v["f_Mo"]) for v in D.values())
    fmax = max(max(v["f_W"], v["f_Mo"]) for v in D.values())
    if fmax > 1:
        raise SystemExit(f"❌ 劑量比 f = {fmax:.3g} > 1(論文的電子數高於基準):停止、回報(協定 §3.4)")
    lo = fmin / 2 if fmin < DOSE_MIN else DOSE_MIN
    return {"seeds": D, "f_min": fmin, "f_max": fmax, "lat_dose_lo": lo, "out_of_range": bool(fmin < DOSE_MIN),
            "n_paper": N_PAPER}


_DOSE = {}


def load_dose():
    if "d" not in _DOSE:
        if not DOSE_JSON.exists():
            raise SystemExit(f"❌ 找不到 {DOSE_JSON}:先跑 python scan_8.py --check(劑量校準寫在這個檔)")
        _DOSE["d"] = json.load(open(DOSE_JSON))
    return _DOSE["d"]


def dose_vec(s, spec, mats):
    if spec["dose"] == "one":
        return torch.ones(len(mats))
    D = load_dose()["seeds"][str(s)]
    return torch.tensor([D["f_W"] if m == 0 else D["f_Mo"] for m in mats], dtype=torch.float64).float()


# ============================================================================
# 一份考卷的資料(同 scan_7a.setup_cond 的介面:迭代法與網路流程可直接使用)
# ============================================================================
def setup_exam(s, dev, exam, n, smoke=False, eta=ETA, obj_seeds=None, noise_seed=None, mats=None, newdef=None):
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[PROBE]
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS["3x3s8"][2])
    bs = beamstop_mask(cfg, device=dev)
    geo = a6.Geom(cfg, pr, dev)
    ops = a7.Ops(pr, geo.W, dev)
    dx = pixel_size(pr, geo.W)
    spec = dict(ESPEC[exam])
    sd = lattice_seeds(s, smoke)
    guard(list(sd.values()) + ([noise_seed] if noise_seed is not None else []))
    if obj_seeds is None:
        obj_seeds = [sd["obj"] + i for i in range(n)]
        if mats is None:
            mats = [MATS[i % 2] for i in range(n)]                                    # 分層:偶數 WS₂、奇數 MoS₂(256 / 256)
    nd = spec["newdef"] if newdef is None else newdef
    nds = [sd["newdef"] + i for i in range(n)] if nd else None
    Ucpu = geo.U.cpu()
    fields, meta = make_lattice(obj_seeds, geo.F, dev, materials=mats, newdef_seeds=nds, U=Ucpu, eta=eta, info=True, dx=dx)
    O = torch.polar(fields[:, 0], fields[:, 1])
    z, sign = draw_errors_L(sd, n, dev)
    mt = torch.tensor(meta["mats"])
    dv = dose_vec(s, spec, meta["mats"])
    ns = sd["noise"] if noise_seed is None else noise_seed
    counts = measure_L(fields, geo, cp, bs, ns, spec, z, sign, ops, dv)
    vac = Vac(meta["cols"], geo.F, Ucpu, fields[:, 1].cpu(), norm=NEWDEF_NORM if nd else "all")
    m = geo.U.to(O.real.dtype)
    cs = (O * m).sum((1, 2)) / m.sum()
    Ec = (((O - cs[:, None, None]).abs() ** 2) * m).sum((1, 2))
    E = ((O.abs() ** 2) * m).sum((1, 2))
    return dict(cfg=cfg, pr=pr, cp=cp, bs=bs, geo=geo, O=O, counts=counts, init=sd["init"], n=n, fields=fields, z=z, sign=sign,
                ops=ops, spec=spec, cond=exam, mats=mt, dose=dv, vac=vac, meta=meta, Ec=Ec, E=E, dx=dx, noise=ns,
                obj_seeds=obj_seeds)


# ============================================================================
# 空缺判讀(協定 §4.3 (b))
# ============================================================================
_OFF = None


def _circle(p, F, U):
    """半徑 R_SAMPLE 的取樣圓:每個中心最多 16 個像素(4 × 4 候選)。回傳 (平攤索引 [N, 16], 權重 [N, 16], 完整落在 U 內 [N])。"""
    base = torch.floor(p - R_SAMPLE).long()                                             # [N, 2]
    oy, ox = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    off = torch.stack([oy.flatten(), ox.flatten()], 1)                                  # [16, 2]
    pix = base[:, None, :] + off[None]                                                  # [N, 16, 2]
    d2 = ((pix.double() - p[:, None, :].double()) ** 2).sum(-1)
    w = (d2 <= R_SAMPLE ** 2 + 1e-9)
    inb = (pix >= 0).all(-1) & (pix < F).all(-1)
    yy, xx = pix[..., 0].clamp(0, F - 1), pix[..., 1].clamp(0, F - 1)
    inu = U[yy, xx] & inb
    ok = (~w | inu).all(1) & (w.sum(1) > 0)
    return yy * F + xx, w.float(), ok


class Vac:
    """一組場的真值 S₂ 柱(取樣圓完整落在 U 內者;協定 §4.3 (b) 2;空心位置的取樣圓只要求在場內)與判讀用的索引。
    真值的 S 數 → 標籤 0 完整、1 SV、2 DV。norm:完整柱的尺度 m 怎麼估 —— "all" = 所有可判讀柱的中位數(協定);
    "gmm" = 三群(0、m/2、m)的高斯混合模型(只用於 L-newdef,見 NEWDEF_NORM);"background" = 排除新缺陷的柱再取中位數(描述用)。"""

    def __init__(self, cols, F, U, ph_true, norm="all"):
        fid, cp, cw, hp, hw, lab, cat, slot, pos = [], [], [], [], [], [], [], [], []
        self.n, self.F = len(cols), F
        Uall = torch.ones_like(U)
        for i, c in enumerate(cols):
            pc, wc, okc = _circle(c["pos"], F, U)
            hs = [_circle(c["hol"][:, k], F, Uall) for k in range(3)]
            ok = okc & hs[0][2] & hs[1][2] & hs[2][2]
            k = torch.where(ok)[0]
            fid.append(torch.full((len(k),), i, dtype=torch.long))
            cp.append(pc[k])
            cw.append(wc[k])
            hp.append(torch.stack([x[0][k] for x in hs], 1))
            hw.append(torch.stack([x[1][k] for x in hs], 1))
            lab.append(2 - c["nS"][k])
            cat.append(c["cat"][k])
            slot.append(torch.arange(len(k)))
            pos.append(c["pos"][k])
        self.fid, self.cp, self.cw = torch.cat(fid), torch.cat(cp), torch.cat(cw)
        self.pos = torch.cat(pos)
        self.hp, self.hw = torch.cat(hp), torch.cat(hw)
        self.lab, self.cat, self.slot = torch.cat(lab).long(), torch.cat(cat).long(), torch.cat(slot)
        self.cmax = int(self.slot.max()) + 1 if len(self.slot) else 1
        self.count = torch.bincount(self.fid, minlength=self.n)
        self.norm = norm
        self.nmask = (self.cat < CAT_LD) if norm == "background" else torch.ones_like(self.cat, dtype=torch.bool)
        self.s_true, self.m_true = self.signal(ph_true)

    def signal(self, ph):
        """柱的訊號 s = 圓內平均相位 − 3 個空心位置的平均相位;每個場的中位數 m。ph:[n, F, F](CPU)。"""
        flat = ph.reshape(ph.shape[0], -1).float()
        v = (flat[self.fid[:, None], self.cp] * self.cw).sum(1) / self.cw.sum(1)
        hb = (flat[self.fid[:, None, None], self.hp] * self.hw).sum(2) / self.hw.sum(2)
        s = v - hb.mean(1)
        M = torch.full((self.n, self.cmax), float("nan"))
        if self.norm == "gmm":
            M[self.fid, self.slot] = s
            return s, gmm_scale(M)
        k = self.nmask
        M[self.fid[k], self.slot[k]] = s[k]
        return s, torch.nanmedian(M, 1).values

    def read(self, ph, fin=None):
        """判讀(正規化不用真值標籤:以該場所有 S₂ 柱的中位數 m 正規化)。無效規則:m ≤ 0.25 × 真值的 m(或非有限)→ 無法判讀,
        所有柱計為判錯(SV / DV 記為漏掉 = 判成完整;完整柱記為誤報 = 判成 SV)。回傳 (每個柱的判定, 可判讀 [n], m [n])。"""
        s, m = self.signal(ph)
        ok = torch.isfinite(m) & (m > M_INVALID * self.m_true)
        if fin is not None:
            ok &= fin
        rho = s / m[self.fid]
        pred = torch.where(rho >= RHO_HI, 0, torch.where(rho >= RHO_LO, 1, 2))
        bad = ~ok[self.fid] | ~torch.isfinite(rho)
        pred = torch.where(bad, torch.where(self.lab == 0, 1, 0), pred)
        return pred.long(), ok, m

    def confusion(self, pred):
        """每個場的 3 × 3 混淆矩陣 [n, 3, 3](列 = 真值、欄 = 判定)與依類別(0 = 孤立 / 背景、1 = 新缺陷 LD / MR)[n, 2, 3, 3]。"""
        C = torch.bincount(self.fid * 9 + self.lab * 3 + pred, minlength=self.n * 9).view(self.n, 3, 3)
        new = (self.cat >= CAT_LD).long()
        Cc = torch.bincount(self.fid * 18 + new * 9 + self.lab * 3 + pred, minlength=self.n * 18).view(self.n, 2, 3, 3)
        return C, Cc


def gmm_scale(M, iters=None):
    """每個場(列)的完整柱尺度 m:柱訊號 = 三群高斯的混合,平均固定在 0、m/2、m(S 數 0、1、2;投影位勢與 S 數成正比)、共同寬度 σ、
    權重自由;EM(由第 90 百分位起算)。M:[n, C](nan = 沒有柱)。回傳 [n](該場沒有柱 → nan)。"""
    iters = iters or GMM_ITERS
    X = M.double()
    ok = torch.isfinite(X)
    Xz = torch.where(ok, X, torch.zeros_like(X))
    cnt = ok.sum(1).clamp_min(1).double()
    m = torch.nanquantile(X, 0.9, dim=1)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    sig = (m.abs() / 6).clamp_min(1e-6)
    w = torch.full((X.shape[0], 3), 1.0 / 3, dtype=torch.float64)
    f = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    for _ in range(iters):
        mu = m[:, None] * f[None]                                                       # [n, 3]
        ll = -0.5 * ((Xz[:, :, None] - mu[:, None, :]) / sig[:, None, None]) ** 2 + torch.log(w.clamp_min(1e-12))[:, None, :]
        r = torch.softmax(ll, -1) * ok[:, :, None]
        den = (r[..., 1] * 0.25 + r[..., 2]).sum(1)
        m = torch.where(den > 1e-12, (r[..., 1] * Xz * 0.5 + r[..., 2] * Xz).sum(1) / den.clamp_min(1e-12), m)
        mu = m[:, None] * f[None]
        sig = torch.sqrt((r * (Xz[:, :, None] - mu[:, None, :]) ** 2).sum((1, 2)) / cnt).clamp_min(1e-6)
        sig = torch.maximum(sig, 0.03 * m.abs())
        w = r.sum(1) / cnt[:, None]
    return torch.where(ok.any(1), m, torch.full_like(m, float("nan"))).float()


def vac_stats(C):
    """由合併的 3 × 3 混淆矩陣(列 = 真值、欄 = 判定)算 SV / DV 的精確率、召回率、F1;二元 F1;誤報率;補回率。"""
    C = np.asarray(C, float).reshape(3, 3)

    def prf(k):
        tp = C[k, k]
        fp = C[:, k].sum() - tp
        fn = C[k].sum() - tp
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)

    sp, sr, sf = prf(1)
    dp, dr, df = prf(2)
    tpb = C[1:, 1:].sum()
    fpb = C[0, 1:].sum()
    fnb = C[1:, 0].sum()
    pb_ = tpb / (tpb + fpb) if tpb + fpb > 0 else 0.0
    rb_ = tpb / (tpb + fnb) if tpb + fnb > 0 else 0.0
    nv = C[1:].sum()
    return {"sv_p": sp, "sv_r": sr, "sv_f1": sf, "dv_p": dp, "dv_r": dr, "dv_f1": df,
            "bin_f1": 2 * pb_ * rb_ / (pb_ + rb_) if pb_ + rb_ > 0 else 0.0,
            "fpr": C[0, 1:].sum() / C[0].sum() if C[0].sum() > 0 else 0.0,
            "fill": C[1:, 0].sum() / nv if nv > 0 else float("nan"), "n_sv": int(C[1].sum()), "n_dv": int(C[2].sum()),
            "n_intact": int(C[0].sum()), "conf": C.astype(int).tolist()}


# ============================================================================
# 對齊與指標(協定 §4.3)
# ============================================================================
@torch.no_grad()
def align_full(est, O, U):
    """= scan_7a.align_err(整體相位 + 平移 + 相位斜坡;同一套格點與「與不做取小者」的規則),另回傳對齊後的估計(含整體相位)。
    回傳 (v_ph, v_sh, v_sr, t, q, 對齊後的估計)。"""
    F = O.shape[-1]
    grid, Ey, ky, kx, qg, Qy, yc = a7._sh_mats(F, O.device)
    m = U.to(O.real.dtype)
    out = [[], [], [], [], [], []]
    S, Sq = grid.numel(), qg.numel()
    for i in range(0, O.shape[0], MCH):
        e, o = est[i:i + MCH], O[i:i + MCH]
        A = torch.fft.fft2(o * m)
        B = torch.fft.fft2(e)
        C = (Ey @ (A.conj() * B) @ Ey.T).abs()
        idx = C.flatten(1).argmax(1)
        t = torch.stack([grid[idx // S], grid[idx % S]], -1)
        ph = 2 * math.pi * (ky * t[:, 0, None, None] + kx * t[:, 1, None, None])
        es = torch.fft.ifft2(B * torch.polar(torch.ones_like(ph), ph))
        v0 = a6.per_sample(e, o, U)
        vs = a6.per_sample(es, o, U)
        b1 = vs < v0
        e1 = torch.where(b1[:, None, None], es, e)
        v1 = torch.where(b1, vs, v0)
        t1 = torch.where(b1[:, None], t, torch.zeros_like(t))
        a = (o * m).conj() * e1
        Cq = (Qy @ a @ Qy.T).abs()
        iq = Cq.flatten(1).argmax(1)
        q = torch.stack([qg[iq // Sq], qg[iq % Sq]], -1)
        er = e1 * a7._ramp_field(q, yc)
        vr = a6.per_sample(er, o, U)
        b2 = vr < v1
        e2 = torch.where(b2[:, None, None], er, e1)
        cc = ((o * m) * (e2 * m).conj()).sum((1, 2))
        eal = e2 * torch.exp(1j * torch.angle(cc))[:, None, None]
        for lst, v in zip(out, (v0, v1, torch.where(b2, vr, v1), t1, torch.where(b2[:, None], q, torch.zeros_like(q)), eal)):
            lst.append(v)
    return tuple(torch.cat(x) for x in out)


def frc_block(U):
    """U 內接的 FRC_N × FRC_N 方塊(以 U 外框的中心為中心)。回傳 (y0, x0, 是否完整落在 U 內)。"""
    ys, xs = torch.where(U)
    cy, cx = (int(ys.min()) + int(ys.max()) + 1) // 2, (int(xs.min()) + int(xs.max()) + 1) // 2
    y0, x0 = cy - FRC_N // 2, cx - FRC_N // 2
    return y0, x0, bool(U[y0:y0 + FRC_N, x0:x0 + FRC_N].all())


_FW = {}


def _frc_win(dev):
    if str(dev) not in _FW:
        n = FRC_N
        x = torch.arange(n, dtype=torch.float64)
        w = torch.ones(n, dtype=torch.float64)
        L = FRC_ALPHA * (n - 1) / 2
        lo = x < L
        hi = x > (n - 1) - L
        w[lo] = 0.5 * (1 + torch.cos(math.pi * (x[lo] / L - 1)))
        w[hi] = 0.5 * (1 + torch.cos(math.pi * ((x[hi] - (n - 1)) / L + 1)))
        k = torch.fft.fftfreq(n, d=1.0 / n)
        ky, kx = torch.meshgrid(k, k, indexing="ij")
        ring = torch.round(torch.sqrt(ky ** 2 + kx ** 2)).long()
        _FW[str(dev)] = ((w[:, None] * w[None, :]).float().to(dev), ring.to(dev))
    return _FW[str(dev)]


@torch.no_grad()
def frc_sums(a, b, y0, x0):
    """FRC 的累加量(合併多個場用):方塊、減平均、Tukey 窗;環寬 1 個頻率格點。a, b:[n, F, F] complex。回傳 (互譜實部, 功率 a, 功率 b) 各 [R]。"""
    win, ring = _frc_win(a.device)
    nr = int(ring.max()) + 1

    def fx(z):
        z = z[:, y0:y0 + FRC_N, x0:x0 + FRC_N]
        z = z - z.mean((1, 2), keepdim=True)
        return torch.fft.fft2(z * win)

    A, B = fx(a), fx(b)
    rr = ring.flatten()
    num = torch.zeros(nr, dtype=torch.float64, device=a.device).index_add_(0, rr, (A * B.conj()).real.sum(0).flatten().double())
    pa = torch.zeros(nr, dtype=torch.float64, device=a.device).index_add_(0, rr, (A.abs() ** 2).sum(0).flatten().double())
    pb = torch.zeros(nr, dtype=torch.float64, device=a.device).index_add_(0, rr, (B.abs() ** 2).sum(0).flatten().double())
    return num.cpu(), pa.cpu(), pb.cpu()


def frc_curve(num, pa, pb):
    """pooled FRC;功率 < 總功率 × 1e-6 的環記為不可定義(nan)。"""
    num, pa, pb = (torch.as_tensor(x, dtype=torch.float64) for x in (num, pa, pb))
    f = num / torch.sqrt(pa * pb).clamp_min(1e-300)
    bad = (pa < FRC_PMIN * pa.sum()) | (pb < FRC_PMIN * pb.sum())
    f[bad] = float("nan")
    return f.numpy()


def reflections(dx):
    """晶格的反射(MoS₂、WS₂ 平均的 a;d ≥ 取樣極限 0.72 Å)→ FRC 方塊的環編號(描述用的標記)。"""
    a = sum(A_LAT.values()) / 2
    out = []
    for hk, s in (("100", 1), ("110", 3), ("200", 4), ("210", 7), ("300", 9), ("220", 12), ("310", 13)):
        d = a / math.sqrt(4.0 / 3.0 * s)
        if d >= 2 * dx:
            out.append({"hk": hk, "d_A": d, "ring": d and FRC_N * dx / d})
    return out


@torch.no_grad()
def residual(est, d):
    """資料一致性殘差(描述):估計物體以名目探針與位置重算的繞射振幅 vs 量測振幅(beamstop 外)。
    R = Σ_j Σ (|F{P·Ô_j}| − √量測)² / Σ 量測振幅²(每個場)。"""
    geo, bs = d["geo"], d["bs"]
    W, P = geo.W, geo.pr["P"]
    meas = a6.measured(d["counts"], d["cp"], est.device)
    out = []
    for i in range(0, est.shape[0], MCH):
        e = est[i:i + MCH]
        mm = meas[i:i + MCH]
        Oj = torch.stack([e[:, y0:y0 + W, x0:x0 + W] for y0, x0 in geo.starts], 1)
        E = torch.fft.fftshift(torch.fft.fft2(P * Oj, norm="ortho"), dim=(-2, -1)).abs()
        out.append((((E - mm) ** 2) * bs).sum((1, 2, 3)) / ((mm ** 2) * bs).sum((1, 2, 3)).clamp_min(1e-30))
    return torch.cat(out)


@torch.no_grad()
def lat_metrics(est, d, extra=False):
    """每個場的指標(CPU tensor):nerr_ph / sh / sr(同階段七)、nerr_c(主)、相位 / 振幅 RMS、失敗旗標、空缺判讀的混淆矩陣。
    extra=True 另算 SSIM(同 7b)、FRC 累加量(依材料)、資料一致性殘差。"""
    O, U, vac = d["O"], d["geo"].U, d["vac"]
    m = U.to(O.real.dtype)
    nU = float(m.sum())
    v0, v1, v2, t, q, eal = align_full(est, O, U)
    fin = torch.isfinite(v0)
    c = ((eal - O).abs() ** 2 * m).sum((1, 2)) / d["Ec"].clamp_min(1e-30)
    dph = torch.angle(eal * O.conj())
    phr = torch.sqrt((dph ** 2 * m).sum((1, 2)) / nU) * 1e3
    ampr = torch.sqrt(((eal.abs() - O.abs()) ** 2 * m).sum((1, 2)) / nU)
    phase = torch.where(fin[:, None, None], torch.angle(eal), torch.full_like(eal.real, float("nan")))
    pred, ok, mrec = vac.read(phase.cpu(), fin.cpu())
    C, Cc = vac.confusion(pred)
    r = {"ph": v0.cpu(), "sh": v1.cpu(), "sr": v2.cpu(), "c": torch.where(fin, c, torch.full_like(c, float("nan"))).cpu(),
         "c_sr": (v2 * d["E"] / d["Ec"].clamp_min(1e-30)).cpu(), "ph_rms": phr.cpu(), "amp_rms": ampr.cpu(), "fin": fin.cpu(),
         "conf": C, "conf_cat": Cc, "readable": ok, "m_rec": mrec, "shift": t.norm(dim=1).cpu()}
    if extra:
        sa, sp = b7.ssim_eval(est, O, U, d["cfg"].phase_max)
        r["ssim_amp"], r["ssim_ph"] = sa.cpu(), sp.cpu()
        y0, x0, _ = frc_block(U)
        e0 = torch.where(fin[:, None, None], eal, torch.zeros_like(eal))
        r["frc"] = {k: frc_sums(e0[d["mats"].to(e0.device) == i], O[d["mats"].to(O.device) == i], y0, x0) for k, i in (("W", 0), ("Mo", 1))}
        r["resid"] = residual(torch.where(fin[:, None, None], est, torch.zeros_like(est)), d).cpu()
    return r


def summarize(r, mats):
    """一組指標 → 存進包絡檔的數字(每個 seed 一個):nerr_c 的總和 / 個數 / 失敗數 / 最大有限值(依材料)、
    nerr_sr(失敗 = 4.0)、混淆矩陣(依材料、依新缺陷類別)、無法判讀的場數。失敗值在彙整時決定(協定 §4.3 (c))。"""
    out = {}
    fin = r["fin"]
    for tag, sel in (("", torch.ones_like(fin)), ("_W", mats == 0), ("_Mo", mats == 1)):
        sf = sel & fin
        out["c_sum" + tag] = float(r["c"][sf].double().sum())
        out["c_n" + tag] = int(sel.sum())
        out["c_fail" + tag] = int((sel & ~fin).sum())
        out["c_max" + tag] = float(r["c"][sf].max()) if bool(sf.any()) else 0.0
        out["conf" + tag] = r["conf"][sel].sum(0).flatten().tolist()
        out["unread" + tag] = int((~r["readable"][sel]).sum())
    out["nerr_sr"] = float(torch.where(fin, r["sr"], torch.full_like(r["sr"], FAIL_SR)).mean())
    out["nerr_ph"] = float(torch.where(fin, r["ph"], torch.full_like(r["ph"], FAIL_SR)).mean())
    out["ph_rms"] = float(r["ph_rms"][fin].mean()) if bool(fin.any()) else float("nan")
    out["amp_rms"] = float(r["amp_rms"][fin].mean()) if bool(fin.any()) else float("nan")
    out["conf_iso"] = r["conf_cat"][:, 0].sum(0).flatten().tolist()
    out["conf_new"] = r["conf_cat"][:, 1].sum(0).flatten().tolist()
    return out


RAW_KEYS = None


def finalize_rec(rec, fail_c):
    """包絡檔的原始數字 → 判讀用的鍵(每個 seed):nerr_c(_W / _Mo;失敗 = fail_c)、sv_miss = 1 − SV-F1(_W / _Mo)。"""
    for tag in ("", "_W", "_Mo"):
        rec["nerr_c" + tag] = [(cs + nf * fail_c) / max(nn, 1) for cs, nf, nn in zip(rec["c_sum" + tag], rec["c_fail" + tag], rec["c_n" + tag])]
        rec["sv_miss" + tag] = [1.0 - vac_stats(cf)["sv_f1"] for cf in rec["conf" + tag]]


# ============================================================================
# 一份考卷的迭代法包絡(協定 §4.2;同 scan_7a.measure_condition 的設定與停止點,指標換成晶格的)
# ============================================================================
@torch.no_grad()
def measure_exam(dev, exam, n, iters, smoke, log=print, partial=None):
    kit = sorted(set(iters) | set(a7.KHIO_N.values()))
    meta = {"exam": exam, "n": n, "iters": list(iters), "md5": me_md5(), "smoke": smoke}
    env = {c: {str(it): {} for it in a7.stops7(c, iters)} for c in a7.configs7()}
    stats, done = [], []
    if partial is not None and partial.exists():
        old = json.load(open(partial))
        if old.get("meta") == meta:
            env, stats, done = old["env"], old["stats"], old["done"]
            log(f"  續跑:暫存檔已有 seed {done}({partial.name})")
        else:
            raise SystemExit(f"❌ {partial} 的設定或程式版本不同:不續跑。先告訴 Claude")

    def put(conf, it, r, mats):
        for k, v in summarize(r, mats).items():
            env[conf][str(it)].setdefault(k, []).append(v)

    t0 = time.time()
    for s in SEEDS:
        if s in done:
            continue
        d = setup_exam(s, dev, exam, n, smoke)
        mats = d["mats"]
        st = exam_stats(d)
        stats.append(st)
        win = a7.khio_windows7(d, dev, kit)
        kw = {}
        for k, wv in win.items():
            Of, cover, _ = a6.merge_windows(wv, d["geo"])
            kw[k] = (Of, cover)
            if k in iters:
                put("K-HIO", k, lat_metrics(torch.where(cover, Of, torch.zeros_like(Of)), d), mats)
        del win
        for kind in s5b.INITS:
            O0 = a6.make_start(kind, d, kw, dev)
            r0 = lat_metrics(O0, d)
            for meth in a7.METHODS7:
                outs = a7.run_method7(meth, d, iters, O0)
                conf = f"{meth}|{kind}"
                put(conf, 0, r0, mats)
                for it in iters:
                    put(conf, it, lat_metrics(outs[it], d), mats)
                del outs
        del kw, d
        done.append(s)
        if partial is not None:
            dump_json({"meta": meta, "env": env, "stats": stats, "done": done}, partial)
        log(f"  [{exam} seed {s}] 完成(經過 {time.time() - t0:.0f} 秒)")
    return env, stats


@torch.no_grad()
def exam_stats(d):
    """考卷的統計(描述與檢查):誤差確實加上、計數 / 期望(劑量)、相位範圍、空缺數。"""
    spec, n = d["spec"], d["n"]
    ex = measure_L(d["fields"], d["geo"], d["cp"], d["bs"], 0, spec, d["z"], d["sign"], d["ops"], d["dose"], poisson=False)
    dd = d["dose"].to(d["O"].device)[:, None, None]
    raw = sum(float((c * dd).sum()) for c in d["counts"])
    exp_ = sum(float((c * dd).sum()) for c in ex)
    lab = d["vac"].lab
    cat = d["vac"].cat
    meta = d["meta"]
    st = {"n": n, "pos_std_px": float((spec["pos"] * d["z"]).std()) if spec["pos"] > 0 else 0.0,
          "df_plus_frac": float((d["sign"] > 0).float().mean()), "raw_vs_expected": raw / max(exp_, 1e-30),
          "mean_counts": float(torch.stack([c.sum((-2, -1)) for c in d["counts"]]).mean()),
          "mean_raw_counts": raw / (n * len(d["counts"])),
          "dose_W": float(d["dose"][d["mats"] == 0].mean()) if bool((d["mats"] == 0).any()) else None,
          "dose_Mo": float(d["dose"][d["mats"] == 1].mean()) if bool((d["mats"] == 1).any()) else None,
          "phase_max": meta["phase_max"], "neg_min": meta["neg_min"], "neg_frac_max": max(meta["neg_frac"]),
          "n_cols": int(len(lab)), "n_sv": int((lab == 1).sum()), "n_dv": int((lab == 2).sum()),
          "n_ld": int(((cat == CAT_LD) & (lab > 0)).sum()), "n_mr": int(((cat == CAT_MR) & (lab > 0)).sum()),
          "n_W": int((d["mats"] == 0).sum()), "cols_per_field": float(len(lab) / n), "dx": d["dx"]}
    return st


def run_env(dev, exam, n, iters, root, smoke):
    me = me_md5()
    t0 = time.time()
    print(f"\n  考卷 {exam}({ELABEL[exam]}):{ESPEC[exam]}", flush=True)
    path = env_json(exam, root)
    partial = Path(str(path) + ".partial")
    env, stats = measure_exam(dev, exam, n, iters, smoke, log=lambda s: print(s, flush=True), partial=partial)
    sp = ESPEC[exam]
    nn_ = stats[0]["n"]
    if sp["pos"] > 0:
        r = np.mean([s["pos_std_px"] for s in stats]) / sp["pos"]
        check(f"[{exam}] 位置誤差的標準差 = 設定 × (1 ± 5%)", abs(r - 1) <= 0.05, f"實際 / 設定 = {r:.3f}")
    if sp["df"] > 0:
        f = np.mean([s["df_plus_frac"] for s in stats])
        check(f"[{exam}] 離焦符號約各半(± 3 個標準誤)", abs(f - 0.5) <= 3 * 0.5 / math.sqrt(nn_), f"+ 的比例 {f:.3f}")
    rr = np.mean([s["raw_vs_expected"] for s in stats])
    check(f"[{exam}] 平均偵測計數 / 期望 = 1 ± 2%(劑量;協定 §3.4 (3))", abs(rr - 1) <= 0.02, f"{rr:.4f}")
    pm = max(s["phase_max"] for s in stats)
    check(f"[{exam}] 所有場的最大相位 < π/2", pm < math.pi / 2, f"最大 {pm:.3f} rad")
    out = {"meta": {"exam": exam, "spec": sp, "n": nn_, "iters": list(iters), "seeds": list(SEEDS), "script_md5": me, "quick": QUICK,
                    "smoke": smoke, "gpu": a7.gpu_name(dev), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0},
           "env": env, "stats": stats, "checks_ok": all_ok()}
    dump_json(out, path)
    partial.unlink(missing_ok=True)
    print(f"  存檔:{path}(經過 {time.time() - t0:.0f} 秒)")
    return out


# ============================================================================
# 8b:微調 P6B4e8-L 與對照 P6B4e8-C(協定 §五)
# ============================================================================
def draw_errors_8(B, gen, lat, lo):
    """= scan_7c.draw_errors_7c 的抽法(同順序、同亂數);晶格樣本的劑量下限改為 lo(協定 §3.4,事先寫定)。"""
    sig = torch.rand(B, generator=gen) * b7.TR_POS_MAX
    z = torch.randn(B, 9, 2, generator=gen).clamp(-b7.Z_CLIP, b7.Z_CLIP)
    idx = torch.randint(0, len(b7.DF_GRID), (B,), generator=gen)
    hi = torch.rand(B, generator=gen) < b7.TR_DOSE_P1
    u = torch.rand(B, generator=gen)
    pos_on = torch.rand(B, generator=gen) < c7.P_POS_ON
    prb_on = torch.rand(B, generator=gen) < c7.P_PRB_ON
    dose = torch.where(hi, torch.ones(B), 10.0 ** (b7.TR_DOSE_LOGMIN * u))
    if bool(lat.any()):
        dl = torch.where(hi, torch.ones(B), 10.0 ** (math.log10(lo) * u))
        dose = torch.where(lat, dl, dose)
    d = sig[:, None, None] * z * pos_on[:, None, None].float()
    d = d - d.mean(1, keepdim=True)
    idx = torch.where(prb_on, idx, torch.full_like(idx, b7.DF_GRID.index(0.0)))
    return d, idx, dose, sig, pos_on, prb_on


class TrainMeas8(b7.TrainMeas):
    def __call__(self, obj, err_seed, noise_seed, lat, lo):
        gc = torch.Generator().manual_seed(int(err_seed))
        d, idx, dose, sig, pon, qon = draw_errors_8(obj.shape[0], gc, lat, lo)
        self.gen.manual_seed(int(noise_seed))
        cnt = b7.measure_rand(obj, self.rel, self.Pf, self.ops, self.cp, self.bs, d, idx, dose, self.gen)
        return cnt, (d, idx, dose, sig, pon, qon)


def lattice_pool(cfg, pr, n, dev, smoke=False):
    """晶格訓練池(方框 80 × 80;材料隨機各半;SV / DV;不含線缺陷、缺 S 區域)。seed = 86,000,000 + 索引(smoke:89,600,000 + 索引)。"""
    S = s5c.geometry(cfg)[1]
    base = SEED_SMOKE_TRAIN if smoke else SEED_TRAIN_OBJ
    f, meta = make_lattice([base + i for i in range(n)], S, dev, dx=pixel_size(pr, cfg.canvas))
    return f, meta


def load_8b(kind, s, cfg, pr, dev, mroot=None):
    md = model_dir(kind, s, mroot)
    meta = json.load(open(md / "result.json"))
    model, _ = d7.build("e", cfg, pr, dev)
    model.load_state_dict(torch.load(md / "final.pt", map_location=dev))
    return model.eval(), meta["norm"], meta


def train_8b(kind, seed, dev, n_fields=None, epochs=None, out=None, src_root=None, smoke=False):
    n_fields = a6.N_TRAIN if n_fields is None else n_fields
    epochs = epochs or g.EPOCHS
    cfg, pr, cp, bs = s5c.setup(seed, PROBE, dev)
    R0 = s5c.r0_box(cfg, pr, dev)
    out = Path(out) if out is not None else model_dir(kind, seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    name = NEW_L if kind == "L" else NEW_C
    org = s5c.make_train_fields(cfg, n_fields, seed).to(dev)                           # 同 P6B4e8 的原本訓練場(同 seed)
    lat, lmeta = (lattice_pool(cfg, pr, n_fields, dev, smoke) if kind == "L" else (None, None))
    lo = load_dose()["lat_dose_lo"]
    print(f"[init] {tg(name)} s{seed}:原本訓練場 {len(org)} 個" + (f";晶格池 {len(lat)} 個(WS₂ {lmeta['mats'].count(0)}、"
          f"最大相位 {lmeta['phase_max']:.3f} rad、負相位最小 {lmeta['neg_min']:.4f})" if lat is not None else "")
          + f";晶格樣本的劑量下限 {lo:.4g}({time.time() - t0:.1f}s)", flush=True)
    if lat is not None and not lmeta["phase_max"] < math.pi / 2:
        raise SystemExit(f"❌ 晶格訓練池有場的最大相位 ≥ π/2({lmeta['phase_max']:.3f}):停止、回報(協定 §2.2)")
    src = d7.model_dir("e", seed, src_root)
    model, norm, smeta = d7.load_e(seed, cfg, pr, dev, src_root, "e")
    ref = torch.load(src / "final.pt", map_location=dev)
    sd_ = model.state_dict()
    w_ok = set(sd_) == set(ref) and all(torch.equal(sd_[k], ref[k]) for k in ref)
    bnorm = json.load(open(s5c.run_dir("B", PROBE, seed, a6.N_TRAIN) / "result.json"))["norm"]
    check("載入檢查:起點的權重 = P6B4e8(逐位元,含 b_k;協定 §八 #10);8 級;輸入正規化 = P6B4e8 的(= B 的)",
          w_ok and model.K == d7.K8 and norm == bnorm, f"權重相同 {w_ok}、級數 {model.K}、正規化相同 {norm == bnorm}")
    del ref
    if not all_ok():
        raise SystemExit("❌ 載入檢查未通過,不訓練")
    tm = TrainMeas8(cfg, pr, cp, bs, dev)
    opt = torch.optim.Adam([{"params": list(model.parameters()), "lr": LR_8B}])
    steps = n_fields // g.BATCH
    half = g.BATCH // 2
    if kind == "L" and steps * half > len(lat):
        raise SystemExit("❌ 晶格池不夠一個 epoch 用")
    total = epochs * steps + 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_8B], total_steps=total)
    print(f"[init] {epochs} epochs × {steps} 步(batch {g.BATCH}:" + (f"晶格 {half} + 原本 {half}" if kind == "L" else "原本 128")
          + f");學習率 {LR_8B:g}(所有權重,含 b_k);位置監督 λ = {c7.LAMBDA_POS}", flush=True)
    meta = {"group": name, "kind": kind, "arch": "P6B4e8 (8 stages, position step)", "K": d7.K8, "probe": PROBE, "seed": seed,
            "n_train": n_fields, "lat_pool": len(lat) if lat is not None else 0, "epochs": epochs, "batch": g.BATCH, "lr": LR_8B,
            "clip": g.CLIP, "norm": norm, "lambda_pos": c7.LAMBDA_POS, "init": f"P6B4e8 s{seed} (all weights)",
            "init_md5": g._md5(src / "final.pt"), "lat_dose_lo": lo, "err_seed_base": TRAIN_ERR_BASE, "order_seed": TRAIN_ORDER_BASE + seed,
            "quick": QUICK, "smoke": smoke, "script_md5": me_md5(), "d7_md5": D7_MD5, "order": smeta.get("order"), "m": smeta.get("m")}
    json.dump(meta, open(out / "config_used.json", "w"), indent=2)
    ck = out / "ckpt.pt"
    start, hist = 0, []
    if ck.exists():
        dck = torch.load(ck, map_location=dev, weights_only=False)
        if dck.get("script_md5") != me_md5():
            raise SystemExit(f"❌ {ck} 是由不同版本的 scan_8.py 存的:不續跑。先告訴 Claude")
        model.load_state_dict(dck["model"])
        opt.load_state_dict(dck["opt"])
        sched.load_state_dict(dck["sched"])
        start, hist = dck["epoch"] + 1, dck["history"]
        print(f"[resume] 從 epoch {start} 續跑", flush=True)
    gen_o = torch.Generator().manual_seed(TRAIN_ORDER_BASE + seed)                     # 資料順序(L 與 C 同一個亂數流)
    nl = n_fields
    perms = []
    for ep in range(epochs):
        perms.append((torch.randperm(len(org), generator=gen_o), torch.randperm(nl, generator=gen_o)))
    gstep = start * steps
    K = model.K
    lat_mask = torch.cat([torch.ones(half, dtype=torch.bool), torch.zeros(g.BATCH - half, dtype=torch.bool)]) if kind == "L" \
        else torch.zeros(g.BATCH, dtype=torch.bool)
    for ep in range(start, epochs):
        model.train()
        po, pl = perms[ep]
        acc = {"loss": 0.0, "obj": 0.0, "pos": 0.0, "final": 0.0, "lat": 0.0, "org": 0.0, "stages": [0.0] * K, "pos_rms": [0.0] * K,
               "sat": [0.0] * K}
        n_ok, n_skip, n_clip, gns, te = 0, 0, 0, [], time.time()
        for i in range(steps):
            if kind == "L":
                obj = torch.cat([lat[pl[i * half:(i + 1) * half].to(dev)], org[po[i * half:(i + 1) * half].to(dev)]])
            else:
                obj = org[po[i * g.BATCH:(i + 1) * g.BATCH].to(dev)]
            es = TRAIN_ERR_BASE + 1_000_000 * seed + gstep
            with torch.no_grad():
                counts, err = tm(obj, es, es, lat_mask, lo)
                inp = g.prep(counts, bs, norm, "P6B4", cp)
            gstep += 1
            want = err[0].to(dev)
            _, outs, ds, sats = model(inp, with_delta=True)
            Ot = torch.polar(obj[:, 0], obj[:, 1])
            l_obj, ls = g.ds_loss(outs, Ot, R0)
            l_pos = torch.stack([(dk - want).pow(2).sum(-1).mean() for dk in ds]).mean()
            loss = l_obj + c7.LAMBDA_POS * l_pos
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip += 1
                if sched.last_epoch + 1 < total:
                    sched.step()
                continue
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), g.CLIP))
            if not np.isfinite(gn):
                n_skip += 1
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch + 1 < total:
                    sched.step()
                continue
            opt.step()
            if sched.last_epoch + 1 < total:
                sched.step()
            n_ok += 1
            n_clip += gn > g.CLIP
            gns.append(gn)
            with torch.no_grad():
                lm = lat_mask.to(dev)
                if kind == "L":
                    acc["lat"] += float(g.ds_loss([o[lm] for o in outs], Ot[lm], R0)[0])
                acc["org"] += float(g.ds_loss([o[~lm] for o in outs], Ot[~lm], R0)[0])
            acc["loss"] += float(loss.detach())
            acc["obj"] += float(l_obj.detach())
            acc["pos"] += float(l_pos.detach())
            acc["final"] += float(ls[-1].detach())
            for k in range(K):
                acc["stages"][k] += float(ls[k].detach())
                acc["pos_rms"][k] += float((ds[k].detach() - want).pow(2).sum(-1).mean(-1).sqrt().mean())
                acc["sat"][k] += float(sats[k])
        for k in ("loss", "obj", "pos", "final", "lat", "org"):
            acc[k] /= max(n_ok, 1)
        for k in ("stages", "pos_rms", "sat"):
            acc[k] = [v / max(n_ok, 1) for v in acc[k]]
        acc.update({"epoch": ep, "skipped": n_skip, "clip_frac": n_clip / max(n_ok, 1),
                    "grad_norm_median": float(np.median(gns)) if gns else None, "alphas": model.net.alphas(),
                    "betas": model.betas(), "sec": time.time() - te, "lr": sched.get_last_lr()})
        hist.append(acc)
        print(f"epoch {ep:03d}  loss={acc['loss']:.5f}  物體={acc['obj']:.5f}(晶格 {acc['lat']:.5f} / 原本 {acc['org']:.5f})"
              f"  位置={acc['pos']:.4f}  最後一級 {acc['final']:.5f}  各級位置 RMS=" + " ".join(f"{v:.3f}" for v in acc["pos_rms"])
              + "  α=" + " ".join(f"{v:.3f}" for v in acc["alphas"]) + "  β=" + " ".join(f"{v:.2f}" for v in acc["betas"])
              + f"  飽和 {100 * np.mean(acc['sat']):.1f}%  梯度中位數 {acc['grad_norm_median'] or 0:.2e}(裁切 {100 * acc['clip_frac']:.0f}%)"
              + f"  跳過 {n_skip}  {acc['sec']:.1f}s", flush=True)
        tmp = out / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "history": hist, "script_md5": meta["script_md5"]}, tmp)
        os.replace(tmp, ck)
    model.eval()
    with torch.no_grad():
        sub = org[:s5c.TRAIN_EVAL_N]
        c_id = s5c.measure_box(sub, PROBE, pr, cp, bs, cfg, seed=seed + 99)
        tr = c7.nerr_on(model, sub, c_id, norm, cp, bs, R0)
        tr_l = None
        if lat is not None:
            subl = lat[:s5c.TRAIN_EVAL_N]
            tr_l = c7.nerr_on(model, subl, s5c.measure_box(subl, PROBE, pr, cp, bs, cfg, seed=seed + 99), norm, cp, bs, R0)
    meta.update({"history": hist, "train_nerr_ph": tr, "train_nerr_ph_lattice": tr_l, "alphas": model.net.alphas(), "betas": model.betas(),
                 "train_sec": time.time() - t0, "skipped_total": sum(x["skipped"] for x in hist)})
    tmp = out / "final.tmp"
    torch.save(model.state_dict(), tmp)
    json.dump(meta, open(out / "result.json", "w"), indent=2)
    os.replace(tmp, out / "final.pt")
    print(f"[done] {tg(name)} s{seed}:訓練場 nerr_ph(理想,方框)原本 {tr:.4f}" + (f"、晶格 {tr_l:.4f}" if tr_l is not None else "")
          + f"  β {' '.join(f'{v:.2f}' for v in model.betas())}  {time.time() - t0:.0f}s → {out}", flush=True)
    return meta


# ============================================================================
# 彙整:網路的載入與計時
# ============================================================================
def roots_of(smoke):
    """模型的位置:正式 → 各自的正式資料夾;smoke → 7b / 7d / 本階段的 smoke 資料夾。"""
    return {"r": b7.SMOKE_DIR, "e": d7.SMOKE_DIR, "m": SMOKE_DIR} if smoke else {"r": None, "e": None, "m": None}


def get_net(nm, s, cfg, pr, dev, roots):
    """回傳 (流程, 模型, 輸入正規化):P6B4 / P6B4r 走 scan_6a 的 G9;8 級估計型走 scan_7c 的 G9(各級估位置)。"""
    if nm == "P6B4":
        net, norm, _ = a6.load_p6b4(s, cfg, pr, dev)
        return "g9", net, norm
    if nm == "P6B4r":
        net, norm = b7.load_net("P6B4r", s, cfg, pr, dev, roots["r"])
        return "g9", net, norm
    if nm == BASE:
        m, norm, _ = d7.load_e(s, cfg, pr, dev, roots["e"], "e")
        return "g9e", m, norm
    m, norm, _ = load_8b("L" if nm == NEW_L else "C", s, cfg, pr, dev, roots["m"])
    return "g9e", m, norm


@torch.no_grad()
def run_net(kind, model, norm, d, pre=None):
    if kind == "g9":
        return a6.run_pipe("G9", model, norm, d, pre)
    return c7.run_g9e(model, norm, d, sg=pre)[0]


@torch.no_grad()
def time_nets8(dev, roots):
    """5 個網路 G9 的端到端時間(ms / 整張;同 scan_7c.time_nets:幾何預先建好、暖機 2 次、取中位數)。三個 8 級網路結構相同,仍各自量。"""
    torch.backends.cudnn.benchmark = True
    res = {nm: {} for nm in NETS}
    for Bl in ["64", "512"]:
        B = int(Bl) if not QUICK else {"64": 4, "512": 8}[Bl]
        d = a7.setup_cond(0, dev, "ideal", B)
        geo, cfg, pr = d["geo"], d["cfg"], d["geo"].pr
        for nm in NETS:
            kind, model, norm = get_net(nm, 0, cfg, pr, dev, roots)
            pre = a6.GlobalStages(model, geo.starts, geo.F, pr, d["bs"], dev) if kind == "g9" else c7.field_geo(model.net, geo, d["bs"], dev)
            fn = (lambda kind=kind, model=model, norm=norm, pre=pre, d=d: run_net(kind, model, norm, d, pre))
            fn()
            fn()
            res[nm][Bl] = a7._time_it(fn, B, dev, 1)
        del d
    return res


# ============================================================================
# 彙整:網路評估(5 份考卷、原本 13 個條件、η = 0 敏感度、泛化落差)
# ============================================================================
PER_KEYS = ("c", "sr", "ph", "ph_rms", "amp_rms", "ssim_amp", "ssim_ph", "resid", "fin", "readable", "m_rec")


@torch.no_grad()
def eval_exams(dev, n, smoke, roots, log=print):
    """5 份考卷 × 3 seeds × 5 個網路(G9):逐場的指標(numpy)、混淆矩陣、FRC 累加量;seed 0 另存影像與判讀(畫圖用)。"""
    R = {e: {nm: {k: [] for k in PER_KEYS + ("conf", "conf_cat", "frc")} for nm in NETS} for e in EXAMS}
    keep = {}
    info = {}
    t0 = time.time()
    for e in EXAMS:
        for s in SEEDS:
            d = setup_exam(s, dev, e, n, smoke)
            if s == SEEDS[0]:
                info[e] = {"mats": d["mats"].numpy(), "theta": np.array(d["meta"]["theta"]), "stats": exam_stats(d),
                           "m_true": d["vac"].m_true.numpy()}
                keep[e] = {"O": d["O"].cpu(), "vac": d["vac"], "est": {}, "pred": {}, "geo": d["geo"], "cfg": d["cfg"],
                           "cols": d["meta"]["cols"], "dx": d["dx"]}
            for nm in NETS:
                kind, model, norm = get_net(nm, s, d["cfg"], d["geo"].pr, dev, roots)
                est = run_net(kind, model, norm, d)
                r = lat_metrics(est, d, extra=True)
                for k in PER_KEYS:
                    R[e][nm][k].append(r[k].numpy())
                R[e][nm]["conf"].append(r["conf"].numpy())
                R[e][nm]["conf_cat"].append(r["conf_cat"].numpy())
                R[e][nm]["frc"].append({k: [x.numpy().tolist() for x in v] for k, v in r["frc"].items()})
                if s == SEEDS[0]:
                    keep[e]["est"][nm] = est.cpu()
                    phase = torch.where(r["fin"][:, None, None].to(est.device), torch.angle(align_full(est, d["O"], d["geo"].U)[5]),
                                        torch.full_like(est.real, float("nan")))
                    keep[e]["pred"][nm] = d["vac"].read(phase.cpu(), r["fin"])[0]
                del model
            del d
        log(f"  [網路評估] {e} 完成(經過 {time.time() - t0:.0f} 秒)")
    return R, keep, info


@torch.no_grad()
def eval_orig(dev, n, roots, log=print):
    """原本 13 個條件(驗證場 80,000,000;量測同 7a / 7b / 7c,逐位元相同的程式)× 3 seeds × 5 個網路(G9):nerr_sr(逐場與平均)。"""
    R = {c: {nm: {"nerr_sr": [], "nerr_ph": [], "ps": []} for nm in NETS} for c in c7.EVAL_CONDS}
    t0 = time.time()
    for c in c7.EVAL_CONDS:
        for s in SEEDS:
            dd = a7.setup_cond(s, dev, c, n)
            O, U = dd["O"], dd["geo"].U
            for nm in NETS:
                kind, model, norm = get_net(nm, s, dd["cfg"], dd["geo"].pr, dev, roots)
                m = a7.metrics7(run_net(kind, model, norm, dd), O, U, keep_ps=True)
                R[c][nm]["nerr_sr"].append(m["nerr_sr"])
                R[c][nm]["nerr_ph"].append(m["nerr_ph"])
                ps = np.array(m["ps_sr"], float)
                R[c][nm]["ps"].append(np.where(np.isfinite(ps), ps, FAIL_SR))
                del model
            del dd
        log(f"  [原本的條件] {c} 完成(經過 {time.time() - t0:.0f} 秒)")
    return R


def org_train_fields(cfg, s, n):
    """原本訓練池的前 n 個場(整張 112 × 112):訓練池 = 每塊 s5c.CHUNK 個場的 make_fields,第一塊的前 n 個(make_fields 的結果與個數有關,
    所以要用同樣的個數產生再取前 n 個)。"""
    if n > s5c.CHUNK:
        raise ValueError("泛化落差只用第一塊")
    return s5.make_fields(cfg, s5c.CHUNK, seed=s5c.train_chunk_seed(s, 0))[:n]


def orig_setup(s, dev, n, fields_seed, noise_seed):
    """原本資料(程序生成)的整張量測(理想條件):網路 seed fields_seed 的訓練池前 n 個場、指定的雜訊 seed(泛化落差用;同 scan_6a.setup 的介面)。"""
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[PROBE]
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS["3x3s8"][2])
    bs = beamstop_mask(cfg, device=dev)
    geo = a6.Geom(cfg, pr, dev)
    fields = org_train_fields(cfg, fields_seed, n).to(dev)
    O = torch.polar(fields[:, 0], fields[:, 1])
    counts = a6.measure49(fields, geo, cp, bs, noise_seed)
    return dict(cfg=cfg, pr=pr, cp=cp, bs=bs, geo=geo, O=O, counts=counts, init=noise_seed + 1, n=n, fields=fields)


@torch.no_grad()
def eval_extra(dev, smoke, roots, log=print):
    """η = 0 敏感度(L-ideal 的前 128 個場,雜訊 87,600,000 + 1000 × seed)與泛化落差(協定 §五;晶格訓練池的前 512 個場、
    L-paper 的條件、雜訊 88,500,000 + 1000 × seed;原本資料:原本訓練池的前 512 個場、理想條件)。"""
    X = {"eta0": {nm: [] for nm in (FROZEN + [NEW_L])}, "eta0_c": {nm: [] for nm in (FROZEN + [NEW_L])},
         "gap_lat": {nm: [] for nm in (BASE, NEW_C, NEW_L)}, "gap_org": {nm: [] for nm in (BASE, NEW_C, NEW_L)}}
    nb = SEED_SMOKE_GAP if smoke else SEED_GAP_NOISE
    tb = SEED_SMOKE_TRAIN if smoke else SEED_TRAIN_OBJ
    ne = (SEED_SMOKE + 900_000) if smoke else SEED_ETA0_NOISE
    t0 = time.time()
    for s in SEEDS:
        d = setup_exam(s, dev, "L-ideal", ETA0_N, smoke, eta=0.0, noise_seed=ne + 1000 * s)
        for nm in X["eta0"]:
            kind, model, norm = get_net(nm, s, d["cfg"], d["geo"].pr, dev, roots)
            r = lat_metrics(run_net(kind, model, norm, d), d)
            X["eta0"][nm].append(r["c"].numpy())
            X["eta0_c"][nm].append(vac_stats(r["conf"].sum(0).numpy())["sv_f1"])
        d = setup_exam(s, dev, "L-paper", GAP_N, smoke, obj_seeds=[tb + i for i in range(GAP_N)], noise_seed=nb + 1000 * s)
        X.setdefault("gap_lat_mats", d["mats"].numpy().tolist())
        for nm in X["gap_lat"]:
            kind, model, norm = get_net(nm, s, d["cfg"], d["geo"].pr, dev, roots)
            X["gap_lat"][nm].append(lat_metrics(run_net(kind, model, norm, d), d)["c"].numpy())
        do = orig_setup(s, dev, GAP_N, s, nb + 1000 * s)
        for nm in X["gap_org"]:
            kind, model, norm = get_net(nm, s, do["cfg"], do["geo"].pr, dev, roots)
            m = a7.metrics7(run_net(kind, model, norm, do), do["O"], do["geo"].U, keep_ps=True)
            X["gap_org"][nm].append(np.array(m["ps_sr"], float))
        log(f"  [η = 0、泛化落差] seed {s} 完成(經過 {time.time() - t0:.0f} 秒)")
    return X


@torch.no_grad()
def iter_est_L(d, conf, it, dev):
    """重算一個迭代法設定(同 scan_7b.iter_est;d 為晶格考卷)。"""
    geo = d["geo"]
    need = [it] if conf == "K-HIO" else ([a7.KHIO_N[conf.split("|")[1]]] if conf.split("|")[1] in a7.KHIO_N else [])
    kw = None
    if need:
        win = a7.khio_windows7(d, dev, sorted(set(need)))
        kw = {k: a6.merge_windows(wv, geo)[:2] for k, wv in win.items()}
        del win
    if conf == "K-HIO":
        Of, cover = kw[it]
        return torch.where(cover, Of, torch.zeros_like(Of))
    meth, kind = conf.split("|")
    O0 = a6.make_start(kind, d, kw, dev)
    return O0 if it == 0 else a7.run_method7(meth, d, [it], O0)[it]


# ============================================================================
# 彙整:統計工具
# ============================================================================
def fail_value(env, R, e):
    """nerr_c 的失敗值 = max(10, 該考卷所有方法中最大有限值 × 1.5)(協定 §4.3 (c))。"""
    mx = 0.0
    for conf in env.values():
        for rec in conf.values():
            mx = max(mx, max(rec.get("c_max", [0.0])))
    for nm in NETS:
        for c in R[e][nm]["c"]:
            f = c[np.isfinite(c)]
            if f.size:
                mx = max(mx, float(f.max()))
    return max(FAIL_C_MIN, FAIL_C_X * mx)


def net_c(R, e, nm, fc, sel=None):
    """每個 seed 的逐場 nerr_c(失敗 = fc);sel:場的布林遮罩(材料)。回傳 [S, n'] array。"""
    out = []
    for c in R[e][nm]["c"]:
        v = np.where(np.isfinite(c), c, fc)
        out.append(v if sel is None else v[sel])
    return np.array(out)


def pooled_conf(R, e, nm, sel=None, seeds=None, cat=None):
    """合併的混淆矩陣(所有 seed、所選的場)。cat:None = 全部;0 = 孤立 / 背景;1 = 新缺陷。"""
    tot = np.zeros((3, 3))
    for i, C in enumerate(R[e][nm]["conf_cat" if cat is not None else "conf"]):
        if seeds is not None and i not in seeds:
            continue
        X = C[:, cat] if cat is not None else C
        tot += (X if sel is None else X[sel]).sum(0)
    return tot


def seed_svf1(R, e, nm, sel=None):
    return [vac_stats(pooled_conf(R, e, nm, sel, seeds=[i]))["sv_f1"] for i in range(len(SEEDS))]


def boot_ratio(A, B, rng):
    """以場為單位的成對 bootstrap(2000 次):比值 = 各 seed 平均誤差比的幾何平均。A, B:[S, n]。回傳 95% 區間。"""
    n = A.shape[1]
    idx = rng.integers(0, n, (BOOT_N, n))
    ma = np.maximum(A[:, idx].mean(-1), 1e-12)
    mb = np.maximum(B[:, idx].mean(-1), 1e-12)
    r = np.exp(np.log(ma / mb).mean(0))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def _f1_batch(C):
    """C:[..., 3, 3] → SV-F1 [...]。"""
    tp = C[..., 1, 1]
    fp = C[..., :, 1].sum(-1) - tp
    fn = C[..., 1, :].sum(-1) - tp
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), 0.0)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)


def boot_svf1(confs, rng, sel=None):
    """SV-F1 的 95% 區間(以場為單位重抽,每個場帶著它的所有柱)。confs:每個 seed 的 [n, 3, 3]。"""
    X = np.stack([c if sel is None else c[sel] for c in confs]).astype(float)                # [S, n, 3, 3]
    n = X.shape[1]
    idx = rng.integers(0, n, (BOOT_N, n))
    f = np.stack([_f1_batch(X[s][idx].sum(1)) for s in range(X.shape[0])]).mean(0)
    return [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]


def boot_fill(confs_cat, rng):
    """補回率(新缺陷、孤立缺陷)與差的 95% 區間。confs_cat:每個 seed 的 [n, 2, 3, 3]。"""
    X = np.stack(confs_cat).astype(float)                                                     # [S, n, 2, 3, 3]
    n = X.shape[1]
    idx = rng.integers(0, n, (BOOT_N, n))
    agg = X[:, idx].sum((0, 2))                                                               # [B, 2, 3, 3]
    fill = agg[..., 1:, 0].sum(-1) / np.maximum(agg[..., 1:, :].sum((-2, -1)), 1)             # [B, 2]
    d = fill[:, 1] - fill[:, 0]
    return {"new": [float(np.percentile(fill[:, 1], 2.5)), float(np.percentile(fill[:, 1], 97.5))],
            "iso": [float(np.percentile(fill[:, 0], 2.5)), float(np.percentile(fill[:, 0], 97.5))],
            "diff": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def t_first(env, key, Q, tB, iters):
    """包絡最早達到 Q(3 seeds 平均 ≤ Q)的 (時間, 設定, 次數) 與下界(前一個停止點;同 scan_7b 的 t_match_conf / t_match_lo,鍵由參數給)。"""
    best = (float("inf"), None, None)
    lo = float("inf")
    for c in a7.configs7():
        prev = None
        hit = False
        for it in a7.stops7(c, iters):
            v = float(a7.env_vals7(env, c, it, key).mean())
            t = a7.cfg_time7(c, it, tB)
            if v <= Q and t < best[0]:
                best = (t, c, it)
            if v <= Q and not hit:
                lo = min(lo, a7.cfg_time7(c, prev if prev is not None else (it if it == 0 else 0), tB))
                hit = True
            prev = it
    return best, lo


def speed(a_mean, env, tB, t_net, iters, key="nerr_c", qs=None, higher=False):
    """加速倍數(同 scan_7b.speed_row):x = 迭代法包絡最早達到 Q 的時間 / 網路的時間;x_lo = 下界。
    higher=True:指標越高越好(SV-F1),以 1 − 值 存成鍵、門檻同樣換算。"""
    out = {}
    for Q in (qs or QS):
        Qe = 1.0 - Q if higher else Q
        (t_it, conf, it), t_lo = t_first(env, key, Qe, tB, iters)
        reach = bool(a_mean >= Q) if higher else bool(a_mean <= Q)
        out[str(Q)] = {"reach": reach, "t_iter": t_it, "t_iter_lo": t_lo, "conf": conf, "it": it,
                       "x": (t_it / t_net) if reach else None, "x_lo": (t_lo / t_net) if reach else None}
    return out


def speed_label(s64, s512):
    if not s64["reach"]:
        return "網路未達"
    if not np.isfinite(s64["x"]) and not np.isfinite(s512["x"]):
        return "迭代法未達(網路達到)"
    if all(x["x_lo"] >= SPEED_MIN for x in (s64, s512)):
        return "穩健加速"
    if any(x["x_lo"] >= SPEED_MIN for x in (s64, s512)):
        return "達到、只在 " + "、".join(f"b{B}" for B, x in (("64", s64), ("512", s512)) if x["x_lo"] >= SPEED_MIN) + " 穩健加速"
    if any(x["x"] >= SPEED_MIN for x in (s64, s512)):
        return "達到、加速不確定(只有上界 ≥ 1.25)"
    return "達到、無明顯加速"


def cmp_label(r):
    return {"較差": "變差", "較準": "改善", "相近": "相近"}[r[0]]


# ============================================================================
# 彙整:報表與判讀(協定 §六,結果出來前寫定)
# ============================================================================
def fmt_x(r):
    return b7.fmt_x(r) if r["x"] is None or np.isfinite(r["x"]) or not r["reach"] else "迭代法未達(網路達到)"


def report(E, R, R13, X, IT, tim, iters, fc, dose, info, J7, smoke):
    Bs = ["64", "512"]
    rng = np.random.default_rng(BOOT_SEED)
    V = {"exam": {}, "fail_c": fc}
    if smoke:
        print("\n" + "!" * 100)
        print("迷你流程:訓練極少、場數少、停止點 ≤ 20 → 數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    # ---- 計時與劑量 ----
    print("\n" + "=" * 100)
    print(f"計時(ms / 整張;本 job 重新量;GPU {tim['gpu']})")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:G9 " + "、".join(f"{tg(nm)} {tim['net'][nm][B]:.4f}" for nm in NETS) + f";AP-C 每次 {tim[B]['AP-C']:.4f}")
    print("\n" + "=" * 100)
    print("劑量(協定 §3.4;--check 算出、此處讀回)")
    print("=" * 100)
    dx = info[MAIN_EXAM]["stats"]["dx"]
    step_a = STEP_A * dx
    for s, v in dose["seeds"].items():
        print(f"  seed {s}:N_inc = {v['N_inc']:.4g} 個電子 / 張;f_W = {v['f_W']:.4g}、f_Mo = {v['f_Mo']:.4g}")
    nin = float(np.mean([v["N_inc"] for v in dose["seeds"].values()]))
    print(f"  晶格樣本的劑量下限(8b):{dose['lat_dose_lo']:.4g}" + ("(f < 0.01 → 論文劑量超出原本的訓練範圍 [0.01, 1];8a 的該材料註明「超出訓練範圍」)"
                                                               if dose["out_of_range"] else ""))
    print(f"  附帶更正(階段七 §十二):以入射電子數換算,基準劑量 = {nin:.3g} / ({step_a:.3f} Å)² = {nin / step_a ** 2:.3g} e/Å²"
          f"(先前以平均偵測計數換算為約 4.8 × 10⁵);doseL / doseH = {nin / step_a ** 2 / 10:.3g} / {nin / step_a ** 2 / 100:.3g} e/Å²"
          f";論文 WS₂ 1.27 × 10⁴ – 5.08 × 10⁴、MoS₂ 4.48 × 10⁵ e/Å²")
    V["dose"] = {"N_inc": nin, "per_area": nin / step_a ** 2, "f": {s: [v["f_W"], v["f_Mo"]] for s, v in dose["seeds"].items()},
                 "lo": dose["lat_dose_lo"], "out_of_range": dose["out_of_range"]}
    # ---- 考卷的統計 ----
    print("\n" + "=" * 100)
    print("考卷的統計(seed 0;描述)")
    print("=" * 100)
    for e in EXAMS:
        st = info[e]["stats"]
        print(f"  {e:<9} 每場 {st['cols_per_field']:.1f} 個可判讀的 S₂ 柱(SV {st['n_sv']}、DV {st['n_dv']}、線缺陷 {st['n_ld']}、缺 S 區域 {st['n_mr']});"
              f"最大相位 {st['phase_max']:.3f} rad;負相位最小 {st['neg_min']:.4f}、< −0.05 rad 的像素比例最大 {st['neg_frac_max']:.2%};"
              f"每張平均偵測計數 {st['mean_counts']:.3g}(原始 {st['mean_raw_counts']:.3g});計數 / 期望 {st['raw_vs_expected']:.4f}")
        if st["neg_frac_max"] > NEG_FRAC:
            print(f"    ⚠️ 有場的負相位(< {NEG_TOL} rad)像素比例 > {NEG_FRAC:.0%}:回報、與使用者討論(協定 §2.2)")
    # ---- 各考卷 ----
    print("\n" + "=" * 100)
    print("各考卷(G9;主指標 nerr_c(失敗值見下);3 seeds;加速倍數 [下界–上界] = 迭代法包絡最早達到 Q 的時間 / 該網路的時間)")
    print("=" * 100)
    for e in EXAMS:
        env = E[e]
        mats = info[e]["mats"]
        V["exam"][e] = {"fail_c": fc[e], "nets": {}}
        print(f"\n  ■ {e}({ELABEL[e]});nerr_c 的失敗值 {fc[e]:.3g}")
        for nm in NETS:
            A = net_c(R, e, nm, fc[e])
            row = {"nerr_c": A.mean(1).tolist(), "nerr_c_W": net_c(R, e, nm, fc[e], mats == 0).mean(1).tolist(),
                   "nerr_c_Mo": net_c(R, e, nm, fc[e], mats == 1).mean(1).tolist(),
                   "nerr_sr": [float(np.where(f, x, FAIL_SR).mean()) for x, f in zip(R[e][nm]["sr"], R[e][nm]["fin"])],
                   "ph_rms": float(np.mean([x[f].mean() for x, f in zip(R[e][nm]["ph_rms"], R[e][nm]["fin"]) if f.any()])),
                   "amp_rms": float(np.mean([x[f].mean() for x, f in zip(R[e][nm]["amp_rms"], R[e][nm]["fin"]) if f.any()])),
                   "ssim": [float(np.mean([x.mean() for x in R[e][nm]["ssim_amp"]])), float(np.mean([x.mean() for x in R[e][nm]["ssim_ph"]]))],
                   "resid": float(np.mean([np.nanmean(x) for x in R[e][nm]["resid"]])), "n_fail": int(sum((~f).sum() for f in R[e][nm]["fin"])),
                   "unread": float(np.mean([(~x).mean() for x in R[e][nm]["readable"]])),
                   "vac": {k: vac_stats(pooled_conf(R, e, nm, sel)) for k, sel in (("all", None), ("W", mats == 0), ("Mo", mats == 1))},
                   "svf1_seed": {k: seed_svf1(R, e, nm, sel) for k, sel in (("all", None), ("W", mats == 0), ("Mo", mats == 1))}}
            row["svf1"] = {k: float(np.mean(v)) for k, v in row["svf1_seed"].items()}   # SV-F1 = 各 seed 的 F1 平均(同迭代法包絡的算法)
            mt = np.stack(R[e][nm]["m_rec"])
            row["contrast_gain"] = float(np.nanmean(mt / np.maximum(info[e]["m_true"][None], 1e-12)))
            line = (f"    {tg(nm):<9}:nerr_c {np.mean(row['nerr_c']):.4f}({' '.join(f'{x:.4f}' for x in row['nerr_c'])});"
                    f"WS₂ {np.mean(row['nerr_c_W']):.4f} / MoS₂ {np.mean(row['nerr_c_Mo']):.4f};nerr_sr {np.mean(row['nerr_sr']):.5f};"
                    f"相位 RMS {row['ph_rms']:.1f} mrad、振幅 RMS {row['amp_rms']:.4f};SSIM {row['ssim'][0]:.3f} / {row['ssim'][1]:.3f};"
                    f"SV-F1 {row['svf1']['all']:.3f}(WS₂ {row['svf1']['W']:.3f} / MoS₂ {row['svf1']['Mo']:.3f})"
                    f";無法判讀 {row['unread']:.1%};對比增益 {row['contrast_gain']:.3f};殘差 {row['resid']:.4f}"
                    + (f";發散 {row['n_fail']} 個" if row["n_fail"] else ""))
            for B in Bs:
                tn = tim["net"][nm][B]
                ev = a7.envelope7(env, tn, tim[B], iters, key="nerr_c")
                r = s5b.ratio(np.array(row["nerr_c"]), ev[2])
                sp = speed(np.mean(row["nerr_c"]), env, tim[B], tn, iters)
                spm = {mk: speed(np.mean(row[f"nerr_c_{mk}"]), env, tim[B], tn, iters, key=f"nerr_c_{mk}") for mk in ("W", "Mo")}
                row[B] = {"t": tn, "opp": [ev[0], ev[1], list(ev[2]), ev[3]], "ratio": list(r), "speed": sp, "speed_mat": spm}
                line += (f"\n       b{B}:{tn:.3f} ms;同時間包絡 {s5b.fmt_conf(ev[0])} ×{ev[1]} {np.mean(ev[2]):.4f} → 比值 {r[1]:.3f}(z {r[2]:+.1f})"
                         f";加速 " + "、".join(f"Q {q}:{fmt_x(sp[str(q)])}" for q in QS))
            print(line)
            V["exam"][e]["nets"][nm] = row
        bo = a7.best_overall(env, iters, key="nerr_c")
        V["exam"][e]["iter_best"] = [bo[0], bo[1], float(bo[2].mean())]
        tq = "、".join(f"Q {q}:" + (f"{t_first(env, 'nerr_c', q, tim['512'], iters)[0][0]:.2f} ms" if np.isfinite(
            t_first(env, "nerr_c", q, tim["512"], iters)[0][0]) else "500 次內到不了") for q in QS)
        fl = [(c, it, sum(rec["c_fail"]), sum(rec["c_n"])) for c, cr in env.items() for it, rec in cr.items()]
        nfail = [x for x in fl if x[2] > 0]
        worst_f = max(fl, key=lambda x: x[2] / max(x[3], 1))
        V["exam"][e]["iter_fail"] = {"n_settings_with_fail": len(nfail), "worst": [worst_f[0], worst_f[1], worst_f[2] / max(worst_f[3], 1)]}
        print(f"    迭代法:nerr_c 最佳 {s5b.fmt_conf(bo[0])} ×{bo[1]} {bo[2].mean():.4f};b512 達到門檻的時間 {tq}")
        print("    失敗率(發散的場):網路 " + "、".join(f"{tg(nm)} {V['exam'][e]['nets'][nm]['n_fail']}" for nm in NETS)
              + f";迭代法 有發散的(設定, 停止點){len(nfail)} 個、最高 {s5b.fmt_conf(worst_f[0])} ×{worst_f[1]} {worst_f[2] / max(worst_f[3], 1):.1%}")
    # ---- 判讀 ----
    print("\n" + "=" * 100)
    print("判讀(階段八協定 §六,結果出來前寫定;主假說只有 h1,其他為次要或探索性,多重比較不另校正)")
    print("=" * 100)
    ex = V["exam"][MAIN_EXAM]
    rl = ex["nets"][NEW_L]
    s64, s512 = rl["64"]["speed"][str(Q_MAIN)], rl["512"]["speed"][str(Q_MAIN)]
    h1 = speed_label(s64, s512)
    V["h1"] = {"nerr_c": float(np.mean(rl["nerr_c"])), "label": h1, "b64": s64, "b512": s512,
               "q05": [rl[B]["speed"]["0.05"] for B in Bs]}
    print(f"  (h1 主判)L-paper 下 P6B4e8-L 的 nerr_c {np.mean(rl['nerr_c']):.4f}、Q = {Q_MAIN}:{h1}(b64 {fmt_x(s64)}、b512 {fmt_x(s512)})"
          + ("  [smoke:無意義]" if smoke else ""))
    if not s64["reach"]:
        q5 = [rl[B]["speed"]["0.05"] for B in Bs]
        print(f"       Q = 0.05:{speed_label(*q5)}(b64 {fmt_x(q5[0])}、b512 {fmt_x(q5[1])})")
    V["h2"] = {}
    for mk, nmk in (("W", "WS₂(最接近論文)"), ("Mo", "MoS₂")):
        sm = [rl[B]["speed_mat"][mk][str(Q_MAIN)] for B in Bs]
        V["h2"][mk] = {"nerr_c": float(np.mean(rl[f"nerr_c_{mk}"])), "label": speed_label(*sm)}
        print(f"  (h2)L-paper {nmk}:P6B4e8-L {np.mean(rl[f'nerr_c_{mk}']):.4f};Q = {Q_MAIN}:{speed_label(*sm)}"
              f"(b64 {fmt_x(sm[0])}、b512 {fmt_x(sm[1])})")
    # (h3) 空缺判讀
    print("  (h3)空缺判讀(L-paper;SV-F1 ≥ 0.9 看得出、0.7–0.9 大致看得出、< 0.7 看不清楚;網路與迭代法分別判定)")
    V["h3"] = {}
    envp = E[MAIN_EXAM]
    mats = info[MAIN_EXAM]["mats"]
    for mk, nmk in (("W", "WS₂"), ("Mo", "MoS₂")):
        sel = mats == (0 if mk == "W" else 1)
        vs = rl["vac"][mk]                                                     # 合併 3 seeds 的混淆矩陣(精確率、召回率等描述)
        f1n = rl["svf1"][mk]                                                    # 判讀用:各 seed 的 SV-F1 平均(同迭代法、同 bootstrap)
        ci = boot_svf1(R[MAIN_EXAM][NEW_L]["conf"], rng, sel)
        lab = lambda f: "看得出單一 S 空缺" if f >= SVF1_GOOD else ("大致看得出" if f >= SVF1_OK else "在本研究的設定下看不清楚")  # noqa: E731
        key = f"sv_miss_{mk}"
        it_t = {B: a7.envelope7(envp, tim["net"][NEW_L][B], tim[B], iters, key=key) for B in Bs}
        bo = a7.best_overall(envp, iters, key=key)
        conf_b = np.array(envp[bo[0]][str(bo[1])][f"conf_{mk}"]).sum(0).reshape(3, 3)
        vb = vac_stats(conf_b)
        f1b = 1.0 - float(bo[2].mean())
        tsp = {B: speed(f1n, envp, tim[B], tim["net"][NEW_L][B], iters, key=key, qs=[f1n, SVF1_GOOD], higher=True)
               for B in Bs}
        V["h3"][mk] = {"net_svf1": f1n, "net_pooled": vs, "net_ci": ci, "net_label": lab(f1n),
                       "iter_at_net_time": {B: [it_t[B][0], it_t[B][1], 1 - float(np.mean(it_t[B][2]))] for B in Bs},
                       "iter_best": [bo[0], bo[1], f1b, vb], "iter_label": lab(f1b), "task_speed": tsp}
        print(f"    {nmk}:P6B4e8-L SV-F1 {f1n:.3f}(各 seed 平均;95% 區間 {ci[0]:.3f}–{ci[1]:.3f})→ {lab(f1n)};合併 3 seeds:精確率 {vs['sv_p']:.3f}、"
              f"召回率 {vs['sv_r']:.3f}、DV-F1 {vs['dv_f1']:.3f}、二元 F1 {vs['bin_f1']:.3f}、誤報率 {vs['fpr']:.3%}")
        print(f"       混淆矩陣(合併 3 seeds;列 = 真值 完整 / SV / DV;欄 = 判定):{vs['conf']}")
        print(f"       迭代法:網路時間下的包絡(依 SV-F1)b64 {s5b.fmt_conf(it_t['64'][0])} ×{it_t['64'][1]} SV-F1 {1 - np.mean(it_t['64'][2]):.3f}、"
              f"b512 {s5b.fmt_conf(it_t['512'][0])} ×{it_t['512'][1]} {1 - np.mean(it_t['512'][2]):.3f};500 次內最佳 {s5b.fmt_conf(bo[0])} ×{bo[1]} "
              f"SV-F1 {f1b:.3f}(合併:精確率 {vb['sv_p']:.3f}、召回率 {vb['sv_r']:.3f})→ {lab(f1b)}")
        print("       任務版的加速(迭代法達到網路的 SV-F1 / 0.9 所需的時間 / 網路時間):"
              + ";".join(f"b{B} " + "、".join(f"SV-F1 {q}:{fmt_x(r)}" for q, r in tsp[B].items()) for B in Bs))
    # (h4)
    print("  (h4)L-combo、L-ideal:見上表(各門檻的加速倍數、空缺判讀;描述)")
    V["h4"] = {e: {nm: {"speed_main": speed_label(V["exam"][e]["nets"][nm]["64"]["speed"][str(Q_MAIN)],
                                                  V["exam"][e]["nets"][nm]["512"]["speed"][str(Q_MAIN)]),
                        "svf1": V["exam"][e]["nets"][nm]["svf1"]["all"]} for nm in NETS} for e in ("L-combo", "L-ideal")}
    for e in ("L-combo", "L-ideal"):
        print(f"    {e}:" + ";".join(f"{tg(nm)} {v['speed_main']}、SV-F1 {v['svf1']:.3f}" for nm, v in V["h4"][e].items()))
    # (h5) 微調的效果與歸因
    print("  (h5)微調的效果與歸因(nerr_c;比值 ≤ 0.8 且 z < −2 = 改善;≥ 1.25 且 z > 2 = 變差;另列以場為單位的 bootstrap 95% 區間)")
    V["h5"] = {}
    for e in SEEN:
        V["h5"][e] = {}
        parts = []
        for a_, b_, nmk in ((NEW_L, BASE, "L / P6B4e8"), (NEW_L, NEW_C, "L / C(晶格資料的貢獻)"), (NEW_C, BASE, "C / P6B4e8(多訓練)")):
            A, Bm = net_c(R, e, a_, fc[e]), net_c(R, e, b_, fc[e])
            r = s5b.ratio(A.mean(1), Bm.mean(1))
            ci = boot_ratio(A, Bm, rng)
            V["h5"][e][nmk] = {"ratio": r[1], "z": r[2], "label": cmp_label(r), "ci": ci}
            parts.append(f"{nmk} {r[1]:.3f}(z {r[2]:+.1f};{ci[0]:.3f}–{ci[1]:.3f})→ {cmp_label(r)}")
        print(f"    {e}:" + ";".join(parts))
    lvc = [e for e in SEEN if V["h5"][e]["L / C(晶格資料的貢獻)"]["label"] == "改善"]
    V["h5"]["attribution"] = ("改善來自晶格資料:" + "、".join(lvc)) if lvc else "L vs C 沒有「改善」→ 寫「整體微調流程的效果」"
    print(f"    ▶ {V['h5']['attribution']}")
    if dose["out_of_range"]:
        V["h5"]["caveat"] = ("論文劑量在原本的訓練範圍之外(f < 0.01):P6B4e8-L 的晶格樣本練過更低的劑量(下限 "
                             f"{dose['lat_dose_lo']:.3g}),P6B4e8-C 沒有 → 在 L-paper / L-combo(論文劑量)上「晶格資料的貢獻」"
                             "同時包含「晶格的樣貌」與「更低的訓練劑量」,無法分開(L-ideal 為劑量 1,不受影響)")
        print(f"    ⚠️ 須註明:{V['h5']['caveat']}")
    # (h6) 原本 13 個條件
    print("  (h6)有沒有退步(原本 13 個條件、驗證場;nerr_sr;比值 ≥ 1.25 且 z > 2 = 退步)")
    V["h6"] = {}
    regs = []
    for nm in (NEW_L, NEW_C):
        V["h6"][nm] = {}
        parts = []
        for c in c7.EVAL_CONDS:
            A, Bm = np.array(R13[c][nm]["ps"]), np.array(R13[c][BASE]["ps"])
            r = s5b.ratio(np.array(R13[c][nm]["nerr_sr"]), np.array(R13[c][BASE]["nerr_sr"]))
            lab = {"較差": "退步", "較準": "改善", "相近": "相近"}[r[0]]
            ci = boot_ratio(A, Bm, rng)
            V["h6"][nm][c] = {"ratio": r[1], "z": r[2], "label": lab, "ci": ci}
            parts.append(f"{c} {r[1]:.3f}(z {r[2]:+.1f};{ci[0]:.3f}–{ci[1]:.3f})→ {lab}")
            if lab == "退步":
                regs.append(f"{tg(nm)}:{c}")
        print(f"    {tg(nm)} / P6B4e8:" + ";".join(parts))
    regL = [x for x in regs if x.startswith(tg(NEW_L))]
    regC = [x for x in regs if x.startswith(tg(NEW_C))]
    V["h6"]["regress"] = regL
    V["h6"]["regress_control"] = regC
    print("    ▶ P6B4e8-L:" + (f"❌ 退步:{', '.join(regL)} → 8b 不算完全成功" if regL else "未檢出退步(不寫「證明不退步」)")
          + ";對照組 P6B4e8-C(描述):" + (f"退步 {', '.join(regC)}" if regC else "未檢出退步"))
    # (h7)
    A, Bm = net_c(R, "L-coh", NEW_L, fc["L-coh"]), net_c(R, "L-coh", BASE, fc["L-coh"])
    r = s5b.ratio(A.mean(1), Bm.mean(1))
    V["h7"] = {"ratio": r[1], "z": r[2], "label": cmp_label(r), "ci": boot_ratio(A, Bm, rng)}
    print(f"  (h7)只測試的部分同調(L-coh):P6B4e8-L / P6B4e8 {r[1]:.3f}(z {r[2]:+.1f})→ {cmp_label(r)};網路 vs 迭代法只作描述(迭代法沒有混合態修正)")
    # (h8) 補回警示
    print("  (h8)只測試的新缺陷(L-newdef):補回率 = 真值有空缺卻被判成完整的比例(同一份考卷內:新缺陷 vs 孤立的 SV / DV)")
    V["h8"] = {"nets": {}}
    for nm in NETS:
        fn, fi = vac_stats(pooled_conf(R, "L-newdef", nm, cat=1))["fill"], vac_stats(pooled_conf(R, "L-newdef", nm, cat=0))["fill"]
        trig = (fn >= FILL_MIN) if fi == 0 else (fn >= FILL_X * fi and fn - fi >= FILL_D)
        V["h8"]["nets"][nm] = {"fill_new": fn, "fill_iso": fi, "pattern": bool(trig), "resid": V["exam"]["L-newdef"]["nets"][nm]["resid"]}
    V["h8"]["ci_L"] = boot_fill(R["L-newdef"][NEW_L]["conf_cat"], rng)
    envn = E["L-newdef"]

    def it_fill(conf, it):
        cn = np.array(envn[conf][str(it)]["conf_new"]).sum(0)
        ci_ = np.array(envn[conf][str(it)]["conf_iso"]).sum(0)
        fn_, fi_ = vac_stats(cn)["fill"], vac_stats(ci_)["fill"]
        return fn_, fi_, bool((fn_ >= FILL_MIN) if fi_ == 0 else (fn_ >= FILL_X * fi_ and fn_ - fi_ >= FILL_D))

    evn = a7.envelope7(envn, tim["net"][NEW_L]["512"], tim["512"], iters, key="nerr_c")
    ev64 = a7.envelope7(envn, tim["net"][NEW_L]["64"], tim["64"], iters, key="nerr_c")
    tgt = float(np.mean(V["exam"]["L-newdef"]["nets"][NEW_L]["nerr_c"]))
    close = min(((c, it) for c in a7.configs7() for it in a7.stops7(c, iters)),
                key=lambda x: abs(float(a7.env_vals7(envn, x[0], x[1], "nerr_c").mean()) - tgt))
    f1_ = it_fill(evn[0], evn[1])
    f0_ = it_fill(ev64[0], ev64[1])
    f2_ = it_fill(*close)
    V["h8"]["iter"] = {"net_time_b512": [evn[0], evn[1], *f1_], "net_time_b64": [ev64[0], ev64[1], *f0_], "closest": [close[0], close[1], *f2_]}
    netL = V["h8"]["nets"][NEW_L]
    warn = netL["pattern"] and not (f1_[2] or f0_[2] or f2_[2])           # 迭代法的三個參考點都沒有同樣的落差
    V["h8"]["label"] = "觸發補回警示" if warn else "未觀察到足夠的補回證據"
    for nm, v in V["h8"]["nets"].items():
        print(f"    {tg(nm):<9}:新缺陷 {v['fill_new']:.1%}、孤立 {v['fill_iso']:.1%}" + ("(符合落差的條件)" if v["pattern"] else "")
              + f";資料一致性殘差 {v['resid']:.4f}")
    ci8 = V["h8"]["ci_L"]
    print(f"    P6B4e8-L 的 95% 區間:新缺陷 {ci8['new'][0]:.1%}–{ci8['new'][1]:.1%}、孤立 {ci8['iso'][0]:.1%}–{ci8['iso'][1]:.1%}、差 "
          f"{ci8['diff'][0]:+.1%}–{ci8['diff'][1]:+.1%}")
    print(f"    迭代法:網路時間的包絡 b512 {s5b.fmt_conf(evn[0])} ×{evn[1]} 新缺陷 {f1_[0]:.1%}、孤立 {f1_[1]:.1%}"
          + ("(符合落差的條件)" if f1_[2] else "") + f"、b64 {s5b.fmt_conf(ev64[0])} ×{ev64[1]} 新缺陷 {f0_[0]:.1%}、孤立 {f0_[1]:.1%}"
          + ("(符合落差的條件)" if f0_[2] else "") + f";nerr_c 最接近網路的停止點 {s5b.fmt_conf(close[0])} ×{close[1]} 新缺陷 {f2_[0]:.1%}、"
          f"孤立 {f2_[1]:.1%}" + ("(符合落差的條件)" if f2_[2] else "")
          + (f";殘差(網路時間的設定,seed 0){IT['newdef_resid']:.4f}" if IT.get("newdef_resid") is not None else ""))
    print(f"    ▶ {V['h8']['label']}(判準:P6B4e8-L 的新缺陷補回率 ≥ 孤立的 2 倍且差 ≥ 10 個百分點,且迭代法的參考點(網路時間 b64 / b512、nerr_c 最接近網路的停止點)都沒有同樣的落差;"
          f"新缺陷的正規化:{NEWDEF_NORM})")
    # (h9)
    print("  (h9)8a 的分布差異(凍結的網路;描述)")
    V["h9"] = {}
    for e in ("L-ideal", "L-paper"):
        V["h9"][e] = {nm: [float(np.mean(V["exam"][e]["nets"][nm]["nerr_c"])), V["exam"][e]["nets"][nm]["svf1"]["all"]] for nm in FROZEN}
        print(f"    {e}:" + ";".join(f"{nm} nerr_c {v[0]:.4f}、SV-F1 {v[1]:.3f}" for nm, v in V["h9"][e].items()))
    # (h10) 方向敏感
    print("  (h10)誤差 vs 晶格旋轉角(L-paper;以 60° 折疊、6 個區間;某區間 ≥ 其他區間平均的 2 倍 → 方向敏感)")
    V["h10"] = {}
    th = np.mod(info[MAIN_EXAM]["theta"], 60.0)
    bins = np.minimum((th / (60.0 / ROT_BINS)).astype(int), ROT_BINS - 1)
    for nm in NETS:
        A = net_c(R, MAIN_EXAM, nm, fc[MAIN_EXAM]).mean(0)
        mb = np.array([A[bins == b].mean() if (bins == b).any() else np.nan for b in range(ROT_BINS)])
        flag = any(mb[b] >= ROT_X * np.nanmean(np.delete(mb, b)) for b in range(ROT_BINS) if np.isfinite(mb[b]))
        V["h10"][nm] = {"bins": mb.tolist(), "flag": bool(flag)}
        print(f"    {tg(nm):<9}:" + " / ".join(f"{v:.4f}" for v in mb) + (" → 方向敏感" if flag else ""))
    # (h11) FRC、SSIM、η = 0
    print("  (h11)FRC(pooled,各布拉格反射處;只描述,不把第一次越過 0.5 稱為解析度)、SSIM 與 nerr_c 的排名一致性、η = 0 敏感度")
    refl = reflections(dx)
    V["h11"] = {"refl": refl, "frc": {}}
    for nm in NETS:
        V["h11"]["frc"][nm] = {}
        for mk in ("W", "Mo"):
            sums = [np.array(fr[mk]) for fr in R[MAIN_EXAM][nm]["frc"]]
            curves = [frc_curve(*sm) for sm in sums]
            cur = np.nanmean(np.stack(curves), 0)
            V["h11"]["frc"][nm][mk] = cur.tolist()
        print(f"    {tg(nm):<9} L-paper FRC:" + ";".join(
            f"{mk} " + "、".join(f"{x['hk']} {np.array(V['h11']['frc'][nm][mk])[int(round(x['ring']))]:.3f}" for x in refl
                                if int(round(x["ring"])) < len(V["h11"]["frc"][nm][mk])) for mk in ("W", "Mo")))
    agree = {"amp": [0, 0], "ph": [0, 0]}
    for e in EXAMS:
        items = [(np.mean(V["exam"][e]["nets"][nm]["nerr_c"]), *V["exam"][e]["nets"][nm]["ssim"]) for nm in NETS]
        items += [(x["nerr_c"], x["ssim_amp"], x["ssim_ph"]) for x in IT.get("ssim", {}).get(e, [])]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] == items[j][0]:
                    continue
                for k, col in (("amp", 1), ("ph", 2)):
                    agree[k][1] += 1
                    agree[k][0] += int((items[i][0] < items[j][0]) == (items[i][col] > items[j][col]))
    V["h11"]["ssim_agree"] = agree
    print(f"    SSIM 與 nerr_c 的排名一致:振幅 {agree['amp'][0]}/{agree['amp'][1]}、相位 {agree['ph'][0]}/{agree['ph'][1]}(網路 3 seeds 平均 + 迭代法 seed 0)")
    V["h11"]["eta0"] = {}
    fc0 = fc["L-ideal"]
    for nm in X["eta0"]:
        a0 = float(np.mean([np.where(np.isfinite(x), x, fc0).mean() for x in X["eta0"][nm]]))
        a1 = float(net_c(R, "L-ideal", nm, fc0)[:, :ETA0_N].mean())
        V["h11"]["eta0"][nm] = {"eta0": a0, "eta01_same_fields": a1, "svf1_eta0": float(np.mean(X["eta0_c"][nm]))}
        print(f"    η = 0:{tg(nm):<9} nerr_c {a0:.4f}(η = 0.1 的同一批場 {a1:.4f};雜訊不同)、SV-F1 {np.mean(X['eta0_c'][nm]):.3f}")
    # (h12) 泛化落差與 §13.2 的規則
    print("  (h12)泛化落差 = 驗證 / 訓練(晶格:L-paper 的條件、nerr_c;原本資料:理想條件、nerr_sr)")
    V["h12"] = {}
    for nm in (BASE, NEW_C, NEW_L):
        lt = float(np.mean([np.where(np.isfinite(x), x, fc[MAIN_EXAM]).mean() for x in X["gap_lat"][nm]]))
        lv = float(np.mean(V["exam"][MAIN_EXAM]["nets"][nm]["nerr_c"]))
        ot = float(np.mean([np.where(np.isfinite(x), x, FAIL_SR).mean() for x in X["gap_org"][nm]]))
        ov = float(np.mean(R13["ideal"][nm]["nerr_sr"]))
        V["h12"][nm] = {"lat_train": lt, "lat_val": lv, "lat_gap": lv / max(lt, 1e-12), "org_train": ot, "org_val": ov, "org_gap": ov / max(ot, 1e-12)}
        print(f"    {tg(nm):<9}:晶格 驗證 {lv:.4f} / 訓練 {lt:.4f} = {lv / max(lt, 1e-12):.3f};原本 驗證 {ov:.5f} / 訓練 {ot:.5f} = {ov / max(ot, 1e-12):.3f}")
    bad8b = (not V["h1"]["q05"][0]["reach"]) or V["h5"]["L-paper"]["L / P6B4e8"]["label"] != "改善"
    rg = V["h12"][NEW_L]["lat_gap"] / max(V["h12"][BASE]["org_gap"], 1e-12)
    V["h12"]["rule"] = {"8b_not_good": bool(bad8b), "gap_ratio": rg,
                        "overfit_possible": bool(bad8b and rg >= GAP_X)}
    print(f"    §13.2 的規則:8b 的結果{'不理想' if bad8b else '理想'}(h1 未達 0.05,或 L / P6B4e8 在 L-paper 未達「改善」);"
          f"P6B4e8-L 的晶格落差比 / P6B4e8 的原本資料落差比 = {rg:.2f}"
          + ((" → ≥ 1.5:過擬合是可能的原因,下一版微調納入調整(另寫版本、與使用者確認)" if rg >= GAP_X else " → < 1.5:過擬合不是主因,另找原因")
             if bad8b else "(規則不適用)"))
    return V


# ============================================================================
# 圖(協定 §九;描述,不改變判讀)
# ============================================================================
def figures(V, E, R, keep, IT, tim, iters, info, fc, mroot, out_dir, where):
    plt = s5._plt()
    if plt is None:
        check("matplotlib 可用(畫圖需要)", False)
        return
    from matplotlib.patches import Circle
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    # 1. 時間 vs nerr_c、時間 vs SV-F1(b512 與 b64)
    for key, ylab, name in (("nerr_c", "nerr_c on U", "c_time_nerr_c"), ("sv_miss", "1 - SV-F1 (pooled)", "c_time_svf1")):
        for B in ("64", "512"):
            fig, axs = plt.subplots(1, len(EXAMS), figsize=(4.2 * len(EXAMS), 3.8), squeeze=False)
            for k, e in enumerate(EXAMS):
                ax = axs[0, k]
                ts, vs = [], []
                best = float("inf")
                for t, v in sorted((a7.cfg_time7(c, it, tim[B]), float(a7.env_vals7(E[e], c, it, key).mean()))
                                   for c in a7.configs7() for it in a7.stops7(c, iters)):
                    best = min(best, v)
                    if t > 0:
                        ts.append(t)
                        vs.append(best)
                ax.plot(ts, vs, color=COL["iter"], lw=1.4, drawstyle="steps-post", label="iterative envelope")
                for nm in NETS:
                    row = V["exam"][e]["nets"][nm]
                    y = float(np.mean(row["nerr_c"])) if key == "nerr_c" else 1 - row["svf1"]["all"]
                    ax.plot([tim["net"][nm][B]], [max(y, 1e-6)], "o", color=COL[nm], ms=6, label=tg(nm))
                if key == "nerr_c":
                    for q in QS:
                        ax.axhline(q, color="#d6452a" if q == Q_MAIN else "#e8a598", lw=0.7, ls="--")
                else:
                    ax.axhline(1 - SVF1_GOOD, color="#d6452a", lw=0.7, ls="--")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(e, fontsize=9)
                ax.set_xlabel("time per whole image (ms)")
                ax.grid(True, color="#e4e3df", which="both", lw=0.5)
            axs[0, 0].set_ylabel(ylab)
            axs[0, 0].legend(fontsize=6.5, frameon=False)
            fig.suptitle(f"{where}: time vs {ylab}, batch {B}", fontsize=10)
            fig.tight_layout()
            save(fig, f"{name}_b{B}.png")
    # 2. FRC
    refl = V["h11"]["refl"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for k, mk in enumerate(("W", "Mo")):
        ax = axs[k]
        for nm in NETS:
            cur = np.array(V["h11"]["frc"][nm][mk])
            ax.plot(np.arange(len(cur)) / FRC_N, cur, color=COL[nm], lw=1.2, label=tg(nm))
        if IT.get("frc") is not None:
            cur = np.array(IT["frc"][mk])
            ax.plot(np.arange(len(cur)) / FRC_N, cur, color=COL["iter"], lw=1.2, ls="--", label="iterative (P6B4e8-L time, b512, seed 0)")
        for x in refl:
            ax.axvline(x["ring"] / FRC_N, color="#bbbbbb", lw=0.6)
            ax.text(x["ring"] / FRC_N, 1.02, x["hk"], fontsize=6, ha="center")
        ax.axvline(0.5, color="#d6452a", lw=0.8, ls=":")
        ax.set_ylim(-0.2, 1.08)
        ax.set_xlabel("spatial frequency (cycles / px); red dotted = sampling limit 0.72 A (Nyquist)")
        ax.set_title(f"L-paper, {'WS2' if mk == 'W' else 'MoS2'}: pooled FRC vs truth (48x48 block in U)", fontsize=8)
        ax.grid(True, color="#e4e3df", lw=0.5)
    axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "c_frc.png")
    # 3. 誤差 vs 旋轉角
    fig, ax = plt.subplots(figsize=(7, 4))
    xb = (np.arange(ROT_BINS) + 0.5) * 60.0 / ROT_BINS
    for nm in NETS:
        ax.plot(xb, V["h10"][nm]["bins"], "o-", color=COL[nm], ms=4, label=tg(nm))
    ax.set_xlabel("lattice rotation mod 60 deg")
    ax.set_ylabel("nerr_c (L-paper, 3 seeds)")
    ax.set_yscale("log")
    ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "c_rotation.png")
    # 4. 訓練曲線
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for s in SEEDS:
        for kind, nm in (("L", NEW_L), ("C", NEW_C)):
            p = model_dir(kind, s, mroot) / "result.json"
            if not p.exists():
                continue
            hist = json.load(open(p)).get("history", [])
            ep = [x["epoch"] + 1 for x in hist]
            if kind == "L":
                axs[0].plot(ep, [x["lat"] for x in hist], color=COL[nm], lw=1.2, label="L: lattice part" if s == SEEDS[0] else None)
            axs[0].plot(ep, [x["org"] for x in hist], ":" if kind == "L" else "-", color=COL[nm], lw=1.0,
                        label=f"{tg(nm)}: original part" if s == SEEDS[0] else None)
            axs[1].plot(ep, [x["pos_rms"][-1] for x in hist], color=COL[nm], lw=1.0, label=tg(nm) if s == SEEDS[0] else None)
    axs[0].set_yscale("log")
    axs[0].set_title("8b training loss (deep supervision, object part)", fontsize=8)
    axs[1].set_title("last-stage position RMS (px, 9 windows)", fontsize=8)
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "c_train.png")
    # 5. 重建圖(依 P6B4e8-L seed 0 的 nerr_c 取百分位、最差、隨機;真值的 SV / DV 以圈標出,判讀以顏色)
    rngf = np.random.default_rng(FIG_SEED)
    for e in (MAIN_EXAM, "L-newdef"):
        kp = keep[e]
        mats = info[e]["mats"]
        cN = np.where(np.isfinite(R[e][NEW_L]["c"][0]), R[e][NEW_L]["c"][0], fc[e])
        groups = (("W", 0), ("Mo", 1)) if e == MAIN_EXAM else (("all", None),)
        for gname, gm in groups:
            idx = np.where(mats == gm)[0] if gm is not None else np.arange(len(mats))
            order = idx[np.argsort(cN[idx], kind="stable")]
            picks = [("p10", order[int(0.1 * (len(order) - 1))]), ("p50", order[int(0.5 * (len(order) - 1))]),
                     ("p90", order[int(0.9 * (len(order) - 1))]), ("worst", order[-1])]
            picks += [(f"rand{k}", int(v)) for k, v in enumerate(rngf.choice(idx, size=min(3, len(idx)), replace=False))]
            cols = [("truth", None)] + [(f"iter {k}", k) for k in IT.get("img", {}).get(e, {})] + [(tg(nm), nm) for nm in (BASE, NEW_C, NEW_L)]
            U = kp["geo"].U.cpu()
            y0, y1, x0, x1 = b7._bbox(U, a7.MARGIN)
            fig, axs = plt.subplots(len(picks), len(cols), figsize=(2.6 * len(cols), 2.5 * len(picks)), squeeze=False, constrained_layout=True)
            vac = kp["vac"]
            for r_, (pn, i) in enumerate(picks):
                o = kp["O"][i]
                sel = vac.fid == i
                for k_, (lab, key) in enumerate(cols):
                    ax = axs[r_, k_]
                    if key is None:
                        z, v = o, None
                    elif key in NETS:
                        z, v = a7.align_show(kp["est"][key][i], o, U)
                    else:
                        z, v = a7.align_show(IT["img"][e][key][i], o, U)
                    hp = ax.imshow(np.angle(z[y0:y1, x0:x1].numpy()), cmap="viridis", vmin=0, vmax=0.9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ttl = lab if v is None else f"{lab}\nnerr_sr {v:.4f}"
                    ax.set_title(ttl if r_ == 0 else ("" if v is None else f"nerr_sr {v:.4f}"), fontsize=6)
                    lab_i = vac.lab[sel]
                    pr_i = kp["pred"][key][sel] if key in NETS else None
                    pos_i = vac.pos[sel]
                    for j in range(len(lab_i)):
                        yy, xx = float(pos_i[j, 0]) - y0, float(pos_i[j, 1]) - x0
                        tl = int(lab_i[j])
                        if tl == 0:
                            if pr_i is not None and int(pr_i[j]) != 0:
                                ax.plot([xx], [yy], "x", color="#ff8c00", ms=4, mew=1.0)
                            continue
                        colr = "white" if pr_i is None else ("#1baf7a" if int(pr_i[j]) != 0 else "#d6452a")
                        ax.add_patch(Circle((xx, yy), 2.2 if tl == 1 else 3.0, fill=False, color=colr, lw=0.8, ls="-" if tl == 1 else "--"))
                axs[r_, 0].set_ylabel(f"#{i} ({pn})", fontsize=7)
            fig.colorbar(hp, ax=list(axs[:, -1]), fraction=0.03, label="phase (rad)")
            fig.suptitle(f"{where}: {e} {gname}; circles = truth SV (solid) / DV (dashed): green caught, red missed (white = no reading); "
                         "orange x = false alarm", fontsize=8)
            save(fig, f"c_recon_{e}_{gname}.png")



# ============================================================================
# 彙整流程
# ============================================================================
def load_inputs(env_root, roots, j7_path, iters, smoke):
    E, bad = {}, []
    me = me_md5()
    for e in EXAMS:
        p = env_json(e, env_root)
        if not p.exists():
            bad.append(f"{p.name} 不存在")
            continue
        E[e] = json.load(open(p))
        m = E[e]["meta"]
        if m["script_md5"] != me:
            bad.append(f"{p.name} 由不同版本的 scan_8.py 產生")
        if not E[e]["checks_ok"]:
            bad.append(f"{p.name} 有檢查未通過")
    if not bad:
        n0 = E[EXAMS[0]]["meta"]["n"]
        bad += [f"{e} 的停止點或場數不同" for e in EXAMS if E[e]["meta"]["iters"] != list(iters) or E[e]["meta"]["n"] != n0]
    check("5 份考卷的包絡檔都在、同一版程式、檢查全過、設定一致", not bad, ";".join(bad))
    J7 = json.load(open(j7_path)) if j7_path.exists() else None
    ok7 = J7 is not None and J7.get("script_md5") == D7_MD5 and J7.get("complete") is True
    check(f"7d 的結果檔 {j7_path} 在、版本正確、完整(原本 13 個條件的重現檢查用)", ok7)
    mb = []
    for kind in ("L", "C"):
        for s in SEEDS:
            md = model_dir(kind, s, roots["m"])
            if not ((md / "final.pt").exists() and (md / "result.json").exists()):
                mb.append(f"{md.name} 缺 final.pt / result.json")
            elif json.load(open(md / "result.json")).get("script_md5") != me:
                mb.append(f"{md.name} 由不同版本的 scan_8.py 訓練")
    for s in SEEDS:
        for md in (d7.model_dir("e", s, roots["e"]), b7.model_dir(s, roots["r"])):
            if not ((md / "final.pt").exists() and (md / "result.json").exists()):
                mb.append(f"{md.name} 缺 final.pt / result.json")
    check("6 個新模型(P6B4e8-L、P6B4e8-C 各 3 seeds)由同一版程式訓練;P6B4e8、P6B4r 都在", not mb, ";".join(mb))
    return (E, J7) if not bad and ok7 and not mb else None


def run_final(dev, env_root, roots, j7_path, out_root, fig_dir, iters, n_orig, smoke):
    where = "lattice exams (smoke)" if smoke else "lattice exams, 7x7 scan"
    L = load_inputs(env_root, roots, j7_path, iters, smoke)
    if L is None:
        return None
    E, J7 = L
    dose = load_dose()
    t0 = time.time()
    print("\n  計時(ms / 整張)", flush=True)
    tim = a7.timing(dev)
    tim["net"] = time_nets8(dev, roots)
    print(f"  (計時完成,經過 {time.time() - t0:.0f} 秒;GPU {tim['gpu']})", flush=True)
    n = E[EXAMS[0]]["meta"]["n"]
    print(f"\n  網路評估:5 份考卷 × 3 seeds × 5 個網路({n} 個場)", flush=True)
    R, keep, info = eval_exams(dev, n, smoke, roots, log=lambda s: print(s, flush=True))
    fc = {e: fail_value(E[e]["env"], R, e) for e in EXAMS}
    env = {}
    for e in EXAMS:
        env[e] = E[e]["env"]
        for conf in env[e].values():
            for rec in conf.values():
                finalize_rec(rec, fc[e])
    print(f"\n  網路評估:原本 13 個條件 × 3 seeds × 5 個網路({n_orig} 個場)", flush=True)
    R13 = eval_orig(dev, n_orig, roots, log=lambda s: print(s, flush=True))
    dm = 0.0
    for c in c7.EVAL_CONDS:
        for nm in FROZEN:
            for i in range(len(SEEDS)):
                dm = max(dm, absdiff(R13[c][nm]["nerr_sr"][i], J7["nets"][c][nm]["nerr_sr"][i]))
    check("重現(協定 §八 #4):凍結的 P6B4、P6B4r、P6B4e8 在原本 13 個條件的 G9 nerr_sr = 7d 的結果檔(< 1e-4)", dm < 1e-4, f"最大差 {dm:.1e}")
    print("\n  η = 0 敏感度與泛化落差", flush=True)
    X = eval_extra(dev, smoke, roots, log=lambda s: print(s, flush=True))
    print("\n  迭代法的關鍵設定(重算,seed 0):SSIM、FRC、資料一致性殘差、重建圖", flush=True)
    IT = iter_extra(dev, env, R, tim, iters, n, smoke, fc)
    V = report(env, R, R13, X, IT, tim, iters, fc, dose, info, J7, smoke)
    final = final_json(out_root)
    res = {"verdict": V, "timing": tim, "checks_ok": all_ok(), "smoke": smoke, "script_md5": me_md5(), "d7_md5": D7_MD5,
           "dose": dose, "orig": {c: {nm: {k: R13[c][nm][k] for k in ("nerr_sr", "nerr_ph")} for nm in NETS} for c in R13},
           "stats": {e: info[e]["stats"] for e in EXAMS}, "complete": False}
    dump_json(res, final)
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        figures(V, env, R, keep, IT, tim, iters, info, fc, roots["m"], fig_dir, where)
    except Exception as ex:                                               # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"畫圖完成(判讀不受影響,已存於 {final.name})", False, f"{type(ex).__name__}: {ex}")
    res["checks_ok"] = all_ok()
    res["complete"] = bool(all_ok())                                     # 只有全部檢查通過才標為完整(否則可重跑)
    dump_json(res, final)
    print(f"  結果:{final}")
    return res


@torch.no_grad()
def iter_extra(dev, env, R, tim, iters, n, smoke, fc):
    """迭代法的關鍵設定(seed 0 重算):各考卷「P6B4e8-L 時間下的包絡設定」(b64、b512;依 nerr_c)的 SSIM;
    L-paper 的 FRC(b512)與重建圖(網路時間、達到 Q = 0.02 的設定);L-newdef 的資料一致性殘差與重建圖。另核對重算的 nerr_c = 包絡檔。"""
    IT = {"ssim": {}, "img": {}, "frc": None, "newdef_resid": None}
    worst = 0.0
    for e in EXAMS:
        d = setup_exam(SEEDS[0], dev, e, n, smoke)
        want = {}
        for B in ("64", "512"):
            ev = a7.envelope7(env[e], tim["net"][NEW_L][B], tim[B], iters, key="nerr_c")
            want.setdefault((ev[0], ev[1]), []).append(f"T b{B}")
        if e == MAIN_EXAM:
            (t_, cq, iq), _ = t_first(env[e], "nerr_c", Q_MAIN, tim["512"], iters)
            if cq is not None:
                want.setdefault((cq, iq), []).append(f"Q {Q_MAIN}")
        IT["ssim"][e] = []
        for (conf, it), tags in want.items():
            est = iter_est_L(d, conf, it, dev)
            r = lat_metrics(est, d, extra=True)
            v = float(np.where(np.isfinite(r["c"].numpy()), r["c"].numpy(), fc[e]).mean())
            worst = max(worst, absdiff(v, env[e][conf][str(it)]["nerr_c"][0]))
            IT["ssim"][e].append({"conf": conf, "it": it, "tags": tags, "nerr_c": v, "ssim_amp": float(r["ssim_amp"].mean()),
                                  "ssim_ph": float(r["ssim_ph"].mean())})
            if e == MAIN_EXAM and "T b512" in tags:
                IT["frc"] = {k: frc_curve(*[x.numpy() for x in v_]).tolist() for k, v_ in r["frc"].items()}
            if e in (MAIN_EXAM, "L-newdef"):
                IT["img"].setdefault(e, {})["/".join(tags) + f" {s5b.fmt_conf(conf)} x{it}"] = est.cpu()
            if e == "L-newdef" and "T b512" in tags:
                IT["newdef_resid"] = float(np.nanmean(r["resid"].numpy()))
            del est
        del d
        print(f"  [迭代法重算] {e}:{len(want)} 個設定", flush=True)
    print(f"  描述性核對(不阻擋):重算的迭代法 nerr_c vs 包絡檔(seed 0),最大差 {worst:.1e}")
    IT["recompute_maxdiff"] = worst
    return IT


# ============================================================================
# 內建檢查(協定 §八;全部通過才量測)
# ============================================================================
def scan_seed_constants(skip):
    """既有腳本中出現的大整數常數(≥ 10⁶;含底線寫法)。"""
    found = {}
    for p in sorted(Path(".").glob("*.py")) + sorted(Path("src").glob("*.py")):
        if p.name in skip:
            continue
        for mm in re.finditer(r"(?<![\w.])(\d{1,3}(?:_\d{3}){2,}|\d{7,})(?![\w.])", p.read_text(errors="ignore")):
            v = int(mm.group(1).replace("_", ""))
            found.setdefault(v, set()).add(p.name)
    return found


@torch.no_grad()
def checks(dev, write_dose=True):
    torch.backends.cudnn.benchmark = True
    print("=" * 70)
    print("階段八:內建檢查")
    print("=" * 70)
    import probe_4_2b as pb
    from src.physics import beamstop_mask, forward_measure
    cfg = s5.load_cfg(0)
    pr = pb.build_probes(cfg, dev)[PROBE]
    W = cfg.canvas
    dx = pixel_size(pr, W)
    lam = 12.264259 / math.sqrt(200e3 * (1 + 0.97847e-6 * 200e3))                      # 相對論波長(Å)
    sig = 2 * math.pi / (lam * 200e3) * (511e3 + 200e3) / (2 * 511e3 + 200e3)
    check(f"物理常數:Δx = {dx:.4f} Å/px(協定 {DX_PROTO},差 < 0.5%);λ、σ 與相對論公式一致(< 0.1%)",
          abs(dx / DX_PROTO - 1) < 5e-3 and abs(lam / LAMBDA - 1) < 1e-3 and abs(sig / SIGMA - 1) < 1e-3,
          f"λ {lam:.6f} vs {LAMBDA:.6f};σ {sig:.4e} vs {SIGMA:.4e};晶格常數 MoS₂ {A_LAT['MoS2'] / dx:.3f} px、WS₂ {A_LAT['WS2'] / dx:.3f} px")
    # (2a) 散射因子表 vs abTEM
    kk = torch.tensor(KREF_K, dtype=torch.float64) ** 2
    err = max(float(((kirk_F(el, kk) - torch.tensor(KREF[el], dtype=torch.float64)).abs() / torch.tensor(KREF[el], dtype=torch.float64)).max())
              for el in KREF)
    check("相位校準 (1):投影散射因子 = abTEM 的 Kirkland 計算(k = 0–2 Å⁻¹,頻帶內相對差 < 1%)", err < 1e-2, f"最大相對差 {err:.1e}")
    # (2b) 孤立柱的峰值與積分相位
    G = 128
    tabs = element_tables(G, dx, dev)
    kp = torch.fft.fftfreq(G, dtype=torch.float64, device=dev)
    iso = {}
    for el, nS in (("Mo", 1), ("W", 1), ("S", 2)):
        row = []
        for off in (0.0, 0.5):
            r0 = G // 2 + off
            ph = torch.polar(torch.ones_like(kp), -2 * math.pi * kp * r0)
            S = (ph[:, None] * ph[None, :]).to(torch.complex64) * nS
            img = torch.fft.ifft2(tabs[el] * S).real
            row.append((float(img.max()), float(img.sum()) * dx * dx))
        iso[el] = row
    txt = ";".join(f"{el}{'₂' if el == 'S' else ''} 峰值 {v[0][0]:.3f}(偏 0.5 px:{v[1][0]:.3f})、積分 {v[0][1]:.3f} rad·Å²" for el, v in iso.items())
    ratio = iso["W"][0][0] / iso["Mo"][0][0]
    check("相位校準 (2):孤立柱的峰值(頻帶限制、含 DW)與審查的估計相符(Mo ≈ 0.59、W ≈ 0.84、S₂ ≈ 0.55 rad,± 5%);次像素偏移的影響(印出)",
          abs(iso["Mo"][0][0] / 0.59 - 1) < 0.05 and abs(iso["W"][0][0] / 0.84 - 1) < 0.05 and abs(iso["S"][0][0] / 0.55 - 1) < 0.05,
          txt + f";W / Mo = {ratio:.3f}")
    # (1) 晶格幾何
    from scipy import stats as sst
    F = 112
    Uf = a6.Geom(cfg, pr, dev).U.cpu()
    res_a = {}
    for mat in MATS:
        f0, mt0 = make_lattice([SEED_CHECK], 256, dev, materials=[mat], fixed={"theta": 0.0, "e": [0, 0, 0], "psv": 0.0, "pdv": 0.0}, dx=dx)
        ph = f0[0, 1].double()
        ph = ph - ph.mean()
        yy, xx = torch.meshgrid(torch.arange(256, dtype=torch.float64, device=dev), torch.arange(256, dtype=torch.float64, device=dev), indexing="ij")
        best = (0.0, 0.0)
        for kmag in np.arange(0.10, 0.16, 0.0002):
            for ang in (0.0, 60.0, 120.0):                                              # θ = 0:(100) 反射在 a1 / a2 的倒晶格方向
                a = math.radians(ang + 30.0)
                v = float(torch.abs((ph * torch.exp(-2j * math.pi * kmag * (math.sin(a) * yy + math.cos(a) * xx))).sum()))
                if v > best[0]:
                    best = (v, kmag)
        a_est = 2.0 / (math.sqrt(3) * best[1])
        res_a[mat] = (a_est, A_LAT[mat] / dx)
    idx, R_ = index_grid(F + 2 * HALO, dx)
    prm = draw_params(SEED_CHECK + 1, idx.shape[0], "MoS2", {"theta": 0.0, "e": [0, 0, 0]})
    metal, s2, hol, _ = geom_of(prm, idx, F, dx)
    dmin = float(torch.cdist(s2[:200], metal).min())
    a_px = A_LAT["MoS2"] / dx
    hd = float((hol[:50] - s2[:50, None]).norm(dim=-1).max() / (a_px / math.sqrt(3)))
    nchk = 4000 if not QUICK else 400
    pp = [draw_params(SEED_CHECK + 10 + i, idx.shape[0], None, {"psv": 0.02, "pdv": 0.01}) for i in range(nchk // 100)]
    nS = torch.cat([p["nS"] for p in pp])
    pSV_exp, pDV_exp = (1 - 0.01) * 2 * 0.02 * 0.98, 0.01 + (1 - 0.01) * 0.02 ** 2
    nn_ = len(nS)
    fSV, fDV = float((nS == 1).float().mean()), float((nS == 0).float().mean())
    se = lambda p: math.sqrt(p * (1 - p) / nn_)                                          # noqa: E731
    th = [draw_params(SEED_CHECK + 100_000 + i, 1)["theta"] for i in range(nchk)]
    ks = sst.kstest(np.array(th) / ROT_MAX, "uniform").pvalue
    Sb = s5c.geometry(cfg)[1]
    fA, _ = make_lattice([SEED_CHECK + 7 + i for i in range(4)], Sb, dev, dx=dx)                 # 訓練的方框大小
    big, _ = make_lattice([SEED_CHECK + 7 + i for i in range(4)], F, dev, dx=dx)                 # 整張(泛化落差用同一個 seed)
    o_ = (F - Sb) // 2
    dd_ = fA[:, 1] - big[:, 1, o_:o_ + Sb, o_:o_ + Sb]
    halo_d = float(dd_.abs().max())
    halo_r = float(dd_.pow(2).mean().sqrt() / big[:, 1].std())
    check("晶格幾何:無應變場的布拉格峰換算的晶格常數 = 8.83 / 8.80 px ± 1%;金屬–S₂ 最近距離 = a/√3 ± 2%;空心位置距 S₂ = a/√3;"
          "SV / DV 的實際比例 = 設定值 ± 3 個標準誤;旋轉角均勻(KS p > 0.01);同一個 seed 在方框(80)與整張(112)是同一片晶格"
          "(差 = 頻帶硬截止的長尾振鈴:最大 < 0.02 rad、RMS < 相位 RMS 的 2%)",
          all(abs(v[0] / v[1] - 1) < 0.01 for v in res_a.values()) and abs(dmin / (a_px / math.sqrt(3)) - 1) < 0.02 and abs(hd - 1) < 1e-6
          and abs(fSV - pSV_exp) < 3 * se(pSV_exp) and abs(fDV - pDV_exp) < 3 * se(pDV_exp) and ks > 0.01 and halo_d < 0.02 and halo_r < 0.02,
          "、".join(f"{m} {v[0]:.3f} vs {v[1]:.3f} px" for m, v in res_a.items())
          + f";最近距離 / (a/√3) {dmin / (a_px / math.sqrt(3)):.4f};SV {fSV:.4f}(預期 {pSV_exp:.4f})、DV {fDV:.4f}(預期 {pDV_exp:.4f});"
            f"KS p {ks:.3f};方框 vs 整張:最大差 {halo_d:.1e} rad、RMS 差 / 相位 RMS {halo_r:.1%}")
    # (2c) 完整晶格的相位範圍、SV 柱 / 完整 S₂ 柱
    nv = 16 if QUICK else 512
    fv, mv = make_lattice([SEED_VAL["obj"] + i for i in range(nv)], F, dev, materials=[MATS[i % 2] for i in range(nv)], info=True, U=Uf, dx=dx)
    nt = 16 if QUICK else 1000
    ft, mt = make_lattice([SEED_TRAIN_OBJ + i for i in range(nt)], s5c.geometry(cfg)[1], dev, dx=dx)
    vac = Vac(mv["cols"], F, Uf, fv[:, 1].cpu())
    flat = fv[:, 1].cpu().reshape(nv, -1)
    peak = torch.stack([flat[vac.fid[:, None], vac.cp].max(1).values])[0]
    sv_ratio = []
    for i in range(nv):
        sl = vac.fid == i
        mi = peak[sl & (vac.lab == 0)].median()
        if bool((sl & (vac.lab == 1)).any()):
            sv_ratio.append(float(peak[sl & (vac.lab == 1)].mean() / mi))
    rs = float(np.mean(sv_ratio)) if sv_ratio else float("nan")
    negf = max(max(mv["neg_frac"]), max(mt["neg_frac"]))
    check("相位校準 (3):完整晶格的最大相位 < π/2(全部驗證場與訓練場的抽樣);SV 柱 / 完整 S₂ 柱的峰值比在 0.4–0.6;負相位(描述)",
          mv["phase_max"] < math.pi / 2 and mt["phase_max"] < math.pi / 2 and 0.4 <= rs <= 0.6,
          f"最大相位 驗證 {mv['phase_max']:.3f}、訓練 {mt['phase_max']:.3f} rad;SV / S₂ 峰值比 {rs:.3f}(場數 {len(sv_ratio)});"
          f"負相位最小 {min(mv['neg_min'], mt['neg_min']):.4f} rad、< {NEG_TOL} rad 的像素比例最大 {negf:.2%}"
          + (f"(> {NEG_FRAC:.0%}:回報、與使用者討論)" if negf > NEG_FRAC else ""))
    # (3) 劑量
    D = dose_all(dev)
    if write_dose:
        if DOSE_JSON.exists():
            old = json.load(open(DOSE_JSON))
            same = all(abs(old["seeds"][s]["N_inc"] / D["seeds"][s]["N_inc"] - 1) < 1e-9 for s in D["seeds"])
            if not same:
                raise SystemExit(f"❌ {DOSE_JSON} 已存在但數值不同:先告訴 Claude")
        dump_json({**D, "script_md5": me_md5(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, DOSE_JSON)
        _DOSE.pop("d", None)
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS["3x3s8"][2])
    bs = beamstop_mask(cfg, device=dev)
    one = torch.ones(nchk // 20 if not QUICK else 8, 2, W, W, device=dev)
    one[:, 1] = 0
    psi = pr["P"] * torch.polar(one[:, 0], one[:, 1])
    cnt = forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), torch.ones_like(bs), cp)
    r1 = float(cnt.sum((-2, -1)).mean()) / D["seeds"]["0"]["N_inc"]
    geo = a6.Geom(cfg, pr, dev)
    ops = a7.Ops(pr, W, dev)
    nd = 8
    fd, md_ = make_lattice([SEED_DOSE_CAL + i for i in range(nd)], F, dev, materials=[MATS[i % 2] for i in range(nd)], dx=dx)
    z0 = torch.zeros(nd, 49, 2, device=dev)
    sg0 = torch.ones(nd, device=dev)
    dv = torch.tensor([D["seeds"]["0"]["f_W"] if i % 2 == 0 else D["seeds"]["0"]["f_Mo"] for i in range(nd)])
    cP = measure_L(fd, geo, cp, bs, SEED_DOSE_CAL, ESPEC["L-paper"], z0, sg0, ops, dv)
    eP = measure_L(fd, geo, cp, bs, SEED_DOSE_CAL, ESPEC["L-paper"], z0, sg0, ops, dv, poisson=False)
    ddv = dv.to(dev)[:, None, None]
    r3 = sum(float((c * ddv).sum()) for c in cP) / sum(float((c * ddv).sum()) for c in eP)
    det = float(torch.stack([c.sum((-2, -1)) for c in eP]).mean())
    check("劑量 (3 種):(1) 透明物體的平均總計數 / N_inc = 1 ± 1%;(2) TMD 場的平均偵測計數(描述);(3) L-paper 的偵測計數 / 期望 = 1 ± 2%",
          abs(r1 - 1) < 0.01 and abs(r3 - 1) < 0.02,
          f"(1) {r1:.4f};(2) 每張 {det:.4g}(劑量 1 時;N_inc {D['seeds']['0']['N_inc']:.4g});(3) {r3:.4f};"
          f"f_W {D['seeds']['0']['f_W']:.4g}、f_Mo {D['seeds']['0']['f_Mo']:.4g} → 晶格樣本的劑量下限 {D['lat_dose_lo']:.4g}"
          + ("(f < 0.01:論文劑量超出原本的訓練範圍,依 §3.4 的規則)" if D["out_of_range"] else ""))
    # (5) seed 不重疊與期末考保護
    found = scan_seed_constants({Path(__file__).name})
    clash = sorted({v for v in found for lo, hi in MY_RANGES if lo <= v <= hi})
    derived = []
    for b, nm in ((50_000_000, "訓練場"), (80_000_000, "驗證場"), (82_000_000, "雜訊"), (83_000_000, "位置"), (84_000_000, "離焦"),
                  (95_000_000, "期中考"), (96_000_000, "期中考"), (97_000_000, "期末考")):
        for s in SEEDS:
            for off in (0, 1000 * s, 1000 * s + 1, 10_000_000 * s):
                v = b + off
                if any(lo <= v <= hi for lo, hi in MY_RANGES):
                    derived.append(v)
    ranges_ok = all(MY_RANGES[i][1] < MY_RANGES[i + 1][0] for i in range(len(MY_RANGES) - 1))
    rej = []
    for fn in (lambda: draw_params(98_000_000, 1), lambda: draw_params(98_999_999, 1),
               lambda: make_lattice([98_123_456], 16, dev, dx=dx), lambda: guard([98_500_000])):
        try:
            fn()
            rej.append(False)
        except ValueError:
            rej.append(True)
    check("seed:本協定的範圍與既有腳本的常數(grep)不重疊、與既有的派生範圍不重疊、彼此不重疊;98,000,000–98,999,999 一律拒絕(只測拒絕)",
          not clash and not derived and ranges_ok and all(rej),
          f"既有腳本中的大整數 {len(found)} 個,落在本協定範圍內的 {clash or '無'};派生 {derived or '無'};拒絕 {rej}")
    # (6) nerr_c 與對齊
    dI = setup_exam(0, dev, "L-ideal", 8, smoke=True)
    O, U = dI["O"], dI["geo"].U
    v_ref = a7.align_err(O * torch.exp(torch.tensor(0.3j, device=dev)) + 0.02 * torch.randn_like(O.real), O, U)
    noisy = O * torch.exp(torch.tensor(0.3j, device=dev)) + 0.02 * torch.randn_like(O.real)
    va = align_full(noisy, O, U)
    vb = a7.align_err(noisy, O, U)
    same = max(float((x - y).abs().max()) for x, y in zip(va[:5], vb))
    r_t = lat_metrics(O.clone(), dI)
    m = U.to(O.real.dtype)
    cs = (O * m).sum((1, 2)) / m.sum()
    r_c = lat_metrics(cs[:, None, None].expand_as(O).contiguous(), dI)
    flat_ = torch.polar(torch.ones_like(O.real), torch.full_like(O.real, float(torch.angle(cs).mean())))
    r_f = lat_metrics(flat_, dI)
    r_n = lat_metrics(noisy, dI)
    ident = float(((r_n["c"] - r_n["c_sr"]) * dI["Ec"].cpu() / dI["E"].cpu()).abs().max())      # 以 nerr_sr 的單位比較(float32 的抵消誤差 ~1e-7)
    check("nerr_c:align_full 的三種誤差與平移 / 斜坡 = scan_7a.align_err(< 1e-6);真值 → 0;最佳複數常數 c* → 1(± 1e-6);"
          "nerr_c × Σ|真值 − c*|² = nerr_sr × Σ|真值|²(同一組對齊參數;差 / Σ|真值|² < 1e-6);振幅 1 的平面 → 約 1(描述)",
          same < 1e-6 and float(r_t["c"].abs().max()) < 1e-6 and float((r_c["c"] - 1).abs().max()) < 1e-6 and ident < 1e-6,
          f"align 差 {same:.1e};真值 {float(r_t['c'].abs().max()):.1e};c* {float(r_c['c'].min()):.6f}–{float(r_c['c'].max()):.6f};"
          f"恆等式 {ident:.1e};振幅 1 的平面 {float(r_f['c'].mean()):.3f}(nerr_sr {float(r_f['sr'].mean()):.4f} ← 為什麼不能直接用 nerr_sr 的門檻)")
    del v_ref
    # (7) 空缺判讀
    Ct = r_t["conf"].sum(0).numpy()
    diag = bool((Ct - np.diag(np.diag(Ct))).sum() == 0)
    st_t = vac_stats(Ct)
    fq = torch.fft.fftfreq(F, d=dx, device=dev)
    lp = ((fq[:, None] ** 2 + fq[None, :] ** 2) <= 1.0).to(torch.complex64)
    O_lp = torch.fft.ifft2(torch.fft.fft2(O) * lp)
    r_lp = lat_metrics(O_lp, dI)
    st_lp = vac_stats(r_lp["conf"].sum(0).numpy())
    neg = torch.polar(torch.ones_like(O.real), -torch.angle(O))
    noise = torch.polar(torch.ones_like(O.real), 0.3 * torch.rand_like(O.real))
    unread = [bool((~lat_metrics(x, dI)["readable"]).all()) for x in (cs[:, None, None].expand_as(O).contiguous(), neg, noise)]
    nlab = [int((dI["vac"].lab == k).sum()) for k in range(3)]
    check("空缺判讀:真值 → 混淆矩陣為對角(SV / DV 召回率 100%、誤報 0);真值低通到 1/(1.0 Å) → SV-F1 ≥ 0.95;常數平面、負對比、純雜訊 → 無法判讀",
          diag and st_t["sv_r"] == 1.0 and st_t["fpr"] == 0.0 and st_lp["sv_f1"] >= 0.95 and all(unread),
          f"柱數 完整 / SV / DV = {nlab};真值:{st_t['conf']};低通:SV-F1 {st_lp['sv_f1']:.3f}、DV-F1 {st_lp['dv_f1']:.3f};無法判讀 {unread}")
    # (8) FRC
    y0, x0, inside = frc_block(U)
    s1 = frc_sums(O, O, y0, x0)
    f_same = frc_curve(*s1)
    rnd = torch.polar(torch.rand_like(O.real), torch.rand_like(O.real))
    nrep = 64
    acc = [torch.zeros_like(s1[0]) for _ in range(3)]
    for k in range(nrep // 8):
        dk = setup_exam(0, dev, "L-ideal", 8, smoke=True, obj_seeds=[SEED_CHECK + 500 + 8 * k + i for i in range(8)]) if k else dI
        rk = torch.polar(torch.rand_like(dk["O"].real), torch.rand_like(dk["O"].real))
        sm = frc_sums(dk["O"], rk, y0, x0)
        acc = [a_ + b_ for a_, b_ in zip(acc, sm)]
    f_rnd = frc_curve(*acc)
    ok_r = np.isfinite(f_rnd)
    check(f"FRC:{FRC_N} × {FRC_N} 方塊完整落在 U 內;真值 vs 真值 = 1;真值 vs 獨立的隨機場(合併 {nrep} 個場):功率足夠的環上 |平均| < 0.05",
          inside and np.nanmax(np.abs(f_same - 1)) < 1e-6 and abs(float(np.mean(f_rnd[ok_r]))) < 0.05,
          f"方塊左上角 ({y0}, {x0});自己 {np.nanmax(np.abs(f_same - 1)):.1e};隨機 平均 {np.mean(f_rnd[ok_r]):+.3f}")
    del rnd
    # (9) 部分同調、量測路徑
    fs = dI["fields"][:4]
    zz = torch.zeros(4, 49, 2, device=dev)
    ss = torch.ones(4, device=dev)
    one4 = torch.ones(4)
    e_ref = measure_L(fs, dI["geo"], dI["cp"], dI["bs"], 0, ESPEC["L-ideal"], zz, ss, ops, one4, poisson=False)
    e_c0 = measure_L(fs, dI["geo"], dI["cp"], dI["bs"], 0, dict(ESPEC["L-ideal"], coh=1e-9), zz, ss, ops, one4, poisson=False)
    e_c5 = measure_L(fs, dI["geo"], dI["cp"], dI["bs"], 0, dict(ESPEC["L-ideal"], coh=0.5), zz, ss, ops, one4, poisson=False)
    cp0 = copy.deepcopy(dI["cp"])
    cp0.add_poisson = False
    s6 = a6.measure49(fs, dI["geo"], cp0, dI["bs"], 0)
    d6 = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(e_ref, s6))
    dcoh = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(e_c0, e_ref))
    tot = sum(float(c.sum()) for c in e_c5) / sum(float(c.sum()) for c in e_ref)
    zi = torch.zeros(4, 49, 2, device=dev)
    zi[..., 0], zi[..., 1] = 2.0, -1.0
    e_sh = measure_L(fs, dI["geo"], dI["cp"], dI["bs"], 0, dict(ESPEC["L-ideal"], pos=1.0), zi, ss, ops, one4, poisson=False)
    ref = []
    for (yy0, xx0) in dI["geo"].starts:
        win = fs[:, :, yy0:yy0 + W, xx0:xx0 + W]
        psi = torch.roll(pr["P"], (2, -1), dims=(-2, -1)) * torch.polar(win[:, 0], win[:, 1])
        ref.append(forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), dI["bs"], cp0))
    dsh = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(e_sh, ref))
    check("量測:期望強度 = 階段六的量測(無雜訊,相對差 < 1e-5);位置誤差方向 = 探針 torch.roll((2, −1))(< 1e-4);"
          "部分同調 σ_s → 0 = 無同調誤差(< 1e-5);σ_s = 0.5 px 的總計數不變(± 3%)",
          d6 < 1e-5 and dsh < 1e-4 and dcoh < 1e-5 and abs(tot - 1) < 0.03, f"{d6:.1e}、{dsh:.1e}、{dcoh:.1e}、總計數比 {tot:.4f}")
    # (10) 訓練:誤差抽樣 = scan_7c(非晶格樣本逐位元相同);晶格樣本的劑量下限;方框的量測可用
    B = 4096
    a = c7.draw_errors_7c(B, torch.Generator().manual_seed(5))
    lat = torch.arange(B) % 2 == 0
    b = draw_errors_8(B, torch.Generator().manual_seed(5), lat, D["lat_dose_lo"])
    b0 = draw_errors_8(B, torch.Generator().manual_seed(5), torch.zeros(B, dtype=torch.bool), D["lat_dose_lo"])
    same_o = all(torch.equal(x[~lat], y[~lat]) for x, y in zip(a, b)) and all(torch.equal(x, y) for x, y in zip(a, b0))
    dl = b[2][lat]
    lo_ok = float(dl.min()) >= D["lat_dose_lo"] * (1 - 1e-5) and float((dl == 1).float().mean()) > 0.4
    tm = TrainMeas8(cfg, pr, cp, bs, dev)
    box = ft[:8].to(dev)
    cnt8, _ = tm(box, 3, 3, torch.ones(8, dtype=torch.bool), D["lat_dose_lo"])
    ok8 = len(cnt8) == 9 and all(torch.isfinite(c).all() for c in cnt8)
    check("訓練的誤差抽樣:非晶格樣本 = scan_7c.draw_errors_7c(逐位元);晶格樣本的劑量 ≥ 下限、一半 = 1;方框的晶格場可量測",
          same_o and lo_ok and ok8, f"相同 {same_o};晶格劑量最小 {float(dl.min()):.4g}(下限 {D['lat_dose_lo']:.4g});量測 {ok8}")
    # 新缺陷
    dN = setup_exam(0, dev, "L-newdef", 8, smoke=True)
    base_same = float((dN["O"] - dI["O"]).abs().max())
    nd_ok = all(1 <= len(x["lines"]) <= 3 and all(l_["in_u"] >= LD_MIN_IN for l_ in x["lines"]) for x in dN["meta"]["newdef"])
    catn = dN["vac"].cat
    check("新缺陷(L-newdef):線缺陷 1–3 條、每條至少 5 個位置在 U 內;缺 S 區域在 U 內;其餘的晶格與 L-ideal 相同(描述:最大差)",
          nd_ok and int(((catn == CAT_MR) | (catn == CAT_LD)).sum()) > 0,
          f"線缺陷的柱 {int((catn == CAT_LD).sum())}、缺 S 區域的柱 {int((catn == CAT_MR).sum())}(可判讀者);與 L-ideal 的場最大差 {base_same:.3f}")
    r_nd = lat_metrics(dN["O"].clone(), dN)
    Cn = r_nd["conf_cat"].sum(0).numpy()
    st_new, st_iso = vac_stats(Cn[1]), vac_stats(Cn[0])
    Ca = r_nd["conf"].sum(0).numpy()
    diag_n = bool((Ca - np.diag(np.diag(Ca))).sum() == 0)
    vf = float((dN["vac"].lab > 0).float().mean())
    Mi = torch.full((dI["vac"].n, dI["vac"].cmax), float("nan"))
    Mi[dI["vac"].fid, dI["vac"].slot] = dI["vac"].s_true
    gm_d = float(((gmm_scale(Mi) - dI["vac"].m_true) / dI["vac"].m_true).abs().max())
    check(f"L-newdef 的判讀在真值上(正規化 {NEWDEF_NORM}):混淆矩陣為對角、新缺陷與孤立缺陷的補回率 = 0",
          diag_n and st_new["fill"] == 0 and st_iso["fill"] == 0,
          f"可判讀柱的空缺比例 {vf:.1%};混淆矩陣 {Ca.astype(int).tolist()};補回率 新缺陷 {st_new['fill']:.1%}、孤立 {st_iso['fill']:.1%};"
          f"描述:L-ideal 的真值上,混合模型的 m 與中位數相對差最大 {gm_d:.1e}")
    # (6 補)對齊的局部細化(描述):在找到的平移 / 斜坡附近 ± 1 個格點內,nerr_sr 有沒有更小的值
    v0_, v1_, v2_, t_, q_, _ = align_full(noisy, O, U)
    grid_, Ey_, ky_, kx_, qg_, Qy_, yc_ = a7._sh_mats(F, dev)
    better = 0.0
    for dy in (-1, 0, 1):
        for dxx in (-1, 0, 1):
            for mode in ("t", "q"):
                if dy == 0 and dxx == 0:
                    continue
                tt = t_ + (torch.tensor([dy, dxx], device=dev) * a7.SH_STEP if mode == "t" else 0)
                qq = q_ + (torch.tensor([dy, dxx], device=dev) * a7.RAMP_STEP if mode == "q" else 0)
                ph_ = 2 * math.pi * (ky_ * tt[:, 0, None, None] + kx_ * tt[:, 1, None, None])
                es_ = torch.fft.ifft2(torch.fft.fft2(noisy) * torch.polar(torch.ones_like(ph_), ph_)) * a7._ramp_field(qq, yc_)
                better = max(better, float((v2_ - a6.per_sample(es_, O, U)).max()))
    check("對齊的局部細化(描述;協定 §八 #6):找到的平移 / 斜坡附近 ± 1 個格點內,nerr_sr 沒有明顯更小的值(差 < 1e-4;記錄差值)",
          better < 1e-4, f"最大改善 {better:.1e}")
    # 原本訓練場的前綴(泛化落差用)
    b0_, S_, _ = s5c.geometry(cfg)
    fo = org_train_fields(cfg, 0, 4)
    ft0 = s5c.make_train_fields(cfg, s5c.CHUNK, 0)[:4]
    check("原本訓練池的前幾個場:泛化落差用的整張場(第一塊的前 n 個)的方框 = 訓練池的前 n 個(逐位元)",
          torch.equal(fo[:, :, b0_:b0_ + S_, b0_:b0_ + S_].contiguous(), ft0))
    print(f"\n     裝置:{dev}({a7.gpu_name(dev)})")


# ============================================================================
# 主程式
# ============================================================================
def check_files(kind):
    ok = d7.check_files("final" if kind == "final" else ("smoke" if kind == "smoke" else "check"))
    m = md5(Path(d7.__file__).resolve())
    check("scan_7d.py = 7d 的版本(P6B4e8 的模型與 13 個條件的結果)", m == D7_MD5, f"md5 {m[:8]}…")
    need = []
    if kind in ("final", "task"):
        need += [d7.model_dir("e", s) / f for s in SEEDS for f in ("final.pt", "result.json")]
    if kind == "final":
        need += [d7.final_json()]
    if kind == "smoke":
        need += [d7.model_dir("e", s, d7.SMOKE_DIR) / f for s in SEEDS for f in ("final.pt", "result.json")] + [d7.final_json(d7.SMOKE_DIR)]
    missing = [str(p) for p in need if not p.exists()]
    check(f"階段八另外需要的 {len(need)} 個檔案都存在(P6B4e8 的模型、7d 的結果檔)", not missing, "缺:" + ", ".join(missing) if missing else "")
    return ok and m == D7_MD5 and not missing


def need_passed(me):
    if not PASSED.exists():
        raise SystemExit(f"❌ 找不到 {PASSED}:先在 dev 節點跑 python scan_8.py --smoke,全部通過才可送件")
    pm = json.load(open(PASSED))["script_md5"]
    if pm != me:
        raise SystemExit(f"❌ smoke 通過時的 scan_8.py(md5 {pm})與現在的({me})不同:用現在的版本重跑 --smoke")
    print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", type=int, choices=range(len(TASKS)))
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = me_md5()
    print(f"scan_8.py md5 {me};GPU {a7.gpu_name(dev)}")
    kind = "final" if a.final else ("smoke" if a.smoke else ("task" if a.task is not None else "check"))
    if not check_files(kind):
        print("\n❌ 缺檔案或版本不同,停下來")
        sys.exit(1)
    if a.check:
        checks(dev)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.smoke:
        print("#" * 70)
        print(f"迷你全流程(--smoke):檢查、5 份考卷({SMOKE_FIELDS} 個場、停止點 {a7.SMOKE_ITERS})、P6B4e8-L / P6B4e8-C 各 3 seeds × "
              f"{SMOKE_EPOCHS} epoch × {SMOKE_TRAIN_N} 個場(起點 = 7d 的 smoke 模型)、彙整;輸出到 {SMOKE_DIR}")
        print("#" * 70)
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)                                      # 只刪這個 smoke 專用資料夾
        SMOKE_DIR.mkdir(parents=True)
        t0 = time.time()
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
        te = {}
        for e in EXAMS:
            t1 = time.time()
            run_env(dev, e, SMOKE_FIELDS, a7.SMOKE_ITERS, SMOKE_DIR, smoke=True)
            te[e] = time.time() - t1
        tt = time.time()
        for k, s in TASKS[len(EXAMS):]:
            train_8b(k, s, dev, SMOKE_TRAIN_N, SMOKE_EPOCHS, model_dir(k, s, SMOKE_DIR), src_root=d7.SMOKE_DIR, smoke=True)
        per_tr = (time.time() - tt) / (len(TASKS) - len(EXAMS))
        tf = time.time()
        run_final(dev, SMOKE_DIR, roots_of(True), d7.final_json(d7.SMOKE_DIR), SMOKE_DIR, SMOKE_DIR / "figs", a7.SMOKE_ITERS,
                  a7.SMOKE_FIELDS, smoke=True)
        tfin = time.time() - tf
        print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒(考卷 " + "、".join(f"{e} {v:.0f}" for e, v in te.items())
              + f" 秒;每個迷你訓練 {per_tr:.0f} 秒;彙整 {tfin:.0f} 秒)")
        nf = s5b.N_FIELDS or s5.load_cfg(0).eval_n
        k_it = sum(a7.SMOKE_ITERS) / max(1, sum(a7.ITERS))
        est_env = np.mean(list(te.values())) * (nf / SMOKE_FIELDS) / max(k_it, 1e-9)
        print(f"  粗估正式的每份考卷:約 {est_env / 3600:.1f} 小時(線性外推:場數 × {nf / SMOKE_FIELDS:.0f}、迭代次數 × {1 / k_it:.0f};"
              f"偏保守,實際以計時為準;job 時限 4 小時)")
        if all_ok():
            json.dump({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
            print("\n✅ 迷你全流程全部通過 → 可以送出 run_scan8_jobs.sh,再送 run_scan8_final.sh")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    if a.task is not None:
        need_passed(me)
        k, v = TASKS[a.task]
        if k == "env":
            p = env_json(v)
            if p.exists():
                raise SystemExit(f"❌ {p} 已存在:這份考卷已經跑完。為避免覆蓋,先告訴 Claude")
        else:
            md = model_dir(k, v)
            if (md / "final.pt").exists():
                raise SystemExit(f"❌ {md}/final.pt 已存在:這個模型已經訓練完。為避免覆蓋,先告訴 Claude")
        load_dose()
        checks(dev, write_dose=False)
        D = dose_all(dev)
        same = all(abs(load_dose()["seeds"][s]["N_inc"] / D["seeds"][s]["N_inc"] - 1) < 1e-9 for s in D["seeds"])
        check("劑量:讀回的 scan8_dose.json = 本程式的計算(同一組 cfg)", same)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有執行。把輸出貼給 Claude")
            sys.exit(1)
        if k == "env":
            run_env(dev, v, s5b.N_FIELDS or s5.load_cfg(0).eval_n, a7.ITERS, RUN_ROOT, smoke=False)
            print("\n" + (f"✅ 考卷 {v} 完成、檢查全過" if all_ok() else f"❌ 考卷 {v} 有檢查未通過"))
        else:
            train_8b(k, v, dev)
            print("\n" + (f"✅ {model_dir(k, v).name} 訓練完成、檢查全過" if all_ok() else f"❌ {model_dir(k, v).name} 有檢查未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.final:
        out = final_json()
        if out.exists() and json.load(open(out)).get("complete"):
            raise SystemExit(f"❌ {out} 已存在且完整:正式結果已經有了。為避免覆蓋,先告訴 Claude")
        need_passed(me)
        checks(dev, write_dose=False)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有彙整。把輸出貼給 Claude")
            sys.exit(1)
        nf = s5b.N_FIELDS or s5.load_cfg(0).eval_n
        res = run_final(dev, RUN_ROOT, roots_of(False), d7.final_json(), RUN_ROOT, FIG_DIR, a7.ITERS, nf, smoke=False)
        print("\n" + ("✅ 內建檢查全部通過" if all_ok() and res else "❌ 有項目未通過(判讀先不要採信)"))
        print("把完整輸出與 figs_scan8/ 的圖傳給 Claude")
        sys.exit(0 if all_ok() and res else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
