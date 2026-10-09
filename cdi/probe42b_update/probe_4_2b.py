#!/usr/bin/env python
"""階段四 4-2b(HIO 部分):複數探針能否打破翻轉?(階段四協定 §九;不需訓練)

背景(4-1、4-2a):平頂圓盤探針(實數、中心對稱)的已知範圍,就是 HIO 缺的 support;
但「共軛 + 180° 旋轉」的翻轉歧異打不破(選翻轉約 0.5)。
問:換成有像差的複數探針(真實 STEM 探針),已知探針能不能打破翻轉?能不能讓 HIO 更準?

探針(7 個;皆為已知,強度質心在 (31.5, 31.5),最大振幅 1)
  DISK  平頂圓盤 r = 12(= 4-1 / 4-2a)
  AMP   = |DEF|(實數、正;振幅與 DEF 完全相同、沒有相位)  ← 相位的對照組
  DEF   孔徑 + 離焦(孔徑邊緣相位 π)
  DEF2  孔徑 + 離焦 2π
  SPH   孔徑 + 離焦 π + 球差 π
  AST   孔徑 + 離焦 π + 像散 π(非旋轉對稱,但仍中心對稱)
  COMA  孔徑 + 離焦 π + 彗差 π(不中心對稱)
  P = IFFT{ A(k) e^{-iχ(k)} },A = 半徑 k_a 的孔徑;χ = 各像差 × (k/k_a)^n(以孔徑邊緣的相位表示)
  k_a 的規則(幾何,事先寫定):由 3.00 起以 0.05 步增加,取第一個「|P| ≥ 0.1 的面積 ≤ 圓盤面積(448 px)」者

HIO(4 種,知道的資訊由少到多;同一份量測、同一個隨機起點)
  A  方框 support + |ψ| ≤ 1(不知道探針的形狀;max|P| = 1 是正規化)。
     註:與 4-1 的 A 不同 —— 4-1 另加了物體相位的約束,複數探針下不成立
  M  知道 |P|:方框內 |ψ| ≤ |P|
  K  知道 P(振幅 + 相位):方框內 ψ = P·O,O 須滿足 |O| ≤ 1、相位 ∈ [0, φmax]
     平頂圓盤時與 4-1 的 P-HIO 逐位元相同(內建檢查)
  S  K + ψ 的真實範圍(上限,診斷用;平頂圓盤時 = 4-1 的 S*ψ)

劑量:每張繞射圖的總光子數與平面波相同(ref_energy 以校準集的能量比縮放,同 4-1)。

翻轉的判定(協定 §9.4):K / S 的輸出帶有已知探針的相位,若在 ψ 空間比對「輸出的翻轉」,隨機的合法輸出也會被判為
「不翻轉」(審查發現的偏誤)。K / S 因此改在**物體空間**判定:ψ̂ 比較像 P·O 還是 P·conj(O(−r))(各取最佳平移與全域相位);
A / M 不知道探針相位,沿用 ψ 空間的對齊。兩種判定對「與真值無關的輸出」皆約 0.5(內建檢查)。

用法(需在計算節點執行;需 probe_4_1.py、probe_4_1.json、realign_eval.py、ambiguity_check.py、src/hio_sw.py):
    python probe_4_2b.py --check   # 檔案 + 探針幾何 + 單元測試(約 1–3 分鐘)
    python probe_4_2b.py           # 完整量測(約 30–60 分鐘;用 run_probe42b.sh 送 job)
"""
import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Cfg                                          # noqa: E402

RUN_ROOT = Path(os.environ.get("CDI_RUN_ROOT", "/work/elviss0915/runs"))
OUT_JSON = RUN_ROOT / "probe42b.json"
REF_41 = RUN_ROOT / "probe_4_1.json"
FIG_DIR = Path("figs_probe42b")
BASE = "ideal_base"
SEEDS = [0, 1, 2]
PROBES = ["DISK", "AMP", "DEF", "DEF2", "SPH", "AST", "COMA"]
PI = math.pi
# 孔徑邊緣的像差相位(rad);AMP = |DEF|,DISK 另計
ABER = {"DEF": dict(defocus=PI), "DEF2": dict(defocus=2 * PI),
        "SPH": dict(defocus=PI, spherical=PI), "AST": dict(defocus=PI, astig=PI),
        "COMA": dict(defocus=PI, coma=PI)}
EVEN = ["DISK", "AMP", "DEF", "DEF2", "SPH", "AST"]                  # 中心對稱的探針
COMPLEX = ["DEF", "DEF2", "SPH", "AST", "COMA"]
DISK_R = 12
TAU = 0.1                                   # 探針範圍的門檻:|P| ≥ 0.1·max(只用於幾何與 k_a 規則)
KA_START, KA_STEP, KA_MAX = 3.0, 0.05, 24.0
VARIANTS = ["A", "M", "K", "S"]
V_LABEL = {"A": "A 方框", "M": "M 知道|P|", "K": "K 知道P", "S": "S 真實範圍"}
ITERS = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
JUDGE_MA = [20, 50, 100]                    # 探針範圍是否仍有效(M − A),同 4-1 的主判準
JUDGE_KM = [50, 100]                        # 探針相位的幫助(K − M)
STRONG_N = 50                               # 距上限(S − K)
MAIN_N, LONG_N = 100, 1000                  # 翻轉判定、跨探針比較
PROTO_N = 50
GOOD_NERR = 0.25                            # 「解出」的樣本(ψ 空間,A / M):對齊後的正規化誤差 < 0.25
GOOD_RHO = 0.875                            # 「解出」的樣本(物體空間,K / S):最佳正規化相關 ≥ 0.875(同範數時 ⟺ 誤差 ≤ 0.25)
OBJ_FLIP = ("K", "S")                       # 以物體空間判翻轉的 HIO
GEO_SHIFT = 4                               # 翻轉可行性的幾何計算:允許翻轉解平移 ±4 px
BS0_VARIANTS = ["M", "K"]                   # 診斷:不擋 beamstop(分開「相位」與「beamstop 擋掉的比例」兩種效果)
BS0_ITERS = [20, 50, 100, 1000]
REPRO_TOL = 0.005
N_OBJ = None                                # None = cfg.eval_n(512)
N_TWIN_GEO = 256                            # 翻轉可行性的幾何計算用幾張校準樣本
DISK_SEED_OFFSET = 3000                     # = 4-1 中 r = 12 的量測 seed(RADII 的第 3 個)

QUICK = os.environ.get("CDI_QUICK") == "1"  # 只供本地測程式流程
if QUICK:
    ITERS, JUDGE_MA, JUDGE_KM, STRONG_N, MAIN_N, LONG_N, PROTO_N = [5, 20, 50], [20, 50], [50], 50, 50, 50, 50
    BS0_ITERS = [20, 50]
    N_OBJ, N_TWIN_GEO = 32, 32
    REF_41 = RUN_ROOT / "probe_4_1_quick.json"

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


# ============================================================================
# 探針
# ============================================================================
def aperture_probe(n, ka, defocus=0.0, spherical=0.0, astig=0.0, coma=0.0):
    """孔徑 + 像差相位的複數探針 [n, n](complex128),強度質心置於 ((n-1)/2, (n-1)/2),最大振幅 1。

    χ(k) = defocus·ρ² + spherical·ρ⁴ + astig·ρ²·cos2θ + coma·ρ³·cosθ,ρ = |k|/k_a(k 以頻率 px 計)。
    偶函數的 χ(離焦、球差、像散)給出中心對稱的 P(P(r) = P(-r));彗差是奇函數,P 不中心對稱。
    置中:頻率空間乘相位斜坡(對頻寬有限的 P 是精確平移);彗差會移動質心,以斜坡補回(迭代至 < 1e-6 px)。
    """
    k = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64)
    ky, kx = torch.meshgrid(k, k, indexing="ij")
    kr = torch.sqrt(kx ** 2 + ky ** 2)
    rho = kr / ka
    th = torch.atan2(ky, kx)
    chi = (defocus * rho ** 2 + spherical * rho ** 4 + astig * rho ** 2 * torch.cos(2 * th)
           + coma * rho ** 3 * torch.cos(th))
    spec = (kr <= ka).to(torch.float64) * torch.exp(-1j * chi)
    c = (n - 1) / 2.0
    ax = torch.arange(n, dtype=torch.float64)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")

    def build(ty, tx):
        return torch.fft.ifft2(spec * torch.exp(-2j * PI * (ky * ty + kx * tx) / n))

    def centroid(p):
        i = p.abs() ** 2
        return float((i * yy).sum() / i.sum()), float((i * xx).sum() / i.sum())

    ty = tx = c
    p = build(ty, tx)
    for _ in range(8):
        cy, cx = centroid(p)
        if abs(cy - c) <= 1e-6 and abs(cx - c) <= 1e-6:
            break
        ty, tx = ty + (c - cy), tx + (c - cx)
        p = build(ty, tx)
    return p / p.abs().max()


def footprint(p):
    return int((p.abs() >= TAU * p.abs().max()).sum())


def tune_ka(n, target, aber):
    """事先寫定的規則:k_a 由 KA_START 起以 KA_STEP 增加,取第一個 footprint ≤ target 者。"""
    steps = int(round((KA_MAX - KA_START) / KA_STEP))
    for i in range(steps + 1):
        ka = round(KA_START + KA_STEP * i, 2)
        p = aperture_probe(n, ka, **aber)
        a = footprint(p)
        if a <= target:
            return ka, a, p
    raise RuntimeError(f"找不到符合規則的 k_a:{aber}")


def build_probes(cfg, dev):
    """回傳 {名稱: dict(P, Pa, Pu, unit, meta)};P/Pu 為 complex64、Pa 為 float32,皆 [H, W]。"""
    import probe_4_1 as p41
    n = cfg.canvas
    D = p41.disk(DISK_R, cfg)                                   # 與 4-1 相同的圓盤
    target = int(D.sum())
    out = {}
    raw = {}
    for name, ab in ABER.items():
        ka, a, p = tune_ka(n, target, ab)
        raw[name] = (p, {"ka": ka, **{k: v / PI for k, v in ab.items()}})
    for name in PROBES:
        if name == "DISK":
            P64 = D.to(torch.float64).to(torch.complex128)
            meta = {"kind": "flat disk", "r": DISK_R}
        elif name == "AMP":
            P64 = raw["DEF"][0].abs().to(torch.complex128)
            meta = {"kind": "|DEF| (no phase)", **raw["DEF"][1]}
        else:
            P64, m = raw[name]
            meta = {"kind": "aperture + aberration (edge phase in units of pi)", **m}
        pa64 = P64.abs()
        unit = bool((P64.imag == 0).all() and (P64.real >= 0).all())   # 實數非負探針(DISK、AMP)
        pu64 = torch.ones_like(P64) if unit else \
            torch.where(pa64 > 0, P64 / pa64.clamp_min(1e-300), torch.ones_like(P64))
        meta["footprint_px"] = footprint(P64)
        out[name] = {"P": P64.to(torch.complex64).to(dev), "Pa": pa64.to(torch.float32).to(dev),
                     "Pu": pu64.to(torch.complex64).to(dev), "unit": unit, "meta": meta, "P64": P64}
    return out


def make_psi(objs, name, pr, cfg):
    """出射波 ψ = P·O,以 [振幅, 相位] 表示。DISK 用 4-1 的 to_psi(逐位元相同)。"""
    if name == "DISK":
        import probe_4_1 as p41
        return p41.to_psi(objs, p41.disk(DISK_R, cfg, objs.device))
    psi = pr["P"] * torch.polar(objs[:, 0], objs[:, 1])
    return torch.stack([psi.abs(), torch.angle(psi)], dim=1)


_RATIO = {}


def probe_cfg(cfg, name, pr):
    """同一份設定,ref_energy 乘上校準集(≠ 測試集)上 ψ 與物體的能量比:總光子數與平面波相同。"""
    import probe_4_1 as p41
    if name == "DISK":
        return p41.psi_cfg(cfg, DISK_R)
    from src.physics import calibrate_flux
    if name not in _RATIO:
        cal = p41.make_home(cfg, p41.CALIB_N, seed=p41.CALIB_SEED).to(pr["P"].device)
        _RATIO[name] = calibrate_flux(make_psi(cal, name, pr, cfg)) / calibrate_flux(cal)
    c = Cfg.from_dict(cfg.to_dict())
    c.ref_energy = cfg.ref_energy * _RATIO[name]
    return c


# ============================================================================
# 知道探針的 HIO
# ============================================================================
def _support(variant, pr, box, true_sup, n):
    if variant == "A":
        s = box
    elif variant in ("M", "K"):
        s = box * (pr["Pa"] > 0).float()
    elif variant == "S":
        s = true_sup * (pr["Pa"] > 0).float()
    else:
        raise ValueError(variant)
    return s.expand(n, *s.shape[-2:]) if s.dim() == 2 else s


def _project(g, variant, pr, S, phmax):
    """目前估計投影到該 HIO 的約束集,輸出 [振幅, 相位]。K/S 在實數探針時與 hio_ext 的輸出逐位元相同。"""
    if variant == "A":
        return torch.stack([g.abs().clamp(0, 1) * S, torch.angle(g) * S], dim=1)
    if variant == "M":
        return torch.stack([torch.minimum(g.abs(), pr["Pa"]) * S, torch.angle(g) * S], dim=1)
    rel = g if pr["unit"] else g * pr["Pu"].conj()
    amp = torch.minimum(rel.abs(), pr["Pa"]).clamp_min(0) * S
    pha = torch.angle(rel).clamp(0, phmax) * S
    if not pr["unit"]:
        pha = pha + torch.angle(pr["Pu"]) * S
    return torch.stack([amp, pha], dim=1)


@torch.no_grad()
def hio_probe(init_obj, counts, bs_mask, cfg, iters, variant, pr, box, true_sup=None, beta=None):
    """回傳 {迭代數: 輸出 [N, 2, H, W]}(同一條軌跡在各迭代數取出,等同分別跑;內建檢查驗證)。

    傅立葉約束與 hio_sw.hio_ext 逐行相同;實空間約束依 variant:
      A  違反 = 方框外,或 |g′| > 1
      M  違反 = 方框外,或 |g′| > |P|
      K  違反 = 方框外,或 O = g′/P 的振幅 > 1、相位 ∉ [0, φmax](以 g′·conj(P/|P|) 判斷,不做除法)
      S  同 K,方框換成 ψ 的真實範圍
    未違反的像素取 g′(K/S 取投影後的值,未違反時即 g′);違反的像素取 g − β·g′(HIO 負回饋)。
    """
    from src.hio import _measured_amp
    dev = init_obj.device
    beta = cfg.hio_beta if beta is None else beta
    S = _support(variant, pr, box, true_sup, init_obj.shape[0])
    Pa, Pu, unit = pr["Pa"], pr["Pu"], pr["unit"]
    bs = bs_mask.to(dev)
    meas_amp = _measured_amp(counts, cfg).to(dev)
    phmax = cfg.phase_max
    want = set(iters)
    g = torch.polar(init_obj[:, 0], init_obj[:, 1])
    outs = {}
    for k in range(max(iters)):
        E = torch.fft.fftshift(torch.fft.fft2(g, norm="ortho"), dim=(-2, -1))
        mag, pha = E.abs(), torch.angle(E)
        new_mag = torch.where(bs > 0, meas_amp, mag)
        g_prime = torch.fft.ifft2(
            torch.fft.ifftshift(torch.polar(new_mag, pha), dim=(-2, -1)), norm="ortho")
        if variant == "A":
            viol = (S <= 0) | (g_prime.abs() > 1.0)
            g_ok = g_prime
        elif variant == "M":
            viol = (S <= 0) | (g_prime.abs() > Pa)
            g_ok = g_prime
        else:
            rel = g_prime if unit else g_prime * Pu.conj()
            amp_p, pha_p = rel.abs(), torch.angle(rel)
            viol = (S <= 0) | (pha_p < 0) | (pha_p > phmax) | (amp_p > Pa)
            g_ok = torch.polar(torch.minimum(amp_p, Pa).clamp_min(0), pha_p.clamp(0, phmax))
            if not unit:
                g_ok = g_ok * Pu
        g = torch.where(viol, g - beta * g_prime, g_ok)
        if (k + 1) in want:
            outs[k + 1] = _project(g, variant, pr, S, phmax)
    return outs


# ============================================================================
# 幾何(不含任何重建)
# ============================================================================
def twin_violation(pt, pr, box, phmax, E, nb=720):
    """翻轉解 pt [N, H, W] 違反約束的能量佔比(除以 E)。回傳 (K:取最佳全域相位, M)。

    M:方框外有值,或 |ψ| > |P|。K:另加 O 的相位 ∈ [0, φmax] —— 像素 r 在全域相位 θ 下合法
    ⟺ θ ∈ [−arg rel(r), −arg rel(r) + φmax](mod 2π),rel = ψ·conj(P/|P|);以 nb 格的覆蓋計數取最佳 θ。
    """
    Pa, Pu = pr["Pa"], pr["Pu"]
    w = pt.abs() ** 2
    live = w > 1e-12
    hard = ((box <= 0) | (pt.abs() > Pa * (1 + 1e-5) + 1e-9)) & live      # 與 θ 無關的違反
    vM = (hard.float() * w).sum((1, 2)) / E
    rel = pt if pr["unit"] else pt * Pu.conj()
    st = torch.remainder(-torch.angle(rel), 2 * PI)
    d = 2 * PI / nb
    j0 = torch.ceil(st / d - 1e-9).long()
    j1 = torch.floor((st + phmax) / d + 1e-9).long()
    gw = torch.where(live & ~hard, w, torch.zeros_like(w)).flatten(1)
    n = pt.shape[0]
    diff = torch.zeros(n, 2 * nb + 2, dtype=w.dtype, device=w.device)
    diff.scatter_add_(1, j0.flatten(1), gw)
    diff.scatter_add_(1, (j1 + 1).flatten(1), -gw)
    cov = torch.cumsum(diff, 1)[:, :2 * nb]
    cov = cov[:, :nb] + cov[:, nb:]
    vK = ((w * live).sum((1, 2)) - cov.amax(1)) / E
    return vK.clamp_min(0), vM


@torch.no_grad()
def geometry(cfg, probes, dev):
    """每個探針的幾何量,以及「翻轉解是否滿足約束」(在校準集上以真值計算,不跑 HIO)。
    翻轉解允許平移 ±GEO_SHIFT px(HIO 在方框內也可能收斂到平移過的翻轉解),取違反最少者。"""
    import probe_4_1 as p41
    from src.physics import beamstop_mask, support_mask
    n = cfg.canvas
    box = support_mask(cfg, device=dev)
    dig = torch.zeros(n, n, device=dev)
    s0 = (n - cfg.digit) // 2
    dig[s0:s0 + cfg.digit, s0:s0 + cfg.digit] = 1
    bs = beamstop_mask(cfg, device=dev)
    cal = p41.make_home(cfg, N_TWIN_GEO, seed=p41.CALIB_SEED).to(dev)
    obj_c = torch.polar(cal[:, 0], cal[:, 1])
    phmax = cfg.phase_max
    geo = {}
    for name, pr in probes.items():
        P, Pa = pr["P"], pr["Pa"]
        I = Pa ** 2
        psi = P * obj_c
        E = (psi.abs() ** 2).sum((1, 2))
        okE = E > 0
        Ec = E.clamp_min(1e-30)
        pt = torch.conj(torch.flip(psi, dims=(-2, -1)))
        vK0, vM0 = twin_violation(pt, pr, box, phmax, Ec)
        vK, vM = vK0.clone(), vM0.clone()
        for dy in range(-GEO_SHIFT, GEO_SHIFT + 1):
            for dx in range(-GEO_SHIFT, GEO_SHIFT + 1):
                if dy == 0 and dx == 0:
                    continue
                a, b = twin_violation(torch.roll(pt, (dy, dx), (-2, -1)), pr, box, phmax, Ec)
                vK, vM = torch.minimum(vK, a), torch.minimum(vM, b)
        # ψ 的能量被 beamstop 擋掉的比例(校準集平均)
        F = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1)).abs() ** 2
        blocked = float(((F * (1 - bs)).sum((1, 2))[okE] / F.sum((1, 2))[okE]).mean())
        geo[name] = {**pr["meta"],
                     "E_in_box": float((I * box).sum() / I.sum()),
                     "E_in_object_region": float((I * dig).sum() / I.sum()),
                     "E_in_footprint": float((I * (Pa >= TAU)).sum() / I.sum()),
                     "mean_I_in_footprint": float(I[Pa >= TAU].mean()),
                     "centro_sym_P": float((pr["P64"] - pr["P64"].flip(0, 1)).abs().max()),
                     "centro_sym_absP": float((pr["P64"].abs() - pr["P64"].abs().flip(0, 1)).abs().max()),
                     "twin_viol_K_median": float(vK[okE].median()),
                     "twin_infeasible_K": float((vK[okE] > 0.05).float().mean()),
                     "twin_viol_M_median": float(vM[okE].median()),
                     "twin_infeasible_M": float((vM[okE] > 0.05).float().mean()),
                     "twin_infeasible_K_noshift": float((vK0[okE] > 0.05).float().mean()),
                     "twin_infeasible_M_noshift": float((vM0[okE] > 0.05).float().mean()),
                     "bs_blocked_frac": blocked}
    return geo


def print_geometry(geo):
    print("\n" + "=" * 100)
    print("探針幾何(不含任何重建;翻轉可行性以校準集真值計算)")
    print("=" * 100)
    print(f"  {'探針':<6}{'k_a':>6}{'範圍px':>8}{'框內能量':>9}{'物體區能量':>10}{'範圍內能量':>10}"
          f"{'|P|對稱差':>10}{'翻轉違反K':>10}{'不可行K':>8}{'翻轉違反M':>10}{'不可行M':>8}{'bs擋掉':>8}")
    for name, g in geo.items():
        print(f"  {name:<6}{g.get('ka', float('nan')):>6.2f}{g['footprint_px']:>8d}{g['E_in_box']:>9.3f}"
              f"{g['E_in_object_region']:>10.3f}{g['E_in_footprint']:>10.3f}{g['centro_sym_absP']:>10.2f}"
              f"{g['twin_viol_K_median']:>10.3f}{g['twin_infeasible_K']:>8.2f}{g['twin_viol_M_median']:>10.3f}"
              f"{g['twin_infeasible_M']:>8.2f}{g['bs_blocked_frac']:>8.3f}")
    print(f"  翻轉違反 = 翻轉解中違反約束的能量佔比(中位數;取最佳全域相位與平移 ±{GEO_SHIFT} px);不可行 = 違反 > 0.05 的樣本比例")
    print("  不平移時的不可行比例 K / M:" + " / ".join(
        f"{k} {g['twin_infeasible_K_noshift']:.2f}/{g['twin_infeasible_M_noshift']:.2f}" for k, g in geo.items()))
    print("  預測(結果出來前):「不可行 K」≈ 1 的探針,K-HIO 的約束排除了翻轉解;M 只有 |P| 不對稱(COMA)時部分排除")


# ============================================================================
# 檔案與單元測試
# ============================================================================
def check_files():
    print("=" * 70)
    print("檔案")
    print("=" * 70)
    need = [REF_41, Path("probe_4_1.py"), Path("src/hio_sw.py"), Path("realign_eval.py"),
            Path("ambiguity_check.py")]
    for s in SEEDS:
        d = RUN_ROOT / f"{BASE}_s{s}"
        need += [d / "config_used.json", d / "final.pt"]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def _margin(psi_c, pr, phmax):
    """把 ψ 移到約束邊界內(振幅 ×0.95;O 的相位 → 0.05φmax + 0.9φ),同 4-1 的固定點測試(協定 §2.9)。"""
    o = psi_c if pr["unit"] else psi_c * pr["Pu"].conj()
    on = o.abs() > 0
    amp = o.abs() * 0.95
    pha = torch.where(on, 0.05 * phmax + 0.9 * torch.angle(o).clamp(0, phmax), torch.zeros_like(amp))
    out = torch.polar(amp, pha)
    return out if pr["unit"] else out * pr["Pu"]


@torch.no_grad()
def unit_tests(dev):
    import probe_4_1 as p41
    from realign_eval import load
    from src.hio import random_init
    from src.hio_sw import hio_ext
    from src.physics import beamstop_mask, forward_measure, support_mask

    print("\n" + "=" * 70)
    print("探針與單元測試")
    print("=" * 70)
    cfg, _ = load(RUN_ROOT / f"{BASE}_s0", dev)
    probes = build_probes(cfg, dev)
    box = support_mask(cfg, device=dev)
    bs = beamstop_mask(cfg, device=dev)
    phmax = cfg.phase_max
    target = int(p41.disk(DISK_R, cfg).sum())

    # (1) 探針幾何
    fp = {k: v["meta"]["footprint_px"] for k, v in probes.items()}
    check(f"孔徑探針的範圍面積 ≤ 圓盤({target} px)且 ≥ 90%",
          all(0.9 * target <= fp[k] <= target for k in COMPLEX), " / ".join(f"{k} {v}" for k, v in fp.items()))
    print("     k_a(規則選定):" + " / ".join(f"{k} {probes[k]['meta']['ka']:.2f}" for k in COMPLEX))
    sym = {k: float((probes[k]["P64"] - probes[k]["P64"].flip(0, 1)).abs().max()) for k in PROBES}
    check("中心對稱的探針 P(r) = P(-r)(差 < 1e-6)", all(sym[k] < 1e-6 for k in EVEN),
          " / ".join(f"{k} {sym[k]:.1e}" for k in EVEN))
    asym = float((probes["COMA"]["P64"].abs() - probes["COMA"]["P64"].abs().flip(0, 1)).abs().max())
    check("COMA 的 |P| 不中心對稱(差 > 0.1)", asym > 0.1, f"{asym:.2f}")
    check("AMP = |DEF|、DISK 與 AMP 為實數非負(走與 4-1 相同的計算路徑)",
          bool(torch.equal(probes["AMP"]["Pa"], probes["DEF"]["Pa"])) and probes["AMP"]["unit"]
          and probes["DISK"]["unit"] and not any(probes[k]["unit"] for k in COMPLEX))
    check("最大振幅皆為 1", all(abs(float(probes[k]["P64"].abs().max()) - 1) < 1e-9 for k in PROBES))

    # (2) 平頂圓盤:K = 4-1 的 P-HIO、S = 4-1 的 S*ψ(逐位元)
    home = p41.make_home(cfg, 64).to(dev)
    D = p41.disk(DISK_R, cfg, dev)
    psi = p41.to_psi(home, D)
    cp = probe_cfg(cfg, "DISK", probes["DISK"])
    torch.manual_seed(0)
    counts = forward_measure(psi, bs, cp)
    init = random_init(counts, cp, seed=1, device=dev)
    a = hio_probe(init, counts, bs, cp, [50], "K", probes["DISK"], box)[50]
    b = hio_ext(init, counts, bs, cp, 50, beta=cp.hio_beta, support=D.expand(len(psi), *D.shape))
    e1 = float((a - b).abs().max())
    check("平頂圓盤:K-HIO = 4-1 的 P-HIO(hio_ext,support = 圓盤),50 次", e1 == 0.0, f"最大差 {e1:.1e}")
    ts = (psi[:, 0] > 0).float()
    a = hio_probe(init, counts, bs, cp, [20], "S", probes["DISK"], box, true_sup=ts)[20]
    b = hio_ext(init, counts, bs, cp, 20, beta=cp.hio_beta, support=ts)
    e2 = float((a - b).abs().max())
    check("平頂圓盤:S-HIO = 4-1 的 S*ψ(hio_ext,support = 真實範圍),20 次", e2 == 0.0, f"最大差 {e2:.1e}")

    # (3) 同一條軌跡取出 = 分別跑(以 DEF 的 K 與 COMA 的 M 測)
    worst = 0.0
    for name, var in [("DEF", "K"), ("COMA", "M"), ("DEF2", "A")]:
        pr = probes[name]
        ps = make_psi(home, name, pr, cfg)
        cpp = probe_cfg(cfg, name, pr)
        torch.manual_seed(0)
        cn = forward_measure(ps, bs, cpp)
        it0 = random_init(cn, cpp, seed=1, device=dev)
        snap = hio_probe(it0, cn, bs, cpp, [5, 20], var, pr, box)
        sep = hio_probe(it0, cn, bs, cpp, [20], var, pr, box)
        worst = max(worst, float((snap[20] - sep[20]).abs().max()))
    check("同一條軌跡在各迭代數取出 = 分別跑", worst == 0.0, f"最大差 {worst:.1e}")

    # (4) 真值為固定點(無雜訊,移到約束邊界內)
    fixed = {}
    for name in PROBES:
        pr = probes[name]
        c0 = Cfg.from_dict(probe_cfg(cfg, name, pr).to_dict())
        c0.add_poisson = False
        ps = make_psi(home, name, pr, cfg)
        t = _margin(torch.polar(ps[:, 0], ps[:, 1]), pr, phmax)
        tt = torch.stack([t.abs(), torch.angle(t)], 1)
        cnt0 = forward_measure(tt, bs, c0)
        ts_ = (t.abs() > 0).float()
        errs = []
        for var in VARIANTS:
            out = hio_probe(tt, cnt0, bs, c0, [100], var, pr, box, true_sup=ts_)[100]
            errs.append(float((torch.polar(out[:, 0], out[:, 1]) - t).abs().max()))
        fixed[name] = max(errs)
    check("真值為固定點(7 個探針 × 4 種 HIO,無雜訊,100 次)", all(v < 1e-3 for v in fixed.values()),
          " / ".join(f"{k} {v:.1e}" for k, v in fixed.items()))

    # (5) 翻轉的機制(無雜訊):以「最佳全域相位的翻轉解」為起點,看 HIO 是否停在那裡
    moved = {}
    thetas = torch.linspace(-PI, PI, 721, device=dev)[:-1]
    for name in PROBES:
        pr = probes[name]
        c0 = Cfg.from_dict(probe_cfg(cfg, name, pr).to_dict())
        c0.add_poisson = False
        ps = make_psi(home, name, pr, cfg)
        t = _margin(torch.polar(ps[:, 0], ps[:, 1]), pr, phmax)
        tw = torch.conj(torch.flip(t, dims=(-2, -1)))
        rel = tw if pr["unit"] else tw * pr["Pu"].conj()
        w = tw.abs() ** 2
        best_v = torch.full((len(tw),), 9.0, device=dev)
        best_t = torch.zeros(len(tw), device=dev)
        for th in thetas:
            ph = torch.angle(rel * torch.exp(1j * th))
            v = (((ph < 0) | (ph > phmax)) & (w > 1e-12)).float().mul(w).sum((1, 2))
            better = v < best_v
            best_v = torch.where(better, v, best_v)
            best_t = torch.where(better, th.expand_as(best_t), best_t)
        if pr["unit"]:
            best_t = torch.full_like(best_t, phmax)                # 實數探針:e^{iφmax} 恰好可行
        g0 = tw * torch.exp(1j * best_t)[:, None, None]
        g0o = torch.stack([g0.abs(), torch.angle(g0)], 1)
        cnt0 = forward_measure(torch.stack([t.abs(), torch.angle(t)], 1), bs, c0)
        nz = (t.abs() ** 2).sum((1, 2)) > 0
        res = {}
        for var in ["M", "K"]:
            out = hio_probe(g0o, cnt0, bs, c0, [100], var, pr, box)[100]
            d = (torch.polar(out[:, 0], out[:, 1]) - g0).abs().pow(2).sum((1, 2)).sqrt()
            rel_d = d / g0.abs().pow(2).sum((1, 2)).sqrt().clamp_min(1e-12)
            res[var] = float((rel_d[nz] > 1e-2).float().mean())
        moved[name] = res
    print("     以翻轉解為起點、100 次後離開的樣本比例(K / M):"
          + " / ".join(f"{k} {v['K']:.2f}/{v['M']:.2f}" for k, v in moved.items()))
    check("實數探針(DISK、AMP):翻轉解是 K 與 M 的固定點(離開比例 0)",
          all(moved[k][v] == 0.0 for k in ["DISK", "AMP"] for v in ["K", "M"]))
    check("中心對稱的探針:翻轉解是 M 的固定點(離開比例 0)", all(moved[k]["M"] == 0.0 for k in EVEN))
    check("複數探針:翻轉解不是 K 的固定點(離開比例 ≥ 0.9)", all(moved[k]["K"] >= 0.9 for k in COMPLEX))

    # (6) 翻轉指標本身(不跑 HIO):對真值 / 物體翻轉解 / 與真值無關的估計,判定是否正確、是否無偏
    from ambiguity_check import align
    cal = p41.make_home(cfg, N_TWIN_GEO, seed=p41.CALIB_SEED).to(dev)
    oc = torch.polar(cal[:, 0], cal[:, 1])
    other = torch.roll(oc, 1, 0)                                        # 另一張樣本的物體
    gen = torch.Generator().manual_seed(7)
    rnd = torch.polar(torch.rand(oc.shape, generator=gen), torch.rand(oc.shape, generator=gen) * phmax).to(dev) * box
    rph = torch.exp(1j * (torch.rand(oc.shape, generator=gen) * 2 - 1) * PI).to(dev)
    null_o, null_p, exact = {}, {}, True
    for name in PROBES:
        pr = probes[name]
        P = pr["P"]
        r1, r2 = orientation(P * oc, P, oc)
        exact &= bool((r2 < r1).all()) and float(r1.min()) > 0.99
        r1, r2 = orientation(P * obj_twin(oc) * math.e ** 0, P, oc)
        exact &= bool((r2 > r1).all())
        fr = []
        for est in (P * rnd, P * other):
            a1, a2 = orientation(est, P, oc)
            fr.append(float((a2 > a1).float().mean()))
        null_o[name] = fr
        _, tw, _ = align(pr["Pa"] * rnd.abs() * rph * box, P * oc)
        null_p[name] = float(tw.float().mean())
    check("物體空間的翻轉判定:真值判為不翻轉(相關 > 0.99)、P·conj(O(−r)) 判為翻轉(7 個探針皆正確)", exact)
    lo, hi = 0.35, 0.65
    check(f"翻轉判定對「與真值無關的估計」無偏:選翻轉比例在 {lo}–{hi}(物體空間:隨機物體 / 他張物體;ψ 空間:隨機)"
          + (";QUICK 張數太少,只列出不判定" if QUICK else ""),
          QUICK or all(lo <= x <= hi for v in null_o.values() for x in v) and all(lo <= x <= hi for x in null_p.values()),
          " / ".join(f"{k} {v[0]:.2f},{v[1]:.2f},{null_p[k]:.2f}" for k, v in null_o.items()))

    # (7) 劑量:ψ 的平均總光子數(不含 beamstop)≈ 平面波;另列偵測到的比例
    full = p41.make_home(cfg, N_OBJ or cfg.eval_n).to(dev)
    bs0 = beamstop_mask(cfg, radius=-1, device=dev)
    ratios, seen = {}, {}
    torch.manual_seed(0)
    cpl0 = forward_measure(full, bs0, cfg)
    for name in PROBES:
        pr = probes[name]
        cpp = probe_cfg(cfg, name, pr)
        ps = make_psi(full, name, pr, cfg)
        torch.manual_seed(0)
        c_all = forward_measure(ps, bs0, cpp)
        ratios[name] = float(c_all.sum((1, 2)).mean() / cpl0.sum((1, 2)).mean())
        seen[name] = float((c_all * bs).sum((1, 2)).mean() / c_all.sum((1, 2)).mean())
    check(f"ψ 的平均總光子數 / 平面波 在 0.95–1.05(不含 beamstop,{len(full)} 張"
          + (";QUICK 張數太少,只列出不判定)" if QUICK else ")"),
          QUICK or all(0.95 <= x <= 1.05 for x in ratios.values()),
          " / ".join(f"{k} {v:.3f}" for k, v in ratios.items()))
    print("     (參考)偵測到的光子比例(beamstop 外):" + " / ".join(f"{k} {v:.3f}" for k, v in seen.items()))

    geo = geometry(cfg, probes, dev)
    print_geometry(geo)
    make_probe_figure(probes, geo)
    print(f"\n     裝置:{dev}")
    return probes, geo


# ============================================================================
# 量測
# ============================================================================
@torch.no_grad()
def repro_4_1(dev):
    """K-HIO(平頂圓盤)重現 probe_4_1.json 的 P-HIO(r = 12、seed 0、50 次),S 重現 S*ψ(20 次)。"""
    import probe_4_1 as p41
    from realign_eval import load, score_all
    from src.hio import random_init
    from src.physics import beamstop_mask, forward_measure, support_mask
    ref = json.load(open(REF_41))
    if DISK_R not in ref["radii"]:
        raise SystemExit(f"{REF_41} 沒有 r = {DISK_R},停下來查")
    ri = ref["radii"].index(DISK_R)
    cfg, model = load(RUN_ROOT / f"{BASE}_s0", dev)
    probes = build_probes(cfg, dev)
    n = N_OBJ or cfg.eval_n
    home = p41.make_home(cfg, n).to(dev)
    psi = make_psi(home, "DISK", probes["DISK"], cfg)
    cp = probe_cfg(cfg, "DISK", probes["DISK"])
    bs = beamstop_mask(cfg, device=dev)
    box = support_mask(cfg, device=dev)
    torch.manual_seed(cfg.test_seed + 0 + 1000 * (ri + 1))
    counts = forward_measure(psi, bs, cp)
    init = random_init(counts, cp, seed=cfg.test_seed, device=dev)
    ts = (psi[:, 0] > 0).float()
    for var, key, it in [("K", "P", 50), ("S", "S", 20)]:
        if it not in ref["iters"]:
            continue
        out = hio_probe(init, counts, bs, cp, [it], var, probes["DISK"], box, true_sup=ts)[it]
        got = score_all(model, psi, bs, cp, out)[0]["aligned"]["frc_gain"]
        exp = ref["results"][str(DISK_R)][0]["rec"][key][str(it)]["aligned"]["frc_gain"]
        if abs(got - exp) > REPRO_TOL:
            raise SystemExit(f"平頂圓盤 {var}-HIO {it} 次:{got:+.4f} 與 4-1 的 {key} {exp:+.4f} 對不上,停下來查")
        check(f"平頂圓盤 {var}-HIO 重現 4-1 的 {key}(r = 12、seed 0、{it} 次)", True, f"{got:+.4f} vs {exp:+.4f}")


def obj_twin(obj_c):
    return torch.conj(torch.flip(obj_c, dims=(-2, -1)))


def orientation(pred_c, P, obj_c, valid=1e-3):
    """物體空間的翻轉判定。回傳 (ρ_原, ρ_翻):ψ̂ 與 P·O(r − s)、P·conj(O(−r − s)) 的正規化相關,
    各取最佳平移 s 與全域相位(float64;平移限於「被照亮的物體能量 ≥ 最大值的 1e-3」處,避免 0/0)。"""
    pred_c, P, obj_c = pred_c.to(torch.complex128), P.to(torch.complex128), obj_c.to(torch.complex128)
    u = pred_c * P.conj()
    Fu = torch.fft.fft2(u)
    FI = torch.fft.fft2(P.abs() ** 2)
    nrm = (pred_c.abs() ** 2).sum((1, 2))
    rhos = []
    for ref in (obj_c, obj_twin(obj_c)):
        num = torch.fft.ifft2(Fu * torch.conj(torch.fft.fft2(ref))).abs()
        den2 = torch.fft.ifft2(FI * torch.conj(torch.fft.fft2(ref.abs() ** 2))).real.clamp_min(0)
        ok = den2 >= valid * den2.amax((1, 2), keepdim=True)
        rho = torch.where(ok, num / (nrm[:, None, None] * den2).sqrt().clamp_min(1e-20), torch.zeros_like(num))
        rhos.append(rho.amax((1, 2)))
    return rhos[0], rhos[1]


def extra_metrics(pred, psi, al, P, obj_c):
    """對齊前後的正規化誤差、只對齊全域相位的誤差、平移 / 翻轉(ψ 空間與物體空間)、「解出」樣本中的翻轉比例。"""
    from ambiguity_check import align, to_c
    pc, tc = to_c(pred), to_c(psi)
    _, tw, sh = align(pc, tc)
    E = (tc.abs() ** 2).sum((1, 2))
    ok = E > 1e-12                                              # 排除空樣本
    ac = to_c(al)
    nerr_al = ((ac - tc).abs() ** 2).sum((1, 2)) / E.clamp_min(1e-12)
    cross = (pc * tc.conj()).sum((1, 2)).abs()
    nerr_ph = ((pc.abs() ** 2).sum((1, 2)) + E - 2 * cross) / E.clamp_min(1e-12)
    good = ok & (nerr_al < GOOD_NERR)
    r1, r2 = orientation(pc, P, obj_c)
    rb = torch.maximum(r1, r2)
    ok_o = ok & torch.isfinite(rb) & (rb > 0)                   # 輸出全為 0 / 非有限時無從判定,不列入
    tw_o = r2 > r1
    good_o = ok_o & (rb >= GOOD_RHO)
    return {"twin_obj": float(tw_o[ok_o].float().mean()) if bool(ok_o.any()) else float("nan"),
            "n_obj_undecided": int((ok & ~ok_o).sum()),
            "rho_best": float(rb[ok_o].mean()) if bool(ok_o.any()) else float("nan"),
            "good_obj_frac": float(good_o.float().sum() / ok_o.float().sum().clamp_min(1)),
            "n_good_obj": int(good_o.sum()),
            "twin_obj_good": float(tw_o[good_o].float().mean()) if bool(good_o.any()) else float("nan"),
            "shift_frac": float((sh.abs().sum(1) > 0).float().mean()),
            "twin_frac_check": float(tw.float().mean()),
            "nerr_al": float(nerr_al[ok].mean()), "nerr_al_median": float(nerr_al[ok].median()),
            "nerr_ph": float(nerr_ph[ok].mean()), "nerr_ph_median": float(nerr_ph[ok].median()),
            "good_frac": float(good.float().sum() / ok.float().sum().clamp_min(1)),
            "n_good": int(good.sum()),
            "twin_good": float(tw[good].float().mean()) if bool(good.any()) else float("nan"),
            "n_empty": int((~ok).sum())}


@torch.no_grad()
def measure(dev):
    import probe_4_1 as p41
    from realign_eval import load, proto_labels, score, score_all
    from src.hio import random_init
    from src.physics import beamstop_mask, forward_measure, support_mask

    res = {p: [] for p in PROBES}
    kinds = None
    for s in SEEDS:
        cfg, model = load(RUN_ROOT / f"{BASE}_s{s}", dev)
        probes = build_probes(cfg, dev)
        n = N_OBJ or cfg.eval_n
        bs = beamstop_mask(cfg, device=dev)
        bs0 = beamstop_mask(cfg, radius=-1, device=dev)
        box = support_mask(cfg, device=dev)
        home = p41.make_home(cfg, n).to(dev)
        labels, kinds = proto_labels(n, cfg, home)
        for pi_, name in enumerate(PROBES):
            pr = probes[name]
            psi = make_psi(home, name, pr, cfg)
            cp = probe_cfg(cfg, name, pr)
            off = DISK_SEED_OFFSET if name == "DISK" else 10_000 + 1000 * pi_
            torch.manual_seed(cfg.test_seed + s + off)
            counts = forward_measure(psi, bs, cp)
            init = random_init(counts, cp, seed=cfg.test_seed + s, device=dev)
            ts = (psi[:, 0] > 0).float()
            obj_c = torch.polar(home[:, 0], home[:, 1])
            rec = {}
            for var in VARIANTS:
                outs = hio_probe(init, counts, bs, cp, ITERS, var, pr, box, true_sup=ts)
                rec[var] = {}
                for it in ITERS:
                    pred = outs[it]
                    sc, al = score_all(model, psi, bs, cp, pred)
                    sc.update(extra_metrics(pred, psi, al, pr["P"], obj_c))
                    if it == PROTO_N:
                        sc["proto"] = {}
                        for i, k in enumerate(kinds):
                            m = labels.to(dev) == i
                            p_ = score(model, psi[m], bs, cp, al[m])
                            sc["proto"][k] = {"frc_gain": p_["frc_gain"]}
                    rec[var][str(it)] = sc
                del outs
            # 診斷:不擋 beamstop(同一批物體、同樣劑量;量測另取 seed)
            torch.manual_seed(cfg.test_seed + s + off + 500)
            counts0 = forward_measure(psi, bs0, cp)
            init0 = random_init(counts0, cp, seed=cfg.test_seed + s, device=dev)
            rec0 = {}
            for var in BS0_VARIANTS:
                outs = hio_probe(init0, counts0, bs0, cp, BS0_ITERS, var, pr, box, true_sup=ts)
                rec0[var] = {}
                for it in BS0_ITERS:
                    sc, al = score_all(model, psi, bs0, cp, outs[it])
                    sc.update(extra_metrics(outs[it], psi, al, pr["P"], obj_c))
                    rec0[var][str(it)] = sc
                del outs
            res[name].append({"run": f"{BASE}_s{s}", "rec": rec, "rec_bs0": rec0, "ref_energy": cp.ref_energy})
            print(f"  [{BASE}_s{s}] {name} 完成", flush=True)
    return res, kinds


# ============================================================================
# 彙整與判讀(判讀準則見階段四協定 §九,結果出來前寫定)
# ============================================================================
def vals(res, p, var, it, field="frc_gain", kind="aligned", rec="rec"):
    return np.array([sd[rec][var][str(it)][kind][field] if kind else sd[rec][var][str(it)][field]
                     for sd in res[p]], float)


def nmean(v):
    """忽略 nan 的平均(沒有「解出」的樣本時 twin_good 為 nan);全為 nan 時回傳 nan、不發警告。"""
    v = np.asarray(v, float)
    return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float("nan")


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def zpair(d):
    m, s = ms(d)
    return m, s, (m / (s / math.sqrt(len(d))) if s > 0 else float("inf") * np.sign(m))


def flip_fields(var):
    """K / S 以物體空間判翻轉,A / M 以 ψ 空間(協定 §9.4)。回傳 (選翻轉, 解出樣本中選翻轉, 解出張數) 的欄位名。"""
    return ("twin_obj", "twin_obj_good", "n_good_obj") if var in OBJ_FLIP else ("twin_frac", "twin_good", "n_good")


def flip_verdict(res, p, var, rec="rec"):
    ft, fg, fn = flip_fields(var)
    t_main = vals(res, p, var, MAIN_N, ft, None, rec).mean()
    t_long = vals(res, p, var, LONG_N, ft, None, rec).mean()
    tg = nmean(vals(res, p, var, LONG_N, fg, None, rec))
    ng = vals(res, p, var, LONG_N, fn, None, rec).mean()
    if t_main < 0.2 and t_long < 0.2:
        v = "打破"
    elif ng >= 30 and tg < 0.2:
        v = "機制上打破(解出的樣本不選翻轉),但解出的比例不足"
    elif t_long > 0.4 and (ng < 30 or tg > 0.4):
        v = "未打破"
    else:
        v = "部分"
    return v, t_main, t_long, tg, ng


def diff_verdict(res, p, a, b, iters, big, small):
    rows, hit, none = [], False, True
    for it in iters:
        d = zpair(vals(res, p, a, it) - vals(res, p, b, it))
        rows.append((it, d))
        hit |= d[0] >= big and d[2] > 2
        none &= abs(d[0]) <= small
    return ("是" if hit else ("否" if none else "部分")), rows


def ratio_verdict(res, p, q, it, rec="rec"):
    """跨探針:nerr_al(K) 的比值 p / q,逐 seed 配對;z 以 log 比值計。"""
    a = vals(res, p, "K", it, "nerr_al", None, rec)
    b = vals(res, q, "K", it, "nerr_al", None, rec)
    r = a / b
    m, s, z = zpair(np.log(r))
    mr = float(np.exp(m))
    v = "較準" if (mr <= 0.8 and z < -2) else ("較差" if (mr >= 1.25 and z > 2) else "相近")
    return v, mr, z


def report(res, kinds, geo):
    verdict = {}
    for p in PROBES:
        g = geo[p]
        print("\n" + "=" * 104)
        print(f"探針 {p}({g['kind']};範圍 {g['footprint_px']} px;翻轉不可行 K {g['twin_infeasible_K']:.2f} / "
              f"M {g['twin_infeasible_M']:.2f})")
        print("對齊後 FRC gain(mean ± std,3 seeds);只在同一探針內比較(不同探針的目標 ψ 不同)")
        print("=" * 104)
        print(f"  {'迭代':>6}" + "".join(f"{V_LABEL[v]:>22}" for v in VARIANTS))
        for it in ITERS:
            print(f"  {it:>6}" + "".join("{:>+14.4f}±{:.4f}".format(*ms(vals(res, p, v, it))) for v in VARIANTS))
        print(f"  機率水準(K {ITERS[-1]} 次的錯配對齊):{vals(res, p, 'K', ITERS[-1], kind='mismatch').mean():+.4f}")
        for it in sorted({MAIN_N, LONG_N}):
            print(f"  {it} 次 選翻轉(ψ 空間 / 物體空間)/ 有平移:" + "   ".join(
                f"{v} {vals(res, p, v, it, 'twin_frac', None).mean():.2f} / "
                f"{vals(res, p, v, it, 'twin_obj', None).mean():.2f} / "
                f"{vals(res, p, v, it, 'shift_frac', None).mean():.2f}" for v in VARIANTS))
        print(f"  {LONG_N} 次 解出的樣本比例 / 其中選翻轉(A、M:ψ 空間誤差 < {GOOD_NERR};K、S:物體空間相關 ≥ {GOOD_RHO}):"
              + "   ".join(
                  f"{v} {vals(res, p, v, LONG_N, 'good_obj_frac' if v in OBJ_FLIP else 'good_frac', None).mean():.2f} / "
                  f"{nmean(vals(res, p, v, LONG_N, flip_fields(v)[1], None)):.2f}" for v in VARIANTS))
        print(f"  各原型({PROTO_N} 次):" + "   ".join(
            f"{k}: " + " / ".join(f"{v} {np.mean([sd['rec'][v][str(PROTO_N)]['proto'][k]['frc_gain'] for sd in res[p]]):+.3f}"
                                  for v in VARIANTS) for k in kinds))
        if p == "DISK":
            print(f"  材料 MAE({MAIN_N} 次):" + " / ".join(
                f"{v} {vals(res, p, v, MAIN_N, 'material_mae').mean():.3f}" for v in VARIANTS))

        fk = flip_verdict(res, p, "K")
        fm = flip_verdict(res, p, "M")
        print(f"  (1) 翻轉(K-HIO,物體空間):{fk[0]}(選翻轉 {MAIN_N} 次 {fk[1]:.2f}、{LONG_N} 次 {fk[2]:.2f};"
              f"解出樣本中 {fk[3]:.2f},平均 {fk[4]:.0f} 張)")
        print(f"      翻轉(M-HIO,ψ 空間,參考):{fm[0]}({fm[1]:.2f} / {fm[2]:.2f};解出樣本中 {fm[3]:.2f},"
              f"平均 {fm[4]:.0f} 張)")
        ma, rows = diff_verdict(res, p, "M", "A", JUDGE_MA, 0.10, 0.02)
        ma_txt = {"是": "有效", "否": "無效", "部分": "部分"}[ma]
        print(f"  (2) 探針範圍仍有效(M − A):{ma_txt} "
              + ";".join(f"{it} 次 {d[0]:+.4f} ± {d[1]:.4f}(z {d[2]:+.1f})" for it, d in rows))
        km, rows = diff_verdict(res, p, "K", "M", JUDGE_KM, 0.05, 0.02)
        km_txt = {"是": "有幫助", "否": "無", "部分": "部分"}[km]
        print(f"  (3) 知道探針相位的幫助(K − M):{km_txt} "
              + ";".join(f"{it} 次 {d[0]:+.4f} ± {d[1]:.4f}(z {d[2]:+.1f})" for it, d in rows))
        fs, fk50 = vals(res, p, "S", STRONG_N).mean(), vals(res, p, "K", STRONG_N).mean()
        near = fs - fk50 <= 0.05
        print(f"  (4) 距上限({STRONG_N} 次):K {fk50:+.4f} vs S {fs:+.4f},差 {fs - fk50:+.4f} → "
              f"{'接近上限' if near else '未接近'}")
        verdict[p] = {"flip_K": fk[0], "flip_M": fm[0], "twin_K_main": fk[1], "twin_K_long": fk[2],
                      "support_useful": ma, "phase_help": km, "near_upper": bool(near), "K50": float(fk50)}

    # (5) 跨探針:K-HIO 的正規化誤差(對齊後 / 只對齊全域相位)
    print("\n" + "=" * 104)
    print("跨探針比較:K-HIO 的正規化誤差 Σ|ψ̂ − ψ|² / Σ|ψ|²(越小越好;目標 ψ 不同,意義相同:錯掉的能量佔比)")
    print("=" * 104)
    show = [it for it in [20, 50, 100, 1000, ITERS[-1]] if it in ITERS]
    print(f"  {'探針':<6}" + "".join(f"{f'{it} 次 對齊/免對齊':>22}" for it in show))
    for p in PROBES:
        print(f"  {p:<6}" + "".join(
            f"{vals(res, p, 'K', it, 'nerr_al', None).mean():>13.3f}/{vals(res, p, 'K', it, 'nerr_ph', None).mean():<8.3f}"
            for it in show))
    print("  免對齊 = 只對齊全域相位(實驗上唯一不需要真值就能做的對齊);翻轉被打破時應接近「對齊」")
    cross = {}
    v, mr, z = ratio_verdict(res, "DEF", "AMP", MAIN_N)
    print(f"  (5a) 相位的效果(DEF vs AMP,振幅相同):{MAIN_N} 次 比值 {mr:.3f}(z {z:+.1f})→ DEF {v}")
    cross["DEF_vs_AMP"] = {"verdict": v, "ratio": mr, "z": z}
    for p in PROBES:
        if p == "DISK":
            continue
        v, mr, z = ratio_verdict(res, p, "DISK", MAIN_N)
        v50 = ratio_verdict(res, p, "DISK", STRONG_N)
        print(f"  (5b) {p} vs DISK:{MAIN_N} 次 比值 {mr:.3f}(z {z:+.1f})→ {v};"
              f"{STRONG_N} 次 {v50[1]:.3f}(z {v50[2]:+.1f})→ {v50[0]}")
        cross[f"{p}_vs_DISK"] = {"verdict": v, "ratio": mr, "z": z, "at50": v50[0]}

    # (6) 診斷:不擋 beamstop —— 分開「相位的約束效果」與「beamstop 擋掉較少」
    print("\n" + "=" * 104)
    print("診斷:beamstop = 0(同樣劑量;K / M-HIO)。探針相位會把 ψ 的頻譜攤開、被 beamstop 擋掉的較少,"
          "此處拿掉 beamstop 分開兩種效果")
    print("=" * 104)
    b0 = [it for it in [MAIN_N, LONG_N] if it in BS0_ITERS]
    print(f"  {'探針':<6}{'bs擋掉':>8}" + "".join(f"{f'K {it} 次 誤差 / 選翻轉':>24}" for it in b0)
          + "".join(f"{f'M {it} 次 選翻轉':>16}" for it in b0))
    for p in PROBES:
        print(f"  {p:<6}{geo[p]['bs_blocked_frac']:>8.3f}" + "".join(
            f"{vals(res, p, 'K', it, 'nerr_al', None, 'rec_bs0').mean():>15.3f} / "
            f"{vals(res, p, 'K', it, 'twin_obj', None, 'rec_bs0').mean():<6.2f}" for it in b0)
            + "".join(f"{vals(res, p, 'M', it, 'twin_frac', None, 'rec_bs0').mean():>16.2f}" for it in b0))
    v0, mr0, z0 = ratio_verdict(res, "DEF", "AMP", b0[0], "rec_bs0")
    v3 = cross["DEF_vs_AMP"]["verdict"]
    print(f"  (6a) 相位的效果,beamstop = 0:DEF vs AMP {b0[0]} 次 比值 {mr0:.3f}(z {z0:+.1f})→ DEF {v0}")
    if v3 == "較準" and v0 == "較準":
        interp = "相位本身(已知相位提供的約束)讓 HIO 更準"
    elif v3 == "較準" and v0 != "較準":
        interp = "好處主要來自 beamstop 擋掉較少(相位把探針頻譜攤開),不是相位約束本身"
    elif v3 != "較準" and v0 == "較準":
        interp = "相位約束有幫助,但在 beamstop = 3 下被其他因素抵銷"
    else:
        interp = "兩種條件下相位都沒有讓 HIO 明顯更準"
    print(f"  判讀:{interp}")
    cross["DEF_vs_AMP_bs0"] = {"verdict": v0, "ratio": mr0, "z": z0, "interpretation": interp}
    print(f"  (6b) 各探針 vs DISK,beamstop = 0({b0[0]} 次;描述用,對照 (5b) 看 beamstop 佔了多少):")
    for p in PROBES:
        if p == "DISK":
            continue
        vb, mrb, zb = ratio_verdict(res, p, "DISK", b0[0], "rec_bs0")
        print(f"       {p}:比值 {mrb:.3f}(z {zb:+.1f})→ {vb}(beamstop 3:{cross[f'{p}_vs_DISK']['verdict']})")
        cross[f"{p}_vs_DISK_bs0"] = {"verdict": vb, "ratio": mrb, "z": zb}

    # 階段五的預設探針(規則見協定 §9.5)
    cand = [p for p in COMPLEX if verdict[p]["flip_K"] == "打破" and cross[f"{p}_vs_DISK"]["verdict"] != "較差"]
    print("\n" + "=" * 104)
    if cand:
        pick = min(cand, key=lambda p: vals(res, p, "K", MAIN_N, "nerr_al", None).mean())
        print(f"結論:打破翻轉且不比平頂圓盤差的複數探針 {cand} → 階段五預設探針建議 **{pick}**"
              f"(其中 K-HIO {MAIN_N} 次誤差最小者);平頂圓盤保留為對照。待與指導教授確認(協定 §六)")
    else:
        pick = "DISK"
        print("結論:沒有「打破翻轉且不比平頂圓盤差」的複數探針 → 階段五預設探針維持 **DISK**")
    print("網路部分維持暫停(協定 §8.15),延至階段五。判讀準則見階段四協定 §九(結果出來前已寫定)")
    verdict["cross"], verdict["candidates"], verdict["pick"] = cross, cand, pick
    return verdict


def check_disk_vs_41(res):
    """完整模式:DISK 的 K / S 全部迭代數與 4-1 的 P / S*ψ 對照(同一份量測,應逐點相同)。"""
    if QUICK or not REF_41.exists():
        return
    ref = json.load(open(REF_41))
    if ref.get("quick") or DISK_R not in ref["radii"] or ref["radii"].index(DISK_R) != 2:
        print("  ⚠️ 4-1 的結果設定不同,略過逐點對照")
        return
    worst = 0.0
    for si, sd in enumerate(res["DISK"]):
        for var, key in [("K", "P"), ("S", "S")]:
            for it in ITERS:
                if str(it) not in ref["results"][str(DISK_R)][si]["rec"][key]:
                    continue
                a = sd["rec"][var][str(it)]["aligned"]["frc_gain"]
                b = ref["results"][str(DISK_R)][si]["rec"][key][str(it)]["aligned"]["frc_gain"]
                worst = max(worst, abs(a - b))
    check("DISK 的 K / S(3 seeds × 全部迭代數)= 4-1 的 P / S*ψ", worst <= REPRO_TOL, f"最大差 {worst:.4f}")


# ============================================================================
# 圖
# ============================================================================
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  ⚠️ 無 matplotlib,略過圖")
        return None


def make_probe_figure(probes, geo):
    plt = _plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    ink = "#0b0b0b"
    n = len(PROBES)
    fig, axs = plt.subplots(3, n, figsize=(2.2 * n, 6.9))
    sl = slice(8, 56)
    for j, name in enumerate(PROBES):
        P = probes[name]["P64"].numpy()
        a, ph = np.abs(P), np.angle(P)
        axs[0, j].imshow(a[sl, sl], cmap="gray", vmin=0, vmax=1)
        axs[0, j].contour(a[sl, sl] >= TAU, levels=[0.5], colors="#eb6834", linewidths=0.8)
        g = geo[name]
        t = name if name in ("DISK", "AMP") else f"{name}  k_a {g['ka']:.2f}"
        axs[0, j].set_title(t, fontsize=9, color=ink)
        axs[1, j].imshow(np.where(a >= 0.02, ph, np.nan)[sl, sl], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axs[2, j].plot(np.arange(64), a[31, :], color="#2a78d6", lw=1.2, label="|P| (row 31)")
        axs[2, j].plot(np.arange(64), a[:, 31], color="#1baf7a", lw=1.0, ls="--", label="|P| (col 31)")
        axs[2, j].set_ylim(0, 1.05)
        axs[2, j].set_title(f"twin infeasible K {g['twin_infeasible_K']:.2f} / M {g['twin_infeasible_M']:.2f}",
                            fontsize=7.5, color=ink)
        for i in range(2):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    axs[0, 0].set_ylabel("|P|  (orange: |P| ≥ 0.1)", fontsize=8)
    axs[1, 0].set_ylabel("phase of P", fontsize=8)
    axs[2, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("4-2b probes (geometry only, no reconstruction). Footprint area matched to the r = 12 disk.",
                 fontsize=10, color=ink)
    fig.tight_layout()
    p = FIG_DIR / "probe42b_probes.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


def make_figures(res):
    plt = _plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    col = {"A": "#2a78d6", "M": "#8a5cd1", "K": "#eb6834", "S": "#1baf7a"}
    sty = {"A": "-", "M": "-", "K": "-", "S": "--"}
    name = {"A": "A: box", "M": "M: knows |P|", "K": "K: knows P", "S": "S: true support (bound)"}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"

    def style(ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=grid, lw=0.8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.set_xscale("log")

    n = len(PROBES)
    fig, axs = plt.subplots(2, n, figsize=(2.9 * n, 6.2), sharex=True)
    for j, p in enumerate(PROBES):
        for v in VARIANTS:
            mu = np.array([vals(res, p, v, it).mean() for it in ITERS])
            axs[0, j].plot(ITERS, mu, sty[v], color=col[v], lw=1.8, marker="o", ms=3, label=name[v])
            tw = np.array([vals(res, p, v, it, flip_fields(v)[0], None).mean() for it in ITERS])
            axs[1, j].plot(ITERS, tw, sty[v], color=col[v], lw=1.6, marker="o", ms=3)
        axs[0, j].set_title(p, color=ink, fontsize=10)
        axs[1, j].set_ylim(-0.02, 1.02)
        axs[1, j].axhline(0.5, color=ink2, lw=0.8, ls=":")
        style(axs[0, j])
        style(axs[1, j])
        axs[1, j].set_xlabel("iterations", color=ink, fontsize=8)
    axs[0, 0].set_ylabel("aligned FRC gain on ψ\n(within-probe only)", color=ink, fontsize=8)
    axs[1, 0].set_ylabel("fraction flipped\n(K, S: object space; A, M: ψ space)", color=ink, fontsize=8)
    axs[0, 0].legend(frameon=False, fontsize=7, labelcolor=ink, loc="lower right")
    fig.suptitle("4-2b: known complex probes and HIO (3 seeds, mean)", color=ink, fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "probe42b_curves.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")

    pcol = {"DISK": "#0b0b0b", "AMP": "#9a9994", "DEF": "#eb6834", "DEF2": "#c43d1b", "SPH": "#2a78d6",
            "AST": "#1baf7a", "COMA": "#8a5cd1"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, field, ttl in [(axs[0], "nerr_al", "aligned (shift + twin + global phase)"),
                           (axs[1], "nerr_ph", "global phase only (no ground truth needed)")]:
        for p in PROBES:
            mu = np.array([vals(res, p, "K", it, field, None).mean() for it in ITERS])
            ax.plot(ITERS, mu, color=pcol[p], lw=1.8, marker="o", ms=3, label=p,
                    ls="--" if p in ("DISK", "AMP") else "-")
        style(ax)
        ax.set_title(ttl, color=ink, fontsize=9)
        ax.set_xlabel("iterations", color=ink)
    axs[0].set_ylabel("K-HIO normalized error  Σ|ψ̂−ψ|² / Σ|ψ|²", color=ink)
    axs[0].legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.suptitle("4-2b: cross-probe comparison (K-HIO)", color=ink, fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "probe42b_nerr.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑檔案檢查、探針幾何與單元測試")
    a = ap.parse_args()
    if not check_files():
        print("\n❌ 缺檔案,停下來")
        sys.exit(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    probes, geo = unit_tests(dev)
    if a.check or not ok_all:
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過,先不要跑完整量測"))
        sys.exit(0 if ok_all else 1)
    print("\n重現 4-1")
    repro_4_1(dev)
    print("\n量測")
    res, kinds = measure(dev)
    raw = RUN_ROOT / ("probe42b_raw.json" if not QUICK else "probe42b_raw_quick.json")
    json.dump({"results": res, "kinds": kinds, "probes": PROBES, "iters": ITERS, "quick": QUICK}, open(raw, "w"))
    print(f"  原始結果先存檔:{raw}(彙整若出錯,資料不會遺失)")
    check_disk_vs_41(res)
    verdict = report(res, kinds, geo)
    json.dump({"results": res, "geometry": geo, "verdict": verdict, "probes": PROBES, "variants": VARIANTS,
               "iters": ITERS, "quick": QUICK},
              open(OUT_JSON if not QUICK else RUN_ROOT / "probe42b_quick.json", "w"))
    make_figures(res)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))
    print("把完整輸出貼給 Claude")


if __name__ == "__main__":
    main()
