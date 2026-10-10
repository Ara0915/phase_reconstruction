#!/usr/bin/env python
"""階段八 8c:診斷(只評估、不訓練;階段八實驗設計協定 §十七,v1.7)。

A   雜訊極限:Cramér–Rao 下限(CRLB)→ 隨劑量變的門檻 Q_k = k × CRLB(k = 2 主判、1.5 敏感度);
    (a2) 從真值出發的 Poisson 最大概似核對(劑量 1;論文劑量另報);事後的 h1′ / h2′。
B   第 4 點(高劑量時漏判 S 單空缺):B1 位置與特徵、B2 量測的可辨識度 D、B2b 網路重建與資料的一致性、B3 劑量掃描。
B4  第 6 點(有誤差時的誤報):位置特徵、局部正規化。
C   第 5 點:論文劑量下的 η 掃描。
D   包絡檔分析:迭代法的空缺判讀(第 9 點)、誤報比較(第 6 點)、以網路自己的品質為門檻的加速倍數。
不修改 scan_8.py:import 它,重用產生器、量測、網路流程、指標與判讀(同一套程式 → 與階段八可比)。

用法(計算節點):
    python scan_8c.py --check     # 內建檢查(§17.6)
    python scan_8c.py --smoke     # 迷你全流程(階段八 smoke 的模型與包絡檔);通過才可送件
    python scan_8c.py --run       # 正式(run_scan8c.sh)
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_8 as s8                                                   # noqa: E402  (階段八;不修改)

d7, c7, b7, a7, a6, s5, s5b = s8.d7, s8.c7, s8.b7, s8.a7, s8.a6, s8.s5, s8.s5b
RUN_ROOT, QUICK, SEEDS, SUF = s8.RUN_ROOT, s8.QUICK, s8.SEEDS, s8.SUF
S8_MD5 = "4821d2607f6e8aa381f0e6b5c5e72513"                           # 階段八正式結果所用的 scan_8.py
check, all_ok, dump_json = s8.check, s8.all_ok, s8.dump_json
NEW_L, NEW_C, BASE = s8.NEW_L, s8.NEW_C, s8.BASE
NETS3 = [BASE, NEW_C, NEW_L]
tg = s8.tg

# ============================================================================
# 設定(協定 §十七)
# ============================================================================
K_MAIN, K_SENS = 2.0, 1.5                    # §16.1:k = 2 主判、1.5 敏感度(nerr 的平方誤差單位)
TAU, TAU_ALT = 1e-6, 1e-8                    # 參數像素:照明 ≥ τ × 最大值
N_CRLB = 2 if QUICK else 64                  # 每個 seed 的場數(前 n 個;偶數 WS₂、奇數 MoS₂)
A2_N, A2_R, A2_MAXIT = (2, 2, 300) if QUICK else (8, 64, 1000)  # (a2):場數、每個場的雜訊次數、L-BFGS 的最多次數
A2_CONV = 1e-3                               # 收斂:白化後的梯度平方和(≈ Newton 減量)< 1e-3
A2_STOP, A2_CHUNK = 1e-6, 50                 # 提早停止(更嚴):每 50 次檢查一次
RHO_LO, RHO_HI = 0.8, 1.25                   # (a2) / CRLB 的通過範圍(劑量 1)
D_CLEAR, D_UNCLEAR = 25.0, 4.0               # B2:漏判 SV 的 D₁ 中位數 ≥ 25 → 分得清楚;< 4 → 分不清
CONC_X, CONC_D, CONC_MIN_N = 2.0, 0.10, 20   # 「集中」:某組 ≥ 整體 × 2 且差 ≥ 10 個百分點(組內至少 20 個)
DOSES = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
ETAS = [0.0, 0.05, 0.1, 0.15]
ETA_X, ETA_REL = 1.25, 1.25                  # C:r ≥ 1.25 且 z > 2,且 r ≥ 1.25 × P6B4e8 的 r
LOC_R_LAT = 3.0                              # 局部正規化的半徑(晶格常數)
FA_FRAC, LOC_FA_RED, LOC_REC_DROP = 0.5, 0.5, 0.02
B3_TURN = 0.9
NBR_R_LAT = 1.05                             # B1 (v):「1 個晶格常數內」的容差
MINI_MC = 100                                # --check 的迷你 Monte Carlo 次數

# seed(§17.7):正式 / smoke(基底 + 200,000)
SEED_B3, SEED_C, SEED_A2 = 88_600_000, 88_700_000, 88_710_000
SMOKE_OFF = 200_000
SEED_MINI = 88_919_000                       # --check 的迷你問題
MY_RANGES_8C = [(88_600_000, 88_919_999)]

OUT_JSON = RUN_ROOT / f"scan8c{SUF}.json"
SMOKE_DIR = RUN_ROOT / f"scan8c_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
FIG_DIR = Path("figs_scan8c")


def me_md5():
    return s8.md5(Path(__file__).resolve())


def seeds_of(smoke):
    off = SMOKE_OFF if smoke else 0
    return {"B3": SEED_B3 + off, "C": SEED_C + off, "A2": SEED_A2 + off}


# ============================================================================
# 量測模型(與 scan_8.measure_L 的期望強度一致;fp64)
# ============================================================================
class Phys:
    """已知名目探針、名目位置、beamstop 的前向模型:λ_m(ξ) = c · |fftshift(fft2(P · O_m, ortho))|² · bs(劑量 1 的期望計數)。"""

    def __init__(self, W, F, starts, P, c, bs, U, dev):
        self.W, self.F, self.starts, self.dev = W, F, list(starts), dev
        self.P = P.to(dev, torch.complex128)
        self.c = float(c)
        self.bs = bs.to(dev, torch.float64)
        self.U = U.to(dev)
        self.out = self.bs.flatten() > 0                                                # beamstop 外的偵測像素
        iy, ix = torch.meshgrid(torch.arange(W, device=dev), torch.arange(W, device=dev), indexing="ij")
        self.win = torch.stack([((y0 + iy) * F + (x0 + ix)).flatten() for y0, x0 in self.starts])   # [49, W²]
        pa = self.P.abs() ** 2
        ill = torch.zeros(F * F, dtype=torch.float64, device=dev)
        for j in range(len(self.starts)):
            ill.index_add_(0, self.win[j], pa.flatten())
        self.ill = ill.view(F, F) / ill.max()
        yy, xx = torch.meshgrid(torch.arange(W, dtype=torch.float64, device=dev), torch.arange(W, dtype=torch.float64, device=dev), indexing="ij")
        cy, cx = float((pa * yy).sum() / pa.sum()), float((pa * xx).sum() / pa.sum())
        self.centers = torch.tensor([[y0 + cy, x0 + cx] for y0, x0 in self.starts], dtype=torch.float64)
        self._D = None
        self._pm = None

    def idx_pm(self):
        """窗內像素對 (j, k) 的平攤索引:(j − k) mod W 與 (j + k) mod W(各維度)[W², W²]。"""
        if self._pm is None:
            W = self.W
            jy = torch.arange(W, device=self.dev).repeat_interleave(W)
            jx = torch.arange(W, device=self.dev).repeat(W)
            im = ((jy[:, None] - jy[None, :]) % W) * W + (jx[:, None] - jx[None, :]) % W
            ip = ((jy[:, None] + jy[None, :]) % W) * W + (jx[:, None] + jx[None, :]) % W
            self._pm = (im, ip)
        return self._pm

    @staticmethod
    def of(d):
        geo, cp = d["geo"], d["cp"]
        W = geo.W
        return Phys(W, geo.F, geo.starts, geo.pr["P"], cp.photons_per_pix * W * W / cp.ref_energy, d["bs"], geo.U, d["O"].device)

    def lam(self, O):
        """O:[B, F, F] complex → λ [B, 49, W, W](fp64;可微分)。"""
        O = O.to(torch.complex128)
        Ow = O.flatten(1)[:, self.win].view(O.shape[0], len(self.starts), self.W, self.W)
        E = torch.fft.fftshift(torch.fft.fft2(self.P * Ow, norm="ortho"), dim=(-2, -1))
        return self.c * E.abs() ** 2 * self.bs

    def dft(self):
        """[ξ(beamstop 外), j] 的 DFT 矩陣(fftshift 後的列順序;ortho)。"""
        if self._D is None:
            W = self.W
            eye = torch.eye(W * W, dtype=torch.complex128, device=self.dev).view(W * W, W, W)
            Ef = torch.fft.fftshift(torch.fft.fft2(eye, norm="ortho"), dim=(-2, -1)).reshape(W * W, W * W)     # [j, ξ]
            self._D = Ef.T[self.out].contiguous()
            del eye, Ef
        return self._D


# ============================================================================
# A:Fisher 資訊與 CRLB(§17.1)
# ============================================================================
def jac_window(ph, Of, j):
    """第 j 個位置的雅可比 [n_ξ, 2W²](參數 = 窗內像素的 Re O、Im O):2√c · Re(conj(u) ∂E/∂θ)。只用於核對 local_fisher。"""
    D = ph.dft()
    Pf = ph.P.flatten()
    psi = Pf * Of.flatten()[ph.win[j]]
    E = D @ psi
    a = E.abs()
    u = torch.where(a > 0, E / a.clamp_min(1e-300), torch.zeros_like(E))
    M = u.conj()[:, None] * D * Pf[None, :]
    return 2.0 * math.sqrt(ph.c) * torch.cat([M.real, -M.imag], 1)


def local_fisher_jac(ph, Of, j):
    J = jac_window(ph, Of, j)
    return J.T @ J


def local_fisher(ph, Of, j):
    """第 j 個位置的 Fisher 區塊 [2W², 2W²](= J_jᵀ J_j,解析的 FFT 形式;= Wei et al. 2020 Eq. (23) 的第一項):
    M = diag(ū) D diag(P) → A = MᴴM(元素只與 j − k 有關:ifft2(a),a = |u|²·bs)、B = MᵀM(只與 j + k 有關:fft2(b)/W²,b = ū²·bs);
    F_xx = 2c Re(A + B)、F_yy = 2c Re(A − B)、F_xy = −2c (Im A + Im B)、F_yx = F_xyᵀ。"""
    W = ph.W
    im, ip = ph.idx_pm()
    Pw = ph.P
    psi = Pw * Of.flatten()[ph.win[j]].view(W, W)
    E = torch.fft.fft2(psi, norm="ortho")                                                # 未 shift 的頻率順序
    a = E.abs()
    u = torch.where(a > 0, E / a.clamp_min(1e-300), torch.zeros_like(E))
    bsu = torch.fft.ifftshift(ph.bs, dim=(-2, -1))
    KA = torch.fft.ifft2((u.abs() ** 2) * bsu).flatten()
    KB = (torch.fft.fft2((u.conj() ** 2) * bsu) / (W * W)).flatten()
    Pf = Pw.flatten()
    A = (Pf.conj()[:, None] * Pf[None, :]) * KA[im]
    B = (Pf[:, None] * Pf[None, :]) * KB[ip]
    c2 = 2.0 * ph.c
    Fxy = -c2 * (A.imag + B.imag)
    return torch.cat([torch.cat([c2 * (A.real + B.real), Fxy], 1), torch.cat([Fxy.T, c2 * (A.real - B.real)], 1)], 0)


def fisher(ph, Of, inc, local=None):
    """劑量 1 的 Fisher 矩陣(參數 = inc 內像素的 Re、Im;[2Np, 2Np] fp64)與像素 → 參數的索引。"""
    local = local or local_fisher
    F = ph.F
    incf = inc.flatten()
    Np = int(incf.sum())
    pid = torch.full((F * F,), -1, dtype=torch.long, device=ph.dev)
    pid[incf] = torch.arange(Np, device=ph.dev)
    Fp = torch.zeros(2 * Np, 2 * Np, dtype=torch.float64, device=ph.dev)
    for j in range(len(ph.starts)):
        Fl = local(ph, Of, j)
        g = pid[ph.win[j]]
        keep = g >= 0
        kk = torch.cat([keep, keep])
        gi = torch.cat([g[keep], g[keep] + Np])
        Fp[gi[:, None], gi[None, :]] += Fl[kk][:, kk]                                   # 同一個位置內的索引不重複
        del Fl
    return Fp, pid, Np


def gauge_vecs(Of, pid, Np, U):
    """U 內的不可辨識方向(複數誤差空間 → 參數向量):整體相位、x / y 相位斜坡、x / y 平移。回傳 [2Np, 5](未正規化)與整張的整體相位 [2Np]。"""
    F = Of.shape[-1]
    dev = Of.device
    yy, xx = torch.meshgrid(torch.arange(F, dtype=torch.float64, device=dev), torch.arange(F, dtype=torch.float64, device=dev), indexing="ij")
    yy, xx = yy - (F - 1) / 2, xx - (F - 1) / 2
    dOx = torch.zeros_like(Of)
    dOy = torch.zeros_like(Of)
    dOx[:, 1:-1] = (Of[:, 2:] - Of[:, :-2]) / 2
    dOy[1:-1, :] = (Of[2:, :] - Of[:-2, :]) / 2
    fields = [1j * Of, 1j * xx * Of, 1j * yy * Of, dOx, dOy]
    sel = pid.view(F, F) >= 0
    idx = pid.view(F, F)[sel]
    Um = U & sel

    def vec(z, mask):
        v = torch.zeros(2 * Np, dtype=torch.float64, device=dev)
        zz = torch.where(mask, z, torch.zeros_like(z))[sel]
        v[idx] = zz.real
        v[idx + Np] = zz.imag
        return v

    G = torch.stack([vec(z, Um) for z in fields], 1)
    gfull = vec(1j * Of, sel)
    return G, gfull


def fisher_chol(ph, Of, tau=TAU):
    """劑量 1 的 Fisher 矩陣 + μĝĝᵀ 的 Cholesky 分解(§17.1 (4))與相關的索引、不可辨識的方向。"""
    Of = Of.to(ph.dev, torch.complex128)
    inc = ph.ill >= tau
    if not bool(inc[ph.U].all()):
        raise RuntimeError("U 內有照明 < τ 的像素")
    Fp, pid, Np = fisher(ph, Of, inc)
    G, gfull = gauge_vecs(Of, pid, Np, ph.U)
    gh = gfull / gfull.norm()
    null = float((Fp @ gh).norm() / Fp.norm())
    mu = float(Fp.diagonal().mean())
    Fp.addr_(gh, gh, alpha=mu)
    L, info = torch.linalg.cholesky_ex(Fp)
    del Fp
    return {"L": L, "info": int(info), "pid": pid, "Np": Np, "G": G, "null": null, "Of": Of}


def crlb_from(ph, ch, extra=True, chunk=2048):
    """T = tr(C_UU) − ĝ_Uᵀ C_UU ĝ_U(§17.1 (5));extra:另算去掉 5 個方向的 T5。"""
    if ch["info"] != 0:
        return {"ok": False, "info": ch["info"], "null": ch["null"], "n_par": 2 * ch["Np"]}
    L, pid, Np, G = ch["L"], ch["pid"], ch["Np"], ch["G"]
    Uidx = pid.view(ph.F, ph.F)[ph.U]
    cols = torch.cat([Uidx, Uidx + Np])
    trC = 0.0
    for i0 in range(0, len(cols), chunk):
        cc = cols[i0:i0 + chunk]
        B = torch.zeros(2 * Np, len(cc), dtype=torch.float64, device=ph.dev)
        B[cc, torch.arange(len(cc), device=ph.dev)] = 1.0
        X = torch.linalg.solve_triangular(L, B, upper=False)
        trC += float((X ** 2).sum())
        del B, X
    g1 = G[:, :1] / G[:, :1].norm()
    T = trC - float((torch.linalg.solve_triangular(L, g1, upper=False) ** 2).sum())
    out = {"ok": True, "T": T, "trC": trC, "null": ch["null"], "n_par": 2 * Np, "n_U": int(len(Uidx))}
    if extra:
        Q, _ = torch.linalg.qr(G)
        out["T5"] = trC - float((torch.linalg.solve_triangular(L, Q, upper=False) ** 2).sum())
    return out


def crlb_field(ph, Of, tau=TAU, extra=True):
    """一個場(劑量 1)的 CRLB(§17.1 (4)(5))。"""
    ch = fisher_chol(ph, Of, tau)
    out = crlb_from(ph, ch, extra)
    del ch
    return out


def ec_of(O, U):
    """nerr_c 的分母 Σ_U |O − c*|²(同 scan_8.setup_exam)。"""
    m = U.to(O.real.dtype)
    cs = (O * m).sum((-2, -1), keepdim=True) / m.sum()
    return ((O - cs).abs() ** 2 * m).sum((-2, -1))


# ============================================================================
# A:(a2) 從真值出發的 Poisson 最大概似(§17.1 (8))
# ============================================================================
def nll_of(ph, O, nraw, dose):
    lam = ph.lam(O) * dose
    return ((lam - nraw * torch.log(lam.clamp_min(1e-300))) * ph.bs).sum()


def ml_from_truth(ph, ch, nraw, dose, maxit):
    """R 組獨立量測(nraw:[R, 49, W, W] 原始計數)各自的 Poisson 最大概似估計(fp64、從真值出發)。
    白化的 L-BFGS:x = x_真值 + d^{−1/2} L^{−ᵀ} z(L = 真值處 Fisher 矩陣 + μĝĝᵀ 的 Cholesky)→ z 空間的 Hessian ≈ 單位矩陣;
    只改變參數化、不改變目標函數(最佳解 = 最大概似)。參數 = CRLB 的同一組像素(其餘固定在真值)。
    收斂指標:每組白化後的梯度平方和(≈ Newton 減量)。回傳 ([R, F, F], 資訊)。"""
    R = nraw.shape[0]
    Of, L, pid, Np = ch["Of"], ch["L"], ch["pid"], ch["Np"]
    F2 = ph.F * ph.F
    ii = torch.where(pid >= 0)[0]
    base = Of.flatten()
    x0 = torch.cat([base[ii].real, base[ii].imag])[None]
    sc = dose ** -0.5

    def O_of(z):
        x = x0 + sc * torch.linalg.solve_triangular(L.T, z.T, upper=True).T
        o = base[None].expand(z.shape[0], F2).clone()
        o[:, ii] = torch.complex(x[:, :Np], x[:, Np:])
        return o.view(-1, ph.F, ph.F)

    def nll_vec(z):
        lam = ph.lam(O_of(z)) * dose
        return ((lam - nraw * torch.log(lam.clamp_min(1e-300))) * ph.bs).sum((1, 2, 3))

    z = torch.zeros(R, 2 * Np, dtype=torch.float64, device=ph.dev, requires_grad=True)
    opt = torch.optim.LBFGS([z], lr=1.0, max_iter=A2_CHUNK, max_eval=int(A2_CHUNK * 1.5), history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-300)

    def closure():
        opt.zero_grad()
        f = nll_vec(z).sum()
        f.backward()
        return f

    done = 0
    with torch.enable_grad():
        while True:
            opt.step(closure)
            done += A2_CHUNK
            g = torch.autograd.grad(nll_vec(z).sum(), z)[0]
            if float((g ** 2).sum(1).max()) < A2_STOP or done >= maxit:
                break
    gz2 = (g ** 2).sum(1)
    st = opt.state[opt._params[0]]
    with torch.no_grad():
        est = O_of(z.detach())
    return est, {"iters": int(st.get("n_iter", done)), "gz2_max": float(gz2.max()), "conv": bool(float(gz2.max()) < A2_CONV)}


def score_q(ph, ch, nraw, dose):
    """Fisher 恆等式的核對:真值處的 score s = −∇NLL;q = sᵀ (d · I_F)⁻¹ s(用 I_F + μĝĝᵀ;s 與 ĝ 正交)。E[q] = 可辨識的參數數 = 2Np − 1。"""
    Of, L, pid, Np = ch["Of"], ch["L"], ch["pid"], ch["Np"]
    F2 = ph.F * ph.F
    ii = torch.where(pid >= 0)[0]
    base = Of.flatten()
    R = nraw.shape[0]
    with torch.enable_grad():
        x = torch.cat([base[ii].real, base[ii].imag])[None].expand(R, 2 * Np).clone().requires_grad_(True)
        o = base[None].expand(R, F2).clone()
        o[:, ii] = torch.complex(x[:, :Np], x[:, Np:])
        lam = ph.lam(o.view(-1, ph.F, ph.F)) * dose
        f = ((lam - nraw * torch.log(lam.clamp_min(1e-300))) * ph.bs).sum()
        g = torch.autograd.grad(f, x)[0]
    q = (torch.linalg.solve_triangular(L, g.T.contiguous(), upper=False) ** 2).sum(0) / dose
    return q.cpu().numpy(), 2 * Np - 1


def err_gphase(est, Of, U):
    """U 內去掉最佳整體相位後的 Σ|δO|²(est:[R, F, F])。"""
    m = U.to(torch.float64)
    cc = ((Of[None] * est.conj()) * m).sum((-2, -1))
    eal = est * torch.exp(1j * torch.angle(cc))[:, None, None]
    return ((eal - Of[None]).abs() ** 2 * m).sum((-2, -1))


# ============================================================================
# B:判讀、特徵、單顆 S 原子、可辨識度
# ============================================================================
@torch.no_grad()
def predict(est, d):
    """網路 / 迭代法的重建 → 逐場 nerr_c、逐柱判定與訊號(同 scan_8.lat_metrics 的對齊與判讀)。"""
    O, U, vac = d["O"], d["geo"].U, d["vac"]
    v0, v1, v2, t, q, eal = s8.align_full(est, O, U)
    fin = torch.isfinite(v0)
    m = U.to(O.real.dtype)
    c = ((eal - O).abs() ** 2 * m).sum((1, 2)) / d["Ec"].clamp_min(1e-30)
    c = torch.where(fin, c, torch.full_like(c, float("nan")))
    phase = torch.where(fin[:, None, None], torch.angle(eal), torch.full_like(eal.real, float("nan"))).cpu()
    pred, ok, mrec = vac.read(phase, fin.cpu())
    s, _ = vac.signal(phase)
    C, Cc = vac.confusion(pred)
    return {"c": c.cpu(), "pred": pred, "ok": ok, "s": s, "m": mrec, "fin": fin.cpu(), "conf": C, "conf_cat": Cc}


def a_px(d):
    """每個柱所在場的晶格常數(px)。"""
    mats = d["mats"]
    ap = torch.tensor([s8.A_LAT[s8.MATS[int(m)]] / d["dx"] for m in mats], dtype=torch.float64)
    return ap[d["vac"].fid]


def dist_out_u(U):
    """每個像素到最近的 U 外像素中心的距離 [F, F](U 外 = 0)。"""
    F = U.shape[-1]
    Uc = U.cpu()
    out = torch.nonzero(~Uc).double()
    yy, xx = torch.meshgrid(torch.arange(F, dtype=torch.float64), torch.arange(F, dtype=torch.float64), indexing="ij")
    pts = torch.stack([yy.flatten(), xx.flatten()], 1)
    dm = torch.empty(F * F, dtype=torch.float64)
    for i0 in range(0, len(pts), 2048):
        dm[i0:i0 + 2048] = torch.cdist(pts[i0:i0 + 2048], out).min(1).values
    return dm.view(F, F)


def col_features(d, ph, dmap):
    """B1 的特徵(每個可判讀柱):到最近掃描中心的距離、照明、到 U 邊界的距離、次像素偏移、1 個晶格常數內的其他空缺數。"""
    vac = d["vac"]
    pos = vac.pos.double()
    F = ph.F
    r = torch.round(pos).long().clamp(0, F - 1)
    dc = torch.cdist(pos, ph.centers.cpu()).min(1).values
    ill = ph.ill.cpu()[r[:, 0], r[:, 1]]
    du = dmap[r[:, 0], r[:, 1]]
    sub = (pos - torch.round(pos)).norm(dim=1)
    ap = a_px(d)
    nbr = torch.zeros(len(pos), dtype=torch.long)
    for i in range(vac.n):
        k = torch.where(vac.fid == i)[0]
        if len(k) == 0:
            continue
        dd = torch.cdist(pos[k], pos[k])
        isv = (vac.lab[k] > 0)
        near = (dd <= NBR_R_LAT * ap[k][:, None]) & (dd > 1e-9) & isv[None, :]
        nbr[k] = near.sum(1)
    return {"dist_center": dc.numpy(), "illum": ill.numpy(), "dist_U": du.numpy(), "subpix": sub.numpy(), "nbr_vac": nbr.numpy()}


FEAT_NAMES = {"dist_center": "到最近掃描中心的距離(px)", "illum": "照明(相對最大值)", "dist_U": "到 U 邊界的距離(px)",
              "subpix": "次像素偏移(px)", "nbr_vac": "1 個晶格常數內的其他空缺數", "loc_err": "局部位置誤差(px)"}


def quart_table(feat, hit, discrete=False):
    """依四分位(離散特徵:0、1、2、≥ 3)分組的比例;「集中」= 某組 ≥ 整體 × 2 且差 ≥ 10 個百分點(組內 ≥ 20 個)。"""
    feat = np.asarray(feat, float)
    hit = np.asarray(hit, bool)
    tot = float(hit.mean()) if hit.size else float("nan")
    if discrete:
        grp = np.minimum(feat, 3).astype(int)
        labels = ["0", "1", "2", "≥3"]
        edges = None
    else:
        edges = np.quantile(feat, [0.25, 0.5, 0.75]) if feat.size else np.zeros(3)
        grp = np.digitize(feat, edges, right=True)
        labels = ["Q1", "Q2", "Q3", "Q4"]
    rows = []
    conc = False
    for gi, lab in enumerate(labels):
        k = grp == gi
        n = int(k.sum())
        rate = float(hit[k].mean()) if n else float("nan")
        c = bool(n >= CONC_MIN_N and np.isfinite(rate) and rate >= CONC_X * tot and rate - tot >= CONC_D)
        conc |= c
        rows.append({"group": lab, "n": n, "rate": rate, "conc": c,
                     "range": None if discrete else [float(feat[k].min()) if n else None, float(feat[k].max()) if n else None]})
    return {"overall": tot, "rows": rows, "concentrated": conc, "edges": None if edges is None else [float(x) for x in edges]}


def s_kernel(d, pos, chunk=64):
    """單顆 S 原子在 pos([K, 2] 場座標)的投影相位 [K, F, F](同產生器:Kirkland + Debye–Waller、方形 Nyquist 頻帶、產生區 F + 2·HALO 再裁)。"""
    F, dx, dev = d["geo"].F, d["dx"], d["O"].device
    H = s8.HALO
    G = F + 2 * H
    tab = s8.element_tables(G, dx, dev)["S"].to(torch.complex128)
    kp = torch.fft.fftfreq(G, dtype=torch.float64, device=dev)
    out = []
    p = pos.to(dev, torch.float64) + H
    for i0 in range(0, len(p), chunk):
        q = p[i0:i0 + chunk]
        Ey = torch.polar(torch.ones(len(q), G, dtype=torch.float64, device=dev), -2 * math.pi * torch.remainder(q[:, 0, None] * kp, 1.0))
        Ex = torch.polar(torch.ones(len(q), G, dtype=torch.float64, device=dev), -2 * math.pi * torch.remainder(q[:, 1, None] * kp, 1.0))
        S = Ey[:, :, None] * Ex[:, None, :]
        out.append(torch.fft.ifft2(tab[None] * S).real[:, H:H + F, H:H + F])
    return torch.cat(out)


def d1_cols(d, ph, cols, eta, chunk=16):
    """B2:每個柱「多一顆 S」vs 真值在劑量 1 的期望 χ² 距離 D₁ = Σ (λ′ − λ)² / λ(beamstop 外)。cols:柱的索引(vac 的順序)。"""
    vac = d["vac"]
    out = torch.zeros(len(cols), dtype=torch.float64)
    fid = vac.fid[cols]
    for i in torch.unique(fid).tolist():
        kk = torch.where(fid == i)[0]
        Oi = d["O"][i].to(torch.complex128)
        lam0 = ph.lam(Oi[None])[0]
        pos = vac.pos[cols[kk]].double()
        phi = s_kernel(d, pos)
        for j0 in range(0, len(kk), chunk):
            Op = Oi[None] * torch.exp((1j - eta) * phi[j0:j0 + chunk])
            lam1 = ph.lam(Op)
            dl = (lam1 - lam0[None]) ** 2 / lam0[None].clamp_min(1e-300)
            dl = torch.where(lam0[None] > 0, dl, torch.zeros_like(dl))
            out[kk[j0:j0 + chunk]] = dl.sum((1, 2, 3)).cpu()
    return out


def raw_counts(d):
    """量測的原始計數 [n, 49, W, W](counts × 劑量,四捨五入)。"""
    dd = d["dose"].to(d["O"].device).double()[:, None, None]
    return torch.stack([torch.round(c.double() * dd) for c in d["counts"]], 1)


@torch.no_grad()
def dchi2_fields(d, ph, est, idx):
    """B2b:Δχ² = 2 [NLL(重建) − NLL(真值)](Poisson、實際計數;逐場)。"""
    nr = raw_counts(d)
    out = []
    for i in idx:
        dose = float(d["dose"][i])
        n_ = nr[i:i + 1]
        a = nll_of(ph, est[i:i + 1], n_, dose)
        b = nll_of(ph, d["O"][i:i + 1], n_, dose)
        out.append(2.0 * float(a - b))
    return np.array(out)


def local_read(d, s, ok, R_lat=LOC_R_LAT):
    """局部正規化的判讀(§17.3):每個柱 ÷ 同一個場、半徑 R 個晶格常數內所有可判讀柱訊號的中位數;門檻同原本;無效的場同原本的規則。"""
    vac = d["vac"]
    pos = vac.pos.double()
    ap = a_px(d)
    rho = torch.full((len(pos),), float("nan"), dtype=torch.float64)
    for i in range(vac.n):
        k = torch.where(vac.fid == i)[0]
        if len(k) == 0:
            continue
        dd = torch.cdist(pos[k], pos[k])
        near = dd <= R_lat * ap[k][:, None]
        sv = s[k].double()
        M = torch.where(near, sv[None, :].expand(len(k), -1), torch.full_like(dd, float("nan")))
        mloc = torch.nanmedian(M, 1).values
        rho[k] = sv / mloc
    pred = torch.where(rho >= s8.RHO_HI, 0, torch.where(rho >= s8.RHO_LO, 1, 2))
    bad = ~ok[vac.fid] | ~torch.isfinite(rho)
    pred = torch.where(bad, torch.where(vac.lab == 0, 1, 0), pred)
    C, _ = vac.confusion(pred.long())
    return pred.long(), C


def loc_pos_err(d, ph):
    """B4 (i):每個柱的局部位置誤差 = Σ_j w_j |pos · z_j| / Σ_j w_j(w_j = 探針 j 在該柱的 |P|²)。"""
    vac = d["vac"]
    pos = vac.pos.double()
    W = ph.W
    pa = (ph.P.abs() ** 2).cpu()
    r = torch.round(pos).long()
    st = torch.tensor(ph.starts, dtype=torch.long)
    ly = r[:, None, 0] - st[None, :, 0]
    lx = r[:, None, 1] - st[None, :, 1]
    inside = (ly >= 0) & (ly < W) & (lx >= 0) & (lx < W)
    w = torch.where(inside, pa[ly.clamp(0, W - 1), lx.clamp(0, W - 1)], torch.zeros(1, dtype=pa.dtype))
    e = (d["spec"]["pos"] * d["z"].cpu().double()).norm(dim=-1)                       # [n, 49]
    ej = e[vac.fid]
    return ((w * ej).sum(1) / w.sum(1).clamp_min(1e-300)).numpy()


# ============================================================================
# 輸入(階段八的正式 / smoke 結果)
# ============================================================================
def ctx_of(smoke):
    if smoke:
        return {"smoke": True, "env_root": s8.SMOKE_DIR, "j8": s8.final_json(s8.SMOKE_DIR), "roots": s8.roots_of(True),
                "out": SMOKE_DIR / f"scan8c{SUF}.json", "fig": SMOKE_DIR / "figs", "seeds": seeds_of(True)}
    return {"smoke": False, "env_root": RUN_ROOT, "j8": s8.final_json(), "roots": s8.roots_of(False), "out": OUT_JSON, "fig": FIG_DIR,
            "seeds": seeds_of(False)}


def load_inputs(ctx):
    bad = []
    m8 = s8.me_md5()
    if m8 != S8_MD5:
        bad.append(f"scan_8.py 的 md5 {m8} ≠ 階段八正式結果所用的 {S8_MD5}")
    J8 = None
    if not ctx["j8"].exists():
        bad.append(f"{ctx['j8']} 不存在")
    else:
        J8 = json.load(open(ctx["j8"]))
        if J8.get("script_md5") != S8_MD5:
            bad.append(f"{ctx['j8'].name} 由不同版本的 scan_8.py 產生")
        if not J8.get("complete"):
            bad.append(f"{ctx['j8'].name} 未標為完整")
        if bool(J8.get("smoke")) != ctx["smoke"]:
            bad.append(f"{ctx['j8'].name} 的 smoke 旗標不符")
    E = {}
    for e in s8.EXAMS:
        p = s8.env_json(e, ctx["env_root"])
        if not p.exists():
            bad.append(f"{p.name} 不存在")
            continue
        E[e] = json.load(open(p))
        if E[e]["meta"]["script_md5"] != S8_MD5 or not E[e]["checks_ok"]:
            bad.append(f"{p.name} 版本不同或檢查未通過")
    mb = []
    for nm in (NEW_L, NEW_C):
        for s in SEEDS:
            md = s8.model_dir("L" if nm == NEW_L else "C", s, ctx["roots"]["m"])
            if not ((md / "final.pt").exists() and (md / "result.json").exists()):
                mb.append(md.name)
    for s in SEEDS:
        md = d7.model_dir("e", s, ctx["roots"]["e"])
        if not ((md / "final.pt").exists() and (md / "result.json").exists()):
            mb.append(md.name)
    check(f"輸入:scan_8.py = 階段八的版本;{ctx['j8'].name} 完整;5 個包絡檔;P6B4e8 / -L / -C 的模型({'smoke' if ctx['smoke'] else '正式'})",
          not bad and not mb, ";".join(bad + [f"缺模型 {x}" for x in mb]))
    if bad or mb:
        return None
    iters = E[s8.EXAMS[0]]["meta"]["iters"]
    n = E[s8.EXAMS[0]]["meta"]["n"]
    fc = J8["verdict"]["fail_c"]
    env = {}
    for e in s8.EXAMS:
        env[e] = E[e]["env"]
        for conf in env[e].values():
            for rec in conf.values():
                s8.finalize_rec(rec, fc[e])
    return {"J8": J8, "V": J8["verdict"], "env": env, "iters": iters, "n": n, "fc": fc, "tim": J8["timing"]}


def net_mean_c(V, e, nm, tag=""):
    return float(np.mean(V["exam"][e]["nets"][nm]["nerr_c" + tag]))


# ============================================================================
# 各部分
# ============================================================================
def part_A(dev, I, ctx, log):
    """CRLB 與 (a2) 核對。階段八的 3 個 seed 用同一組物體(scan_8.lattice_seeds 的物體 seed 不隨 seed 變)、同一個探針,
    只差在劑量 1 的換算 c(各 seed 的 ref_energy)→ 在 seed 0 算 3 × N_CRLB 個**不同的物體**,其他 seed 以 T₁ ∝ 1/c 換算,
    並在每個 seed 的第 1 個場直接算一次核對(物體、探針相同;比值 = c₀/c_s,相對差 < 1e-6)。"""
    smoke = ctx["smoke"]
    nC = min(N_CRLB * len(SEEDS) if not smoke else 2, I["n"])
    out = {"T1": {}, "a2": {}, "c": {}, "scale": {}}
    t0 = time.time()
    s0 = SEEDS[0]
    for grp, exam in (("ideal", "L-ideal"), ("newdef", "L-newdef")):
        rows = []
        d = s8.setup_exam(s0, dev, exam, nC, smoke)
        ph = Phys.of(d)
        out["c"][str(s0)] = ph.c
        Ec = d["Ec"].double().cpu()
        for i in range(nC):
            r = crlb_field(ph, d["O"][i])
            if not r["ok"]:
                check(f"[A] {exam} 場 {i}:Cholesky 成功", False, f"info {r['info']}")
                continue
            rows.append({"i": i, "mat": int(d["mats"][i]), "T": r["T"], "T5": r["T5"], "trC": r["trC"], "Ec": float(Ec[i]),
                         "null": r["null"], "n_par": r["n_par"], "n_U": r["n_U"]})
            if (i + 1) % 32 == 0:
                log(f"  [A] {exam}:{i + 1} / {nC} 個場的 CRLB(經過 {time.time() - t0:.0f} 秒)")
        out["T1"][grp] = rows
        sc = []
        for s in SEEDS[1:]:
            ds = s8.setup_exam(s, dev, exam, 1, smoke)
            phs = Phys.of(ds)
            out["c"][str(s)] = phs.c
            same = bool(torch.equal(ds["O"][0], d["O"][0])) and bool(torch.equal(phs.P, ph.P)) and bool(torch.equal(phs.bs, ph.bs))
            Ts = crlb_field(phs, ds["O"][0], extra=False)["T"]
            ratio = Ts / (rows[0]["T"] * ph.c / phs.c)
            sc.append({"seed": s, "same_obj_probe": same, "ratio": ratio})
            del ds, phs
        out["scale"][grp] = sc
        check(f"[A] {exam}:其他 seed 的物體與探針 = seed 0;T₁ 依 c 換算(直接算 / 換算 − 1 < 1e-6)",
              all(x["same_obj_probe"] and abs(x["ratio"] - 1) < 1e-6 for x in sc), "、".join(f"seed {x['seed']} {x['ratio'] - 1:+.1e}" for x in sc))
        del d, ph
        log(f"  [A] {exam}:{len(rows)} 個不同物體的 CRLB 完成(經過 {time.time() - t0:.0f} 秒)")
    # (a2)
    nA, R = (A2_N, A2_R) if not smoke else (1, 2)
    maxit = A2_MAXIT
    s = SEEDS[0]
    d = s8.setup_exam(s, dev, "L-ideal", max(nA, 1), smoke)
    ph = Phys.of(d)
    fW = float(s8.load_dose()["seeds"][str(s)]["f_W"])
    rows_by = {"one": [], "paper": []}
    for i in range(nA):
        ch = fisher_chol(ph, d["O"][i])
        Ti = crlb_from(ph, ch, extra=False)["T"]
        Of = ch["Of"]
        for di, (tag, dose) in enumerate((("one", 1.0), ("paper", fW))):
            lam = ph.lam(Of[None])[0] * dose
            nr = []
            for rep in range(R):
                sd = ctx["seeds"]["A2"] + 1000 * di + 100 * i + rep
                s8.guard([sd])
                gen = torch.Generator(device=dev).manual_seed(sd)
                nr.append(torch.poisson(lam, generator=gen))
            nr = torch.stack(nr)
            est, info = ml_from_truth(ph, ch, nr, dose, maxit)
            e1 = err_gphase(est, Of, ph.U).cpu().numpy()
            eal = s8.align_full(est.to(torch.complex64), Of[None].expand(R, -1, -1).to(torch.complex64), ph.U)
            m = ph.U.to(torch.float32)
            e2 = ((eal[5] - Of[None].to(torch.complex64)).abs() ** 2 * m).sum((1, 2)).cpu().numpy()
            rows_by[tag].append({"i": i, "mat": int(d["mats"][i]), "T_dose": Ti / dose, "err": e1.tolist(), "err_align": e2.tolist(), "info": info})
        del ch
        log(f"  [A] (a2) 場 {i}:劑量 1 與論文劑量 × {R} 次完成(經過 {time.time() - t0:.0f} 秒)")
    for tag, dose in (("one", 1.0), ("paper", fW)):
        out["a2"][tag] = {"dose": dose, "rows": rows_by[tag]}
    return out


def run_nets(d, s, dev, roots, nets):
    res = {}
    for nm in nets:
        kind, model, norm = s8.get_net(nm, s, d["cfg"], d["geo"].pr, dev, roots)
        res[nm] = s8.run_net(kind, model, norm, d)
        del model
    return res


def part_B(dev, I, ctx, log):
    """B1、B2、B2b(L-ideal、L-newdef、L-paper;P6B4e8-L)與重現核對。"""
    smoke, n = ctx["smoke"], I["n"]
    out = {}
    t0 = time.time()
    for e in ("L-ideal", "L-newdef", "L-paper"):
        feats, miss, caught, D1, conf_all, dchi, sumD = [], [], [], [], [], [], []
        D_s0, pos_s0 = None, None
        for s in SEEDS:
            d = s8.setup_exam(s, dev, e, n, smoke)
            ph = Phys.of(d)
            dmap = dist_out_u(d["geo"].U)
            est = run_nets(d, s, dev, ctx["roots"], [NEW_L])[NEW_L]
            p = predict(est, d)
            conf_all.append(p["conf"].sum(0).numpy())
            vac = d["vac"]
            sv = (vac.lab == 1) & (vac.cat == s8.CAT_ISO)
            cols = torch.where(sv)[0]
            f = col_features(d, ph, dmap)
            feats.append({k: v[cols.numpy()] for k, v in f.items()})
            ms = (p["pred"][cols] == 0).numpy()
            miss.append(ms)
            caught.append((p["pred"][cols] == 1).numpy())
            if e != "L-paper":                                            # L-paper 的物體 = L-ideal → D₁ 相同(× 劑量)
                pos = vac.pos[cols]
                if D_s0 is not None and torch.equal(pos, pos_s0):
                    D = D_s0                                              # 3 個 seed 的物體相同 → D₁ 相同(只依物體)
                else:
                    D = d1_cols(d, ph, cols, s8.ETA).numpy()
                    D_s0, pos_s0 = D, pos.clone()
                D1.append(D)
                fids = vac.fid[cols].numpy()
                idx = np.unique(fids)
                dc = dchi2_fields(d, ph, est.to(torch.complex128), idx.tolist())
                sd = np.array([D[(fids == i) & ms].sum() for i in idx])
                dchi.append(dc)
                sumD.append(sd)
            del d, est, ph
        F_ = {k: np.concatenate([x[k] for x in feats]) for k in feats[0]}
        M_ = np.concatenate(miss)
        K_ = np.concatenate(caught)
        tab = {k: quart_table(F_[k], M_, discrete=(k == "nbr_vac")) for k in F_}
        rec = {"n_sv": int(M_.size), "n_sv_objects": int(M_.size // len(SEEDS)), "miss_rate": float(M_.mean()) if M_.size else float("nan"), "features": tab,
               "conf": np.sum(conf_all, 0).astype(int).tolist()}
        if D1:
            D_ = np.concatenate(D1)
            rec["D1"] = {"missed": _qs(D_[M_]), "caught": _qs(D_[K_]), "all": _qs(D_)}
            rec["D1_raw"] = {"missed": D_[M_].tolist(), "caught": D_[K_].tolist()}
            dc_, sd_ = np.concatenate(dchi), np.concatenate(sumD)
            hasm = sd_ > 0
            rec["dchi2"] = {"n_fields": int(dc_.size), "mean": float(dc_.mean()), "median": float(np.median(dc_)),
                            "frac_ge_sumD": float((dc_[hasm] >= sd_[hasm]).mean()) if hasm.any() else float("nan"),
                            "n_fields_with_miss": int(hasm.sum())}
        out[e] = rec
        log(f"  [B] {e} 完成(經過 {time.time() - t0:.0f} 秒)")
    return out


def svf1_seeds(confs):
    """SV-F1:階段八的定義(各 seed 的 F1 平均)與合併混淆矩陣的 F1。confs:每個 seed 的 3 × 3。"""
    per = [s8.vac_stats(np.array(c, float))["sv_f1"] for c in confs]
    return float(np.mean(per)), s8.vac_stats(np.sum(np.array(confs, float), 0))["sv_f1"]


def _qs(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return {"n": int(x.size), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]), "mean": float(x.mean())}


def part_B3(dev, I, ctx, log):
    """B3:劑量掃描(L-ideal 的物體;3 個網路)。"""
    smoke, n = ctx["smoke"], I["n"]
    out = {str(x): {nm: {"c": [], "conf": []} for nm in NETS3} for x in DOSES}
    t0 = time.time()
    for s in SEEDS:
        d0 = s8.setup_exam(s, dev, "L-ideal", n, smoke)
        nets = {nm: s8.get_net(nm, s, d0["cfg"], d0["geo"].pr, dev, ctx["roots"]) for nm in NETS3}
        for di, dose in enumerate(DOSES):
            ns = ctx["seeds"]["B3"] + 10_000 * di + 1000 * s
            s8.guard([ns])
            dv = torch.full((n,), float(dose))
            d = dict(d0)
            d["counts"] = s8.measure_L(d0["fields"], d0["geo"], d0["cp"], d0["bs"], ns, d0["spec"], d0["z"], d0["sign"], d0["ops"], dv)
            d["dose"] = dv
            for nm, (kind, model, norm) in nets.items():
                p = predict(s8.run_net(kind, model, norm, d), d)
                c = p["c"].numpy()
                out[str(dose)][nm]["c"].append(np.where(np.isfinite(c), c, I["fc"]["L-ideal"]).tolist())
                out[str(dose)][nm]["conf"].append(p["conf"].sum(0).numpy().astype(int).tolist())
            del d
        del d0, nets
        log(f"  [B3] seed {s} 完成(經過 {time.time() - t0:.0f} 秒)")
    return out


def part_B4(dev, I, ctx, log):
    """B4:L-combo 的誤報特徵(3 seeds)與局部正規化(網路 3 seeds;迭代法 seed 0);L-paper 對照。"""
    smoke, n = ctx["smoke"], I["n"]
    out = {}
    t0 = time.time()
    for e in ("L-combo", "L-paper"):
        it_conf = D_iter_conf(I, e, "512")                               # 網路時間(b512)下依 SV-F1 的包絡設定
        feats, fa = [], []
        Cn_g, Cn_l, Ci_g, Ci_l = [], [], [], []
        for s in SEEDS:
            d = s8.setup_exam(s, dev, e, n, smoke)
            ph = Phys.of(d)
            dmap = dist_out_u(d["geo"].U)
            p = predict(run_nets(d, s, dev, ctx["roots"], [NEW_L])[NEW_L], d)
            vac = d["vac"]
            intact = torch.where(vac.lab == 0)[0]
            f = col_features(d, ph, dmap)
            le = loc_pos_err(d, ph)
            feats.append({"loc_err": le[intact.numpy()], "dist_U": f["dist_U"][intact.numpy()], "illum": f["illum"][intact.numpy()]})
            fa.append((p["pred"][intact] > 0).numpy())
            _, Cl = local_read(d, p["s"], p["ok"])
            Cn_g.append(p["conf"].sum(0).numpy())
            Cn_l.append(Cl.sum(0).numpy())
            if s == SEEDS[0] and it_conf is not None:
                est = s8.iter_est_L(d, it_conf[0], it_conf[1], dev)
                pi = predict(est, d)
                _, Cli = local_read(d, pi["s"], pi["ok"])
                Ci_g.append(pi["conf"].sum(0).numpy())
                Ci_l.append(Cli.sum(0).numpy())
            del d, ph
        F_ = {k: np.concatenate([x[k] for x in feats]) for k in feats[0]}
        A_ = np.concatenate(fa)
        rec = {"n_intact": int(A_.size), "fa_rate": float(A_.mean()) if A_.size else float("nan"),
               "features": {k: quart_table(F_[k], A_) for k in F_},
               "net_global": s8.vac_stats(np.sum(Cn_g, 0)), "net_local": s8.vac_stats(np.sum(Cn_l, 0)),
               "iter_conf": it_conf}
        if Ci_g:
            rec["iter_global_s0"] = s8.vac_stats(np.sum(Ci_g, 0))
            rec["iter_local_s0"] = s8.vac_stats(np.sum(Ci_l, 0))
        out[e] = rec
        log(f"  [B4] {e} 完成(經過 {time.time() - t0:.0f} 秒)")
    return out


def part_C(dev, I, ctx, log):
    """C:L-paper 的物體,η = 0、0.05、0.1、0.15(同一組雜訊 seed);3 個網路;迭代法參考(seed 0)。"""
    smoke, n = ctx["smoke"], I["n"]
    out = {str(x): {nm: {"c": [], "conf": []} for nm in NETS3} for x in ETAS}
    opp = I["V"]["exam"]["L-paper"]["nets"][NEW_L]["512"]["opp"]
    itc = (opp[0], int(opp[1]))
    t0 = time.time()
    for x in ETAS:
        out[str(x)]["iter"] = None
    for s in SEEDS:
        ns = ctx["seeds"]["C"] + 1000 * s
        s8.guard([ns])
        for x in ETAS:
            d = s8.setup_exam(s, dev, "L-paper", n, smoke, eta=x, noise_seed=ns)
            ests = run_nets(d, s, dev, ctx["roots"], NETS3)
            for nm in NETS3:
                p = predict(ests[nm], d)
                c = p["c"].numpy()
                out[str(x)][nm]["c"].append(np.where(np.isfinite(c), c, I["fc"]["L-paper"]).tolist())
                out[str(x)][nm]["conf"].append(p["conf"].sum(0).numpy().astype(int).tolist())
            if s == SEEDS[0]:
                p = predict(s8.iter_est_L(d, itc[0], itc[1], dev), d)
                c = p["c"].numpy()
                out[str(x)]["iter"] = {"conf": list(itc), "c": float(np.where(np.isfinite(c), c, I["fc"]["L-paper"]).mean()),
                                       "svf1": s8.vac_stats(p["conf"].sum(0).numpy())["sv_f1"]}
            del d, ests
        log(f"  [C] seed {s} 完成(經過 {time.time() - t0:.0f} 秒)")
    return out


# ============================================================================
# D:包絡檔分析(不需 GPU)
# ============================================================================
def D_iter_conf(I, e, B):
    """網路(P6B4e8-L)時間下,依 SV-F1(sv_miss)的迭代法包絡設定 (設定, 次數)。"""
    t = I["tim"]["net"][NEW_L][B]
    ev = a7.envelope7(I["env"][e], t, I["tim"][B], I["iters"], key="sv_miss")
    return None if ev is None else (ev[0], int(ev[1]))


def pooled_env_conf(env, conf, it, tag=""):
    return np.sum(np.array(env[conf][str(it)]["conf" + tag], float).reshape(-1, 3, 3), 0)


def lower_bound_time(tB, iters):
    """迭代法 500 次內達不到時的下界:min_設定(最後停止點的時間)。"""
    return min(a7.cfg_time7(c, a7.stops7(c, iters)[-1], tB) for c in a7.configs7())


def part_D(I):
    V, env, tim, iters = I["V"], I["env"], I["tim"], I["iters"]
    out = {"vac": {}, "own_q": {}}
    for e in s8.EXAMS:
        row = {"net": {}}
        for nm in (BASE, NEW_L):
            row["net"][nm] = s8.vac_stats(np.array(V["exam"][e]["nets"][nm]["vac"]["all"]["conf"], float))
        for B in ("64", "512"):
            cf = D_iter_conf(I, e, B)
            row["iter_T" + B] = None if cf is None else {"conf": list(cf), **s8.vac_stats(pooled_env_conf(env[e], cf[0], cf[1]))}
        best = None
        for c in a7.configs7():
            for it in a7.stops7(c, iters):
                v = float(a7.env_vals7(env[e], c, it, "sv_miss").mean())
                if best is None or v < best[2]:
                    best = (c, it, v)
        row["iter_best"] = {"conf": [best[0], best[1]], **s8.vac_stats(pooled_env_conf(env[e], best[0], best[1]))}
        out["vac"][e] = row
        oq = {}
        for nm in (BASE, NEW_L):
            for tag in ("", "_W", "_Mo"):
                if tag and e != s8.MAIN_EXAM:
                    continue
                Q = net_mean_c(V, e, nm, tag)
                r = {}
                for B in ("64", "512"):
                    tn = tim["net"][nm][B]
                    sp = s8.speed(Q, env[e], tim[B], tn, iters, key="nerr_c" + tag, qs=[Q])[str(Q)]
                    lb = lower_bound_time(tim[B], iters) / tn
                    r[B] = {"x": sp["x"], "x_lo": sp["x_lo"], "conf": sp["conf"], "it": sp["it"], "reach_iter": bool(np.isfinite(sp["t_iter"])),
                            "lb": lb}
                oq[nm + tag] = {"Q": Q, **r}
        out["own_q"][e] = oq
    return out


# ============================================================================
# 報表與判讀(§十七,事先寫定)
# ============================================================================
def fx(r):
    if r["reach_iter"]:
        return f"× {r['x_lo']:.2f}–{r['x']:.2f}"
    return f"迭代法 500 次內達不到 → ≥ × {r['lb']:.2f}"


def report(R, I, ctx):
    V, env, tim, iters = I["V"], I["env"], I["tim"], I["iters"]
    dose = s8.load_dose()
    out = {}
    if ctx["smoke"]:
        print("\n" + "!" * 100)
        print("迷你流程:場數極少、模型與包絡檔是 smoke 的 → 數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    # ---------------- A ----------------
    print("\n" + "=" * 100)
    print("A  雜訊極限(CRLB;§17.1)與事後的門檻(相對於階段八為事後分析;k 只依文獻)")
    print("=" * 100)
    A = R["A"]
    a2 = A["a2"]
    rho = {}
    for tag in ("one", "paper"):
        rows = a2[tag]["rows"]
        err = np.array([np.mean(r["err"]) for r in rows])
        era = np.array([np.mean(r["err_align"]) for r in rows])
        T = np.array([r["T_dose"] for r in rows])
        rng = np.random.default_rng(s8.BOOT_SEED)
        bi = rng.integers(0, len(rows), (s8.BOOT_N, len(rows)))
        rb = err[bi].mean(1) / T[bi].mean(1)
        rho[tag] = {"rho": float(err.mean() / T.mean()), "rho_align": float(era.mean() / T.mean()),
                    "ci": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                    "conv": all(r["info"]["conv"] for r in rows), "gz2_max": float(max(r["info"]["gz2_max"] for r in rows)),
                    "per_field": (err / T).tolist()}
    ok_a2 = RHO_LO <= rho["one"]["rho"] <= RHO_HI and rho["one"]["conv"]
    check(f"(a2) 核對:劑量 1 的最大概似誤差 / CRLB = {rho['one']['rho']:.3f} 在 {RHO_LO}–{RHO_HI}(CRLB 的實作與量測模型一致)", ok_a2 or ctx["smoke"],
          f"95% 區間 {rho['one']['ci'][0]:.3f}–{rho['one']['ci'][1]:.3f};用 align_full 的誤差 {rho['one']['rho_align']:.3f};"
          f"全部收斂 {rho['one']['conv']}(白化梯度平方和最大 {rho['one']['gz2_max']:.1e}、門檻 {A2_CONV:g})" + ("  [smoke:不判定]" if ctx["smoke"] else ""))
    print(f"  (a2) 論文劑量(f_W = {a2['paper']['dose']:.6f}):誤差 / CRLB = {rho['paper']['rho']:.3f}(95% 區間 {rho['paper']['ci'][0]:.3f}–"
          f"{rho['paper']['ci'][1]:.3f};全部收斂 {rho['paper']['conv']}、白化梯度平方和最大 {rho['paper']['gz2_max']:.1e};"
          f"描述:Wei et al. 2020 預期低計數時 < 1,估計被起點拉住)")
    out["a2"] = rho
    valid = ok_a2 and not ctx["smoke"]
    # CRLB per exam
    crl = {}
    for grp, exs in (("ideal", ("L-ideal", "L-paper", "L-combo", "L-coh")), ("newdef", ("L-newdef",))):
        rows = A["T1"][grp]
        T1 = np.array([r["T"] for r in rows])
        T5 = np.array([r["T5"] for r in rows])
        Ec = np.array([r["Ec"] for r in rows])
        mat = np.array([r["mat"] for r in rows])
        nul = max(r["null"] for r in rows)
        c0 = A["c"][str(SEEDS[0])]
        print(f"  [{grp}] {len(rows)} 個不同的物體(seed 0;其他 seed 物體相同、以 c 換算:{'、'.join(f'seed {x[0]} 比值 − 1 = {x[1] - 1:+.1e}' for x in [(y['seed'], y['ratio']) for y in A['scale'][grp]])});"
              f"每場參數 {rows[0]['n_par']}、U 內像素 {rows[0]['n_U']};零空間 ‖I_F ĝ‖/‖I_F‖ 最大 {nul:.1e};去掉 5 個方向 vs 只去整體相位:T5 / T 平均 {np.mean(T5 / T1):.4f}")
        for e in exs:
            spec = s8.ESPEC[e]
            per_obj = np.zeros(len(rows))
            for sd_ in SEEDS:
                Dd = dose["seeds"][str(sd_)]
                dd = np.ones(len(rows)) if spec["dose"] == "one" else np.where(mat == 0, Dd["f_W"], Dd["f_Mo"])
                per_obj += T1 * (c0 / A["c"][str(sd_)]) / (dd * Ec) / len(SEEDS)        # 該物體在 3 個 seed 的平均
            rng = np.random.default_rng(s8.BOOT_SEED)
            bi = rng.integers(0, len(per_obj), (s8.BOOT_N, len(per_obj)))
            row = {"crlb_c": float(per_obj.mean()), "ci": [float(np.percentile(per_obj[bi].mean(1), 2.5)), float(np.percentile(per_obj[bi].mean(1), 97.5))],
                   "n_obj": int(len(rows))}
            for tagm, mm in (("_W", 0), ("_Mo", 1)):
                row["crlb_c" + tagm] = float(per_obj[mat == mm].mean()) if (mat == mm).any() else float("nan")
            crl[e] = row
    out["crlb"] = crl
    print("\n  各考卷的 CRLB(nerr_c 單位;去掉整體相位;CRLB ∝ 1 / 劑量)與門檻 Q_k = k × CRLB")
    h1p = {}
    for e in s8.EXAMS:
        r = crl[e]
        note = {"L-combo": "(假設位置 / 探針已知 → 偏嚴的參考;描述)", "L-coh": "(同調模型下的參考;描述)"}.get(e, "")
        print(f"    {e:<9}:CRLB {r['crlb_c']:.5f}({r['n_obj']} 個物體的 bootstrap 95% 區間 {r['ci'][0]:.5f}–{r['ci'][1]:.5f};WS₂ {r['crlb_c_W']:.5f} / MoS₂ {r['crlb_c_Mo']:.5f})"
              f";Q₂ = {K_MAIN * r['crlb_c']:.5f}、Q₁.₅ = {K_SENS * r['crlb_c']:.5f}{note}")
        h1p[e] = {}
        for k in (K_MAIN, K_SENS):
            for nm in s8.NETS:
                for tag in (("", "_W", "_Mo") if e == s8.MAIN_EXAM else ("",)):
                    Q = k * r["crlb_c" + tag]
                    a = net_mean_c(V, e, nm, tag)
                    sp = {B: s8.speed(a, env[e], tim[B], tim["net"][nm][B], iters, key="nerr_c" + tag, qs=[Q])[str(Q)] for B in ("64", "512")}
                    lab = s8.speed_label(sp["64"], sp["512"])
                    lb = {B: lower_bound_time(tim[B], iters) / tim["net"][nm][B] for B in ("64", "512")}
                    h1p[e][f"k{k}|{nm}{tag}"] = {"Q": Q, "net": a, "ratio_to_crlb": a / max(r["crlb_c" + tag], 1e-30),
                                                  "label": lab if valid else None, "label_unverified": lab,
                                                  "descriptive": e in ("L-combo", "L-coh"), "b64": sp["64"], "b512": sp["512"], "lb": lb}
    out["h1p"] = h1p
    print("\n  事後的 h1′ / h2′(門檻 Q_k;判讀規則同 §六 h1;" + ("CRLB 已由 (a2) 核對" if valid else "⚠️ CRLB 未通過 (a2) 核對 → 只列數字、不判讀") + ")")
    for e in s8.EXAMS:
        for k in (K_MAIN, K_SENS):
            for tag in (("", "_W", "_Mo") if e == s8.MAIN_EXAM else ("",)):
                parts = []
                for nm in s8.NETS:
                    h = h1p[e][f"k{k}|{nm}{tag}"]
                    xs = []
                    for B in ("64", "512"):
                        sp = h["b" + B]
                        if sp["reach"]:
                            xs.append(f"b{B} " + (s8.fmt_x(sp) if np.isfinite(sp["t_iter"]) else f"迭代法未達(≥ × {h['lb'][B]:.2f})"))
                    tagx = ("〔未驗證〕" if not valid else "") + ("〔描述〕" if h["descriptive"] else "")
                    parts.append(f"{tg(nm)} {h['net']:.4f}(= CRLB × {h['ratio_to_crlb']:.2f})→ {tagx}{h['label_unverified']}" + (f"〔{'、'.join(xs)}〕" if xs else ""))
                mk = {"": "", "_W": " WS₂", "_Mo": " MoS₂"}[tag]
                print(f"    {e}{mk} k = {k}(Q = {h1p[e][f'k{k}|{NEW_L}{tag}']['Q']:.5f}):" + ";".join(parts))
    out["A_valid"] = valid
    # ---------------- B ----------------
    print("\n" + "=" * 100)
    print("B  第 4 點:P6B4e8-L 漏判 S 單空缺(§17.2)")
    print("=" * 100)
    B_ = R["B"]
    for e, rec in B_.items():
        print(f"  ■ {e}:孤立的 SV {rec['n_sv']} 個(= {rec['n_sv_objects']} 個柱 × 3 seeds:物體相同、雜訊與網路不同),漏判率 {rec['miss_rate']:.1%}")
        for k, tb in rec["features"].items():
            cells = "、".join(f"{r_['group']} {r_['rate']:.1%}(n {r_['n']})" for r_ in tb["rows"])
            print(f"     B1 {FEAT_NAMES[k]}:{cells} → {'集中' if tb['concentrated'] else '不集中'}")
        if "D1" in rec:
            dm, dcg = rec["D1"]["missed"], rec["D1"]["caught"]
            print(f"     B2 D₁(劑量 1):漏判 中位數 {dm.get('median', float('nan')):.3g}(四分位 {dm.get('q25', float('nan')):.3g}–{dm.get('q75', float('nan')):.3g},"
                  f"n {dm['n']});判對 中位數 {dcg.get('median', float('nan')):.3g}(n {dcg['n']})")
            dc = rec["dchi2"]
            print(f"     B2b Δχ²(網路的重建 − 真值,實際計數):平均 {dc['mean']:.3g}、中位數 {dc['median']:.3g};有漏判的 {dc['n_fields_with_miss']} 個場中,"
                  f"Δχ² ≥ 漏判柱的 ΣD 的比例 {dc['frac_ge_sumD']:.1%}")
    dmed = B_["L-ideal"]["D1"]["missed"].get("median", float("nan"))
    lab4 = ("量測分得清楚,網路沒有照資料" if dmed >= D_CLEAR else ("量測本身分不清,網路靠先驗補上" if dmed < D_UNCLEAR else "邊界")) if np.isfinite(dmed) else "無漏判"
    print(f"  ▶ 判讀(L-ideal、劑量 1、漏判 SV 的 D₁ 中位數 {dmed:.3g};≥ {D_CLEAR:g} / < {D_UNCLEAR:g}):{lab4}")
    if "D1" in B_["L-newdef"]:
        dn = B_["L-newdef"]["D1"]["missed"].get("median", float("nan"))
        print(f"    (L-newdef 的孤立 SV:漏判的 D₁ 中位數 {dn:.3g};描述)")
    out["B_label"] = lab4
    # B3
    B3 = R["B3"]
    print("\n  B3 劑量掃描(L-ideal 的物體;SV 召回率 / 精確率(合併 3 seeds)/ SV-F1(各 seed 平均,同階段八)/ 補回率 / nerr_c;D = D₁ 中位數 × 劑量)")
    turn = None
    dmed_all = B_["L-ideal"]["D1"]["all"].get("median", float("nan"))
    b3 = {}
    for dose_ in DOSES:
        cells = []
        for nm in NETS3:
            st = s8.vac_stats(np.sum(np.array(B3[str(dose_)][nm]["conf"], float), 0))
            f1m, _ = svf1_seeds(B3[str(dose_)][nm]["conf"])
            c = float(np.mean([np.mean(x) for x in B3[str(dose_)][nm]["c"]]))
            b3.setdefault(str(dose_), {})[nm] = {**st, "nerr_c": c, "sv_f1_seedmean": f1m}
            cells.append(f"{tg(nm)} {st['sv_r']:.3f} / {st['sv_p']:.3f} / {f1m:.3f} / {st['fill']:.3f} / {c:.4f}")
        print(f"    劑量 {dose_:<6g}(D ≈ {dmed_all * dose_:.3g}):" + ";".join(cells))
    for dose_ in sorted(DOSES):
        if b3[str(dose_)][NEW_L]["sv_r"] < B3_TURN:
            turn = dose_
            break
    print(f"  ▶ 轉折劑量(由低往高,P6B4e8-L 的 SV 召回率第一次 < {B3_TURN}):{turn if turn is not None else '無'}")
    ref = s8.vac_stats(np.array(V["exam"]["L-ideal"]["nets"][NEW_L]["vac"]["all"]["conf"], float))["sv_r"]
    print(f"    核對(描述):劑量 1 的召回率 {b3['1.0'][NEW_L]['sv_r']:.3f} vs 階段八 L-ideal {ref:.3f}(差 {abs(b3['1.0'][NEW_L]['sv_r'] - ref):.3f};"
          f"< 0.05 為一致)")
    rc = s8.vac_stats(np.array(B_["L-ideal"]["conf"], float))["sv_r"]
    print(f"    核對(描述):重現階段八的考卷(同 seed)P6B4e8-L 的 SV 召回率 {rc:.3f} vs 階段八 {ref:.3f}")
    out["B3"] = {"table": b3, "turn": turn}
    # ---------------- B4 + D2 ----------------
    print("\n" + "=" * 100)
    print("B4 / D  第 6 點:有誤差時的誤報(§17.3);第 9 點:迭代法的空缺判讀(§17.5)")
    print("=" * 100)
    D = R["D"]
    for e in s8.EXAMS:
        row = D["vac"][e]
        nl = row["net"][NEW_L]
        parts = [f"P6B4e8-L SV-F1 {nl['sv_f1']:.3f}(精確率 {nl['sv_p']:.3f}、召回率 {nl['sv_r']:.3f}、誤報率 {nl['fpr']:.2%})"]
        for B in ("64", "512"):
            it = row["iter_T" + B]
            if it:
                parts.append(f"迭代法 網路時間 b{B} {s5b.fmt_conf(it['conf'][0])} ×{it['conf'][1]} SV-F1 {it['sv_f1']:.3f}(精確率 {it['sv_p']:.3f}、"
                             f"召回率 {it['sv_r']:.3f}、誤報率 {it['fpr']:.2%})")
        ib = row["iter_best"]
        parts.append(f"500 次內最佳 {s5b.fmt_conf(ib['conf'][0])} ×{ib['conf'][1]} SV-F1 {ib['sv_f1']:.3f}(誤報率 {ib['fpr']:.2%})")
        print(f"  {e}:" + ";".join(parts))
    B4 = R["B4"]
    for e, rec in B4.items():
        print(f"  ■ {e}:P6B4e8-L 的誤報率 {rec['fa_rate']:.2%}(完整柱 {rec['n_intact']} 個)")
        for k, tb in rec["features"].items():
            cells = "、".join(f"{r_['group']} {r_['rate']:.2%}(n {r_['n']})" for r_ in tb["rows"])
            print(f"     {FEAT_NAMES[k]}:{cells} → {'集中' if tb['concentrated'] else '不集中'}")
        g, l_ = rec["net_global"], rec["net_local"]
        print(f"     局部正規化(網路,3 seeds):誤報 {g['conf'][0][1] + g['conf'][0][2]} → {l_['conf'][0][1] + l_['conf'][0][2]};"
              f"SV 召回率 {g['sv_r']:.3f} → {l_['sv_r']:.3f}")
        if "iter_global_s0" in rec:
            gi, li = rec["iter_global_s0"], rec["iter_local_s0"]
            print(f"     局部正規化(迭代法 {s5b.fmt_conf(rec['iter_conf'][0])} ×{rec['iter_conf'][1]},seed 0):誤報 {gi['conf'][0][1] + gi['conf'][0][2]} → "
                  f"{li['conf'][0][1] + li['conf'][0][2]};SV 召回率 {gi['sv_r']:.3f} → {li['sv_r']:.3f}")
    rc_ = D["vac"]["L-combo"]
    fn = rc_["net"][NEW_L]["fpr"]
    f_it = [rc_["iter_T" + B]["fpr"] if rc_["iter_T" + B] else float("nan") for B in ("64", "512")]
    g, l_ = B4["L-combo"]["net_global"], B4["L-combo"]["net_local"]
    nfa_g = g["conf"][0][1] + g["conf"][0][2]
    nfa_l = l_["conf"][0][1] + l_["conf"][0][2]
    red = 1 - nfa_l / nfa_g if nfa_g > 0 else float("nan")
    drop = g["sv_r"] - l_["sv_r"]
    if fn > 0 and all(np.isfinite(x) and x >= FA_FRAC * fn for x in f_it):
        lab6 = "誤差本身造成(共同)→ 寫成限制、不修"
    elif fn > 0 and np.isfinite(red) and red >= LOC_FA_RED and drop <= LOC_REC_DROP:
        lab6 = "判讀方法(局部正規化可改善;另寫版本)"
    elif fn == 0:
        lab6 = "網路沒有誤報(不適用)"
    else:
        lab6 = "網路本身(需要重新微調;另外決定)"
    print(f"  ▶ 第 6 點的判讀(L-combo):網路誤報率 {fn:.2%};迭代法(網路時間 b64 / b512){f_it[0]:.2%} / {f_it[1]:.2%}"
          f"(門檻 ≥ 網路 × {FA_FRAC});局部正規化 誤報減少 {red:.0%}、SV 召回率下降 {drop * 100:.1f} 個百分點 → {lab6}")
    out["lab6"] = lab6
    # D3
    print("\n  D (3) 以網路自己的品質為門檻的加速倍數(同品質比時間;事後分析)")
    for e in s8.EXAMS:
        for key, r in D["own_q"][e].items():
            nm = key.split("_")[0]
            mk = {"": "", "_W": " WS₂", "_Mo": " MoS₂"}["_" + key.split("_")[1] if "_" in key else ""]
            print(f"    {e}{mk} {tg(nm)}:Q = {r['Q']:.4f};b64 {fx(r['64'])};b512 {fx(r['512'])}")
    # ---------------- C ----------------
    print("\n" + "=" * 100)
    print("C  第 5 點:論文劑量下的 η 掃描(L-paper 的物體;同一組雜訊 seed;§17.4)")
    print("=" * 100)
    C_ = R["C"]
    cm = {}
    for x in ETAS:
        cells = []
        for nm in NETS3:
            arr = np.array([np.mean(v) for v in C_[str(x)][nm]["c"]])
            f1m, f1p = svf1_seeds(C_[str(x)][nm]["conf"])
            cm.setdefault(nm, {})[str(x)] = arr
            cells.append(f"{tg(nm)} nerr_c {arr.mean():.4f} / SV-F1 {f1m:.3f}(合併 {f1p:.3f})")
        it = C_[str(x)]["iter"]
        if it:
            cells.append(f"迭代法(seed 0)nerr_c {it['c']:.4f} / SV-F1 {it['svf1']:.3f}")
        print(f"    η = {x:<5g}:" + ";".join(cells))
    lab5 = "論文劑量下未見明顯依賴"
    eta_rows = {}
    for x in ETAS:
        if x == 0.1:
            continue
        rL = s5b.ratio(cm[NEW_L][str(x)], cm[NEW_L]["0.1"])
        rB = s5b.ratio(cm[BASE][str(x)], cm[BASE]["0.1"])
        eta_rows[str(x)] = {"L": list(rL), "base": list(rB)}
        flag = x < 0.1 and rL[1] >= ETA_X and rL[2] > 2 and rL[1] >= ETA_REL * rB[1]
        if flag:
            lab5 = "論文劑量下依賴 η(之後考慮訓練時 η 隨機取值)"
        print(f"    r(η = {x:g}) = nerr_c(η) / nerr_c(0.1):P6B4e8-L {rL[1]:.3f}(z {rL[2]:+.1f});P6B4e8 {rB[1]:.3f}(z {rB[2]:+.1f})" + ("  ← 達到判準" if flag else ""))
    print(f"  ▶ 判讀:{lab5}")
    ref = net_mean_c(V, "L-paper", NEW_L)
    print(f"    核對(描述):η = 0.1 的 P6B4e8-L {cm[NEW_L]['0.1'].mean():.4f} vs 階段八 L-paper {ref:.4f}(差 {abs(cm[NEW_L]['0.1'].mean() / ref - 1):.1%};< 10% 為一致)")
    out["eta"] = eta_rows
    out["lab5"] = lab5
    return out


# ============================================================================
# 圖(描述)
# ============================================================================
def figures(R, I, rep, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    col = {BASE: "#2a78d6", NEW_C: "#8a5cd1", NEW_L: "#1baf7a", "iter": "#0b0b0b"}

    def save(fig, name):
        p = fig_dir / name
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    # 1. CRLB vs 劑量 + 劑量掃描的 nerr_c
    rows = R["A"]["T1"]["ideal"]
    c0 = R["A"]["c"][str(SEEDS[0])]
    base = np.mean([r["T"] / r["Ec"] for r in rows]) * np.mean([c0 / R["A"]["c"][str(x)] for x in SEEDS])
    ds = np.array(sorted(DOSES))
    dz = s8.load_dose()["seeds"]
    f_pap = float(np.mean([(dz[str(x)]["f_W"] + dz[str(x)]["f_Mo"]) / 2 for x in SEEDS]))

    def it_best(e):
        return min(float(a7.env_vals7(I["env"][e], c, it, "nerr_c").mean()) for c in a7.configs7() for it in a7.stops7(c, I["iters"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ds, base / ds, color="#d6452a", label="CRLB (nerr_c)")
    ax[0].fill_between(ds, K_SENS * base / ds, K_MAIN * base / ds, color="#d6452a", alpha=0.15, label="Q: k = 1.5–2")
    for nm in NETS3:
        y = [rep["B3"]["table"][str(x)][nm]["nerr_c"] for x in ds]
        ax[0].plot(ds, y, "o-", color=col[nm], label=tg(nm))
    for nm in NETS3:
        ax[0].plot([f_pap], [net_mean_c(I["V"], "L-paper", nm)], "D", color=col[nm], ms=6, mfc="none")
    ax[0].plot([1.0, f_pap], [it_best("L-ideal"), it_best("L-paper")], "s", color=col["iter"], ms=6, label="iterative best (500 it; stage 8)")
    ax[0].text(f_pap, net_mean_c(I["V"], "L-paper", NEW_L), "  L-paper (stage 8)", fontsize=6, va="center")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("dose (relative to base)")
    ax[0].set_ylabel("nerr_c (L-ideal objects)")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, which="both", lw=0.4, color="#e4e3df")
    for nm in NETS3:
        y = [rep["B3"]["table"][str(x)][nm]["sv_r"] for x in ds]
        ax[1].plot(ds, y, "o-", color=col[nm], label=tg(nm))
    ax[1].axhline(B3_TURN, color="#d6452a", lw=0.7, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("dose (relative to base)")
    ax[1].set_ylabel("SV recall")
    ax[1].legend(fontsize=7)
    ax[1].grid(True, which="both", lw=0.4, color="#e4e3df")
    fig.suptitle("8c: CRLB and dose sweep (lattice, 7x7 scan)", fontsize=10)
    fig.tight_layout()
    save(fig, "c8_dose.png")
    # 2. D₁ 分布
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, e in enumerate(("L-ideal", "L-newdef")):
        rec = R["B"][e]
        if "D1_raw" not in rec:
            continue
        allv = np.array(rec["D1_raw"]["missed"] + rec["D1_raw"]["caught"])
        if allv.size == 0:
            continue
        bins = np.logspace(np.log10(max(allv.min(), 1e-6)), np.log10(allv.max() + 1e-12), 40)
        ax[k].hist(rec["D1_raw"]["caught"], bins=bins, color="#1baf7a", alpha=0.6, label="caught SV")
        ax[k].hist(rec["D1_raw"]["missed"], bins=bins, color="#d6452a", alpha=0.6, label="missed SV")
        for v in (D_UNCLEAR, D_CLEAR):
            ax[k].axvline(v, color="#555", lw=0.7, ls="--")
        ax[k].set_xscale("log")
        ax[k].set_xlabel("D1 (expected chi^2 distance, dose 1)")
        ax[k].set_title(f"{e}: P6B4e8-L", fontsize=9)
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "c8_D1.png")
    # 3. η 掃描
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for nm in NETS3:
        y = [np.mean([np.mean(v) for v in R["C"][str(x)][nm]["c"]]) for x in ETAS]
        f1 = [s8.vac_stats(np.sum(np.array(R["C"][str(x)][nm]["conf"], float), 0))["sv_f1"] for x in ETAS]
        ax[0].plot(ETAS, y, "o-", color=col[nm], label=tg(nm))
        ax[1].plot(ETAS, f1, "o-", color=col[nm], label=tg(nm))
    yi = [R["C"][str(x)]["iter"]["c"] if R["C"][str(x)]["iter"] else np.nan for x in ETAS]
    ax[0].plot(ETAS, yi, "s--", color=col["iter"], label="iterative (seed 0)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("eta (absorption)")
    ax[0].set_ylabel("nerr_c (L-paper objects, paper dose)")
    ax[1].set_xlabel("eta (absorption)")
    ax[1].set_ylabel("SV-F1")
    for a_ in ax:
        a_.axvline(0.1, color="#999", lw=0.6, ls=":")
        a_.legend(fontsize=7)
        a_.grid(True, lw=0.4, color="#e4e3df")
    fig.tight_layout()
    save(fig, "c8_eta.png")


# ============================================================================
# 內建檢查(§17.6)
# ============================================================================
def mini_problem(dev, seed=SEED_MINI, counts_per_pos=1e6):
    """小問題(W = 16、F = 24、3 × 3 掃描、步距 4、beamstop 半徑 1.5):檢查 CRLB 的程式本身。"""
    s8.guard([seed])
    gen = torch.Generator().manual_seed(seed)
    W, F, st = 16, 24, 4
    yy, xx = torch.meshgrid(torch.arange(W, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    r2 = (yy - W / 2 + 0.5) ** 2 + (xx - W / 2 + 0.5) ** 2
    P = torch.polar(torch.exp(-r2 / (2 * 3.0 ** 2)), 0.02 * r2)
    starts = [(st * i, st * j) for i in range(3) for j in range(3)]
    bs = (torch.sqrt((yy - W / 2) ** 2 + (xx - W / 2) ** 2) > 1.5).double()
    U = torch.zeros(F, F, dtype=torch.bool)
    U[6:18, 6:18] = True
    O = torch.polar(0.8 + 0.2 * torch.rand(F, F, generator=gen, dtype=torch.float64), 0.5 * torch.rand(F, F, generator=gen, dtype=torch.float64))
    c = counts_per_pos / float((P.abs() ** 2).sum())
    return Phys(W, F, starts, P, c, bs, U, dev), O.to(dev)


@torch.no_grad()
def checks(dev, ctx):
    print("=" * 70)
    print("階段八 8c:內建檢查")
    print("=" * 70)
    # (1) 迷你問題:雅可比 vs autograd、零空間、Cholesky vs 特徵值分解、Monte Carlo
    ph, O = mini_problem(dev)
    inc = ph.ill >= TAU
    Fp, pid, Np = fisher(ph, O, inc)
    Fj, _, _ = fisher(ph, O, inc, local=local_fisher_jac)
    F2 = ph.F * ph.F
    with torch.enable_grad():
        th0 = torch.cat([O.real.flatten(), O.imag.flatten()]).requires_grad_(True)

        def lam_flat(th):
            z = torch.complex(th[:F2], th[F2:]).view(ph.F, ph.F)
            return ph.lam(z[None]).flatten()

        Jt = torch.autograd.functional.jacobian(lam_flat, th0)
    l0 = lam_flat(th0.detach())
    pos = l0 > 0
    Fa = Jt[pos].T @ (Jt[pos] / l0[pos][:, None])
    ii = torch.where(inc.flatten())[0]
    sel = torch.cat([ii, ii + F2])
    Fa = Fa[sel][:, sel]
    ej = max(float((Fp - Fa).norm() / Fa.norm()), float((Fj - Fa).norm() / Fa.norm()))
    G, gfull = gauge_vecs(O, pid, Np, ph.U)
    gh = gfull / gfull.norm()
    nul = float((Fp @ gh).norm() / Fp.norm())
    ev, Vv = torch.linalg.eigh(Fp)
    nz = int((ev < 1e-9 * ev.max()).sum())
    ch = fisher_chol(ph, O)
    r = crlb_from(ph, ch)
    keep = ev >= 1e-9 * ev.max()
    C = (Vv[:, keep] / ev[keep]) @ Vv[:, keep].T
    Uidx = pid.view(ph.F, ph.F)[ph.U]
    cols = torch.cat([Uidx, Uidx + Np])
    Cuu = C[cols][:, cols]
    g1 = G[cols, 0] / G[cols, 0].norm()
    Tp = float(Cuu.trace() - g1 @ Cuu @ g1)
    et = abs(r["T"] / Tp - 1)
    check("CRLB 的程式(迷你問題):解析雅可比的 Fisher 矩陣 = autograd(< 1e-6);整體相位是零空間(< 1e-10)且只有 1 個近零特徵值;"
          "Cholesky 的路徑 = 特徵值分解的偽反矩陣(< 1e-6)", ej < 1e-6 and nul < 1e-10 and nz == 1 and et < 1e-6,
          f"雅可比 {ej:.1e};零空間 {nul:.1e};近零特徵值 {nz} 個;T 相對差 {et:.1e};條件數(去零空間){float(ev[keep].max() / ev[keep].min()):.1e}")
    lam = ph.lam(O[None])[0]
    nr = torch.stack([torch.poisson(lam, generator=torch.Generator(device=dev).manual_seed(SEED_MINI + 1 + k)) for k in range(MINI_MC)])
    q, nexp = score_q(ph, ch, nr, 1.0)
    qm, qse = float(q.mean() / nexp), float(q.std() / math.sqrt(len(q)) / nexp)
    est, info = ml_from_truth(ph, ch, nr, 1.0, A2_MAXIT)
    e1 = err_gphase(est, O, ph.U).cpu().numpy()
    rho = float(e1.mean() / r["T"])
    se = float(e1.std() / math.sqrt(len(e1)) / r["T"])
    check(f"Fisher 恆等式(迷你問題,{MINI_MC} 次量測):真值處的 score s 滿足 E[sᵀ I_F⁻¹ s] = 可辨識的參數數(相對差 < 3%)", abs(qm - 1) < 0.03,
          f"{qm:.4f} ± {qse:.4f}(參數 {nexp + 1},可辨識 {nexp});描述:從真值出發的最大概似誤差 / CRLB {rho:.3f} ± {se:.3f}"
          f"(收斂 {info['conv']}、次數 {info['iters']}、白化梯度平方和最大 {info['gz2_max']:.1e})")
    # (2) 真實幾何:前向模型 = measure_L、一個場的 CRLB、τ 的敏感度
    d = s8.setup_exam(0, dev, "L-ideal", 4, smoke=True)
    phr = Phys.of(d)
    ref = s8.measure_L(d["fields"], d["geo"], d["cp"], d["bs"], 0, d["spec"], d["z"], d["sign"], d["ops"], torch.ones(4), poisson=False)
    mine = phr.lam(d["O"])
    ef = max(float((mine[:, j].float() - ref[j]).abs().max() / ref[j].abs().max()) for j in range(len(ref)))
    O0 = d["O"][0].to(torch.complex128)
    fl1, fl2 = local_fisher(phr, O0, 24), local_fisher_jac(phr, O0, 24)
    efl = float((fl1 - fl2).norm() / fl2.norm())
    del fl1, fl2
    check("真實幾何:解析的 FFT 形式的 Fisher 區塊 = 雅可比相乘(中央的掃描位置,< 1e-10)", efl < 1e-10, f"{efl:.1e}")
    t0 = time.time()
    chr_ = fisher_chol(phr, d["O"][0])
    rr = crlb_from(phr, chr_)
    tsec = time.time() - t0
    lamr = phr.lam(chr_["Of"][None])[0]
    nrr = torch.stack([torch.poisson(lamr, generator=torch.Generator(device=dev).manual_seed(SEED_MINI + 200 + k)) for k in range(16)])
    qr, nexpr = score_q(phr, chr_, nrr, 1.0)
    qrm, qrse = float(qr.mean() / nexpr), float(qr.std() / 4 / nexpr)
    fW0 = float(s8.load_dose()["seeds"]["0"]["f_W"])
    nrp = torch.stack([torch.poisson(lamr * fW0, generator=torch.Generator(device=dev).manual_seed(SEED_MINI + 300 + k)) for k in range(16)])
    qp, _ = score_q(phr, chr_, nrp, fW0)
    qpm, qpse = float(qp.mean() / nexpr), float(qp.std() / 4 / nexpr)
    del chr_
    check("Fisher 恆等式(真實幾何,一個場、各 16 次量測):E[sᵀ (d·I_F)⁻¹ s] = 可辨識的參數數(相對差 < 3%),劑量 1 與論文劑量(核對 Fisher ∝ 劑量與計數的換算)",
          abs(qrm - 1) < 0.03 and abs(qpm - 1) < 0.03, f"劑量 1:{qrm:.4f} ± {qrse:.4f};論文劑量 f_W = {fW0:.5f}:{qpm:.4f} ± {qpse:.4f}")
    inc6, inc8 = phr.ill >= TAU, phr.ill >= TAU_ALT
    r8 = rr if bool(torch.equal(inc6, inc8)) else crlb_field(phr, d["O"][0], tau=TAU_ALT, extra=False)
    etau = abs(r8["T"] / rr["T"] - 1)
    null_exact = r8["null"] if bool(inc8.all()) else float("nan")
    cr1 = rr["T"] / float(d["Ec"][0])
    D = s8.load_dose()["seeds"]["0"]
    check("真實幾何:前向模型 = scan_8.measure_L 的期望強度(< 1e-5);一個場的 CRLB(Cholesky 成功);整體相位的零空間:所有像素都是參數時(τ = 1e-8)< 1e-10、"
          "τ = 1e-6(少數照明極弱的像素固定在真值,零空間只是近似)< 1e-6;τ = 1e-6 vs 1e-8 的 T 相對差 < 1%",
          ef < 1e-5 and rr["ok"] and r8["ok"] and null_exact < 1e-10 and rr["null"] < 1e-6 and etau < 0.01,
          f"前向 {ef:.1e};參數 {rr['n_par']}(照明 ≥ τ 的像素 {int(inc6.sum())} / {phr.F ** 2};τ = 1e-8:{int(inc8.sum())});U 內 {rr['n_U']};"
          f"零空間 τ = 1e-8:{null_exact:.1e}、τ = 1e-6:{rr['null']:.1e};"
          f"τ 相對差 {etau:.1e};耗時 {tsec:.1f} 秒;CRLB_c 劑量 1 {cr1:.2e}、論文劑量(WS₂){cr1 / D['f_W']:.2e};T5 / T {rr['T5'] / rr['T']:.4f}")
    # (3) 單顆 S 原子:峰值、多一顆 S 的 SV 判成完整、D 對劑量線性
    k1 = s_kernel(d, torch.tensor([[d["geo"].F / 2.0, d["geo"].F / 2.0]], dtype=torch.float64))
    pk = float(k1.max())
    vac = d["vac"]
    sv = torch.where(vac.lab == 1)[0]
    phs = d["fields"][:, 1].double().cpu().clone()
    if len(sv):
        kk = s_kernel(d, vac.pos[sv].double()).cpu()
        for j, c_ in enumerate(sv.tolist()):
            phs[int(vac.fid[c_])] += kk[j]
        pred, ok, _ = vac.read(phs.float())
        frac = float((pred[sv] == 0).float().mean())
        Dv = d1_cols(d, phr, sv[:2], s8.ETA)
    else:
        frac, Dv = float("nan"), torch.zeros(1)
    lam0 = phr.lam(d["O"][:1])
    lin = 0.0
    if len(sv):
        i0 = int(vac.fid[sv[0]])
        Op = d["O"][i0].to(torch.complex128)[None] * torch.exp((1j - s8.ETA) * s_kernel(d, vac.pos[sv[:1]].double()))
        l0_, l1_ = phr.lam(d["O"][i0:i0 + 1]), phr.lam(Op)
        D2 = float((((2 * l1_ - 2 * l0_) ** 2) / (2 * l0_).clamp_min(1e-300) * (l0_ > 0)).sum())
        lin = abs(D2 / (2 * float(Dv[0])) - 1)
    # 相位核 = 產生器:同一個 seed 不放空缺(fixed psv = pdv = 0;draw_params 的亂數順序不變 → 幾何相同)的場 − 原本的場 = 所有空缺處少掉的 S 原子的相位核之和
    #(只比 U 內、兩者相位都 > 0.05 rad 的像素:避開「截在 ≥ 0」與場外的空缺;相對 S 原子的峰值)
    seeds4 = d["obj_seeds"][:4]
    f_int, _ = s8.make_lattice(seeds4, d["geo"].F, dev, materials=[s8.MATS[int(m)] for m in d["mats"][:4]], U=d["geo"].U.cpu(), eta=s8.ETA, dx=d["dx"],
                               fixed={"psv": 0.0, "pdv": 0.0})
    gen_diff, ker_sum, msk = [], [], []
    for i in range(4):
        k_ = torch.where(vac.fid == i)[0]
        cols_all = d["meta"]["cols"][i]
        miss_n = (2 - cols_all["nS"]).double()
        ks = torch.zeros(d["geo"].F, d["geo"].F, dtype=torch.float64, device=dev)
        kv = torch.where(miss_n > 0)[0]
        if len(kv):
            kk_ = s_kernel(d, cols_all["pos"][kv].double())
            ks = (kk_ * miss_n[kv].to(dev)[:, None, None]).sum(0)
        gen_diff.append(f_int[i, 1].double() - d["fields"][i, 1].double())
        ker_sum.append(ks)
        msk.append((f_int[i, 1] > 0.05) & (d["fields"][i, 1] > 0.05) & d["geo"].U)
        del k_
    gd, kd, mk_ = torch.stack(gen_diff), torch.stack(ker_sum), torch.stack(msk)
    egen = float(((gd - kd).abs() * mk_).max() / pk) if bool(mk_.any()) else float("nan")
    check("單顆 S 原子的相位核:峰值 ≈ S₂ 柱的一半(0.277 ± 5%);= 產生器(不放空缺的同一片晶格 − 原本 = 空缺處少掉的 S 的相位核之和;相對差 < 1e-4);"
          "真值的 SV 加上一顆 S 後判成完整(≥ 95%);D 與劑量成正比(< 1e-6)",
          abs(pk / 0.277 - 1) < 0.05 and egen < 1e-4 and (frac >= 0.95 if np.isfinite(frac) else False) and lin < 1e-6,
          f"峰值 {pk:.3f} rad;與產生器的相對差 {egen:.1e};判成完整 {frac:.1%}(SV {len(sv)} 個);D₁ 例 {float(Dv[0]):.3g};線性 {lin:.1e};lam0 {float(lam0.sum()):.3g}")
    # (4) 特徵與局部正規化
    dmap = dist_out_u(d["geo"].U)
    f = col_features(d, phr, dmap)
    cen = phr.centers[24:25]
    dc0 = float(torch.cdist(cen, phr.centers).min())
    inside_ok = bool((f["dist_U"] > 0).all())
    p = predict(d["O"].clone(), d)
    _, Cl = local_read(d, p["s"], p["ok"])
    Ct = Cl.sum(0).numpy()
    diag = bool((Ct - np.diag(np.diag(Ct))).sum() == 0)
    le = loc_pos_err(dict(d, spec=dict(d["spec"], pos=0.5)), phr)
    zz = (0.5 * d["z"].cpu().double()).norm(dim=-1)
    le_ok = bool(np.all(le >= float(zz.min()) - 1e-9) and np.all(le <= float(zz.max()) + 1e-9))
    check("特徵:掃描中心到自己的距離 = 0;可判讀柱都在 U 內(到邊界 > 0);局部位置誤差在 |pos·z| 的範圍內;局部正規化在真值上 = 對角",
          dc0 < 1e-9 and inside_ok and le_ok and diag, f"中心 {dc0:.1e};U 內 {inside_ok};局部位置誤差 {le_ok};真值 局部正規化 {Ct.astype(int).tolist()}")
    # (5) seed 不重疊
    found = s8.scan_seed_constants({"scan_8c.py"})
    clash = sorted(v for v in found if any(lo <= v <= hi for lo, hi in MY_RANGES_8C))
    mine_ = [SEED_B3, SEED_C, SEED_A2, SEED_MINI] + [x + SMOKE_OFF for x in (SEED_B3, SEED_C, SEED_A2)]
    over8 = [x for x in mine_ if any(lo <= x <= hi for lo, hi in s8.MY_RANGES)]
    hi_used = max(SEED_B3 + 10_000 * (len(DOSES) - 1) + 1000 * max(SEEDS), SEED_C + 1000 * max(SEEDS),
                  SEED_A2 + 1000 + 100 * (A2_N - 1) + (A2_R - 1)) + SMOKE_OFF
    inside = hi_used <= MY_RANGES_8C[0][1] and SEED_MINI + 316 <= MY_RANGES_8C[0][1]
    rej = []
    for x in (s8.FINAL_LO, s8.FINAL_HI):
        try:
            s8.guard([x])
            rej.append(False)
        except ValueError:
            rej.append(True)
    check("seed:8c 的區間 88,600,000–88,919,999 與既有腳本的常數、階段八的區間都不重疊;所有派生的 seed 在區間內;期末考區間仍拒絕",
          not clash and not over8 and inside and all(rej), f"重疊的常數 {clash};與階段八重疊 {over8};最大 {hi_used:,};拒絕 {rej}")
    del d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = me_md5()
    print(f"scan_8c.py md5 {me};scan_8.py md5 {s8.me_md5()};GPU {a7.gpu_name(dev)}")
    smoke = bool(a.smoke)
    ctx = ctx_of(smoke)
    if a.check:
        checks(dev, ctx)
        I = load_inputs(ctx_of(False))
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if smoke:
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)                                      # 只刪這個 smoke 專用資料夾
        SMOKE_DIR.mkdir(parents=True)
        checks(dev, ctx)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
    elif a.run:
        if not PASSED.exists() or json.load(open(PASSED)).get("script_md5") != me:
            raise SystemExit(f"❌ 找不到 {PASSED} 或 smoke 時的版本不同:先在 dev 節點用現在的版本跑 python scan_8c.py --smoke")
        print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")
    else:
        ap.print_help()
        sys.exit(2)
    I = load_inputs(ctx)
    if I is None:
        print("\n❌ 輸入不完整,停下來")
        sys.exit(1)
    if smoke:
        I["n"] = min(I["n"], 4 if QUICK else 8)
    part_path = Path(str(ctx["out"]) + ".partial")
    meta = {"md5": me, "s8_md5": S8_MD5, "smoke": smoke, "n": I["n"], "quick": QUICK}
    R, Rok = {}, {}
    if part_path.exists():
        old = json.load(open(part_path))
        if old.get("meta") == meta:
            R, Rok = old["parts"], old.get("parts_ok", {})
            print(f"  續跑:暫存檔已有 {list(R)}")
            for k, v in Rok.items():
                if not v:
                    check(f"[續跑] 部分 {k} 在先前的執行中有檢查未通過", False)
        else:
            raise SystemExit(f"❌ {part_path} 的程式版本或設定不同:不續跑。先告訴 Claude")
    log = lambda s: print(s, flush=True)                                  # noqa: E731
    t0 = time.time()
    tp = {}
    for name, fn in (("D", lambda: part_D(I)), ("A", lambda: part_A(dev, I, ctx, log)), ("B", lambda: part_B(dev, I, ctx, log)),
                     ("B3", lambda: part_B3(dev, I, ctx, log)), ("B4", lambda: part_B4(dev, I, ctx, log)), ("C", lambda: part_C(dev, I, ctx, log))):
        if name in R:
            continue
        t1 = time.time()
        print(f"\n  [{name}] 開始", flush=True)
        R[name] = json.loads(json.dumps(fn(), default=_js))
        Rok[name] = bool(all_ok())
        tp[name] = time.time() - t1
        dump_json({"meta": meta, "parts": R, "parts_ok": Rok}, part_path)
        print(f"  [{name}] 完成(經過 {tp[name]:.0f} 秒)", flush=True)
    rep = report(R, I, ctx)
    res = {"meta": meta, "parts": R, "report": json.loads(json.dumps(rep, default=_js)), "checks_ok": all_ok(), "complete": False,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0, "part_seconds": tp}
    dump_json(res, ctx["out"])
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        figures(R, I, rep, ctx["fig"])
    except Exception as ex:                                               # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("畫圖完成(判讀不受影響)", False, f"{type(ex).__name__}: {ex}")
    res["checks_ok"] = all_ok()
    res["complete"] = bool(all_ok())
    dump_json(res, ctx["out"])
    part_path.unlink(missing_ok=True)
    print(f"  結果:{ctx['out']}(總耗時 {time.time() - t0:.0f} 秒;各部分 " + "、".join(f"{k} {v:.0f}" for k, v in tp.items()) + " 秒)")
    if smoke:
        if all_ok():
            json.dump({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
            print("\n✅ 迷你全流程全部通過 → 可以送出 run_scan8c.sh")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    print("\n" + ("✅ 內建檢查全部通過" if all_ok() else "❌ 有檢查未通過"))
    print("把完整輸出與 figs_scan8c/ 的圖傳給 Claude")


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(str(type(o)))


if __name__ == "__main__":
    main()
