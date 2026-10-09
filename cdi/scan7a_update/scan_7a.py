#!/usr/bin/env python
"""階段七 7a-1:貼近真實條件的壓力測試(第一層誤差)(階段七協定)。不重新訓練,只用期中考凍結的 P6B4。

問:加入真實條件的誤差後,網路(G9 主力、G25 參考)和整張的迭代法誰退步得多?
條件(8 個;§2.2):ideal(= 階段六)、posL / posH(位置誤差 σ 0.5 / 1.0 px)、prbL / prbH(離焦 ± 10% / 30%)、
      doseL / doseH(劑量 ÷ 10 / ÷ 100)、combo(posL + prbL + doseL,主判)。
量測來自「真的」條件,重建只知道「名目」條件(位置、探針);劑量已知(計數除以劑量比)。
對手:整張的迭代法包絡 = 平常版(4 演算法 × 6 起點 + K-HIO,同階段六)
      + 修正版(ePIE-C、AP-C × {位置、位置 + 退火搜尋、探針、兩者、兩者 + 退火搜尋} × 6 起點)。
主指標 nerr_sr:U 上對齊一個整體相位 + 整體平移(|t| ≤ 2 px)+ 線性相位斜坡(探針修正的歧異)。

用法(需在計算節點執行):
    python scan_7a.py --check          # 只跑內建檢查
    python scan_7a.py --smoke          # 迷你全流程(少量場、停止點 ≤ 20;dev 節點);通過才可正式執行
    python scan_7a.py --cond posL      # 一個條件(run_scan7a_cond.sh 的 array job)
    python scan_7a.py --final          # 彙整:計時、判讀、圖(run_scan7a_final.sh,等 8 個條件都完成)
"""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_6a as a6                                                 # noqa: E402  (階段六;不修改)

h, g = a6.h, a6.g
s5, s5b, s5c, s5d, s5e = a6.s5, a6.s5b, a6.s5c, a6.s5d, a6.s5e
RUN_ROOT = a6.RUN_ROOT
QUICK = a6.QUICK
PROBE = a6.PROBE
SEEDS = a6.SEEDS
GROUP = a6.GROUP
FIELD_SEED = a6.FIELD_SEED                   # 80,000,000(驗證場)
NOISE_BASE = a6.NOISE_BASE                   # 82,000,000(= 階段六)
POS_BASE = 83_000_000                        # 位置誤差的標準常態亂數
DF_BASE = 84_000_000                         # 離焦誤差的符號
CONDS = ["ideal", "posL", "posH", "prbL", "prbH", "doseL", "doseH", "combo"]
SPEC = {"ideal": dict(pos=0.0, df=0.0, dose=1.0),
        "posL": dict(pos=0.5, df=0.0, dose=1.0), "posH": dict(pos=1.0, df=0.0, dose=1.0),
        "prbL": dict(pos=0.0, df=0.10, dose=1.0), "prbH": dict(pos=0.0, df=0.30, dose=1.0),
        "doseL": dict(pos=0.0, df=0.0, dose=0.1), "doseH": dict(pos=0.0, df=0.0, dose=0.01),
        "combo": dict(pos=0.5, df=0.10, dose=0.1)}
MAIN_COND = "combo"
Z_CLIP = 3.0                                 # 位置誤差截在 ± 3σ
PIPES = ["G9", "G25"]
PRIMARY = "G9"
MAIN = "nerr_sr"                            # 主指標:整體相位 + 平移 + 線性相位斜坡
BASE_METHODS = list(s5b.METHODS)             # 平常版:ePIE-C、ePIE、AP-C、AP
CORR_BASE = ["ePIE-C", "AP-C"]
MODES = ["pos", "posA", "prb", "both", "bothA"]    # A = 另加退火式的位置搜尋(大誤差用)
CORR_METHODS = [f"{m}:{x}" for m in CORR_BASE for x in MODES]
METHODS7 = BASE_METHODS + CORR_METHODS
START = 5                                    # 修正在第 5 次掃描之後開始
POS_BETA = 2.0                               # 位置的 Gauss–Newton 步長倍數(線性化低估位移;本地小測試選定,執行前)
POS_STEP = 1.0                               # 每次掃描最多移 1 px(梯度步)
POS_MAX = 6.0                                # 總量上限 6 px
ANNEAL = 20                                  # A 模式:修正開始後的前 20 次掃描做退火式搜尋(cf. Maiden et al. 2012)
RHO0, RHO_MIN = 3.0, 0.3                     # 搜尋半徑由 3 px 線性縮到 0.3 px
DIRS = [(math.sin(a), math.cos(a)) for a in np.arange(8) * math.pi / 4]   # 8 個方向
SH_MAX, SH_STEP = 2.0, 0.05                  # 平移對齊:|t_y|、|t_x| ≤ 2 px、0.05 px 格點
RAMP_MAX, RAMP_STEP = 0.002, 0.0001          # 相位斜坡:|q| ≤ 0.002 週期 / px(≈ 0.0126 rad/px)、0.0001 格點
FAIL_VAL = 4.0                               # 發散的樣本記為 4.0(高於先前觀察到的最大誤差 3.8)
MCH = 128                                    # 指標一次處理的場數(記憶體)
ITERS = list(s5b.ITERS)
SMOKE_ITERS = [i for i in s5b.ITERS if i <= 20]
KHIO_N = dict(s5b.KHIO_N)
PS_TOL, REPRO_TOL = 1e-6, 1e-5
SUF = "_quick" if QUICK else ""
SMOKE_FIELDS = 4 if QUICK else 16
SMOKE_DIR = RUN_ROOT / f"scan7a_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
S6_ENV = RUN_ROOT / f"scan6a_env{SUF}.json"
S6_OUT = RUN_ROOT / f"scan6a{SUF}.json"
FIG_DIR = Path("figs_scan7a")
RAND_SEED = 2027
MARGIN = 4
AMP_MIN = 0.05
COL = {"G9": "#1baf7a", "G25": "#2a78d6", "iter_b64": "#9a9994", "iter_b512": "#0b0b0b", "plain": "#d6452a"}
CLABEL = {"ideal": "ideal", "posL": "position sigma 0.5px", "posH": "position sigma 1.0px", "prbL": "defocus ±10%",
          "prbH": "defocus ±30%", "doseL": "dose ÷10", "doseH": "dose ÷100", "combo": "all three (light)"}

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and a6.all_ok()


def md5(path):
    m = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


def absdiff(a, b):
    d = abs(float(a) - float(b))
    return d if np.isfinite(d) else float("inf")


def cond_json(c, root=None):
    return (root or RUN_ROOT) / f"scan7a_cond_{c}{SUF}.json"


def dump_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, default=float)
    os.replace(tmp, path)


def gpu_name(dev):
    return torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"


# ============================================================================
# 探針的平移、真探針、孔徑投影
# ============================================================================
class Ops:
    """頻率空間的工具(視窗 W × W):平移(可分離的相位斜坡)、導數、已知孔徑的投影 + 總能量固定。"""

    def __init__(self, pr, W, dev):
        k = torch.fft.fftfreq(W, device=dev)                                      # 週期 / px
        self.k1 = k
        ky, kx = torch.meshgrid(k, k, indexing="ij")
        self.ky, self.kx = ky, kx
        self.Dy = (2j * math.pi * ky).to(torch.complex64)
        self.Dx = (2j * math.pi * kx).to(torch.complex64)
        kp = torch.fft.fftfreq(W, d=1.0 / W, dtype=torch.float64)                 # 頻率 px(同 aperture_probe)
        kyp, kxp = torch.meshgrid(kp, kp, indexing="ij")
        self.ka = float(pr["meta"]["ka"])
        self.aper = (torch.sqrt(kxp ** 2 + kyp ** 2) <= self.ka).to(torch.complex64).to(dev)
        self.E0 = float((pr["P"].abs() ** 2).sum())
        self.P0f = torch.fft.fft2(pr["P"])                                        # 名目探針的頻譜(只算一次)
        gy, gx = self.grad_f(self.P0f)
        self.S0 = float((gy.abs() ** 2 + gx.abs() ** 2).sum())                    # 位置步的正則化尺度(只看探針)
        self.W = W

    def ramp(self, dy, dx):
        """exp(−2πi(k_y dy + k_x dx)):乘在頻譜上 = 內容往 +(dy, dx) 移動。dy, dx 為任意形狀的張量(可分離,省記憶體與時間)。"""
        one = torch.ones_like(self.k1)
        ey = torch.polar(one.expand(*dy.shape, -1), -2 * math.pi * self.k1 * dy[..., None])
        ex = torch.polar(one.expand(*dx.shape, -1), -2 * math.pi * self.k1 * dx[..., None])
        return ey[..., :, None] * ex[..., None, :]

    def shift(self, x, dy, dx):
        return torch.fft.ifft2(torch.fft.fft2(x) * self.ramp(dy, dx))

    def grad_f(self, Xf):
        """由頻譜直接得 ∂/∂y、∂/∂x(對頻寬有限的探針是精確的)。"""
        return torch.fft.ifft2(Xf * self.Dy), torch.fft.ifft2(Xf * self.Dx)

    def grad(self, x):
        return self.grad_f(torch.fft.fft2(x))

    def project(self, P):
        """投影到已知孔徑(頻率空間孔徑外為 0),再把總能量固定為名目探針的(束流已知)。P:[..., W, W]。"""
        P = torch.fft.ifft2(torch.fft.fft2(P) * self.aper)
        e = (P.abs() ** 2).sum((-2, -1), keepdim=True).clamp_min(1e-30)
        return P * torch.sqrt(self.E0 / e)


def true_probe_pair(pr, level, dev):
    """離焦 = 名目 × (1 ± level) 的真探針(同一孔徑、總能量 = 名目)。回傳 (P+, P−) complex64 [W, W]。"""
    import probe_4_2b as pb
    W = pr["P"].shape[-1]
    ka = float(pr["meta"]["ka"])
    d0 = float(pr["meta"]["defocus"]) * math.pi
    E0 = float((pr["P64"].abs() ** 2).sum())
    out = []
    for sgn in (1.0, -1.0):
        p = pb.aperture_probe(W, ka, defocus=d0 * (1.0 + sgn * level))
        p = p * math.sqrt(E0 / float((p.abs() ** 2).sum()))
        out.append(p.to(torch.complex64).to(dev))
    return out[0], out[1]


def draw_errors(s, n, dev):
    """位置誤差的標準常態亂數 z [n, 49, 2](截在 ± 3)與離焦符號 [n](± 1);同 seed 的條件共用 → 成對比較。"""
    gz = torch.Generator().manual_seed(POS_BASE + 1000 * s)
    z = torch.randn(n, a6.NPOS * a6.NPOS, 2, generator=gz).clamp(-Z_CLIP, Z_CLIP)
    gd = torch.Generator().manual_seed(DF_BASE + 1000 * s)
    sign = torch.where(torch.rand(n, generator=gd) < 0.5, -1.0, 1.0)
    return z.to(dev), sign.to(dev)


def is_ideal(spec):
    return spec["pos"] == 0 and spec["df"] == 0 and spec["dose"] == 1


@torch.no_grad()
def measure_cond(fields, geo, cp, bs, seed, spec, z, sign, ops, force_generic=False):
    """49 張繞射圖(真的條件)。理想條件直接用階段六的 measure49(逐位元相同);
    其他條件:探針放在名目位置 + δ(頻率空間平移)、離焦依符號、計數 = Poisson(劑量比 × 期望)/ 劑量比。"""
    if is_ideal(spec) and not force_generic:
        return a6.measure49(fields, geo, cp, bs, seed)
    from src.physics import forward_measure
    W = geo.W
    P = geo.pr["P"]
    if spec["df"] > 0:
        pp, pm = true_probe_pair(geo.pr, spec["df"], fields.device)
        Pt = torch.where(sign[:, None, None] > 0, pp, pm)                          # [n, W, W]
    else:
        Pt = P
    dose = float(spec["dose"])
    cpd = cp
    if dose != 1.0:
        cpd = copy.deepcopy(cp)
        cpd.photons_per_pix = cp.photons_per_pix * dose
    torch.manual_seed(seed)                                                        # 同 s5.measure_scan
    counts = []
    for j, (y0, x0) in enumerate(geo.starts):
        win = s5.crop(fields, y0, x0, W)
        if spec["pos"] > 0:
            Pj = ops.shift(Pt if Pt.dim() == 3 else Pt.expand(fields.shape[0], W, W),
                           spec["pos"] * z[:, j, 0], spec["pos"] * z[:, j, 1])
        else:
            Pj = Pt
        psi = Pj * torch.polar(win[:, 0], win[:, 1])
        c = forward_measure(torch.stack([psi.abs(), torch.angle(psi)], dim=1), bs, cpd)
        counts.append(c / dose if dose != 1.0 else c)
    return counts


def setup_cond(s, dev, cond, n=None, force_generic=False):
    """同 scan_6a.setup,但量測來自條件 cond。重建用的 pr / cp / geo 都是名目的。"""
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[PROBE]
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS["3x3s8"][2])
    bs = beamstop_mask(cfg, device=dev)
    geo = a6.Geom(cfg, pr, dev)
    n = n or (s5b.N_FIELDS or cfg.eval_n)
    sd = a6.seeds_of(s)
    fields = s5.make_fields(cfg, n, seed=sd["field"], device=dev)
    O = torch.polar(fields[:, 0], fields[:, 1])
    z, sign = draw_errors(s, n, dev)
    ops = Ops(pr, geo.W, dev)
    spec = SPEC[cond]
    counts = measure_cond(fields, geo, cp, bs, sd["noise"], spec, z, sign, ops, force_generic)
    return dict(cfg=cfg, pr=pr, cp=cp, bs=bs, geo=geo, O=O, counts=counts, init=sd["init"], n=n, fields=fields,
                z=z, sign=sign, ops=ops, spec=spec, cond=cond)


# ============================================================================
# 迭代法的修正版(位置修正、探針修正)
# ============================================================================
def _fourier(psi, meas, bs):
    """傅立葉約束(同 scan_5b.ap / scan_5.epie:beamstop 內保留模型的振幅)。"""
    E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    new_mag = torch.where(bs > 0, meas, E.abs())
    return torch.fft.ifft2(torch.fft.ifftshift(torch.polar(new_mag, torch.angle(E)), dim=(-2, -1)), norm="ortho")


def _pos_step(Pj, Oj, dd, ops, Sf=None):
    """梯度式位置修正:ψ(r_j + Δ) ≈ ψ − Δ·(∇P_j)·O_j,對 ψ′ 做最小平方(2 × 2 的 Gauss–Newton),乘 POS_BETA。
    Sf:P_j 的頻譜(有的話直接用,省一次 FFT)。正則化只依探針(每個場、每個位置各自獨立,與 batch 無關)。
    回傳 [..., 2](px,已截在 ± POS_STEP)。"""
    dPy, dPx = ops.grad_f(Sf) if Sf is not None else ops.grad(Pj)
    gy, gx = dPy * Oj, dPx * Oj
    reg = 1e-4 * ops.S0
    a = (gy.abs() ** 2).sum((-2, -1)) + reg
    c = (gx.abs() ** 2).sum((-2, -1)) + reg
    b = (gy.conj() * gx).real.sum((-2, -1))
    uy = -(gy.conj() * dd).real.sum((-2, -1))
    ux = -(gx.conj() * dd).real.sum((-2, -1))
    det = a * c - b * b
    step = torch.stack([(c * uy - b * ux) / det, (a * ux - b * uy) / det], -1)
    return (POS_BETA * step).clamp(-POS_STEP, POS_STEP)


def _anneal(Bf, Oj, meas, bs, r, rho, ops):
    """退火式搜尋:目前位置與 8 個方向、距離 rho 的候選位置中,取傅立葉振幅殘差(beamstop 外)最小者。
    Bf:探針的頻譜(可廣播到 Oj 的形狀);r:[..., 2]。回傳新的 r。"""
    m = bs > 0
    best, pick = None, None
    for cy, cx in [(0.0, 0.0)] + DIRS:
        ry, rx = r[..., 0] + rho * cy, r[..., 1] + rho * cx
        Pc = torch.fft.ifft2(Bf * ops.ramp(ry, rx))
        E = torch.fft.fftshift(torch.fft.fft2(Pc * Oj, norm="ortho"), dim=(-2, -1)).abs()
        res = (((E - meas) ** 2) * m).sum((-2, -1))
        cand = torch.stack([ry, rx], -1)
        if best is None:
            best, pick = res, cand
        else:
            better = res < best
            best = torch.where(better, res, best)
            pick = torch.where(better[..., None], cand, pick)
    return pick


def _project_obj(O, phmax):
    return torch.polar(O.abs().clamp(max=1.0), torch.angle(O).clamp(0, phmax))


@torch.no_grad()
def ap_corr(counts, bs, cp, pr, starts, iters, constrain, O0, mode, ops, start=START, state=False):
    """AP(同 scan_5b.ap)+ 修正。mode:"pos"(位置)、"posA"(位置 + 退火搜尋)、"prb"(探針)、"both"、"bothA"。
    修正在第 start 次掃描之後開始(A:其後 ANNEAL 次掃描先做退火搜尋);
    修正開始前與 scan_5b.ap 逐位元相同(內建檢查)。每個場各自估計自己的探針與位置。"""
    from src.hio import _measured_amp
    W = cp.canvas
    dev = O0.device
    meas = torch.stack([_measured_amp(c, cp).to(dev) for c in counts], 1)             # [N, J, W, W]
    N, J = O0.shape[0], len(starts)
    use_pos, use_prb, use_ann = mode in ("pos", "posA", "both", "bothA"), mode in ("prb", "both", "bothA"), mode.endswith("A")
    P0, Pc0 = pr["P"], pr["P"].conj()
    wsum = s5b.illum_sum(cp, pr, starts, dev)
    delta = s5b.AP_DELTA * float(wsum.max())
    den0 = wsum + delta
    phmax = cp.phase_max
    O = O0.clone()
    P = None                                                                          # 各場的探針(探針修正開始後)
    r = torch.zeros(N, J, 2, device=dev)
    moved = False
    want, outs = set(iters), {}
    for k in range(max(iters)):
        pos_on, prb_on = use_pos and k >= start, use_prb and k >= start
        ann_on = use_ann and start <= k < start + ANNEAL
        Oj = torch.stack([O[:, y0:y0 + W, x0:x0 + W] for y0, x0 in starts], 1)
        Bf = ops.P0f if P is None else torch.fft.fft2(P)[:, None]                     # 探針頻譜 [W, W] 或 [N, 1, W, W]
        if ann_on:                                                                    # 退火式搜尋(先於本次的更新)
            rho = RHO0 + (RHO_MIN - RHO0) * (k - start) / max(1, ANNEAL - 1)
            r = _anneal(Bf, Oj, meas, bs, r, rho, ops)
            r = (r - r.mean(1, keepdim=True)).clamp(-POS_MAX, POS_MAX)
            moved = True
        plain = P is None and not moved
        Sf = None
        if plain:
            Pj = P0
        elif moved:
            Sf = Bf * ops.ramp(r[..., 0], r[..., 1])                                  # [N, J, W, W]
            Pj = torch.fft.ifft2(Sf)
        else:
            Pj = P[:, None]
        psi = Pj * Oj
        psi2 = _fourier(psi, meas, bs)
        if plain and not pos_on and not prb_on:                                       # = scan_5b.ap
            gg = Pc0 * psi2
            num = delta * O
            for j, (y0, x0) in enumerate(starts):
                num[:, y0:y0 + W, x0:x0 + W] += gg[:, j]
            O = num / den0
        else:
            Pjj = Pj.expand(N, J, W, W)
            gg = Pjj.conj() * psi2
            wj = Pjj.abs() ** 2
            num = delta * O
            den = torch.full_like(O.real, delta)
            for j, (y0, x0) in enumerate(starts):
                num[:, y0:y0 + W, x0:x0 + W] += gg[:, j]
                den[:, y0:y0 + W, x0:x0 + W] += wj[:, j]
            Onew = num / den
            if prb_on:                                                                # 以新的物體做探針的最小平方
                On_j = torch.stack([Onew[:, y0:y0 + W, x0:x0 + W] for y0, x0 in starts], 1)
                q2 = psi2
                if moved:                                                             # 移到探針座標(平移 −r_j)
                    On_j = ops.shift(On_j, -r[..., 0], -r[..., 1])
                    q2 = ops.shift(psi2, -r[..., 0], -r[..., 1])
                pn = (On_j.conj() * q2).sum(1)
                pd = (On_j.abs() ** 2).sum(1)
                P = ops.project(pn / (pd + 1e-3 * pd.amax((-2, -1), keepdim=True) + 1e-12))
            if pos_on:
                dr = _pos_step(Pjj, Oj, psi2 - psi, ops, Sf if Sf is not None else Bf)
                r = (r + dr) - (r + dr).mean(1, keepdim=True)
                r = r.clamp(-POS_MAX, POS_MAX)
                moved = True
            O = Onew
        if constrain:
            O = _project_obj(O, phmax)
        if (k + 1) in want:
            outs[k + 1] = O.clone()
    if state:
        return outs, {"r": r, "P": P}
    return outs


@torch.no_grad()
def epie_corr(counts, bs, cp, pr, starts, iters, constrain, O0, seed, mode, ops, start=START, state=False):
    """ePIE(同 scan_5.epie,位置順序同一個亂數)+ 修正:位置(梯度式;A:另加退火搜尋)、探針(ePIE 的探針更新,
    Maiden & Rodenburg 2009)。每次掃描後探針投影到已知孔徑並固定總能量、位置減去平均。修正開始前與 scan_5.epie 逐位元相同。"""
    from src.hio import _measured_amp
    W = cp.canvas
    dev = O0.device
    meas = [_measured_amp(c, cp).to(dev) for c in counts]
    N, J = O0.shape[0], len(starts)
    use_pos, use_prb, use_ann = mode in ("pos", "posA", "both", "bothA"), mode in ("prb", "both", "bothA"), mode.endswith("A")
    P0 = pr["P"]
    Pc0 = P0.conj() / (pr["Pa"] ** 2).max()
    phmax = cp.phase_max
    O = O0.clone()
    P = None
    r = torch.zeros(N, J, 2, device=dev)
    moved = False
    gen = torch.Generator().manual_seed(seed)
    want, outs = set(iters), {}
    for k in range(max(iters)):
        pos_on, prb_on = use_pos and k >= start, use_prb and k >= start
        ann_on = use_ann and start <= k < start + ANNEAL
        rho = RHO0 + (RHO_MIN - RHO0) * (k - start) / max(1, ANNEAL - 1)
        for j in torch.randperm(J, generator=gen).tolist():
            y0, x0 = starts[j]
            Oj = O[:, y0:y0 + W, x0:x0 + W]
            corr = P is not None or moved or ann_on
            if corr:
                Bf = ops.P0f if P is None else torch.fft.fft2(P)
                if ann_on:
                    r[:, j] = _anneal(Bf, Oj, meas[j], bs, r[:, j], rho, ops)
                    moved = True
            Sf = None
            if P is None and not moved:
                Pj, Pc = P0, Pc0
            else:
                if moved:
                    Sf = Bf * ops.ramp(r[:, j, 0], r[:, j, 1])
                    Pj = torch.fft.ifft2(Sf)
                else:
                    Pj = P
                Pc = Pj.conj() / (Pj.abs() ** 2).amax((-2, -1), keepdim=True).clamp_min(1e-12)
            psi = Pj * Oj
            psi2 = _fourier(psi, meas[j], bs)
            On = Oj + Pc * (psi2 - psi)
            if prb_on or pos_on:
                dd = psi2 - psi
                Pjj = Pj if Pj.dim() == 3 else Pj.expand(N, W, W)
                if prb_on:
                    Pn = Pjj + Oj.conj() / (Oj.abs() ** 2).amax((-2, -1), keepdim=True).clamp_min(1e-12) * dd
                    P = ops.shift(Pn, -r[:, j, 0], -r[:, j, 1]) if moved else Pn
                if pos_on:
                    if Sf is None:
                        Sf = ops.P0f if Pj is P0 else torch.fft.fft2(Pjj)            # 本步用的(舊)探針的頻譜
                    r[:, j] = r[:, j] + _pos_step(Pjj, Oj, dd, ops, Sf)
                    moved = True
            if constrain:
                On = torch.polar(On.abs().clamp(max=1.0), torch.angle(On).clamp(0, phmax))
            O[:, y0:y0 + W, x0:x0 + W] = On
        if prb_on:
            P = ops.project(P)
        if pos_on:
            r = (r - r.mean(1, keepdim=True)).clamp(-POS_MAX, POS_MAX)
        if (k + 1) in want:
            outs[k + 1] = O.clone()
    if state:
        return outs, {"r": r, "P": P}
    return outs


def run_method7(meth, d, iters, O0, start=START, state=False):
    """一個迭代法設定。state=True 時修正版另回傳最後的位置與探針估計(平常版回傳 None)。"""
    base, _, mode = meth.partition(":")
    geo = d["geo"]
    if not mode:
        outs = s5b.run_method(base, d["counts"], d["bs"], d["cp"], geo.pr, geo.starts, iters, O0, d["init"])
        return (outs, None) if state else outs
    constrain = base.endswith("-C")
    if base.startswith("ePIE"):
        return epie_corr(d["counts"], d["bs"], d["cp"], geo.pr, geo.starts, iters, constrain, O0, d["init"], mode,
                         d["ops"], start, state)
    return ap_corr(d["counts"], d["bs"], d["cp"], geo.pr, geo.starts, iters, constrain, O0, mode, d["ops"], start, state)


# ============================================================================
# 指標:整體相位(nerr_ph)、+ 平移(nerr_sh)、+ 相位斜坡(nerr_sr,主)
# ============================================================================
_SH = {}


def _sh_mats(F, dev):
    key = (F, str(dev))
    if key not in _SH:
        grid = torch.arange(-SH_MAX, SH_MAX + 1e-9, SH_STEP, dtype=torch.float64)
        k = torch.fft.fftfreq(F, dtype=torch.float64)
        Ey = torch.exp(2j * math.pi * grid[:, None] * k[None, :]).to(torch.complex64).to(dev)       # [S, F]
        kf = torch.fft.fftfreq(F, device=dev)
        ky, kx = torch.meshgrid(kf, kf, indexing="ij")
        qg = torch.arange(-RAMP_MAX, RAMP_MAX + 1e-12, RAMP_STEP, dtype=torch.float64)
        yc = torch.arange(F, dtype=torch.float64) - (F - 1) / 2
        Qy = torch.exp(-2j * math.pi * qg[:, None] * yc[None, :]).to(torch.complex64).to(dev)      # [Sq, F]
        _SH[key] = (grid.to(torch.float32).to(dev), Ey, ky, kx, qg.to(torch.float32).to(dev), Qy,
                    yc.to(torch.float32).to(dev))
    return _SH[key]


def _ramp_field(q, yc):
    """exp(−2πi(q_y (y − c) + q_x (x − c))),q:[n, 2](週期 / px)。"""
    ey = torch.polar(torch.ones_like(yc).expand(q.shape[0], -1), -2 * math.pi * q[:, 0, None] * yc[None])
    ex = torch.polar(torch.ones_like(yc).expand(q.shape[0], -1), -2 * math.pi * q[:, 1, None] * yc[None])
    return ey[:, :, None] * ex[:, None, :]


@torch.no_grad()
def align_err(est, O, U):
    """每個樣本的三種正規化誤差(U 上,皆含整體相位對齊):
      v_ph:只對齊整體相位(= 階段六的 nerr_ph);
      v_sh:再對齊整體平移 t(|t_y|、|t_x| ≤ 2 px、0.05 px 格點);t 由 C(t) = Σ_r conj(O·U)(r)·est(r + t) 的最大值決定,
            誤差在 t 處重算(頻率空間平移),與 t = 0 取小者;
      v_sr(主):再對齊線性相位斜坡 q(|q_y|、|q_x| ≤ RAMP_MAX 週期 / px);q 由 |Σ conj(O·U)·est_t·e^{−2πi q·(r − c)}| 的最大值決定,
            與不加斜坡取小者。→ v_sr ≤ v_sh ≤ v_ph。回傳 (v_ph, v_sh, v_sr, t [N, 2], q [N, 2])。"""
    F = O.shape[-1]
    grid, Ey, ky, kx, qg, Qy, yc = _sh_mats(F, O.device)
    m = U.to(O.real.dtype)
    out = [[], [], [], [], []]
    S, Sq = grid.numel(), qg.numel()
    for i in range(0, O.shape[0], MCH):
        e, o = est[i:i + MCH], O[i:i + MCH]
        A = torch.fft.fft2(o * m)
        B = torch.fft.fft2(e)
        C = (Ey @ (A.conj() * B) @ Ey.T).abs()                                    # [n, S, S]
        idx = C.flatten(1).argmax(1)
        t = torch.stack([grid[idx // S], grid[idx % S]], -1)                      # [n, 2](y, x)
        ph = 2 * math.pi * (ky * t[:, 0, None, None] + kx * t[:, 1, None, None])
        es = torch.fft.ifft2(B * torch.polar(torch.ones_like(ph), ph))             # est(r + t)
        v0 = a6.per_sample(e, o, U)
        vs = a6.per_sample(es, o, U)
        b1 = vs < v0
        e1 = torch.where(b1[:, None, None], es, e)
        v1 = torch.where(b1, vs, v0)
        t1 = torch.where(b1[:, None], t, torch.zeros_like(t))
        a = (o * m).conj() * e1
        Cq = (Qy @ a @ Qy.T).abs()                                                # [n, Sq, Sq]
        iq = Cq.flatten(1).argmax(1)
        q = torch.stack([qg[iq // Sq], qg[iq % Sq]], -1)
        vr = a6.per_sample(e1 * _ramp_field(q, yc), o, U)
        b2 = vr < v1
        for lst, v in zip(out, (v0, v1, torch.where(b2, vr, v1), t1, torch.where(b2[:, None], q, torch.zeros_like(q)))):
            lst.append(v)
    return tuple(torch.cat(x) for x in out)


def shift_err(est, O, U):
    """(相容用)只看平移對齊:回傳 (v_sh, t)。"""
    _, vs, _, t, _ = align_err(est, O, U)
    return vs, t


def metrics7(est, O, U, keep_ps=False):
    """三種誤差的平均(只排除「真值在 U 內是空的」樣本);估計發散(NaN / inf)的樣本記為 FAIL_VAL(算失敗,不丟掉)。"""
    mU = U.to(O.real.dtype)
    valid = ((O.abs() ** 2) * mU).sum((1, 2)) > 1e-12
    v0, vs, vr, t, q = align_err(est, O, U)
    bad = valid & ~torch.isfinite(v0)
    fix = [torch.where(valid & ~torch.isfinite(v), torch.full_like(v, FAIL_VAL), v) for v in (v0, vs, vr)]
    v0, vs, vr = fix
    m = {"nerr_ph": float(v0[valid].mean()), "nerr_sh": float(vs[valid].mean()), "nerr_sr": float(vr[valid].mean()),
         "shift": float(t[valid].norm(dim=1).mean()), "ramp": float((2 * math.pi * q[valid]).norm(dim=1).mean()),
         "n_fail": int(bad.sum())}
    if keep_ps:
        for k, v in (("ps_ph", v0), ("ps_sh", vs), ("ps_sr", vr), ("ps_t", t), ("ps_q", q)):
            m[k] = v.cpu().numpy().tolist()
    return m


MKEYS = ("nerr_sr", "nerr_sh", "nerr_ph", "shift", "ramp", "n_fail")


# ============================================================================
# 一個條件的量測:網路 + 整張的迭代法(平常版 + 修正版)
# ============================================================================
@torch.no_grad()
def khio_windows7(d, dev, kit):
    """= scan_6a.khio_windows,但次數清單由參數給(smoke 用較少的停止點;K-HIO 在第 k 次的結果與最多跑幾次無關)。"""
    import probe_4_2b as pb
    geo, cp, bs, counts = d["geo"], d["cp"], d["bs"], d["counts"]
    W = geo.W
    ones = torch.ones(W, W, device=dev)
    B = counts[0].shape[0]
    res = {k: torch.zeros(len(geo.starts), B, W, W, dtype=torch.complex64, device=dev) for k in kit}
    P = geo.pr["P"]
    Psafe = torch.where(geo.fp, P, torch.ones_like(P))
    step = 7
    for j0 in range(0, len(geo.starts), step):
        js = list(range(j0, min(j0 + step, len(geo.starts))))
        init = torch.cat([s5.k_init(counts[j], cp, d["init"] + 100 + j, dev) for j in js])
        cc = torch.cat([counts[j] for j in js])
        outs = pb.hio_probe(init, cc, bs, cp, kit, "K", geo.pr, ones)
        for k in kit:
            psi = torch.polar(outs[k][:, 0], outs[k][:, 1]).reshape(len(js), B, W, W)
            res[k][js] = torch.where(geo.fp, psi / Psafe, torch.zeros_like(psi))
        del outs
    return res


def configs7(subset="all"):
    meths = BASE_METHODS if subset == "plain" else METHODS7
    return [f"{m}|{i}" for m in meths for i in s5b.INITS] + ["K-HIO"]


def stops7(conf, iters):
    return list(iters) if conf == "K-HIO" else [0] + list(iters)


def true_probes(d):
    """各場的真探針 [n, W, W](理想 / 無離焦誤差時 = 名目)。"""
    P = d["geo"].pr["P"]
    n = d["n"]
    if d["spec"]["df"] > 0:
        pp, pm = true_probe_pair(d["geo"].pr, d["spec"]["df"], P.device)
        return torch.where(d["sign"][:, None, None] > 0, pp, pm)
    return P.expand(n, *P.shape)


def probe_err(Pe, Pt):
    """探針的相對誤差(對齊整體相位)[n]。Pe:[W, W] 或 [n, W, W]。"""
    Pe = Pe.expand_as(Pt)
    c = (Pe * Pt.conj()).sum((-2, -1))
    al = Pe * torch.exp(-1j * torch.angle(c))[:, None, None]
    return ((al - Pt).abs() ** 2).sum((-2, -1)) / (Pt.abs() ** 2).sum((-2, -1))


@torch.no_grad()
def measure_condition(dev, cond, n=None, iters=None, log=print, partial=None):
    """3 seeds:網路(G9、G25)與所有迭代法設定在各停止點的誤差;修正版另記下最後(最多次數)的位置 / 探針誤差;
    條件的統計(內建檢查 1 的後半)。partial:每個 seed 完成後存的暫存檔(job 中斷時續跑;各 seed 的亂數各自固定,與先後無關)。"""
    iters = iters or ITERS
    kit = sorted(set(iters) | set(KHIO_N.values()))
    me = md5(Path(__file__).resolve())
    meta = {"cond": cond, "n": n, "iters": list(iters), "md5": me}
    env = {c: {str(it): {q: [] for q in MKEYS} for it in stops7(c, iters)} for c in configs7()}
    for c in configs7():
        if ":" in c:
            env[c]["diag"] = {"pos_rms": [], "prb_err": []}
    nets = {p: {q: [] for q in MKEYS + ("ps_ph", "ps_sh", "ps_sr", "ps_t", "ps_q")} for p in PIPES}
    stats, done = [], []
    if partial is not None and partial.exists():
        old = json.load(open(partial))
        if old.get("meta") == meta:
            env, nets, stats, done = old["env"], old["nets"], old["stats"], old["done"]
            log(f"  續跑:暫存檔已有 seed {done}({partial.name})")
    t0 = time.time()
    last = max(iters)
    for s in SEEDS:
        if s in done:
            continue
        d = setup_cond(s, dev, cond, n)
        geo, O, U = d["geo"], d["O"], d["geo"].U
        spec = d["spec"]
        Pt = true_probes(d)
        want = spec["pos"] * d["z"]
        want = want - want.mean(1, keepdim=True)                                      # 位置修正的目標(去掉平均)
        # 條件的統計
        st = {"pos_std_px": float((spec["pos"] * d["z"]).std()) if spec["pos"] > 0 else 0.0,
              "pos_max_px": float((spec["pos"] * d["z"]).abs().max()) if spec["pos"] > 0 else 0.0,
              "df_plus_frac": float((d["sign"] > 0).float().mean()), "n": d["n"],
              "pos_rms0": float(want.norm(dim=-1).pow(2).mean(1).sqrt().mean()),
              "prb_err0": float(probe_err(geo.pr["P"], Pt).mean())}
        if not is_ideal(spec):
            ideal = a6.measure49(d["fields"], geo, d["cp"], d["bs"], a6.seeds_of(s)["noise"])
            mc = float(torch.stack([c.sum((-2, -1)) for c in d["counts"]]).mean())
            mi = float(torch.stack([c.sum((-2, -1)) for c in ideal]).mean())
            st["raw_count_ratio"] = mc * spec["dose"] / mi                             # 原始計數 / 理想(應 = 劑量比)
            st["counts_differ"] = bool(max(float((a - b).abs().max()) for a, b in zip(d["counts"], ideal)) > 0)
            del ideal
        st["mean_counts"] = float(torch.stack([c.sum((-2, -1)) for c in d["counts"]]).mean())
        stats.append(st)
        # 網路
        net, norm, _ = a6.load_p6b4(s, d["cfg"], geo.pr, dev)
        gs = a6.GlobalStages(net, geo.starts, geo.F, geo.pr, d["bs"], dev)
        for pipe in PIPES:
            m = metrics7(a6.run_pipe(pipe, net, norm, d, gs), O, U, keep_ps=True)
            for k in nets[pipe]:
                nets[pipe][k].append(m[k])
        del net, gs
        # 迭代法
        win = khio_windows7(d, dev, kit)
        kw = {}
        for k, wv in win.items():
            Of, cover, _ = a6.merge_windows(wv, geo)
            kw[k] = (Of, cover)
            if k in iters:
                m = metrics7(torch.where(cover, Of, torch.zeros_like(Of)), O, U)
                for q in MKEYS:
                    env["K-HIO"][str(k)][q].append(m[q])
        del win
        for kind in s5b.INITS:
            O0 = a6.make_start(kind, d, kw, dev)
            m0 = metrics7(O0, O, U)
            for meth in METHODS7:
                outs, stt = run_method7(meth, d, iters, O0, state=True)
                conf = f"{meth}|{kind}"
                for it in iters:
                    m = metrics7(outs[it], O, U)
                    for q in MKEYS:
                        env[conf][str(it)][q].append(m[q])
                for q in MKEYS:
                    env[conf]["0"][q].append(m0[q])
                if stt is not None:                                                  # 修正後({last} 次)的位置 / 探針誤差
                    env[conf]["diag"]["pos_rms"].append(float((stt["r"] - want).norm(dim=-1).pow(2).mean(1).sqrt().mean()))
                    env[conf]["diag"]["prb_err"].append(float(probe_err(stt["P"] if stt["P"] is not None else geo.pr["P"],
                                                                        Pt).mean()))
                del outs
        del kw, d
        done.append(s)
        if partial is not None:
            dump_json({"meta": meta, "env": env, "nets": nets, "stats": stats, "done": done}, partial)
        log(f"  [{cond} seed {s}] 完成(經過 {time.time() - t0:.0f} 秒;修正版的位置 / 探針誤差是第 {last} 次的)")
    return env, nets, stats


# ============================================================================
# 計時(彙整 job;同一張 GPU)
# ============================================================================
def _time_it(fn, B, dev, per):
    ts = []
    for _ in range(s5b.TIME_REPS):
        s5._sync(dev)
        t0 = time.perf_counter()
        fn()
        s5._sync(dev)
        ts.append((time.perf_counter() - t0) / per / B * 1e3)
    return float(np.median(ts))


@torch.no_grad()
def timing(dev):
    """ms / 整張影像。平常版每次;修正版每次分「修正開始前(off)/ 後(on)」;K-HIO(49 張)一次;合併;網路流程端到端。"""
    import probe_4_2b as pb
    torch.backends.cudnn.benchmark = True
    res = {}
    for Bl in ["64", "512"]:
        B = int(Bl) if not QUICK else {"64": 4, "512": 8}[Bl]
        d = setup_cond(0, dev, "ideal", B)
        d["init"] = 0
        geo, cp, bs, counts = d["geo"], d["cp"], d["bs"], d["counts"]
        row = {}
        O0 = s5.epie_init(B, geo.F, d["cfg"].phase_max, 0, dev)
        for meth in BASE_METHODS:
            run_method7(meth, d, [2], O0)
            row[meth] = _time_it(lambda: run_method7(meth, d, [10], O0), B, dev, 10)
        for meth in CORR_METHODS:
            # off = 修正開始前;on = 修正進行中(A 模式量的是退火階段;退火結束後與非 A 模式相同,沿用其 on)
            for tag, st in (("off", 10 ** 9), ("on", 0)):
                run_method7(meth, d, [2], O0, start=st)
                key = f"{meth}|{'anneal' if (tag == 'on' and meth.endswith('A')) else tag}"
                row[key] = _time_it(lambda: run_method7(meth, d, [10], O0, start=st), B, dev, 10)
        ones = torch.ones(geo.W, geo.W, device=dev)
        init = torch.cat([s5.k_init(counts[j], cp, j, dev) for j in range(len(geo.starts))])
        cc = torch.cat(counts)
        pb.hio_probe(init, cc, bs, cp, [3], "K", geo.pr, ones)
        row["K-HIO49"] = _time_it(lambda: pb.hio_probe(init, cc, bs, cp, [20], "K", geo.pr, ones), B, dev, 20)
        win = torch.polar(torch.rand(len(geo.starts), B, geo.W, geo.W, device=dev),
                          torch.rand(len(geo.starts), B, geo.W, geo.W, device=dev))
        a6.merge_windows(win, geo)
        row["merge49"] = _time_it(lambda: a6.merge_windows(win, geo), B, dev, 1)
        net, norm, _ = a6.load_p6b4(0, d["cfg"], geo.pr, dev)
        gs = a6.GlobalStages(net, geo.starts, geo.F, geo.pr, bs, dev)
        for pipe in PIPES:
            a6.run_pipe(pipe, net, norm, d, gs)
            a6.run_pipe(pipe, net, norm, d, gs)
            row[f"net:{pipe}"] = _time_it(lambda: a6.run_pipe(pipe, net, norm, d, gs), B, dev, 1)
        res[Bl] = row
        del d, win, net, gs
    res["gpu"] = gpu_name(dev)
    return res


def cfg_time7(conf, it, tB):
    tK, tM = tB["K-HIO49"], tB["merge49"]
    if conf == "K-HIO":
        return it * tK + tM
    meth, kind = conf.split("|")
    t0 = KHIO_N[kind] * tK + tM if kind in KHIO_N else 0.0
    if ":" not in meth:
        return t0 + it * tB[meth]
    t = t0 + min(it, START) * tB[f"{meth}|off"]
    if meth.endswith("A"):                                              # 退火階段(ANNEAL 次)+ 之後同非 A 模式
        na = min(max(0, it - START), ANNEAL)
        return t + na * tB[f"{meth}|anneal"] + max(0, it - START - ANNEAL) * tB[f"{meth[:-1]}|on"]
    return t + max(0, it - START) * tB[f"{meth}|on"]


def env_vals7(env, conf, it, key=MAIN):
    return np.array(env[conf][str(it)][key], float)


def envelope7(env, T, tB, iters, subset="all", key=MAIN):
    best = None
    for conf in configs7(subset):
        for it in stops7(conf, iters):
            t = cfg_time7(conf, it, tB)
            if t > T * (1 + 1e-9):
                continue
            v = env_vals7(env, conf, it, key)
            if best is None or v.mean() < best[2].mean():
                best = (conf, it, v, t)
    return best


def time_grid(tB, iters, subset="all"):
    return sorted({cfg_time7(c, it, tB) for c in configs7(subset) for it in stops7(c, iters)} - {0.0})


# ============================================================================
# 內建檢查
# ============================================================================
@torch.no_grad()
def checks(dev):
    torch.backends.cudnn.benchmark = True
    print("=" * 70)
    print("階段七 7a-1:內建檢查")
    print("=" * 70)
    n = 8 if not QUICK else 4
    d0 = a6.setup(0, dev, n)
    dz = setup_cond(0, dev, "ideal", n, force_generic=True)
    geo, pr, cp, bs, O, ops = dz["geo"], dz["pr"], dz["cp"], dz["bs"], dz["O"], dz["ops"]
    W = geo.W
    # (1) 量測
    e1 = max(float((a - b).abs().max()) for a, b in zip(d0["counts"], dz["counts"]))
    differ = {}
    for c in CONDS[1:]:
        dc = setup_cond(0, dev, c, n)
        differ[c] = max(float((a - b).abs().max()) for a, b in zip(d0["counts"], dc["counts"]))
    check("量測:誤差 0、劑量 1 時本程式的量測 = 階段六 measure49(逐位元);其他 7 個條件的量測都與理想不同",
          e1 == 0.0 and all(v > 0 for v in differ.values()),
          f"理想差 {e1:.1e};" + "、".join(f"{k} {v:.2g}" for k, v in differ.items()))
    # (2) 探針的平移與真探針
    P = pr["P"]
    e_roll = float((ops.shift(P, torch.tensor(3.0, device=dev), torch.tensor(-2.0, device=dev))
                    - torch.roll(P, (3, -2), dims=(-2, -1))).abs().max())
    dy, dx = torch.tensor(0.37, device=dev), torch.tensor(-1.61, device=dev)
    e_back = float((ops.shift(ops.shift(P, dy, dx), -dy, -dx) - P).abs().max())
    pp, pm = true_probe_pair(pr, 0.3, dev)
    e_en = max(abs(float((q.abs() ** 2).sum()) - ops.E0) / ops.E0 for q in (pp, pm))
    out_ap = max(float((torch.fft.fft2(q) * (1 - ops.aper)).abs().max()) / float(torch.fft.fft2(q).abs().max())
                 for q in (pp, pm, P))
    e_proj = float((ops.project(P) - P).abs().max())
    differs = float((pp - P).abs().max())
    from src.physics import forward_measure
    cp0 = copy.deepcopy(cp)
    cp0.add_poisson = False
    zi = torch.zeros(n, len(geo.starts), 2, device=dev)
    zi[..., 0], zi[..., 1] = 2.0, -1.0
    cg = measure_cond(dz["fields"], geo, cp0, bs, 0, dict(pos=1.0, df=0.0, dose=1.0), zi, dz["sign"], ops)
    ref = []
    for (y0, x0) in geo.starts:
        win = s5.crop(dz["fields"], y0, x0, W)
        psi = torch.roll(P, (2, -1), dims=(-2, -1)) * torch.polar(win[:, 0], win[:, 1])
        ref.append(forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), bs, cp0))
    e_sign = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(cg, ref))
    check("量測的位置誤差加在探針上、方向正確:δ = (2, −1) px 的量測 = 探針 torch.roll((2, −1)) 的量測(無雜訊,相對差 < 1e-4)",
          e_sign < 1e-4, f"相對差 {e_sign:.1e}")
    check("探針:整數平移 = torch.roll、+δ 再 −δ = 原探針(< 1e-5);真探針總能量 = 名目(< 1e-5)、孔徑外為 0(< 1e-5);"
          "名目探針經孔徑投影不變(< 1e-5);真探針 ≠ 名目",
          e_roll < 1e-5 and e_back < 1e-5 and e_en < 1e-5 and out_ap < 1e-5 and e_proj < 1e-5 and differs > 0.05,
          f"roll {e_roll:.1e}、來回 {e_back:.1e}、能量 {e_en:.1e}、孔徑外 {out_ap:.1e}、投影 {e_proj:.1e}、真 − 名目 {differs:.2f}")
    # (3) 修正版在修正關閉時 = 平常版
    Oi = s5.epie_init(n, geo.F, cp.phase_max, 7, dev)
    it3 = [1, 3]
    e3 = {}
    for mode in MODES:
        a = ap_corr(dz["counts"], bs, cp, pr, geo.starts, it3, True, Oi, mode, ops, start=10 ** 9)
        b = s5b.ap(dz["counts"], bs, cp, pr, geo.starts, it3, True, Oi)
        c = epie_corr(dz["counts"], bs, cp, pr, geo.starts, it3, True, Oi, 11, mode, ops, start=10 ** 9)
        e = s5.epie(dz["counts"], bs, cp, pr, geo.starts, it3, True, Oi, 11)
        e3[mode] = max(max(float((a[k] - b[k]).abs().max()) for k in it3), max(float((c[k] - e[k]).abs().max()) for k in it3))
    check("修正版在修正關閉時 = 平常版(AP-C = scan_5b.ap、ePIE-C = scan_5.epie;< 1e-5)", max(e3.values()) < 1e-5,
          "、".join(f"{k} {v:.1e}" for k, v in e3.items()))
    # (4) 修正有效(真值物體、無雜訊)
    from src.physics import forward_measure
    nt = min(n, 4)
    Ot, ft = O[:nt], dz["fields"][:nt]
    cp0 = copy.deepcopy(cp)
    cp0.add_poisson = False
    J = len(geo.starts)

    def meas_pos(dtrue):
        out = []
        for j, (y0, x0) in enumerate(geo.starts):
            win = s5.crop(ft, y0, x0, W)
            psi = ops.shift(P.expand(nt, W, W), dtrue[:, j, 0], dtrue[:, j, 1]) * torch.polar(win[:, 0], win[:, 1])
            out.append(forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), bs, cp0))
        return out

    gen = torch.Generator().manual_seed(9)
    d1 = torch.zeros(nt, J, 2, device=dev)
    pick = torch.randperm(J, generator=gen)[:12]
    d1[:, pick] = (torch.rand(nt, 12, 2, generator=gen) * 2 - 1).to(dev)                      # 12 個位置、± 1 px
    cnt_p = meas_pos(d1)
    want1 = (d1 - d1.mean(1, keepdim=True))[:, pick]
    e_init1 = float(want1.norm(dim=-1).mean())
    res_pos = {}
    for name, fn in (("AP", lambda: ap_corr(cnt_p, bs, cp0, pr, geo.starts, [60], False, Ot.clone(), "pos", ops, 0, True)),
                     ("ePIE", lambda: epie_corr(cnt_p, bs, cp0, pr, geo.starts, [30], False, Ot.clone(), 5, "pos", ops, 0,
                                                True))):
        _, stt = fn()
        res_pos[name] = float((stt["r"][:, pick] - want1).norm(dim=-1).mean())
    d2 = (torch.randn(nt, J, 2, generator=gen).clamp(-Z_CLIP, Z_CLIP) * 1.0).to(dev)           # 全部位置、σ = 1 px
    cnt_a = meas_pos(d2)
    want2 = d2 - d2.mean(1, keepdim=True)
    e_init2 = float(want2.norm(dim=-1).mean())
    res_ann = {}
    for mode in ("pos", "posA"):
        _, stt = ap_corr(cnt_a, bs, cp0, pr, geo.starts, [45], False, Ot.clone(), mode, ops, 0, True)
        res_ann[mode] = float((stt["r"] - want2).norm(dim=-1).mean())
    pp3, _ = true_probe_pair(pr, 0.3, dev)
    cnt_q = []
    for j, (y0, x0) in enumerate(geo.starts):
        win = s5.crop(ft, y0, x0, W)
        psi = pp3 * torch.polar(win[:, 0], win[:, 1])
        cnt_q.append(forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), bs, cp0))

    def perr(Pe):
        c = (Pe * pp3.conj()).sum((-2, -1))
        al = Pe * torch.exp(-1j * torch.angle(c))[..., None, None]
        return float((((al - pp3).abs() ** 2).sum((-2, -1)) / (pp3.abs() ** 2).sum()).max())

    p0 = perr(P.expand(nt, W, W))
    res_prb = {}
    for name, fn in (("AP", lambda: ap_corr(cnt_q, bs, cp0, pr, geo.starts, [60], False, Ot.clone(), "prb", ops, 0, True)),
                     ("ePIE", lambda: epie_corr(cnt_q, bs, cp0, pr, geo.starts, [30], False, Ot.clone(), 5, "prb", ops, 0,
                                                True))):
        _, stt = fn()
        res_prb[name] = perr(stt["P"])
    _, stt = epie_corr(cnt_a, bs, cp0, pr, geo.starts, [30], False, Ot.clone(), 5, "posA", ops, 0, True)
    res_ann["ePIE-posA"] = float((stt["r"] - want2).norm(dim=-1).mean())
    cnt_b = []                                                                                  # 位置 ± 1 px(12 個)+ 離焦 +30%
    for j, (y0, x0) in enumerate(geo.starts):
        win = s5.crop(ft, y0, x0, W)
        psi = ops.shift(pp3.expand(nt, W, W), d1[:, j, 0], d1[:, j, 1]) * torch.polar(win[:, 0], win[:, 1])
        cnt_b.append(forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), bs, cp0))
    res_both = {}
    for name, fn in (("AP-C", lambda: ap_corr(cnt_b, bs, cp0, pr, geo.starts, [60], True, Ot.clone(), "both", ops, 0, True)),
                     ("ePIE-C", lambda: epie_corr(cnt_b, bs, cp0, pr, geo.starts, [30], True, Ot.clone(), 5, "both", ops, 0,
                                                  True))):
        _, stt = fn()
        res_both[name] = (float((stt["r"][:, pick] - want1).norm(dim=-1).mean()), perr(stt["P"]))
    check("修正有效(兩者同時;受限版本):位置 ± 1 px + 離焦 +30% → 位置平均誤差、探針誤差都 < 起始的 1/3;"
          "ePIE 的退火搜尋(σ = 1 px)< 起始的 1/2",
          all(v[0] < e_init1 / 3 and v[1] < p0 / 3 for v in res_both.values()) and res_ann["ePIE-posA"] < e_init2 / 2,
          "、".join(f"{k}:位置 {v[0]:.3f} / 探針 {v[1]:.4f}" for k, v in res_both.items())
          + f"(起始 {e_init1:.3f} / {p0:.3f});ePIE 退火 {res_ann['ePIE-posA']:.3f}(起始 {e_init2:.3f})")
    check("修正有效(真值起點、無雜訊):位置 12 個偏 ± 1 px → 平均誤差 < 起始的 1/3;全部位置 σ = 1 px → 退火搜尋的平均誤差 "
          "< 起始的 1/2 且小於只用梯度;離焦 +30% → 探針誤差 < 起始的 1/3",
          max(res_pos.values()) < e_init1 / 3 and res_ann["posA"] < e_init2 / 2 and res_ann["posA"] < res_ann["pos"]
          and max(res_prb.values()) < p0 / 3,
          f"位置(平均誤差;起始 {e_init1:.3f} px):" + "、".join(f"{k} {v:.3f}" for k, v in res_pos.items())
          + f";σ = 1 px(起始 {e_init2:.3f}):梯度 {res_ann['pos']:.3f}、退火 {res_ann['posA']:.3f}"
          + f";探針誤差(起始 {p0:.3f}):" + "、".join(f"{k} {v:.4f}" for k, v in res_prb.items()))
    # (5) 對齊指標
    U = geo.U
    Fz = geo.F
    kf = torch.fft.fftfreq(Fz, device=dev)
    ky, kx = torch.meshgrid(kf, kf, indexing="ij")
    ty, tx = 0.3, -1.2
    ph = -2 * math.pi * (ky * ty + kx * tx)
    sh = torch.fft.ifft2(torch.fft.fft2(O) * torch.polar(torch.ones_like(ph), ph)) * torch.exp(torch.tensor(0.9j, device=dev))
    _, _, _, _, _, _, yc = _sh_mats(Fz, dev)
    qq = torch.tensor([[0.0005, -0.0008]], device=dev).expand(O.shape[0], 2)
    est = sh * _ramp_field(qq, yc).conj()                                                    # 平移 + 整體相位 + 相位斜坡
    v0, v1, v2, t, q = align_err(est, O, U)
    vz = align_err(O * torch.exp(torch.tensor(0.4j, device=dev)), O, U)
    noisy = O + 0.05 * torch.randn_like(O.real)
    vn = align_err(noisy, O, U)
    bad = noisy.clone()
    bad[0] = float("nan")
    mb = metrics7(bad, O, U)
    mx = lambda v: float(np.nanmax(v.cpu().numpy()))                                           # noqa: E731
    ok5 = (mx(v2) < 1e-3 and mx(v1) > 10 * mx(v2) and float(np.nanmin(v0.cpu().numpy())) > 1e-2
           and max(mx(x) for x in vz[:3]) < 1e-6
           and bool(((vn[2] <= vn[1] + 1e-12) & (vn[1] <= vn[0] + 1e-12))[torch.isfinite(vn[0])].all())
           and mb["n_fail"] == 1 and mb["nerr_ph"] > FAIL_VAL / O.shape[0] * 0.99)
    check("對齊指標:已知平移 (0.3, −1.2) px + 整體相位 + 相位斜坡 → nerr_sr < 1e-3(只對齊平移時明顯較大、nerr_ph > 1e-2);"
          "只差整體相位 → 0;nerr_sr ≤ nerr_sh ≤ nerr_ph;發散(NaN)的樣本算失敗(記為 FAIL_VAL)",
          ok5, f"nerr_sr 最大 {mx(v2):.1e}、nerr_sh 最大 {mx(v1):.1e}(找到 t ≈ {[round(x, 3) for x in t[0].tolist()]}、"
               f"q ≈ {[round(x, 5) for x in q[0].tolist()]})、nerr_ph 最小 {float(np.nanmin(v0.cpu().numpy())):.3f};"
               f"只差相位 {max(mx(x) for x in vz[:3]):.1e};NaN 樣本:失敗 {mb['n_fail']} 個")
    # (6) 網路路徑(本程式的理想量測 → 與階段六相同的網路輸出)
    net, norm, _ = a6.load_p6b4(0, d0["cfg"], d0["geo"].pr, dev)
    gs = a6.GlobalStages(net, geo.starts, geo.F, pr, bs, dev)
    dI = setup_cond(0, dev, "ideal", n)
    e6 = max(float((a6.run_pipe(p, net, norm, d0, gs) - a6.run_pipe(p, net, norm, dI, gs)).abs().max()) for p in PIPES)
    check("網路路徑:理想條件的 G9 / G25 = 以階段六的 setup 得到的輸出(差 0;正式 job 另與階段六的結果檔比對)", e6 == 0.0,
          f"最大差 {e6:.1e}")
    # (7) 指標
    est = O * torch.exp(torch.tensor(0.7j, device=dev)) + 0.05 * torch.randn_like(O.real)
    fm = s5.field_metrics(est, O, U)["nerr_ph"]
    pm_ = metrics7(est, O, U)["nerr_ph"]
    check("指標:逐樣本平均 = field_metrics 的 nerr_ph(差 / max(1, 值) < 1e-6)", absdiff(pm_, fm) / max(1.0, fm) < PS_TOL,
          f"{pm_:.6f} vs {fm:.6f}")
    print(f"\n     裝置:{dev}({gpu_name(dev)})")


# ============================================================================
# 判讀
# ============================================================================
def netv(C, c, p, key=MAIN):
    return np.array(C[c]["nets"][p][key], float)


def verdict_row(C, c, p, tim, iters, B, T=None, subset="all"):
    a = netv(C, c, p)
    tn = tim[B][f"net:{p}"] if T is None else T
    e = envelope7(C[c]["env"], tn, tim[B], iters, subset)
    r = s5b.ratio(a, e[2])
    return {"t": tn, "opp": [e[0], e[1], list(e[2]), e[3]], "ratio": list(r)}


def points(env, tB, iters, subset="all", key=MAIN):
    """所有 (時間, 3 seeds 平均誤差),依時間排序。"""
    pts = [(cfg_time7(c, it, tB), float(env_vals7(env, c, it, key).mean())) for c in configs7(subset) for it in stops7(c, iters)]
    return sorted(pts)


def env_curve(env, tB, iters, subset="all"):
    """包絡曲線:各時間點以前的最小誤差(與 envelope7 相同的定義,只是一次算完)。"""
    ts, best, out = [], float("inf"), []
    for t, v in points(env, tB, iters, subset):
        best = min(best, v)
        if t > 0:
            ts.append(t)
            out.append(best)
    return ts, out


def t_match(env, target, tB, iters):
    """包絡降到 target(3 seeds 平均)所需的最短時間 = 誤差 ≤ target 的設定中最短的時間;500 次內到不了 → inf。"""
    ok = [t for t, v in points(env, tB, iters) if v <= target]
    return min(ok) if ok else float("inf")


def best_overall(env, iters, key=MAIN):
    best = None
    for conf in configs7():
        for it in stops7(conf, iters):
            v = env_vals7(env, conf, it, key)
            if best is None or v.mean() < best[2].mean():
                best = (conf, it, v)
    return best


def report(C, tim, iters, smoke):
    Bs = ["64", "512"]
    V = {"cond": {}}
    if smoke:
        print("\n" + "!" * 100)
        print("迷你流程:場數少、停止點 ≤ 20 → 數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    print("\n" + "=" * 100)
    print(f"計時(ms / 整張影像;本 job 重新量;GPU {tim['gpu']})")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {v:.4f}" for k, v in tim[B].items()))
    print("\n" + "=" * 100)
    print("條件的統計(內建檢查 1 的後半:誤差確實加上)")
    print("=" * 100)
    for c in CONDS:
        sp, st = SPEC[c], C[c]["stats"]
        print(f"  {c:<6} 位置 σ 設定 {sp['pos']:.2f} → 實際 {np.mean([x['pos_std_px'] for x in st]):.3f} px(最大 "
              f"{max(x['pos_max_px'] for x in st):.2f});離焦 + 的比例 {np.mean([x['df_plus_frac'] for x in st]):.2f};"
              f"每張平均計數 {np.mean([x['mean_counts'] for x in st]):.3g}"
              + (f";原始計數 / 理想 {np.mean([x['raw_count_ratio'] for x in st]):.4f}(設定 {sp['dose']})" if c != "ideal" else ""))
    print("\n" + "=" * 100)
    print(f"各條件:網路 vs 整張包絡(主指標 {MAIN}:整體相位 + 平移 + 相位斜坡;3 seeds;比值 ≤ 0.8 且 z < −2 = 較準)")
    print("=" * 100)
    for c in CONDS:
        V["cond"][c] = {}
        print(f"\n  ■ {c}({CLABEL[c]})")
        for p in PIPES:
            a = netv(C, c, p)
            row = {"nerr": list(a), "nerr_ph": list(netv(C, c, p, "nerr_ph")),
                   "shift": float(np.mean(C[c]["nets"][p]["shift"]))}
            line = f"    {p}:{a.mean():.4f}({' '.join(f'{x:.4f}' for x in a)})"
            for B in Bs:
                vr = verdict_row(C, c, p, tim, iters, B)
                vp = verdict_row(C, c, p, tim, iters, B, subset="plain")
                vm = verdict_row(C, c, p, tim, iters, B, T=1.1 * tim[B][f"net:{p}"])
                row[B] = {"all": vr, "plain": vp, "t1.1": vm}
                line += (f"\n       b{B}:{vr['t']:.3f} ms vs {s5b.fmt_conf(vr['opp'][0])} ×{vr['opp'][1]} "
                         f"{np.mean(vr['opp'][2]):.4f} → 比值 {vr['ratio'][1]:.3f}(z {vr['ratio'][2]:+.1f})→ {vr['ratio'][0]}"
                         f";只用平常版 {vp['ratio'][1]:.3f};時間 × 1.1 處 {vm['ratio'][1]:.3f}")
            row["useful"] = row["64"]["all"]["ratio"][0] == "較準"
            row["robust"] = row["useful"] and row["512"]["all"]["ratio"][0] == "較準"
            print(line + f"\n       ▶ {'穩健有用' if row['robust'] else ('有用(只在 b64)' if row['useful'] else '未達有用')}")
            V["cond"][c][p] = row
    # (q3) 相對退步
    print("\n" + "=" * 100)
    print("判讀(階段七協定 §六,結果出來前寫定)")
    print("=" * 100)
    g9c = V["cond"][MAIN_COND][PRIMARY]
    q1 = g9c["robust"]
    print(f"  (q1 主判)combo 下 G9:{'穩健有用' if q1 else ('只在 b64 有用' if g9c['useful'] else '未達有用')}"
          + ("  [smoke:無意義]" if smoke else ""))
    V["q1"] = q1
    print("  (q3) 相對退步 = 條件下的比值 / 理想時的比值(< 0.8 網路相對變強;> 1.25 網路相對變弱;其他相近;描述)")
    q3 = {}
    for c in CONDS[1:]:
        q3[c] = {}
        parts = []
        for p in PIPES:
            for B in Bs:
                rc, ri = V["cond"][c][p][B]["all"]["ratio"][1], V["cond"]["ideal"][p][B]["all"]["ratio"][1]
                rel = rc / max(ri, 1e-12)
                lab = "網路相對變強" if rel < 0.8 else ("網路相對變弱" if rel > 1.25 else "相近")
                q3[c][f"{p}_b{B}"] = {"rel": rel, "label": lab}
                parts.append(f"{p} b{B} {rel:.2f}({lab})")
        dn = netv(C, c, PRIMARY).mean() / netv(C, "ideal", PRIMARY).mean()
        de = {B: np.mean(V["cond"][c][PRIMARY][B]["all"]["opp"][2]) / np.mean(V["cond"]["ideal"][PRIMARY][B]["all"]["opp"][2])
              for B in Bs}
        q3[c]["deg"] = {"net": dn, "env64": de["64"], "env512": de["512"]}
        print(f"    {c:<6} " + "、".join(parts) + f";G9 誤差 × {dn:.2f}、同時間包絡 × {de['64']:.2f}(b64)/ × {de['512']:.2f}(b512)")
    V["q3"] = q3
    print("  (q4) 修正的價值:同上表的「只用平常版」比值 vs 全部對手(描述)")
    print("  (q5) 時間餘裕:同上表的「時間 × 1.1 處」比值(描述)")
    print("  (q6) 同品質所需時間(迭代法包絡降到 G9 的誤差所需時間 / G9 的時間)與迭代法的最佳誤差(描述)")
    q6 = {}
    for c in CONDS:
        env = C[c]["env"]
        tgt = netv(C, c, PRIMARY).mean()
        bo = best_overall(env, iters)
        q6[c] = {"best": [bo[0], bo[1], float(bo[2].mean())]}
        parts = []
        for B in Bs:
            tm = t_match(env, tgt, tim[B], iters)
            q6[c][B] = {"t_match": tm, "x": tm / tim[B][f"net:{PRIMARY}"]}
            parts.append(f"b{B} {tm:.2f} ms(× {tm / tim[B][f'net:{PRIMARY}']:.2f})" if np.isfinite(tm) else f"b{B} 500 次內到不了")
        print(f"    {c:<6} " + "、".join(parts) + f";迭代法最佳 {s5b.fmt_conf(bo[0])} ×{bo[1]} {bo[2].mean():.2e}")
    V["q6"] = q6
    print("  (q7) 對齊的影響(nerr_ph → + 平移 nerr_sh → + 相位斜坡 nerr_sr;網路與同時間包絡所選的設定;描述)")
    for c in CONDS:
        parts = [f"{p} {netv(C, c, p, 'nerr_ph').mean():.4f} → {netv(C, c, p, 'nerr_sh').mean():.4f} → {netv(C, c, p).mean():.4f}"
                 f"(平移 {np.mean(C[c]['nets'][p]['shift']):.2f} px、斜坡 {np.mean(C[c]['nets'][p]['ramp']):.4f} rad/px)"
                 for p in PIPES]
        conf, it = V["cond"][c][PRIMARY]["512"]["all"]["opp"][:2]
        e = C[c]["env"][conf][str(it)]
        parts.append(f"迭代法 b512({s5b.fmt_conf(conf)} ×{it}){np.mean(e['nerr_ph']):.4f} → {np.mean(e['nerr_sh']):.4f} → "
                     f"{np.mean(e['nerr_sr']):.4f}")
        print(f"    {c:<6} " + ";".join(parts))
    print(f"  (q8) 修正版的效果(第 {max(iters)} 次;所有修正版設定中最好的;3 seeds 平均;描述)")
    q8 = {}
    for c in CONDS:
        st = C[c]["stats"]
        p0, e0 = np.mean([x["pos_rms0"] for x in st]), np.mean([x["prb_err0"] for x in st])
        dg = [(k, np.mean(v["diag"]["pos_rms"]), np.mean(v["diag"]["prb_err"])) for k, v in C[c]["env"].items() if "diag" in v]
        bp = min(dg, key=lambda x: x[1])
        bq = min(dg, key=lambda x: x[2])
        q8[c] = {"pos_rms0": p0, "prb_err0": e0, "best_pos": [bp[0], bp[1]], "best_prb": [bq[0], bq[2]]}
        tp = f"位置 RMS {p0:.3f} → {bp[1]:.3f} px({s5b.fmt_conf(bp[0])})" if p0 > 0 else "無位置誤差"
        tq = f"探針誤差 {e0:.4f} → {bq[2]:.4f}({s5b.fmt_conf(bq[0])})" if e0 > 1e-8 else "無探針誤差"
        print(f"    {c:<6} {tp};{tq}")
    V["q8"] = q8
    # 主力的選法
    rob = [p for p in PIPES if V["cond"][MAIN_COND][p]["robust"]]
    choice = PRIMARY
    if rob:
        best = min(rob, key=lambda p: netv(C, MAIN_COND, p).mean())
        close = [p for p in rob if p != best and s5b.ratio(netv(C, MAIN_COND, p), netv(C, MAIN_COND, best))[0] == "相近"]
        choice = min([best] + close, key=lambda p: tim["64"][f"net:{p}"])
    V["choice"] = choice
    print(f"\n  ▶ 主力(事先寫定的選法,combo 下):{choice}" + ("(兩者都不穩健有用 → 維持 G9)" if not rob else ""))
    rob_ideal = V["cond"]["ideal"][choice]["robust"]
    weak = [c for c in CONDS[1:] if any(q3[c][f"{choice}_b{B}"]["label"] == "網路相對變弱" for B in Bs)
            or (rob_ideal and not V["cond"][c][choice]["robust"])]
    kinds = sorted({k for c in weak for k, key in (("位置", "pos"), ("探針", "df"), ("劑量", "dose"))
                    if (SPEC[c][key] != (1.0 if key == "dose" else 0.0))})
    V["for_7b"] = {"network": choice, "conds": weak, "kinds": kinds, "robust_ideal": rob_ideal}
    print(f"  ▶ 7b 的依據(依 {choice};「網路相對變弱」,或理想時穩健有用、在該條件失去穩健有用):"
          + (f"條件 {', '.join(weak)} → 誤差類型 {'、'.join(kinds)}" if weak else "無 → 7b 不是必要,與使用者討論")
          + ("" if rob_ideal else f"(註:{choice} 在理想條件下本來就不是穩健有用)"))
    if any(q3["posH"][f"{p}_b{B}"]["label"] == "網路相對變強" for p in PIPES for B in Bs):
        print("  ⚠️ posH 出現「網路相對變強」:須註明迭代法的位置修正在此誤差下能力有限(本研究的實作;見 (q8) 與協定 §十),"
              "不可直接寫成網路在大位置誤差下較好")
        V["posH_caveat"] = True
    return V


# ============================================================================
# 圖(描述,不改變判讀)
# ============================================================================
def align_show(e, o, U):
    """顯示用:與主指標相同的對齊(整體相位 + 平移 + 相位斜坡)。e, o:[F, F]。"""
    _, _, v, t, q = align_err(e[None], o[None], U)
    F = o.shape[-1]
    _, _, ky, kx, _, _, yc = _sh_mats(F, o.device)
    ph = 2 * math.pi * (ky * t[0, 0] + kx * t[0, 1])
    es = torch.fft.ifft2(torch.fft.fft2(e) * torch.polar(torch.ones_like(ph), ph)) * _ramp_field(q, yc)[0]
    m = U.to(o.real.dtype)
    c = ((o * m) * (es * m).conj()).sum()
    return es * torch.exp(1j * torch.angle(c)), float(v[0])


@torch.no_grad()
def recon(dev, C, tim, iters, n):
    """seed 0:每個條件重算網路與「G9 的時間(b512)」的迭代法,只留事先寫定的兩個樣本。"""
    ps = np.array(C[MAIN_COND]["nets"][PRIMARY]["ps_sr"][0], float)
    ok = np.where(np.isfinite(ps))[0]
    med = int(ok[np.argsort(ps[ok], kind="stable")][len(ok) // 2])
    rng = np.random.default_rng(RAND_SEED)
    rnd = int(rng.choice(len(ps), size=1)[0])
    picks = {"median": med, "random": rnd}
    keep = {"picks": picks, "img": {}, "diff": {}}
    for c in CONDS:
        d = setup_cond(0, dev, c, n)
        geo, O, U = d["geo"], d["O"], d["geo"].U
        net, norm, _ = a6.load_p6b4(0, d["cfg"], geo.pr, dev)
        gs = a6.GlobalStages(net, geo.starts, geo.F, geo.pr, d["bs"], dev)
        ests = {p: a6.run_pipe(p, net, norm, d, gs) for p in PIPES}
        del net, gs
        conf, it = envelope7(C[c]["env"], tim["512"][f"net:{PRIMARY}"], tim["512"], iters)[:2]
        kw = None
        if conf == "K-HIO" or conf.split("|")[1] in KHIO_N:
            kit = sorted(set(iters) | set(KHIO_N.values()))
            win = khio_windows7(d, dev, kit)
            kw = {k: a6.merge_windows(wv, geo)[:2] for k, wv in win.items()}
            del win
        if conf == "K-HIO":
            Of, cover = kw[it]
            est = torch.where(cover, Of, torch.zeros_like(Of))
        else:
            meth, kind = conf.split("|")
            O0 = a6.make_start(kind, d, kw, dev)
            est = O0 if it == 0 else run_method7(meth, d, [it], O0)[it]
        ests["iter"] = est
        v = float(metrics7(est, O, U)[MAIN])
        keep["diff"][c] = {"conf": conf, "it": it, "recomputed": v, "env": float(env_vals7(C[c]["env"], conf, it)[0])}
        keep["img"][c] = {name: {**{k: ests[k][i].cpu() for k in ests}, "O": O[i].cpu()} for name, i in picks.items()}
        if c == CONDS[0]:
            keep["geo"], keep["cfg"] = geo, d["cfg"]
        if c in ("posH", "ideal", "doseL", "doseH"):
            keep.setdefault("diffr", {})[c] = d["counts"][a6.CEN49][0].cpu()
        if c == "posH":
            keep["pos"] = (SPEC[c]["pos"] * d["z"][0]).cpu()
        del d, ests, kw
    worst = max(absdiff(x["recomputed"], x["env"]) for x in keep["diff"].values())
    print(f"  描述性核對(不阻擋):圖用的迭代法重算 vs 條件檔(seed 0)最大差 {worst:.1e}")
    return keep


def figures(C, V, tim, iters, keep, out_dir, where, dev):
    plt = s5._plt()
    if plt is None:
        check("matplotlib 可用(畫圖需要)", False)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    geo, cfg = keep["geo"], keep["cfg"]
    U = geo.U.cpu()
    pm = cfg.phase_max
    ys, xs = torch.where(U)
    y0, y1 = max(0, int(ys.min()) - MARGIN), min(geo.F, int(ys.max()) + 1 + MARGIN)
    x0, x1 = max(0, int(xs.min()) - MARGIN), min(geo.F, int(xs.max()) + 1 + MARGIN)
    Uc = U[y0:y1, x0:x1].numpy()

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    def blank(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    # 1. 條件的樣子
    pr = geo.pr
    pp, pmm = true_probe_pair(pr, SPEC["prbH"]["df"], dev)
    probes = [("nominal (DEF2)", pr["P"].cpu()), ("true, defocus -30%", pmm.cpu()), ("true, defocus +30%", pp.cpu())]
    fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for c_, (nm, q) in enumerate(probes):
        a = q.abs().numpy()
        axs[0, c_].imshow(a, cmap="gray")
        axs[0, c_].set_title(f"|P| {nm}", fontsize=8)
        axs[1, c_].imshow(np.where(a > 0.1 * a.max(), np.angle(q.numpy()), np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axs[1, c_].set_title(f"phase(P) {nm}", fontsize=8)
        blank(axs[0, c_])
        blank(axs[1, c_])
    ax = axs[2, 0]
    st = np.array(geo.starts, float) + geo.W / 2 - 0.5
    dpos = keep["pos"].numpy()
    ax.plot(st[:, 1], st[:, 0], "o", color="#9a9994", ms=4, label="nominal")
    ax.quiver(st[:, 1], st[:, 0], dpos[:, 1], dpos[:, 0], angles="xy", scale_units="xy", scale=1 / 3, color="#d6452a",
              width=0.004, label="true - nominal (x3)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"position errors, posH\n(sigma {SPEC['posH']['pos']} px; seed 0, field 0)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    for ax, c in zip(axs[2, 1:], ["ideal", "doseH"]):
        im = ax.imshow(np.log10(keep["diffr"][c].numpy() + 1), cmap="magma")
        ax.set_title(f"centre diffraction pattern, {CLABEL[c]}\nlog10(counts / dose ratio + 1)", fontsize=7)
        blank(ax)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"{where}: the first-layer realistic errors", fontsize=10)
    save(fig, "c_conditions.png")

    # 2. 總覽
    xs_ = np.arange(len(CONDS))
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, B in zip(axs[:2], ["64", "512"]):
        for k, p in enumerate(PIPES):
            m = [np.mean(V["cond"][c][p]["nerr"]) for c in CONDS]
            e = [np.mean(V["cond"][c][p][B]["all"]["opp"][2]) for c in CONDS]
            ax.plot(xs_ + (k - 0.5) * 0.18, m, "o", color=COL[p], ms=7, label=f"{p} (network)")
            ax.plot(xs_ + (k - 0.5) * 0.18, e, "x", color=COL[p], ms=7, label=f"iterative envelope at {p}'s time")
        ax.set_yscale("log")
        ax.set_xticks(xs_)
        ax.set_xticklabels([CLABEL[c] for c in CONDS], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"batch {B}: error ({MAIN}) per condition", fontsize=9)
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
        ax.legend(fontsize=6.5, frameon=False)
    ax = axs[2]
    for p in PIPES:
        for B, ls in (("64", ":"), ("512", "-")):
            ax.plot(xs_, [V["cond"][c][p][B]["all"]["ratio"][1] for c in CONDS], ls, marker="o", color=COL[p], ms=4,
                    label=f"{p} / envelope, b{B}")
    for yv in (0.8, 1.25):
        ax.axhline(yv, color="#0b0b0b", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(xs_)
    ax.set_xticklabels([CLABEL[c] for c in CONDS], rotation=30, ha="right", fontsize=7)
    ax.set_title("ratio network / envelope at the same time (< 0.8 = network better)", fontsize=9)
    ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"{where}: frozen networks vs whole-image iterative envelope (incl. position / probe correction)", fontsize=10)
    fig.tight_layout()
    save(fig, "c_summary.png")

    # 3. 時間 vs 誤差
    for B in ["64", "512"]:
        fig, axs = plt.subplots(2, 4, figsize=(18, 8.4), sharey=True)
        for ax, c in zip(axs.ravel(), CONDS):
            env = C[c]["env"]
            for sub, col, ls, lab in (("all", "#0b0b0b", "-", "all opponents"), ("plain", COL["plain"], "--", "plain only (stage 6)")):
                ts, vs = env_curve(env, tim[B], iters, sub)
                ax.plot(ts, vs, ls, color=col, lw=1.6, drawstyle="steps-post", label=lab)
            for p in PIPES:
                a = np.mean(V["cond"][c][p]["nerr"])
                ax.plot([tim[B][f"net:{p}"]], [a], "o", color=COL[p], ms=7, zorder=4,
                        label=f"{p} {a:.4f} (ratio {V['cond'][c][p][B]['all']['ratio'][1]:.2f})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(CLABEL[c], fontsize=9)
            ax.grid(True, color="#e4e3df", which="both", lw=0.5)
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        for ax in axs[1]:
            ax.set_xlabel("time per whole image (ms)")
        for ax in axs[:, 0]:
            ax.set_ylabel(f"{MAIN} on U")
        fig.suptitle(f"{where}: time vs error, batch {B} (envelope = best of all settings within the time)", fontsize=10)
        fig.tight_layout()
        save(fig, f"c_vs_envelope_b{B}.png")

    # 4. 重建圖
    cols = [("O", "truth"), ("G9", "G9"), ("G25", "G25"), ("iter", "iterative (G9's time, b512)")]
    for name, i in keep["picks"].items():
        fig, axs = plt.subplots(len(CONDS), 7, figsize=(15, 2.05 * len(CONDS) + 0.8), squeeze=False, constrained_layout=True)
        for r, c in enumerate(CONDS):
            im = keep["img"][c][name]
            o = im["O"]
            t = o[y0:y1, x0:x1].numpy()
            for k, (key, lab) in enumerate(cols):
                if key == "O":
                    z, v = o, None
                else:
                    z, v = align_show(im[key], o, U)
                zc = z[y0:y1, x0:x1].numpy()
                hp = axs[r, k].imshow(np.where(np.abs(t) > AMP_MIN, np.angle(zc), np.nan), cmap="viridis", vmin=0, vmax=pm)
                axs[r, k].imshow(np.where(Uc, np.nan, 0.0), cmap="Greys", vmin=0, vmax=1, alpha=0.55)
                blank(axs[r, k])
                ttl = lab if v is None else (f"{lab}\n{MAIN} {v:.4f}" if r == 0 else f"{MAIN} {v:.4f}")
                axs[r, k].set_title(ttl if (r == 0 or v is not None) else "", fontsize=6.5)
                if key != "O":
                    he = axs[r, 3 + k].imshow(np.where(Uc, np.abs(zc - t), np.nan), cmap="magma", vmin=0, vmax=0.5)
                    blank(axs[r, 3 + k])
                    if r == 0:
                        axs[r, 3 + k].set_title(f"|error| {lab}", fontsize=6.5)
            axs[r, 0].set_ylabel(CLABEL[c], fontsize=7)
        fig.colorbar(hp, ax=list(axs[:, 3]), fraction=0.03, label="phase (rad)")
        fig.colorbar(he, ax=list(axs[:, 6]), fraction=0.03, label="|error|")
        what = "median sample of G9 in 'all three'" if name == "median" else f"random sample (numpy seed {RAND_SEED})"
        fig.suptitle(f"{where}: sample #{i} ({what}); seed-0 networks; phase shown, aligned by one global phase + shift + phase ramp (= the main metric)", fontsize=9)
        save(fig, f"c_recon_{name}.png")


# ============================================================================
# 流程
# ============================================================================
def run_cond(dev, cond, n, iters, root, smoke):
    """一個條件:量測 → 條件的統計檢查 →(理想條件、正式)與階段六比對 → 存檔。"""
    me = md5(Path(__file__).resolve())
    t0 = time.time()
    print(f"\n  條件 {cond}({CLABEL[cond]}):{SPEC[cond]}", flush=True)
    path = cond_json(cond, root)
    partial = Path(str(path) + ".partial")
    env, nets, stats = measure_condition(dev, cond, n, iters, log=lambda s: print(s, flush=True), partial=partial)
    sp = SPEC[cond]
    nn = stats[0]["n"]
    if sp["pos"] > 0:
        r = np.mean([s["pos_std_px"] for s in stats]) / sp["pos"]
        check(f"[{cond}] 位置誤差的標準差 = 設定 × (1 ± 5%)", abs(r - 1) <= 0.05, f"實際 / 設定 = {r:.3f}")
    if sp["df"] > 0:
        f = np.mean([s["df_plus_frac"] for s in stats])
        check(f"[{cond}] 離焦符號約各半(± 3 個標準誤)", abs(f - 0.5) <= 3 * 0.5 / math.sqrt(nn), f"+ 的比例 {f:.3f}")
    if sp["dose"] != 1:
        rr = np.mean([s["raw_count_ratio"] for s in stats]) / sp["dose"]
        check(f"[{cond}] 平均總計數比 = 劑量比(± 1%)", abs(rr - 1) <= 0.01, f"實際 / 設定 = {rr:.4f}")
    if cond != "ideal":
        check(f"[{cond}] 量測與理想不同", all(s["counts_differ"] for s in stats))
    extra = {}

    def save():
        out = {"meta": {"cond": cond, "spec": sp, "n": nn, "iters": list(iters), "seeds": list(SEEDS), "script_md5": me,
                        "quick": QUICK, "smoke": smoke, "gpu": gpu_name(dev), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed_s": time.time() - t0},
               "env": env, "nets": nets, "stats": stats, "extra": extra, "checks_ok": all_ok()}
        dump_json(out, path)
        return out

    out = save()                                                          # 先存(下面的比對出錯也不會丟掉結果)
    if cond == "ideal" and not smoke:
        try:
            s6 = json.load(open(S6_OUT))["results"]
            dn = max(absdiff(nets[p]["nerr_ph"][i], s6[p][i]["nerr_ph"]) for p in PIPES for i in range(len(SEEDS)))
            check(f"網路路徑(理想條件):G9 / G25 的 nerr_ph = 階段六的結果檔(< {REPRO_TOL:.0e})", dn < REPRO_TOL, f"最大差 {dn:.1e}")
            s6e = json.load(open(S6_ENV))["envelope"][PROBE]
            de = 0.0
            for conf in configs7("plain"):
                for it in stops7(conf, iters):
                    for i in range(len(SEEDS)):
                        de = max(de, absdiff(env[conf][str(it)]["nerr_ph"][i], s6e[i]["rec"][conf][str(it)]["nerr_ph"]))
            extra["s6_env_maxdiff"] = de
            print(f"  描述性核對(不阻擋):理想條件的平常版包絡 vs 階段六的包絡檔,nerr_ph 最大差 {de:.1e}")
        except Exception as e:                                            # noqa: BLE001
            check("與階段六的比對能執行", False, f"{type(e).__name__}: {e}")
        out = save()
    if partial.exists():
        os.remove(partial)                                                # 本程式自己的暫存檔
    print(f"  存檔:{path}(經過 {time.time() - t0:.0f} 秒)")
    for p in PIPES:
        print(f"    {p}:{MAIN} {np.mean(nets[p][MAIN]):.4f}(nerr_ph {np.mean(nets[p]['nerr_ph']):.4f})")
    return out


def run_final(dev, root, fig_dir, iters, smoke):
    me = md5(Path(__file__).resolve())
    where = "validation fields (smoke)" if smoke else "validation fields, 7x7 scan"
    C = {}
    bad = []
    for c in CONDS:
        p = cond_json(c, root)
        if not p.exists():
            bad.append(f"{p.name} 不存在")
            continue
        C[c] = json.load(open(p))
        m = C[c]["meta"]
        if m["script_md5"] != me:
            bad.append(f"{p.name} 的程式 md5 {m['script_md5'][:8]}… ≠ 現在的 {me[:8]}…")
        if not C[c]["checks_ok"]:
            bad.append(f"{p.name} 有檢查未通過")
        if m["iters"] != list(iters) or m["n"] != C[CONDS[0]]["meta"]["n"]:
            bad.append(f"{p.name} 的停止點或場數不同")
    check("8 個條件的結果檔都在、同一版程式、檢查全過、設定一致", not bad, ";".join(bad))
    if bad:
        return None
    n = C[CONDS[0]]["meta"]["n"]
    t0 = time.time()
    print("\n  計時(ms / 整張影像)", flush=True)
    tim = timing(dev)
    print(f"  (計時完成,經過 {time.time() - t0:.0f} 秒;GPU {tim['gpu']})", flush=True)
    dump_json({"timing": tim, "script_md5": me}, root / f"scan7a_tim{SUF}.json")
    if smoke:                                                             # 預估正式 job 的時間(用本 job 的 b512 計時)
        per = sum(cfg_time7(c, max(ITERS), tim["512"]) for c in configs7()) * 1e-3                 # 秒 / 張影像
        est_h = per * (s5b.N_FIELDS or 512) * len(SEEDS) / 3600
        print(f"  預估正式的每個條件:迭代法約 {est_h:.2f} 小時(512 場 × 3 seeds;不含指標與 K-HIO 的額外時間,約再加 10–20%);"
              f"條件 job 的時限 6 小時")
    V = report(C, tim, iters, smoke)
    final = root / f"scan7a{SUF}.json"
    res = {"verdict": V, "timing": tim, "checks_ok": all_ok(), "smoke": smoke, "script_md5": me,
           "gpus": {c: C[c]["meta"]["gpu"] for c in CONDS}}
    dump_json(res, final)
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        keep = recon(dev, C, tim, iters, n)
        res["viz_diff"] = keep["diff"]
        res["viz_picks"] = keep["picks"]
        figures(C, V, tim, iters, keep, fig_dir, where, dev)
    except Exception as e:                                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"畫圖完成(判讀不受影響,已存於 {final.name})", False, f"{type(e).__name__}: {e}")
    res["checks_ok"] = all_ok()
    dump_json(res, final)
    print(f"  結果:{final}")
    return res


def check_files(formal=False):
    need = [Path(f) for f in ("scan_6a.py", "scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py", "scan_5e.py", "scan_5g.py",
                              "scan_5h.py", "probe_4_2b.py", "probe_4_1.py", "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    need += [s5c.run_dir(GROUP, PROBE, s, a6.N_TRAIN) / f for s in SEEDS for f in ("final.pt", "result.json")]
    if formal:
        need += [S6_ENV, S6_OUT]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在(含 P6B4 的 3 個模型" + ("、階段六的結果檔" if formal else "") + ")", not missing,
          "缺:" + ", ".join(missing) if missing else "")
    return not missing


def need_passed(me):
    if not PASSED.exists():
        raise SystemExit(f"❌ 找不到 {PASSED}:先在 dev 節點跑 python scan_7a.py --smoke,全部通過才可送件")
    pm = json.load(open(PASSED))["script_md5"]
    if pm != me:
        raise SystemExit(f"❌ smoke 通過時的 scan_7a.py(md5 {pm})與現在的({me})不同:用現在的版本重跑 --smoke")
    print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cond", choices=CONDS)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = md5(Path(__file__).resolve())
    print(f"scan_7a.py md5 {me};GPU {gpu_name(dev)}")
    formal = bool(a.cond or a.final)
    if not check_files(formal):
        print("\n❌ 缺檔案,停下來")
        sys.exit(1)
    if a.check:
        checks(dev)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.smoke:
        print("#" * 70)
        print(f"迷你全流程(--smoke):{SMOKE_FIELDS} 個場、停止點 {SMOKE_ITERS};8 個條件 + 彙整;輸出到 {SMOKE_DIR}")
        print("#" * 70)
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)                                      # 只刪這個 smoke 專用資料夾
        SMOKE_DIR.mkdir(parents=True)
        t0 = time.time()
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
        for c in CONDS:
            run_cond(dev, c, SMOKE_FIELDS, SMOKE_ITERS, SMOKE_DIR, smoke=True)
        run_final(dev, SMOKE_DIR, SMOKE_DIR / "figs", SMOKE_ITERS, smoke=True)
        print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒")
        if all_ok():
            json.dump({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
            print("\n✅ 迷你全流程全部通過 → 可以送出正式的 job(先 run_scan7a_cond.sh 的 array,再 run_scan7a_final.sh)")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    if a.cond:
        p = cond_json(a.cond)
        if p.exists():
            raise SystemExit(f"❌ {p} 已存在:這個條件已經跑完。為避免覆蓋,先告訴 Claude")
        need_passed(me)
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有量測。把輸出貼給 Claude")
            sys.exit(1)
        run_cond(dev, a.cond, None, ITERS, RUN_ROOT, smoke=False)
        print("\n" + (f"✅ 條件 {a.cond} 完成、檢查全過" if all_ok() else f"❌ 條件 {a.cond} 有檢查未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.final:
        out = RUN_ROOT / f"scan7a{SUF}.json"
        if out.exists():
            raise SystemExit(f"❌ {out} 已存在:正式結果已經有了。為避免覆蓋,先告訴 Claude")
        need_passed(me)
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有彙整。把輸出貼給 Claude")
            sys.exit(1)
        res = run_final(dev, RUN_ROOT, FIG_DIR, ITERS, smoke=False)
        print("\n" + ("✅ 內建檢查全部通過" if all_ok() and res else "❌ 有項目未通過(判讀先不要採信)"))
        print("把完整輸出與 figs_scan7a/ 的圖傳給 Claude")
        sys.exit(0 if all_ok() and res else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
