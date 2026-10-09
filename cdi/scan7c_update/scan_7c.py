#!/usr/bin/env python
"""階段七 7c:估計型 P6B4e(每一級加位置步)與對照組 P6B4c(階段七協定 §十六)。

P6B4e = P6B4 + 每一級的位置步(可微分的 2 × 2 Gauss–Newton,同 scan_7a._pos_step 的算法;步長 β_k = 4·sigmoid(b_k)),
        AP 步改用平移後的探針;訓練時另加位置監督(λ = 0.1,px²)。
P6B4c = 同樣的起點(P6B4 全部權重)、同樣的誤差配比與學習率、同樣的訓練預算,但沒有位置步(歸因用的對照組)。
訓練的誤差:位置、探針誤差各自以 1/3 的機率不加;劑量同 7b。
訓練前的可行性檢查(--feas):固定公式(β = 2、α = 1、不用 CNN),從凍結 P6B4 的 G9 草稿出發,比較更新順序與每級的位置內迴圈數;
        通過(posL 4 級後的位置 RMS < 起始的 1/2)才訓練,並選出最省的設定。
評估:7b 的 12 個條件 + 診斷條件 comboNP(combo 拿掉探針誤差;只評估網路);只看 G9。

用法(需在計算節點執行):
    python scan_7c.py --check          # 內建檢查
    python scan_7c.py --feas           # 可行性檢查(dev 節點,約 10–20 分鐘;go / no-go,結果存 scan7c_feas.json)
    python scan_7c.py --smoke          # 迷你全流程(dev 節點);通過才可正式執行
    python scan_7c.py --task 0..5      # run_scan7c_train.sh:0–2 = P6B4e seed 0–2;3–5 = P6B4c seed 0–2
    python scan_7c.py --final          # 彙整(run_scan7c_final.sh)
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_7b as b7                                                 # noqa: E402  (7b;不修改。也擴充了 7a 的條件)

a7, a6, h, g = b7.a7, b7.a6, b7.h, b7.g
s5, s5b, s5c = b7.s5, b7.s5b, b7.s5c
RUN_ROOT = b7.RUN_ROOT
QUICK = b7.QUICK
PROBE = b7.PROBE
SEEDS = b7.SEEDS
SUF = b7.SUF
B7_MD5 = "e030d9e42da69a1848e28854a5bffe8a"  # 7b 正式結果所用的 scan_7b.py
A7_MD5 = b7.A7_MD5
NEW_E, NEW_C = "P6B4e", "P6B4c"
NETS = ["P6B4", "P6B4r", NEW_C, NEW_E]
MAIN = a7.MAIN
DIAG = "comboNP"                             # 診斷條件:combo 拿掉探針誤差(只評估網路)
a7.SPEC[DIAG] = dict(pos=0.5, df=0.0, dose=0.1)
a7.CLABEL[DIAG] = "combo without probe error"
CONDS = list(b7.CONDS)                       # 12 個(有迭代法包絡)
EVAL_CONDS = CONDS + [DIAG]
POS_CONDS = ["posL", "posH", "combo", DIAG, "posX"]
CLEAN = ["ideal", "doseL", "doseH", "prbL"]
# ---- 位置步(§16.1)----
BETA_SCALE = 4.0                             # β_k = 4·sigmoid(b_k);b_k = 0 → β = 2(= scan_7a.POS_BETA)
INC_MAX = 1.0                                # 每次增量每個分量最多 1 px
POS_MAX = 6.0                                # 累積 ±6 px(= scan_7a.POS_MAX)
REG_REL = 1e-4                               # 正則化 = 1e-4 · S0(同 scan_7a._pos_step)
LAMBDA_POS = 0.1                             # 位置監督的權重(px²)
# ---- 訓練(§16.2)----
P_POS_ON, P_PRB_ON = 2.0 / 3.0, 2.0 / 3.0   # 各自以 2/3 的機率加誤差
LR_OLD, LR_NEW = g.LR_INIT, g.LR_NEW         # 已訓練的權重 2e-4;新參數 b_k 2e-3
ERR_SEED_BASE = 1_600_000_000
EVAL_ERR_SEED = 1_950_000_000
# ---- 可行性檢查(§16.6b)----
FEAS_CONDS = ["posL", "posH", "combo"]
FEAS_ORDERS = ["pa", "ap"]                   # pa = 位置 → AP;ap = AP → 位置
FEAS_M = [1, 2, 3]
FEAS_STAGES = 10
FEAS_N = 4 if QUICK else 128
FEAS_GATE = 0.5                              # 4 級後的位置 RMS < 起始的 1/2
# ---- 判讀(§16.4)----
E1_GOOD, E1_PART = 0.05, 0.10
E3_GATE = 1.0 / 3.0
QS, Q_MAIN, SPEED_MIN = b7.QS, b7.Q_MAIN, b7.SPEED_MIN
# ---- 檔案 ----
TASKS = [("e", 0), ("e", 1), ("e", 2), ("c", 0), ("c", 1), ("c", 2)]
SMOKE_DIR = RUN_ROOT / f"scan7c_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
SMOKE_TRAIN_N = s5c.N_TRAIN if QUICK else 2000
SMOKE_EPOCHS = 1
FIG_DIR = Path("figs_scan7c")
COL = {"P6B4": "#9a9994", "P6B4r": "#e8a33d", NEW_C: "#2a78d6", NEW_E: "#1baf7a", "iter": "#0b0b0b"}

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and b7.all_ok()


md5, dump_json, absdiff = a7.md5, a7.dump_json, a7.absdiff


def me_md5():
    return md5(Path(__file__).resolve())


def feas_json(root=None):
    return (root or RUN_ROOT) / f"scan7c_feas{SUF}.json"


def final_json(root=None):
    return (root or RUN_ROOT) / f"scan7c{SUF}.json"


class no_tf32:
    """暫時關掉 TF32(GPU 上的卷積 / 矩陣乘法預設用 TF32,不同計算路徑的捨入差可達 ~3e-5)。只用於「兩條路徑是否等價」的檢查。"""

    def __enter__(self):
        self.s = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *a):
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = self.s


EQ_TOL = 1e-4                                # 等價檢查(關掉 TF32):實作錯誤(例如分母錯)造成的差 ≫ 1e-4


def model_dir(kind, s, mroot=None):
    name = NEW_E if kind == "e" else NEW_C
    return (mroot / f"{name}_s{s}") if mroot is not None else s5c.run_dir(name, PROBE, s, a6.N_TRAIN)


# ============================================================================
# 位置步與平移探針的 AP 步(可微分;方框與整張共用同一套函式)
# ============================================================================
class SGeo:
    """一組幾何:視窗位置 starts、物體大小 F、小 CNN 的權重圖 wn、AP 的 δ。"""

    def __init__(self, starts, F, wn, delta_ap):
        self.starts, self.F, self.wn, self.delta_ap = list(starts), F, wn, delta_ap


def box_geo(net):
    return SGeo(net.starts, net.wsum.shape[-1], net.wn, net.delta)


def field_geo(net, geo, bs, dev):
    gs = a6.GlobalStages(net, geo.starts, geo.F, geo.pr, bs, dev)
    return SGeo(geo.starts, geo.F, gs.wn, gs.delta)


def windows(O, starts, W):
    return torch.stack([O[:, y0:y0 + W, x0:x0 + W] for y0, x0 in starts], 1)


def fourier_proj(psi, meas, bs, eps):
    """傅立葉約束(同 P6Net.ap_step:E·meas/√(|E|² + ε),beamstop 內保留模型)。"""
    E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    mag = torch.sqrt(E.real * E.real + E.imag * E.imag + eps)
    E2 = torch.where(bs > 0, E * (meas / mag), E)
    return torch.fft.ifft2(torch.fft.ifftshift(E2, dim=(-2, -1)), norm="ortho")


def pos_gn(Sf, Oj, meas, bs, ops, eps, reg):
    """2 × 2 Gauss–Newton(= scan_7a._pos_step 的算法,不乘 β、不截):ψ(δ + Δ) ≈ ψ − Δ·∇P·O,
    以傅立葉約束後的 ψ' 為目標做最小平方(實數位移:Re(JᴴJ) + reg、Re(Jᴴr))。Sf:平移後探針的頻譜 [N, J, W, W]。回傳 [N, J, 2]。"""
    Pj = torch.fft.ifft2(Sf)
    psi = Pj * Oj
    dd = fourier_proj(psi, meas, bs, eps) - psi
    gy = torch.fft.ifft2(Sf * ops.Dy) * Oj
    gx = torch.fft.ifft2(Sf * ops.Dx) * Oj
    a = (gy.real ** 2 + gy.imag ** 2).sum((-2, -1)) + reg
    c = (gx.real ** 2 + gx.imag ** 2).sum((-2, -1)) + reg
    b = (gy.conj() * gx).real.sum((-2, -1))
    uy = -(gy.conj() * dd).real.sum((-2, -1))
    ux = -(gx.conj() * dd).real.sum((-2, -1))
    det = a * c - b * b
    return torch.stack([(c * uy - b * ux) / det, (a * ux - b * uy) / det], -1)


def limit_update(delta, raw):
    """增量每個分量截在 ±INC_MAX → 累積截在 ±POS_MAX → 減平均(零平均是精確的)。回傳 (新 δ, 飽和比例)。"""
    inc = raw.clamp(-INC_MAX, INC_MAX)
    sat = (raw.detach().abs() > INC_MAX).float().mean()
    d = (delta + inc).clamp(-POS_MAX, POS_MAX)
    return d - d.mean(1, keepdim=True), sat


def ap_shift(O, meas, Pj, sg, bs, eps):
    """AP 重疊投影,探針為各位置各自平移的 Pj [N, J, W, W];分母 = Σ_j |Pj|² + δ_AP(逐場)。不含物體約束。"""
    W = Pj.shape[-1]
    Oj = windows(O, sg.starts, W)
    psi2 = fourier_proj(Pj * Oj, meas, bs, eps)
    gg = Pj.conj() * psi2
    wj = Pj.real ** 2 + Pj.imag ** 2
    num = sg.delta_ap * O
    den = torch.full_like(O.real, sg.delta_ap)
    for j, (y0, x0) in enumerate(sg.starts):
        num[:, y0:y0 + W, x0:x0 + W] = num[:, y0:y0 + W, x0:x0 + W] + gg[:, j]
        den[:, y0:y0 + W, x0:x0 + W] = den[:, y0:y0 + W, x0:x0 + W] + wj[:, j]
    return num / den


class P6B4e(nn.Module):
    """P6Net(P6B4)外加每一級的位置步。use_pos = False 時前向 = P6B4(內建檢查)。"""

    def __init__(self, cfg, pr, dev, order="pa", m=1, use_pos=True):
        super().__init__()
        self.net = h.build_model("P6B4", cfg, pr).to(dev)
        self.K = self.net.K
        self.b = nn.Parameter(torch.zeros(self.K, device=dev))
        self.order, self.m, self.use_pos = order, int(m), use_pos
        self.ops = a7.Ops(pr, cfg.canvas, dev)
        self.reg = REG_REL * self.ops.S0
        self.beta_override = None                                       # 檢查用

    def betas(self):
        return [float(v) for v in BETA_SCALE * torch.sigmoid(self.b.detach())]

    def beta(self, k):
        if self.beta_override is not None:
            return self.beta_override
        return BETA_SCALE * torch.sigmoid(self.b[k])

    def pos_update(self, k, O, meas, sg, delta, bs, beta=None):
        bk = self.beta(k) if beta is None else beta
        W = self.net.W
        sats = []
        for _ in range(self.m):
            Oj = windows(O, sg.starts, W)
            Sf = self.ops.P0f * self.ops.ramp(delta[..., 0], delta[..., 1])
            d = pos_gn(Sf, Oj, meas, bs, self.ops, self.net.eps_e, self.reg)
            delta, sat = limit_update(delta, bk * d)
            sats.append(sat)
        return delta, torch.stack(sats).mean()

    def stages(self, O, meas, sg, bs, delta=None, n_stages=None, cnn=True, alpha=None, beta=None):
        """回傳 (各級輸出 [amp, ph] 的清單, 各級的 δ 清單, 各級的飽和比例)。n_stages > K 只用於可行性檢查(不用 CNN)。"""
        net = self.net
        n_stages = n_stages or self.K
        N, J = O.shape[0], len(sg.starts)
        if delta is None:
            delta = torch.zeros(N, J, 2, device=O.device)
        outs, ds, sats = [], [], []
        for k in range(n_stages):
            sat = torch.zeros((), device=O.device)
            if self.use_pos and self.order == "pa":
                delta, sat = self.pos_update(k, O, meas, sg, delta, bs, beta)
            al = (torch.sigmoid(net.a[k]) if net.alpha_override is None else net.alpha_override) if alpha is None else alpha
            if self.use_pos:
                Pj = torch.fft.ifft2(self.ops.P0f * self.ops.ramp(delta[..., 0], delta[..., 1]))
                O = O + al * (ap_shift(O, meas, Pj, sg, bs, net.eps_e) - O)
            else:
                O = O + al * (ap_plain(net, O, meas, sg, bs) - O)
            if self.use_pos and self.order == "ap":
                delta, sat = self.pos_update(k, O, meas, sg, delta, bs, beta)
            if cnn:
                d = net.refine[k](torch.cat([O.real[:, None], O.imag[:, None], sg.wn.expand(O.shape[0], -1, -1, -1)], 1))
                O = O + torch.complex(d[:, 0], d[:, 1])
            amp, ph = net.project(O)
            O = torch.polar(amp, ph)
            outs.append(torch.stack([amp, ph], 1))
            ds.append(delta)
            sats.append(sat)
        return outs, ds, sats

    def forward(self, inp, all_stages=False, with_delta=False):
        O = self.net.initial(inp)
        O0 = O
        outs, ds, sats = self.stages(O, inp["meas"], box_geo(self.net), self.net.bs)
        if with_delta:
            return O0, outs, ds, sats
        return (O0, outs) if all_stages else outs[-1]


def ap_plain(net, O, meas, sg, bs):
    """未平移探針的 AP 步(= P6Net.ap_step / GlobalStages.ap_step,幾何由 sg 給)。"""
    W = net.W
    Oj = windows(O, sg.starts, W)
    psi2 = fourier_proj(net.P * Oj, meas, bs, net.eps_e)
    gg = net.Pc * psi2
    num = sg.delta_ap * O
    wsum = torch.zeros_like(O.real[0])
    for j, (y0, x0) in enumerate(sg.starts):
        num[:, y0:y0 + W, x0:x0 + W] = num[:, y0:y0 + W, x0:x0 + W] + gg[:, j]
        wsum[y0:y0 + W, x0:x0 + W] += net.P.real ** 2 + net.P.imag ** 2
    return num / (wsum + sg.delta_ap)


# ============================================================================
# 整張的 G9 流程(同 scan_6a.run_pipe 的 G9;第 ② 段改用 P6B4e 的級)
# ============================================================================
@torch.no_grad()
def g9_draft(net, norm, d):
    geo, cp, bs, counts = d["geo"], d["cp"], d["bs"], d["counts"]
    st = a6.net_starts(net, norm, counts, geo, cp, bs, a6.PATCH9)
    origins = [geo.origin(i, j) for (i, j) in a6.PATCH9]
    O0, cover, _ = a6.merge(st, origins, geo.R0box, geo.Wbox, geo.F, sync=True)
    a0 = s5b.const_amp(counts[a6.CEN49], bs, cp, geo.pr)
    c = torch.polar(a0[:, None, None].expand(O0.shape).contiguous(), torch.full(O0.shape, net.phase_max / 2, device=O0.device))
    return torch.where(cover, O0, c)


@torch.no_grad()
def run_g9e(model, norm, d, O0=None, keep_delta=False, sg=None, **kw):
    """P6B4e 的整張 G9:9 塊草稿(名目位置)→ 合併 → 4 級(49 個位置各估各的 δ)。
    sg:預先建好的整張幾何(計時時重用,同 scan_7a.timing 預建 GlobalStages)。
    回傳 (整張, 各級 δ [級, n, 49, 2] 或 None, 各級的飽和比例(依場數加權))。"""
    geo, bs = d["geo"], d["bs"]
    dev = bs.device
    O0 = g9_draft(model.net, norm, d) if O0 is None else O0
    sg = sg or field_geo(model.net, geo, bs, dev)
    meas = a6.measured(d["counts"], d["cp"], dev)
    outs, dl, sat = [], [], None
    for i in range(0, O0.shape[0], a6.MCHUNK):
        o, ds, ss = model.stages(O0[i:i + a6.MCHUNK].clone(), meas[i:i + a6.MCHUNK], sg, bs, **kw)
        outs.append(torch.polar(o[-1][:, 0], o[-1][:, 1]))
        w = torch.stack(ss) * o[-1].shape[0]
        sat = w if sat is None else sat + w
        if keep_delta:
            dl.append(torch.stack(ds))                                   # [級, n, 49, 2]
    return torch.cat(outs), (torch.cat(dl, 1) if keep_delta else None), [float(v) for v in sat / O0.shape[0]]


def draft_offset_rms(d):
    """草稿的位置誤差:G9 的 9 塊草稿各自用 3 × 3 的名目位置,每塊的內容約略偏移「該塊 9 個視窗真實位移的平均」;
    回傳各塊偏移(減去 9 塊的平均)的 RMS(px,每個場再平均)。"""
    want = true_delta(d)
    geo = d["geo"]
    off = torch.stack([want[:, geo.nbr(i, j)].mean(1) for (i, j) in a6.PATCH9], 1)       # [n, 9, 2]
    off = off - off.mean(1, keepdim=True)
    return float(off.pow(2).sum(-1).mean(-1).sqrt().mean())


def true_delta(d):
    """整張 49 個位置的真實位移(減平均;= scan_7a.measure_condition 的 want)。"""
    w = d["spec"]["pos"] * d["z"]
    return w - w.mean(1, keepdim=True)


def pos_rms(delta, want):
    """每個場的位置 RMS(49 個位置)再平均。兩者都已減平均 → 與整體平移(規範)無關。"""
    return float((delta - want).pow(2).sum(-1).mean(-1).sqrt().mean())


# ============================================================================
# 載入
# ============================================================================
def load_e(s, cfg, pr, dev, mroot=None, kind="e"):
    """P6B4e:回傳 (模型, 輸入正規化, result.json)。P6B4c:回傳 (P6Net, 輸入正規化, result.json)。"""
    md = model_dir(kind, s, mroot)
    meta = json.load(open(md / "result.json"))
    if kind == "c":
        net = h.build_model("P6B4", cfg, pr).to(dev)
        net.load_state_dict(torch.load(md / "final.pt", map_location=dev))
        return net.eval(), meta["norm"], meta
    model = P6B4e(cfg, pr, dev, order=meta["order"], m=meta["m"])
    model.load_state_dict(torch.load(md / "final.pt", map_location=dev))
    return model.eval(), meta["norm"], meta


def load_p6b4_into(model, s, dev):
    """把 P6B4 的全部權重載入 P6B4e 的內部網路(起點)。"""
    md = s5c.run_dir("P6B4", PROBE, s, a6.N_TRAIN)
    model.net.load_state_dict(torch.load(md / "final.pt", map_location=dev))
    return md


# ============================================================================
# 訓練的誤差抽樣與量測
# ============================================================================
def draw_errors_7c(B, gen):
    """位置、探針誤差各自以 2/3 的機率加(旗標);劑量同 7b。回傳 δ [B, 9, 2](減平均)、探針庫編號、劑量、σ、兩個旗標。"""
    sig = torch.rand(B, generator=gen) * b7.TR_POS_MAX
    z = torch.randn(B, 9, 2, generator=gen).clamp(-b7.Z_CLIP, b7.Z_CLIP)
    idx = torch.randint(0, len(b7.DF_GRID), (B,), generator=gen)
    hi = torch.rand(B, generator=gen) < b7.TR_DOSE_P1
    u = torch.rand(B, generator=gen)
    pos_on = torch.rand(B, generator=gen) < P_POS_ON
    prb_on = torch.rand(B, generator=gen) < P_PRB_ON
    dose = torch.where(hi, torch.ones(B), 10.0 ** (b7.TR_DOSE_LOGMIN * u))
    d = sig[:, None, None] * z * pos_on[:, None, None].float()
    d = d - d.mean(1, keepdim=True)
    idx = torch.where(prb_on, idx, torch.full_like(idx, b7.DF_GRID.index(0.0)))
    return d, idx, dose, sig, pos_on, prb_on


class TrainMeas7c(b7.TrainMeas):
    def __call__(self, obj, err_seed, noise_seed):
        gc = torch.Generator().manual_seed(int(err_seed))
        d, idx, dose, sig, pon, qon = draw_errors_7c(obj.shape[0], gc)
        self.gen.manual_seed(int(noise_seed))
        cnt = b7.measure_rand(obj, self.rel, self.Pf, self.ops, self.cp, self.bs, d, idx, dose, self.gen)
        return cnt, (d, idx, dose, sig, pon, qon)


# ============================================================================
# 訓練(同 scan_7b.train_r 的骨架;起點、配比、學習率、位置監督依 §16.2)
# ============================================================================
def nerr_on(model, sub, counts, norm, cp, bs, R0):
    pred = g.predict(model, counts, bs, norm, "P6B4", cp)
    O = torch.polar(sub[:, 0], sub[:, 1])
    Et = ((O.abs() ** 2) * R0).sum((1, 2))
    tr = [float(s5c.nerr_loss(pred[i:i + 1], O[i:i + 1], R0)) for i in range(len(sub)) if Et[i] > 1e-12]
    return float(np.mean(tr))


def train_7c(kind, seed, dev, feas, n_fields=None, epochs=None, out=None):
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
    norm = json.load(open(s5c.run_dir("B", PROBE, seed, a6.N_TRAIN) / "result.json"))["norm"]
    fields = s5c.make_train_fields(cfg, n_fields, seed).to(dev)
    name = NEW_E if kind == "e" else NEW_C
    print(f"[init] {name} s{seed}:訓練場 {len(fields)} 個({time.time() - t0:.1f}s)", flush=True)
    order, m = feas["order"], feas["m"]
    model = P6B4e(cfg, pr, dev, order=order, m=m, use_pos=(kind == "e"))
    p_md = load_p6b4_into(model, seed, dev)
    ref, rnorm, _ = a6.load_p6b4(seed, cfg, pr, dev)
    model.eval()
    with torch.no_grad(), no_tf32():
        cnt = s5c.measure_box(fields[:32], PROBE, pr, cp, bs, cfg, seed=seed + 98)
        inp = g.prep(cnt, bs, norm, "P6B4", cp)
        model.beta_override = 0.0
        e_i = float((torch.polar(*model(inp).unbind(1)) - torch.polar(*ref(inp).unbind(1))).abs().max())
        model.beta_override = None
    check(f"載入檢查:起點 = P6B4 的全部權重(位置步關閉時前向 = P6B4,關掉 TF32 比較 < {EQ_TOL:.0e};輸入正規化 = P6B4 的)",
          e_i < EQ_TOL and rnorm == norm, f"最大差 {e_i:.1e}")
    del ref
    if not all_ok():
        raise SystemExit("❌ 載入檢查未通過,不訓練")
    tm = TrainMeas7c(cfg, pr, cp, bs, dev)
    old = list(model.net.parameters())
    groups = [{"params": old, "lr": LR_OLD}]
    max_lr = [LR_OLD]
    if kind == "e":
        groups.append({"params": [model.b], "lr": LR_NEW})
        max_lr.append(LR_NEW)
    else:
        model.b.requires_grad_(False)
    opt = torch.optim.Adam(groups)
    steps = len(fields) // g.BATCH
    total = epochs * steps + 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total)
    print(f"[init] {epochs} epochs × {steps} 步;位置步 {'開(順序 ' + order + '、每級 ' + str(m) + ' 次)' if kind == 'e' else '關(對照組)'};"
          f"位置 / 探針誤差各 {P_POS_ON:.2f} 的機率;學習率 {LR_OLD:g}(已訓練)/ {LR_NEW:g}(b_k)", flush=True)
    meta = {"group": name, "kind": kind, "arch": "P6B4" if kind == "c" else "P6B4e", "order": order, "m": m, "probe": PROBE,
            "seed": seed, "n_train": n_fields, "epochs": epochs, "batch": g.BATCH, "lr": max_lr, "clip": g.CLIP, "norm": norm,
            "lambda_pos": LAMBDA_POS if kind == "e" else 0.0, "init": "P6B4 all weights", "init_md5": g._md5(p_md / "final.pt"),
            "p_pos_on": P_POS_ON, "p_prb_on": P_PRB_ON, "err_seed_base": ERR_SEED_BASE, "quick": QUICK, "script_md5": me_md5(),
            "feas": {k: feas[k] for k in ("order", "m", "go")}}
    json.dump(meta, open(out / "config_used.json", "w"), indent=2)
    ck = out / "ckpt.pt"
    start, hist = 0, []
    if ck.exists():
        dck = torch.load(ck, map_location=dev, weights_only=False)
        if dck.get("script_md5") != me_md5():
            raise SystemExit(f"❌ {ck} 是由不同版本的 scan_7c.py 存的:不續跑。先告訴 Claude")
        model.load_state_dict(dck["model"])
        opt.load_state_dict(dck["opt"])
        sched.load_state_dict(dck["sched"])
        start, hist = dck["epoch"] + 1, dck["history"]
        print(f"[resume] 從 epoch {start} 續跑", flush=True)
    gstep = start * steps
    K = model.K
    for ep in range(start, epochs):
        model.train()
        gen = torch.Generator().manual_seed(seed * 100_003 + ep)            # 同 B / P6B4 的 batch 順序公式
        perm = torch.randperm(len(fields), generator=gen)
        acc = {"loss": 0.0, "obj": 0.0, "pos": 0.0, "final": 0.0, "stages": [0.0] * K, "pos_rms": [0.0] * K, "sat": [0.0] * K}
        n_ok, n_skip, n_clip, gns, te = 0, 0, 0, [], time.time()
        for i in range(steps):
            idx = perm[i * g.BATCH:(i + 1) * g.BATCH].to(dev)
            obj = fields[idx]
            with torch.no_grad():
                counts, err = tm(obj, ERR_SEED_BASE + seed * 10_000_019 + gstep, s5c.NOISE_SEED_BASE + seed * 10_000_019 + gstep)
                inp = g.prep(counts, bs, norm, "P6B4", cp)
            gstep += 1
            want = err[0].to(dev)
            _, outs, ds, sats = model(inp, with_delta=True)
            l_obj, ls = g.ds_loss(outs, torch.polar(obj[:, 0], obj[:, 1]), R0)
            l_pos = torch.stack([(dk - want).pow(2).sum(-1).mean() for dk in ds]).mean()
            loss = l_obj + (LAMBDA_POS * l_pos if kind == "e" else 0.0)
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
            acc["loss"] += float(loss.detach())
            acc["obj"] += float(l_obj.detach())
            acc["pos"] += float(l_pos.detach())
            acc["final"] += float(ls[-1].detach())
            for k in range(K):
                acc["stages"][k] += float(ls[k].detach())
                acc["pos_rms"][k] += float((ds[k].detach() - want).pow(2).sum(-1).mean(-1).sqrt().mean())
                acc["sat"][k] += float(sats[k])
        for k in ("loss", "obj", "pos", "final"):
            acc[k] /= max(n_ok, 1)
        for k in ("stages", "pos_rms", "sat"):
            acc[k] = [v / max(n_ok, 1) for v in acc[k]]
        acc.update({"epoch": ep, "skipped": n_skip, "clip_frac": n_clip / max(n_ok, 1),
                    "grad_norm_median": float(np.median(gns)) if gns else None, "alphas": model.net.alphas(),
                    "betas": model.betas(), "sec": time.time() - te, "lr": sched.get_last_lr()})
        hist.append(acc)
        print(f"epoch {ep:03d}  loss={acc['loss']:.5f}  物體={acc['obj']:.5f}  位置={acc['pos']:.4f}  各級物體="
              + " ".join(f"{v:.4f}" for v in acc["stages"]) + "  各級位置 RMS=" + " ".join(f"{v:.3f}" for v in acc["pos_rms"])
              + "  α=" + " ".join(f"{v:.3f}" for v in acc["alphas"]) + "  β=" + " ".join(f"{v:.2f}" for v in acc["betas"])
              + f"  飽和 {100 * np.mean(acc['sat']):.1f}%  梯度中位數 {acc['grad_norm_median'] or 0:.2e}(裁切 {100 * acc['clip_frac']:.0f}%)"
              + f"  跳過 {n_skip}  {acc['sec']:.1f}s", flush=True)
        tmp = out / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "history": hist, "script_md5": meta["script_md5"]}, tmp)
        os.replace(tmp, ck)
    model.eval()
    with torch.no_grad():
        sub = fields[:s5c.TRAIN_EVAL_N]
        c_id = s5c.measure_box(sub, PROBE, pr, cp, bs, cfg, seed=seed + 99)
        tr = nerr_on(model, sub, c_id, norm, cp, bs, R0)
        c_er, _ = tm(sub, EVAL_ERR_SEED + seed, seed + 99)
        tr_e = nerr_on(model, sub, c_er, norm, cp, bs, R0)
    meta.update({"history": hist, "train_nerr_ph": tr, "train_nerr_ph_rand": tr_e, "alphas": model.net.alphas(),
                 "betas": model.betas(), "train_sec": time.time() - t0, "skipped_total": sum(x["skipped"] for x in hist)})
    tmp = out / "final.tmp"
    torch.save(model.net.state_dict() if kind == "c" else model.state_dict(), tmp)       # P6B4c:存 P6Net 格式
    json.dump(meta, open(out / "result.json", "w"), indent=2)
    os.replace(tmp, out / "final.pt")                                     # final.pt 最後才出現(= 完成的標記)
    print(f"[done] {name} s{seed}:訓練場 nerr_ph 理想 {tr:.4f} / 隨機誤差 {tr_e:.4f}  β {' '.join(f'{v:.2f}' for v in model.betas())}"
          f"  {time.time() - t0:.0f}s → {out}", flush=True)
    return meta


# ============================================================================
# 可行性檢查(§16.6b;只推論、不訓練)
# ============================================================================
@torch.no_grad()
def feasibility(dev, n=None, log=print):
    n = n or FEAS_N
    res = {"n": n, "conds": FEAS_CONDS, "runs": {}}
    cfgs = [(o, m) for o in FEAS_ORDERS for m in FEAS_M]
    t0 = time.time()
    for c in FEAS_CONDS:
        d = a7.setup_cond(0, dev, c, n)
        geo, O, U = d["geo"], d["O"], d["geo"].U
        net, norm, _ = a6.load_p6b4(0, d["cfg"], geo.pr, dev)
        want = true_delta(d)
        r0 = pos_rms(torch.zeros_like(want), want)
        draft = g9_draft(net, norm, d)
        starts = {"draft": draft, "truth": O.clone()}
        res["runs"][c] = {"rms0": r0, "draft_rms": draft_offset_rms(d), "draft_nerr": float(a7.metrics7(draft, O, U)[MAIN])}
        for (o, m) in cfgs:
            model = P6B4e(d["cfg"], geo.pr, dev, order=o, m=m)
            model.net.load_state_dict(net.state_dict())
            model.eval()
            for sname, S0 in starts.items():
                est, dl, sat = run_g9e(model, norm, d, O0=S0, keep_delta=True, n_stages=FEAS_STAGES, cnn=False, alpha=1.0, beta=2.0)
                rms = [pos_rms(dl[k], want) for k in range(FEAS_STAGES)]
                inc = dl[0]                                                  # 第 1 級的 δ(起始 0)= 第 1 級的更新
                cosv = float((inc * want).sum() / (inc.norm() * want.norm()).clamp_min(1e-12))
                res["runs"][c][f"{sname}|{o}|{m}"] = {"rms": rms, "cos1": cosv, "sat": sat, "nerr10": float(a7.metrics7(est, O, U)[MAIN])}
        log(f"  [可行性] {c} 完成(經過 {time.time() - t0:.0f} 秒)")
        del d, net
    # 判準(事先寫定)
    r0 = res["runs"]["posL"]["rms0"]
    ok = [(m, 0 if o == "pa" else 1, o) for (o, m) in cfgs if res["runs"]["posL"][f"draft|{o}|{m}"]["rms"][3] < FEAS_GATE * r0]
    if ok:
        m, _, o = min(ok)
        res.update({"go": True, "order": o, "m": m})
    else:
        res.update({"go": False, "order": None, "m": None})
    return res


def feas_report(res):
    print("\n" + "=" * 100)
    print(f"可行性檢查(§16.6b;固定公式:β = 2、α = 1、不用 CNN;seed 0、{res['n']} 個場;位置 RMS 單位 px,49 個位置、與整體平移無關)")
    print("=" * 100)
    for c in res["conds"]:
        r = res["runs"][c]
        print(f"\n  ■ {c}:起始位置 RMS {r['rms0']:.3f} px;草稿各塊的平均偏移 RMS {r['draft_rms']:.3f} px;G9 草稿的 {MAIN} {r['draft_nerr']:.4f}")
        print("    起點 | 順序 | 每級次數 | 位置 RMS:第 1 / 2 / 3 / 4 級 … 第 10 級 | 4 級後 / 起始 | 第 1 級更新與真實位移的 cos | "
              "前 4 級的飽和比例 | 10 級後的 " + MAIN)
        for k, v in r.items():
            if "|" not in k:
                continue
            s_, o, m = k.split("|")
            rm = v["rms"]
            print(f"    {s_:<5} | {o} | {m} | {rm[0]:.3f} / {rm[1]:.3f} / {rm[2]:.3f} / {rm[3]:.3f} … {rm[-1]:.3f} | "
                  f"{rm[3] / max(r['rms0'], 1e-12):.2f} | {v['cos1']:+.2f} | {100 * np.mean(v['sat'][:4]):.1f}% | {v['nerr10']:.4f}")
    print("\n  判準:從 P6B4 草稿出發、posL 4 級後的位置 RMS < 起始的 1/2 的設定中,選最省的(每級次數少者優先;同次數選「位置 → AP」)")
    if res["go"]:
        print(f"  ▶ go:訓練用 順序 {res['order']}({'位置 → AP' if res['order'] == 'pa' else 'AP → 位置'})、每級 {res['m']} 次")
    else:
        print("  ▶ no-go:沒有設定達到判準 → 不訓練,把以上輸出完整回報給使用者討論")


# ============================================================================
# 內建檢查(§16.7)
# ============================================================================
def checks(dev):
    torch.backends.cudnn.benchmark = True
    print("=" * 70)
    print("階段七 7c:內建檢查")
    print("=" * 70)
    n = 8 if not QUICK else 4
    cfg, pr, cp, bs = s5c.setup(0, PROBE, dev)
    ref, norm, _ = a6.load_p6b4(0, cfg, pr, dev)
    fb = s5c.make_train_fields(cfg, n, 0).to(dev)
    cnt = s5c.measure_box(fb, PROBE, pr, cp, bs, cfg, seed=3)
    inp = g.prep(cnt, bs, norm, "P6B4", cp)
    with torch.no_grad(), no_tf32():
        # (1) 位置步關閉 = P6B4(方框與整張);關掉 TF32 比較(兩條路徑的加總順序不同)
        mod = P6B4e(cfg, pr, dev)
        mod.net.load_state_dict(ref.state_dict())
        mod.eval()
        mod.beta_override = 0.0
        y_ref = ref(inp)
        e_box = float((torch.polar(*mod(inp).unbind(1)) - torch.polar(*y_ref.unbind(1))).abs().max())
        modc = P6B4e(cfg, pr, dev, use_pos=False)
        modc.net.load_state_dict(ref.state_dict())
        modc.eval()
        e_boxc = float((torch.polar(*modc(inp).unbind(1)) - torch.polar(*y_ref.unbind(1))).abs().max())
        d = a7.setup_cond(0, dev, "ideal", n)
        geo = d["geo"]
        est_ref = a6.run_pipe("G9", ref, norm, d)
        est_e, _, _ = run_g9e(mod, norm, d)
        e_fld = float((est_e - est_ref).abs().max())
        mod.beta_override = None
    with torch.no_grad():                                                 # 描述:TF32 開著時的差(GPU 預設)
        mod.beta_override = 0.0
        t_box = float((torch.polar(*mod(inp).unbind(1)) - torch.polar(*ref(inp).unbind(1))).abs().max())
        mod.beta_override = None
    check(f"位置步關閉(β = 0)時,P6B4e 的方框前向與整張 G9 = P6B4;use_pos = False(對照組)的前向 = P6B4(關掉 TF32 比較,< {EQ_TOL:.0e})",
          e_box < EQ_TOL and e_fld < EQ_TOL and e_boxc < EQ_TOL,
          f"方框 {e_box:.1e}、整張 {e_fld:.1e}、對照組 {e_boxc:.1e}(描述:TF32 開著時方框 {t_box:.1e})")
    # (2) 平移探針的 AP 步
    with torch.no_grad():
        sg = box_geo(mod.net)
        O = mod.net.initial(inp)
        meas = inp["meas"]
        P = mod.net.P
        z = torch.zeros(n, 9, 2, device=dev)
        Pj0 = torch.fft.ifft2(mod.ops.P0f * mod.ops.ramp(z[..., 0], z[..., 1]))
        e_ap0 = float((ap_shift(O, meas, Pj0, sg, bs, mod.net.eps_e) - mod.net.ap_step(O, meas)).abs().max())
        zi = z.clone()
        zi[..., 0], zi[..., 1] = 2.0, -1.0
        Pji = torch.fft.ifft2(mod.ops.P0f * mod.ops.ramp(zi[..., 0], zi[..., 1]))
        Pr = torch.roll(P, (2, -1), dims=(-2, -1)).expand(n, 9, *P.shape)
        e_api = float((ap_shift(O, meas, Pji, sg, bs, mod.net.eps_e) - ap_shift(O, meas, Pr, sg, bs, mod.net.eps_e)).abs().max())
    check("平移探針的 AP 步:δ = 0 時 = P6Net.ap_step(< 1e-5);δ = (2, −1) px 時 = 探針 torch.roll((2, −1))(< 1e-4)",
          e_ap0 < 1e-5 and e_api < 1e-4, f"δ = 0:{e_ap0:.1e};整數平移:{e_api:.1e}")
    # (3) 位置步有效(真值物體、無雜訊)+ 次像素導數
    import copy
    from src.physics import forward_measure
    cp0 = copy.deepcopy(d["cp"])
    cp0.add_poisson = False
    nt = min(n, 4)
    ft = d["fields"][:nt]
    Ot = d["O"][:nt]
    J = len(geo.starts)
    gen = torch.Generator().manual_seed(9)
    dtrue = torch.zeros(nt, J, 2)
    pick = torch.randperm(J, generator=gen)[:12]
    dtrue[:, pick] = torch.rand(nt, 12, 2, generator=gen) * 2 - 1
    dtrue = (dtrue - dtrue.mean(1, keepdim=True)).to(dev)
    ops = mod.ops
    with torch.no_grad():
        cnts = []
        for j, (y0, x0) in enumerate(geo.starts):
            win = s5.crop(ft, y0, x0, geo.W)
            psi = ops.shift(P.expand(nt, geo.W, geo.W), dtrue[:, j, 0], dtrue[:, j, 1]) * torch.polar(win[:, 0], win[:, 1])
            cnts.append(forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), d["bs"], cp0))
        measf = a6.measured(cnts, cp0, dev)
        sgf = field_geo(mod.net, geo, d["bs"], dev)
        delta = torch.zeros(nt, J, 2, device=dev)
        r = [pos_rms(delta, dtrue)]
        modp = P6B4e(d["cfg"], geo.pr, dev, m=1)
        for _ in range(4):
            delta, _ = modp.pos_update(0, Ot, measf, sgf, delta, d["bs"], beta=2.0)
            r.append(pos_rms(delta, dtrue))
    # 次像素導數:∂ψ/∂δ_y(解析 −∇_y P·O)vs 有限差分
    hh = 1e-2
    Oj = windows(Ot, geo.starts, geo.W)[:, :3]
    dy0 = torch.full((nt, 3), 0.37, device=dev)
    dx0 = torch.full((nt, 3), -0.21, device=dev)
    psi_p = torch.fft.ifft2(ops.P0f * ops.ramp(dy0 + hh, dx0)) * Oj
    psi_m = torch.fft.ifft2(ops.P0f * ops.ramp(dy0 - hh, dx0)) * Oj
    fd = (psi_p - psi_m) / (2 * hh)
    an = -torch.fft.ifft2(ops.P0f * ops.ramp(dy0, dx0) * ops.Dy) * Oj
    e_fd = float((fd - an).abs().max() / an.abs().max())
    check("位置步有效(真值物體、無雜訊,12 個位置偏 ± 1 px):每步單調下降、4 步後位置 RMS < 起始的 1/3;"
          "次像素導數(解析 −∇P·O)= 有限差分(相對差 < 1e-3)",
          all(r[k + 1] < r[k] for k in range(4)) and r[4] < r[0] / 3 and e_fd < 1e-3,
          "位置 RMS(px):" + " → ".join(f"{v:.3f}" for v in r) + f";導數相對差 {e_fd:.1e}")
    # (4) 整張 vs 方框:整張物體只用方框那 9 個位置時,結果 = 方框版
    with torch.no_grad():
        b0, S, rel = s5c.geometry(cfg)
        F = geo.F
        Of = torch.polar(torch.rand(n, F, F, device=dev) * 0.5 + 0.5, torch.rand(n, F, F, device=dev))
        Ob = Of[:, b0:b0 + S, b0:b0 + S].clone()
        st_f = [(y + b0, x + b0) for (y, x) in rel]
        wsum = torch.zeros(F, F, device=dev)
        for (y0, x0) in st_f:
            wsum[y0:y0 + geo.W, x0:x0 + geo.W] += pr["Pa"] ** 2
        sg_f = SGeo(st_f, F, (wsum / wsum.max())[None, None], mod.net.delta)
        dl0 = (torch.randn(n, 9, 2, device=dev) * 0.4)
        dl0 = dl0 - dl0.mean(1, keepdim=True)
        mod.beta_override = None
        ob, db, _ = mod.stages(Ob, meas, box_geo(mod.net), bs, delta=dl0.clone(), cnn=False)        # CNN 的感受野會跨過方框邊界
        of, df, _ = mod.stages(Of, meas, sg_f, bs, delta=dl0.clone(), cnn=False)
        e_bf = max(float((db[-1] - df[-1]).abs().max()),
                   float((torch.polar(*ob[-1].unbind(1)) - torch.polar(*of[-1].unbind(1))[:, b0:b0 + S, b0:b0 + S]).abs().max()))
    check("整張 vs 方框:整張物體只用方框那 9 個位置時,位置步與 AP 步(不含 CNN)的結果 = 方框版(< 1e-4)", e_bf < 1e-4, f"最大差 {e_bf:.1e}")
    # (5) 可微分:位置 loss → b_k 與 U-Net;物體 loss → b_k
    mod.train()
    mod.zero_grad()
    want = torch.zeros(n, 9, 2, device=dev)
    want[:, :, 0] = torch.linspace(-0.5, 0.5, 9, device=dev)
    want = want - want.mean(1, keepdim=True)
    _, outs, ds, _ = mod(inp, with_delta=True)
    lp = torch.stack([(dk - want).pow(2).sum(-1).mean() for dk in ds]).mean()
    lp.backward(retain_graph=True)
    gb = float(mod.b.grad.abs().sum())
    gu = float(sum(p.grad.abs().sum() for p in mod.net.init.parameters() if p.grad is not None))
    finite = all(torch.isfinite(p.grad).all() for p in mod.parameters() if p.grad is not None)
    mod.zero_grad()
    lo, _ = g.ds_loss(outs, torch.polar(fb[:, 0], fb[:, 1]), s5c.r0_box(cfg, pr, dev))
    lo.backward()
    gbo = float(mod.b.grad.abs().sum())
    mod.zero_grad()
    mod.eval()
    check("可微分:位置 loss 對 b_k 與 U-Net 的梯度有限且非零;物體 loss 也經由 δ 回傳到 b_k", finite and gb > 0 and gu > 0 and gbo > 0,
          f"|∂L_pos/∂b| {gb:.2e}、|∂L_pos/∂U-Net| {gu:.2e}、|∂L_obj/∂b| {gbo:.2e}")
    # (6) 訓練的誤差抽樣
    N = 30000
    dd, idx, dose, sig, pon, qon = draw_errors_7c(N, torch.Generator().manual_seed(11))
    d2 = draw_errors_7c(N, torch.Generator().manual_seed(11))
    rep = all(torch.equal(a, b) for a, b in zip((dd, idx, dose, pon, qon), (d2[0], d2[1], d2[2], d2[4], d2[5])))
    fr = {"none": float((~pon & ~qon).float().mean()), "pos": float((pon & ~qon).float().mean()),
          "prb": float((~pon & qon).float().mean()), "both": float((pon & qon).float().mean())}
    exp = {"none": 1 / 9, "pos": 2 / 9, "prb": 2 / 9, "both": 4 / 9}
    i0 = b7.DF_GRID.index(0.0)
    zero_ok = bool((dd[~pon].abs().max() == 0) and (idx[~qon] == i0).all())
    on_cov = int(torch.unique(idx[qon]).numel())
    zm = float(dd.mean(1).abs().max())
    check("訓練的誤差抽樣:同 seed 可重現;四種組合 ≈ 1/9、2/9、2/9、4/9(各 ± 2%);關閉位置誤差 → δ = 0、關閉探針誤差 → 名目探針(旗標);"
          "開啟探針誤差時 61 個探針都會抽到;每個樣本的 δ 平均 = 0",
          rep and all(abs(fr[k] - exp[k]) < 0.02 for k in fr) and zero_ok and on_cov == len(b7.DF_GRID) and zm < 1e-5,
          "、".join(f"{k} {v:.3f}" for k, v in fr.items()) + f";關閉時正確 {zero_ok};探針種類 {on_cov};平均最大 {zm:.1e}")
    # (7) 限幅
    delta = torch.zeros(2, 5, 2)
    delta[0, 0] = torch.tensor([5.5, -5.8])
    raw = torch.tensor([[[3.0, -0.4], [0.2, 9.0], [-7.0, 0.1], [0.5, 0.5], [-0.3, -2.0]]] * 2)
    nd, sat = limit_update(delta, raw)
    inc_ok = bool(((nd - delta) - (nd - delta).mean(1, keepdim=True)).abs().max() < 1e9)
    pre = (delta + raw.clamp(-INC_MAX, INC_MAX)).clamp(-POS_MAX, POS_MAX)
    lim_ok = (float(nd.mean(1).abs().max()) < 1e-6 and float(pre.abs().max()) <= POS_MAX and
              float((pre - delta).abs().max()) <= INC_MAX + POS_MAX and float(sat) > 0 and inc_ok
              and torch.allclose(nd, pre - pre.mean(1, keepdim=True)))
    check(f"位置更新的限幅(非對稱飽和案例):增量每分量截 ±{INC_MAX:g} → 累積截 ±{POS_MAX:g} → 減平均後平均 = 0;飽和比例 > 0",
          lim_ok, f"減平均後的平均最大 {float(nd.mean(1).abs().max()):.1e}、累積最大 {float(pre.abs().max()):.2f}、飽和 {float(sat):.2f}")
    print(f"\n     裝置:{dev}({a7.gpu_name(dev)})")


# ============================================================================
# 彙整
# ============================================================================
@torch.no_grad()
def time_nets(dev, mroot, r_mroot):
    """4 個網路 G9 的端到端時間(ms / 整張;同 scan_7a.timing 的量法:幾何預先建好、暖機 2 次、取中位數)。回傳 {網路: {b: ms}}。"""
    torch.backends.cudnn.benchmark = True
    res = {nm: {} for nm in NETS}
    for Bl in ["64", "512"]:
        B = int(Bl) if not QUICK else {"64": 4, "512": 8}[Bl]
        d = a7.setup_cond(0, dev, "ideal", B)
        geo, cfg, pr = d["geo"], d["cfg"], d["geo"].pr
        for nm in NETS:
            if nm == NEW_E:
                model, norm, _ = load_e(0, cfg, pr, dev, mroot, kind="e")
                sg = field_geo(model.net, geo, d["bs"], dev)
                fn = (lambda model=model, norm=norm, sg=sg, d=d: run_g9e(model, norm, d, sg=sg))
            else:
                if nm == "P6B4":
                    net, norm, _ = a6.load_p6b4(0, cfg, pr, dev)
                elif nm == "P6B4r":
                    net, norm = b7.load_net("P6B4r", 0, cfg, pr, dev, r_mroot)
                else:
                    net, norm, _ = load_e(0, cfg, pr, dev, mroot, kind="c")
                gs = a6.GlobalStages(net, geo.starts, geo.F, pr, d["bs"], dev)
                fn = (lambda net=net, norm=norm, gs=gs, d=d: a6.run_pipe("G9", net, norm, d, gs))
            fn()
            fn()
            res[nm][Bl] = a7._time_it(fn, B, dev, 1)
        del d
    return res


@torch.no_grad()
def eval_all(dev, C, mroot, r_root, log=print):
    """13 個條件 × 3 seeds × 4 個網路(G9):三種誤差 + SSIM;P6B4e 另記各級的位置 RMS 與飽和。"""
    R = {c: {nm: {k: [] for k in a7.MKEYS + ("ssim_amp", "ssim_ph")} for nm in NETS} for c in EVAL_CONDS}
    P = {c: {"rms0": [], "draft_rms": [], "rms": [], "sat": []} for c in POS_CONDS}
    keep = {"img": {}}
    t0 = time.time()
    n = C[CONDS[0]]["meta"]["n"]
    for c in EVAL_CONDS:
        for s in SEEDS:
            dd = a7.setup_cond(s, dev, c, n)
            geo, O, U = dd["geo"], dd["O"], dd["geo"].U
            phmax = dd["cfg"].phase_max
            for nm in NETS:
                dl = None
                if nm == "P6B4":
                    net, norm, _ = a6.load_p6b4(s, dd["cfg"], geo.pr, dev)
                    est = a6.run_pipe("G9", net, norm, dd)
                elif nm == "P6B4r":
                    net, norm = b7.load_net("P6B4r", s, dd["cfg"], geo.pr, dev, None if r_root == RUN_ROOT else r_root)
                    est = a6.run_pipe("G9", net, norm, dd)
                elif nm == NEW_C:
                    net, norm, _ = load_e(s, dd["cfg"], geo.pr, dev, mroot, kind="c")
                    est = a6.run_pipe("G9", net, norm, dd)
                else:
                    net, norm, _ = load_e(s, dd["cfg"], geo.pr, dev, mroot, kind="e")
                    est, dl, sat = run_g9e(net, norm, dd, keep_delta=True)
                m = a7.metrics7(est, O, U, keep_ps=(s == 0))
                sa, sp = b7.ssim_mean(est, O, U, phmax)
                for k in a7.MKEYS:
                    R[c][nm][k].append(m[k])
                R[c][nm]["ssim_amp"].append(sa)
                R[c][nm]["ssim_ph"].append(sp)
                if s == 0:
                    R[c][nm]["ps_sr0"] = m["ps_sr"]
                    if c in ("posL", "combo", "posH", "ideal"):
                        keep["img"].setdefault(c, {})[nm] = est.cpu()
                if nm == NEW_E and c in POS_CONDS:
                    want = true_delta(dd)
                    P[c]["rms0"].append(pos_rms(torch.zeros_like(want), want))
                    P[c]["rms"].append([pos_rms(dl[k], want) for k in range(dl.shape[0])])
                    P[c]["draft_rms"].append(draft_offset_rms(dd))
                    P[c]["sat"].append(sat)
                del net
            if s == 0 and c in keep["img"]:
                keep["img"][c]["O"] = O.cpu()
                if c == "ideal":
                    keep["geo"], keep["cfg"] = geo, dd["cfg"]
            del dd
        log(f"  [網路評估] {c} 完成(經過 {time.time() - t0:.0f} 秒)")
    return R, P, keep


def report(C, R, P, tim, iters, smoke):
    Bs = ["64", "512"]
    V = {"cond": {}}
    if smoke:
        print("\n" + "!" * 100)
        print("迷你流程:訓練極少、場數少、停止點 ≤ 20 → 數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    print("\n" + "=" * 100)
    print(f"計時(ms / 整張;本 job 重新量;GPU {tim['gpu']})")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:G9 " + "、".join(f"{nm} {tim['net'][nm][B]:.4f}" for nm in NETS)
              + f"(P6B4e / P6B4 × {tim['net'][NEW_E][B] / tim['net']['P6B4'][B]:.2f});AP-C 每次 {tim[B]['AP-C']:.4f}"
              + f";參考:scan_7a.timing 的 P6B4 G9 {tim[B]['net:G9']:.4f}")
    print("\n" + "=" * 100)
    print(f"各條件(G9;{MAIN};3 seeds;SSIM 振幅 / 相位;加速倍數 [下界–上界] = 迭代法包絡最早達到 Q 的時間 / 該網路的時間)")
    print("=" * 100)
    for c in EVAL_CONDS:
        V["cond"][c] = {}
        tag = "(只測試)" if c in b7.HELDOUT else ("(超出訓練範圍)" if c in b7.OUTRANGE else ("(診斷用,沒有迭代法包絡)" if c == DIAG else ""))
        print(f"\n  ■ {c}({a7.CLABEL[c]}){tag}")
        for nm in NETS:
            a = np.array(R[c][nm][MAIN], float)
            row = {"nerr": list(a), "ssim_amp": R[c][nm]["ssim_amp"], "ssim_ph": R[c][nm]["ssim_ph"],
                   "n_fail": int(np.sum(R[c][nm]["n_fail"]))}
            line = (f"    {nm:<6}:{a.mean():.4f}({' '.join(f'{x:.4f}' for x in a)});SSIM {np.mean(row['ssim_amp']):.3f} / "
                    f"{np.mean(row['ssim_ph']):.3f}" + (f";發散 {row['n_fail']} 個" if row["n_fail"] else ""))
            if c != DIAG:
                env = C[c]["env"]
                for B in Bs:
                    tn = tim["net"][nm][B]                                 # 各網路用自己的實測時間
                    e = a7.envelope7(env, tn, tim[B], iters)
                    r = s5b.ratio(a, e[2])
                    sr = b7.speed_row(a.mean(), env, tim[B], tn, iters)
                    row[B] = {"t": tn, "opp": [e[0], e[1], list(e[2]), e[3]], "ratio": list(r), "speed": sr}
                    line += (f"\n       b{B}:{tn:.3f} ms;同時間包絡 {np.mean(e[2]):.4f} → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]};加速 "
                             + "、".join(f"Q {q}:{b7.fmt_x(sr[str(q)])}" for q in QS))
            print(line)
            V["cond"][c][nm] = row
    # ---- 位置估計 ----
    print("\n" + "=" * 100)
    print("位置估計(P6B4e,整張 49 個位置;RMS 單位 px;估計與真值都已減平均 → 最佳整體平移 = 0,即「扣掉整體平移後的 RMS」)")
    print("=" * 100)
    V["pos"] = {}
    for c in POS_CONDS:
        r0 = float(np.mean(P[c]["rms0"]))
        rk = np.mean(np.array(P[c]["rms"], float), 0)
        ratio = float(rk[-1] / max(r0, 1e-12))
        dr = float(np.mean(P[c]["draft_rms"]))
        st = list(np.mean(np.array(P[c]["sat"], float), 0))
        V["pos"][c] = {"rms0": r0, "rms": list(rk), "ratio": ratio, "pass": bool(ratio < E3_GATE), "draft_rms": dr, "sat": st}
        print(f"    {c:<7} 起始 {r0:.3f} → 各級 " + " / ".join(f"{v:.3f}" for v in rk)
              + f";最後 / 起始 {ratio:.2f} → {'有效估計' if ratio < E3_GATE else '未達有效估計'}(門檻 < {E3_GATE:.2f})"
              + f";草稿各塊的平均偏移 RMS {dr:.3f};各級飽和 " + " / ".join(f"{100 * v:.1f}%" for v in st))
    # ---- 判讀 ----
    print("\n" + "=" * 100)
    print("判讀(階段七協定 §16.4 / §16.6,結果出來前寫定)")
    print("=" * 100)
    eL = float(np.mean(V["cond"]["posL"][NEW_E]["nerr"]))
    e3L = V["pos"]["posL"]["pass"]
    if eL < E1_GOOD and e3L:
        e1 = "位置步有效"
    elif eL < E1_GOOD:
        e1 = "影像達標、但位置沒估對(不算位置步有效)"
    elif eL < E1_PART:
        e1 = "部分有效"
    else:
        e1 = "無效"
    V["e1"] = {"posL": eL, "e3": e3L, "label": e1}
    print(f"  (e1 主判)posL 下 P6B4e 的 G9 {eL:.4f}(門檻 {E1_GOOD});位置估計 {'有效' if e3L else '未達'} → {e1}"
          + ("  [smoke:無意義]" if smoke else ""))
    r2 = V["cond"]["combo"][NEW_E]
    s64, s512 = r2["64"]["speed"][str(Q_MAIN)], r2["512"]["speed"][str(Q_MAIN)]
    if not s64["reach"]:
        e2 = "達不到"
    elif all(x["x_lo"] >= SPEED_MIN for x in (s64, s512)):
        e2 = "穩健加速"
    elif any(x["x_lo"] >= SPEED_MIN for x in (s64, s512)):
        e2 = "達到、只在 " + "、".join(f"b{B}" for B, x in (("64", s64), ("512", s512)) if x["x_lo"] >= SPEED_MIN) + " 穩健加速"
    elif any(x["x"] >= SPEED_MIN for x in (s64, s512)):
        e2 = "達到、加速不確定"
    else:
        e2 = "達到、無明顯加速"
    V["e2"] = e2
    print(f"  (e2)combo 下 P6B4e 的 G9 {np.mean(r2['nerr']):.4f}、Q = {Q_MAIN}:{e2}(b64 {b7.fmt_x(s64)}、b512 {b7.fmt_x(s512)})")
    print("  (e3)位置估計:見上表(posL、combo 為判準;其餘描述)")

    def cmp(a_nm, b_nm, conds):
        out = {}
        for c in conds:
            r = s5b.ratio(np.array(V["cond"][c][a_nm]["nerr"]), np.array(V["cond"][c][b_nm]["nerr"]))
            out[c] = {"ratio": r[1], "z": r[2], "label": {"較差": "退步", "較準": "改善", "相近": "相近"}[r[0]]}
        return out

    V["e4"] = cmp(NEW_E, "P6B4", CLEAN)
    reg = [c for c, v in V["e4"].items() if v["label"] == "退步"]
    print("  (e4 不退步)P6B4e / P6B4:" + ";".join(f"{c} {v['ratio']:.3f}(z {v['z']:+.1f})→ {v['label']}" for c, v in V["e4"].items())
          + (f" → ❌ 退步:{', '.join(reg)}" if reg else " → ✅ 沒有退步"))
    V["e5"] = {"e_vs_c": cmp(NEW_E, NEW_C, ["posL", "posH", "combo", DIAG]), "c_vs_r": cmp(NEW_C, "P6B4r", ["posL", "posH", "combo", DIAG]
                                                                                             + CLEAN)}
    print("  (e5 位置步的貢獻)P6B4e / P6B4c(≤ 0.8 且 z < −2 = 位置步帶來改善):"
          + ";".join(f"{c} {v['ratio']:.3f}(z {v['z']:+.1f})→ {v['label']}" for c, v in V["e5"]["e_vs_c"].items()))
    print("       另:P6B4c / P6B4r(配比、起點、學習率的貢獻):"
          + ";".join(f"{c} {v['ratio']:.3f}(z {v['z']:+.1f})→ {v['label']}" for c, v in V["e5"]["c_vs_r"].items()))
    V["e6"] = {"vs_P6B4": cmp(NEW_E, "P6B4", b7.HELDOUT + list(b7.OUTRANGE)), "vs_r": cmp(NEW_E, "P6B4r", b7.HELDOUT + list(b7.OUTRANGE))}
    print("  (e6)部分同調、超出範圍:P6B4e / P6B4:" + ";".join(f"{c} {v['ratio']:.3f} → {v['label']}" for c, v in V["e6"]["vs_P6B4"].items())
          + ";P6B4e / P6B4r:" + ";".join(f"{c} {v['ratio']:.3f} → {v['label']}" for c, v in V["e6"]["vs_r"].items()))
    for cx, cin in b7.OUTRANGE.items():
        ex, ei = np.mean(V["cond"][cx][NEW_E]["nerr"]), np.mean(V["cond"][cin][NEW_E]["nerr"])
        print(f"       {cx}:P6B4e {ei:.4f}({cin})→ {ex:.4f}({cx}),× {ex / max(ei, 1e-12):.2f}")
    V["e7"] = {B: tim["net"][NEW_E][B] / tim["net"]["P6B4"][B] for B in Bs}
    print(f"  (e7 時間)P6B4e / P6B4:b64 × {V['e7']['64']:.2f}、b512 × {V['e7']['512']:.2f}(加速倍數已用各自的時間計算)")
    # ---- 探針步的決定(§16.6)----
    cmb = float(np.mean(V["cond"]["combo"][NEW_E]["nerr"]))
    cnp = float(np.mean(V["cond"][DIAG][NEW_E]["nerr"]))
    e3c = V["pos"]["combo"]["pass"]
    if eL >= E1_GOOD:
        pr = "不做(位置步不夠有效,探針步救不了位置問題)"
    elif cmb < Q_MAIN:
        pr = "不需要(combo 已達 0.02)"
    elif e3c and cnp < Q_MAIN:
        pr = "提出探針步(剩下的誤差來自探針)→ 另寫 §十七,使用者確認後才做"
    else:
        pr = "不提探針步(combo 的位置沒估對,或拿掉探針誤差後仍未達 0.02)→ 回報討論"
    V["probe_rule"] = {"posL": eL, "combo": cmb, "comboNP": cnp, "combo_pos_ok": e3c, "decision": pr}
    print(f"  ▶ 探針步(§16.6):posL {eL:.4f}、combo {cmb:.4f}、comboNP {cnp:.4f}、combo 的位置估計 {'有效' if e3c else '未達'} → {pr}")
    return V


def ssim_rank(V, R, S7):
    """(e8) SSIM 與 nerr_sr 的排名一致性:4 個網路 + 7b 重算過的迭代法關鍵點(同量測、同設定)。"""
    print("\n  (e8) SSIM 與 nerr_sr 的排名一致性(4 個網路 + 7b 的迭代法關鍵點;兩兩比較)")
    tot = {"amp": [0, 0], "ph": [0, 0]}
    out = {}
    for c in CONDS:
        items = [(nm, np.mean(R[c][nm][MAIN]), np.mean(R[c][nm]["ssim_amp"]), np.mean(R[c][nm]["ssim_ph"])) for nm in NETS]
        for x in (S7 or {}).get(c, []):
            items.append((f"iter[{','.join(x['tags'])}]", np.mean(x["nerr_sr"]), np.mean(x["ssim_amp"]), np.mean(x["ssim_ph"])))
        ag = {"amp": [0, 0], "ph": [0, 0]}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][1] == items[j][1]:
                    continue
                for k, col in (("amp", 2), ("ph", 3)):
                    ag[k][1] += 1
                    ag[k][0] += int((items[i][1] < items[j][1]) == (items[i][col] > items[j][col]))
        for k in ag:
            tot[k][0] += ag[k][0]
            tot[k][1] += ag[k][1]
        out[c] = ag
        print(f"    {c:<6} 振幅 {ag['amp'][0]}/{ag['amp'][1]}、相位 {ag['ph'][0]}/{ag['ph'][1]}")
    print(f"    合計:振幅 {tot['amp'][0]}/{tot['amp'][1]}、相位 {tot['ph'][0]}/{tot['ph'][1]}(描述)")
    out["total"] = tot
    V["e8"] = out


def figures(V, R, P, tim, keep, mroot, out_dir, where):
    plt = s5._plt()
    if plt is None:
        check("matplotlib 可用(畫圖需要)", False)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    xs = np.arange(len(EVAL_CONDS))
    lab = [a7.CLABEL[c] for c in EVAL_CONDS]
    # 1. 總覽
    fig, ax = plt.subplots(figsize=(14, 5))
    for k, nm in enumerate(NETS):
        ax.plot(xs + (k - 1.5) * 0.15, [np.mean(V["cond"][c][nm]["nerr"]) for c in EVAL_CONDS], "o", color=COL[nm], ms=6, label=f"{nm} G9")
    ax.plot(xs[:-1], [np.mean(V["cond"][c]["P6B4"]["512"]["opp"][2]) for c in CONDS], "x", color=COL["iter"], ms=8,
            label="iterative envelope at P6B4's time (b512)")
    for q in QS:
        ax.axhline(q, color="#d6452a" if q == Q_MAIN else "#e8a598", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(f"{MAIN} (G9, 3 seeds)")
    ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.set_title(f"{where}: frozen P6B4, tolerant P6B4r, control P6B4c, estimate-type P6B4e (dashed = quality thresholds)", fontsize=9)
    fig.tight_layout()
    save(fig, "c_summary.png")
    # 2. 位置估計
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in POS_CONDS:
        r0 = V["pos"][c]["rms0"]
        ax.plot(range(len(V["pos"][c]["rms"]) + 1), [r0] + V["pos"][c]["rms"], "o-", ms=4, label=a7.CLABEL[c])
    ax.set_xlabel("stage (0 = draft, nominal positions)")
    ax.set_ylabel("position RMS error (px, 49 positions)")
    ax.set_yscale("log")
    ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{where}: P6B4e position estimates per stage", fontsize=9)
    fig.tight_layout()
    save(fig, "c_positions.png")
    # 3. 加速倍數(Q 主)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.2
    for k, nm in enumerate(["P6B4", NEW_E]):
        for j, B in enumerate(["64", "512"]):
            xp = np.arange(len(CONDS)) + (2 * k + j - 1.5) * w
            vals = []
            for i, c in enumerate(CONDS):
                r = V["cond"][c][nm][B]["speed"][str(Q_MAIN)]
                vals.append(r["x"] if (r["reach"] and np.isfinite(r["x"])) else np.nan)
                if not r["reach"]:
                    ax.plot([xp[i]], [0.12], "x", color=COL[nm], ms=5)
                elif not np.isfinite(r["x"]):
                    ax.plot([xp[i]], [20], "^", color=COL[nm], ms=5)
            ax.bar(xp, vals, width=w, color=COL[nm], alpha=1.0 if B == "512" else 0.55, label=f"{nm} b{B}")
    ax.axhline(1.0, color="#0b0b0b", lw=0.7)
    ax.axhline(SPEED_MIN, color="#0b0b0b", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.08, 40)
    ax.set_xticks(np.arange(len(CONDS)))
    ax.set_xticklabels([a7.CLABEL[c] for c in CONDS], rotation=35, ha="right", fontsize=7)
    ax.set_title(f"{where}: speed-up (upper value) at {MAIN} <= {Q_MAIN}; x = network does not reach, triangle = iterative never reaches", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    fig.tight_layout()
    save(fig, "c_speedup.png")
    # 4. 訓練曲線
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for s in SEEDS:
        for kind, nm in (("e", NEW_E), ("c", NEW_C)):
            p = model_dir(kind, s, mroot) / "result.json"
            if not p.exists():
                continue
            hist = json.load(open(p)).get("history", [])
            ep = [x["epoch"] + 1 for x in hist]
            axs[0].plot(ep, [x["final"] for x in hist], color=COL[nm], lw=1.2, label=nm if s == SEEDS[0] else None)
            if kind == "e":
                axs[1].plot(ep, [x["pos_rms"][-1] for x in hist], color=COL[nm], lw=1.2, label="last stage" if s == SEEDS[0] else None)
                axs[1].plot(ep, [x["pos_rms"][0] for x in hist], ":", color=COL[nm], lw=1.0, label="first stage" if s == SEEDS[0] else None)
    axs[0].set_yscale("log")
    axs[0].set_title("training loss of the last stage (object nerr on R0, with random errors)", fontsize=8)
    axs[1].set_title("P6B4e training: position RMS (px, 9 windows)", fontsize=8)
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "c_train.png")
    # 5. 重建圖
    geo, cfg = keep["geo"], keep["cfg"]
    U = geo.U.cpu()
    pm = cfg.phase_max
    y0, y1, x0, x1 = b7._bbox(U, a7.MARGIN)
    Uc = U[y0:y1, x0:x1].numpy()
    ps = np.array(R["combo"][NEW_E]["ps_sr0"], float)
    okk = np.where(np.isfinite(ps))[0]
    i = int(okk[np.argsort(ps[okk], kind="stable")][len(okk) // 2])
    conds = [c for c in ("ideal", "posL", "combo", "posH") if c in keep["img"]]
    cols = ["O", "P6B4", "P6B4r", NEW_C, NEW_E]
    fig, axs = plt.subplots(len(conds), len(cols), figsize=(3 * len(cols), 2.9 * len(conds)), squeeze=False, constrained_layout=True)
    for r, c in enumerate(conds):
        im = keep["img"][c]
        o = im["O"][i]
        t = o[y0:y1, x0:x1].numpy()
        for k, key in enumerate(cols):
            if key == "O":
                z, v = o, None
            else:
                z, v = a7.align_show(im[key][i], o, U)
            zc = z[y0:y1, x0:x1].numpy()
            hp = axs[r, k].imshow(np.where(np.abs(t) > a7.AMP_MIN, np.angle(zc), np.nan), cmap="viridis", vmin=0, vmax=pm)
            axs[r, k].imshow(np.where(Uc, np.nan, 0.0), cmap="Greys", vmin=0, vmax=1, alpha=0.55)
            axs[r, k].set_xticks([])
            axs[r, k].set_yticks([])
            axs[r, k].set_title(("truth" if key == "O" else key) + ("" if v is None else f"\n{MAIN} {v:.4f}"), fontsize=7)
        axs[r, 0].set_ylabel(a7.CLABEL[c], fontsize=7)
    fig.colorbar(hp, ax=list(axs[:, -1]), fraction=0.03, label="phase (rad)")
    fig.suptitle(f"{where}: sample #{i} (median of P6B4e in 'all three'); seed 0; aligned like the main metric", fontsize=9)
    save(fig, "c_recon.png")


def load_inputs(old_root, env_root, r_root, mroot, feas_path, iters):
    C, bad = {}, []
    for c in CONDS:
        p = a7.cond_json(c, old_root) if c in b7.OLD_CONDS else b7.env_json(c, env_root)
        if not p.exists():
            bad.append(f"{p.name} 不存在")
            continue
        C[c] = json.load(open(p))
        m = C[c]["meta"]
        want = A7_MD5 if c in b7.OLD_CONDS else B7_MD5
        if m["script_md5"] != want:
            bad.append(f"{p.name} 的程式 md5 {m['script_md5'][:8]}… ≠ 預期 {want[:8]}…")
        if not C[c]["checks_ok"]:
            bad.append(f"{p.name} 有檢查未通過")
    if not bad:
        n0 = C[CONDS[0]]["meta"]["n"]
        for c in CONDS:
            if C[c]["meta"]["iters"] != list(iters) or C[c]["meta"]["n"] != n0:
                bad.append(f"{c} 的停止點或場數不同")
    check("12 個條件檔都在(7a 的 8 個 + 7b 的 4 個)、版本正確、檢查全過、設定一致", not bad, ";".join(bad))
    r7 = b7.final_json(r_root)
    S7, J7 = None, None
    okr = r7.exists()
    if okr:
        J7 = json.load(open(r7))
        okr = J7.get("script_md5") == B7_MD5 and J7.get("complete") is True
        S7 = J7.get("iter_ssim")
    check(f"7b 的結果檔 {r7.name} 在、版本正確、完整(P6B4r 的重現檢查與迭代法 SSIM 用)", okr)
    mb = []
    for kind in ("e", "c"):
        for s in SEEDS:
            md = model_dir(kind, s, mroot)
            if not ((md / "final.pt").exists() and (md / "result.json").exists()):
                mb.append(f"{md.name} 缺 final.pt / result.json")
                continue
            r = json.load(open(md / "result.json"))
            if r.get("script_md5") != me_md5():
                mb.append(f"{md.name} 由不同版本的 scan_7c.py 訓練")
    for s in SEEDS:
        md = b7.model_dir(s, r_root if r_root != RUN_ROOT else None)
        if not (md / "final.pt").exists():
            mb.append(f"{md.name}(P6B4r)不存在")
    fz = json.load(open(feas_path)) if feas_path.exists() else None
    if fz is None or not fz.get("go") or fz.get("script_md5") != me_md5():
        mb.append("可行性檢查的結果不存在、不是 go,或版本不同")
    check("6 個新模型(P6B4e、P6B4c 各 3 seeds)與 P6B4r 都在、由同一版程式訓練;可行性檢查 = go(同一版本)", not mb, ";".join(mb))
    return (C, J7, S7, fz) if not bad and okr and not mb else None


def run_final(dev, old_root, env_root, r_root, mroot, feas_path, fig_dir, iters, smoke):
    me = me_md5()
    where = "validation fields (smoke)" if smoke else "validation fields, 7x7 scan"
    L = load_inputs(old_root, env_root, r_root, mroot, feas_path, iters)
    if L is None:
        return None
    C, J7, S7, fz = L
    t0 = time.time()
    print("\n  計時(ms / 整張)", flush=True)
    tim = a7.timing(dev)

    tim["net"] = time_nets(dev, mroot, None if r_root == RUN_ROOT else r_root)
    print(f"  (計時完成,經過 {time.time() - t0:.0f} 秒;GPU {tim['gpu']})", flush=True)
    print("\n  網路評估(13 個條件 × 3 seeds × 4 個網路,G9)", flush=True)
    R, P, keep = eval_all(dev, C, mroot, r_root, log=lambda s: print(s, flush=True))
    d4 = max(absdiff(R[c]["P6B4"]["nerr_ph"][i], C[c]["nets"]["G9"]["nerr_ph"][i]) for c in CONDS for i in range(len(SEEDS)))
    dr = max(absdiff(R[c]["P6B4r"]["nerr_ph"][i], J7["nets"][c]["P6B4r"]["G9"]["nerr_ph"][i]) for c in CONDS for i in range(len(SEEDS)))
    check(f"重現:凍結 P6B4 的 G9 = 條件檔、P6B4r 的 G9 = 7b 的結果(12 個條件、3 seeds;nerr_ph < {a7.REPRO_TOL:.0e})",
          d4 < a7.REPRO_TOL and dr < a7.REPRO_TOL, f"P6B4 最大差 {d4:.1e}、P6B4r 最大差 {dr:.1e}")
    V = report(C, R, P, tim, iters, smoke)
    ssim_rank(V, R, S7)
    final = final_json(mroot)
    res = {"verdict": V, "nets": R, "pos": P, "timing": tim, "feas": fz, "checks_ok": all_ok(), "smoke": smoke, "script_md5": me,
           "complete": False}
    dump_json(res, final)
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        figures(V, R, P, tim, keep, mroot, fig_dir, where)
    except Exception as e:                                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"畫圖完成(判讀不受影響,已存於 {final.name})", False, f"{type(e).__name__}: {e}")
    res["checks_ok"] = all_ok()
    res["complete"] = True
    dump_json(res, final)
    print(f"  結果:{final}")
    return res


def check_files(kind):
    need = [Path(f) for f in ("scan_7b.py", "scan_7a.py", "scan_6a.py", "scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py",
                              "scan_5e.py", "scan_5g.py", "scan_5h.py", "probe_4_2b.py", "probe_4_1.py", "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    need += [s5c.run_dir(gp, PROBE, s, a6.N_TRAIN) / f for gp in ("B", "P6B4") for s in SEEDS for f in ("final.pt", "result.json")]
    if kind == "final":
        need += [a7.cond_json(c) for c in b7.OLD_CONDS] + [b7.env_json(c) for c in b7.NEW_CONDS] + [b7.final_json()]
        need += [b7.model_dir(s) / "final.pt" for s in SEEDS]
    if kind == "smoke":
        need += [a7.cond_json(c, a7.SMOKE_DIR) for c in b7.OLD_CONDS] + [b7.env_json(c, b7.SMOKE_DIR) for c in b7.NEW_CONDS]
        need += [b7.final_json(b7.SMOKE_DIR)] + [b7.model_dir(s, b7.SMOKE_DIR) / "final.pt" for s in SEEDS]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing, "缺:" + ", ".join(missing) if missing else "")
    m7 = md5(Path(b7.__file__).resolve())
    m7a = md5(Path(a7.__file__).resolve())
    check("scan_7b.py、scan_7a.py = 7b / 7a-1 正式結果所用的版本", m7 == B7_MD5 and m7a == A7_MD5, f"md5 {m7[:8]}… / {m7a[:8]}…")
    return not missing and m7 == B7_MD5 and m7a == A7_MD5


def need_passed(me):
    if not PASSED.exists():
        raise SystemExit(f"❌ 找不到 {PASSED}:先在 dev 節點跑 python scan_7c.py --smoke,全部通過才可送件")
    pm = json.load(open(PASSED))["script_md5"]
    if pm != me:
        raise SystemExit(f"❌ smoke 通過時的 scan_7c.py(md5 {pm})與現在的({me})不同:用現在的版本重跑 --smoke")
    print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")


def need_feas(me, path):
    if not path.exists():
        raise SystemExit(f"❌ 找不到 {path}:先跑 python scan_7c.py --feas")
    fz = json.load(open(path))
    if fz.get("script_md5") != me:
        raise SystemExit("❌ 可行性檢查是由不同版本的 scan_7c.py 跑的:用現在的版本重跑 --feas")
    if not fz.get("go"):
        raise SystemExit("❌ 可行性檢查是 no-go:不訓練。把 --feas 的輸出貼給 Claude")
    return fz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--feas", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", type=int, choices=range(len(TASKS)))
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = me_md5()
    print(f"scan_7c.py md5 {me};GPU {a7.gpu_name(dev)}")
    kind = "final" if a.final else ("smoke" if a.smoke else "check")
    if not check_files(kind):
        print("\n❌ 缺檔案或版本不同,停下來")
        sys.exit(1)
    if a.check:
        checks(dev)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.feas:
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有做可行性檢查。把輸出貼給 Claude")
            sys.exit(1)
        res = feasibility(dev, log=lambda s: print(s, flush=True))
        res.update({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": a7.gpu_name(dev)})
        dump_json(res, feas_json())
        feas_report(res)
        print(f"\n  結果:{feas_json()}(不管 go / no-go,都把完整輸出貼給 Claude)")
        sys.exit(0)
    if a.smoke:
        print("#" * 70)
        print(f"迷你全流程(--smoke):檢查、可行性(少量場)、P6B4e / P6B4c 各 3 seeds × {SMOKE_EPOCHS} epoch × {SMOKE_TRAIN_N} 個場、彙整;"
              f"輸出到 {SMOKE_DIR}")
        print("#" * 70)
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)                                      # 只刪這個 smoke 專用資料夾
        SMOKE_DIR.mkdir(parents=True)
        t0 = time.time()
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
        fz = feasibility(dev, n=4 if QUICK else 8, log=lambda s: print(s, flush=True))
        fz.update({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": a7.gpu_name(dev)})
        feas_report(fz)
        fz_s = dict(fz, go=True, order=fz["order"] or "pa", m=fz["m"] or 1)          # smoke:不管 go 與否都走完流程
        dump_json(fz_s, feas_json(SMOKE_DIR))
        for kind_, s in TASKS:
            train_7c(kind_, s, dev, fz_s, SMOKE_TRAIN_N, SMOKE_EPOCHS, model_dir(kind_, s, SMOKE_DIR))
        run_final(dev, a7.SMOKE_DIR, b7.SMOKE_DIR, b7.SMOKE_DIR, SMOKE_DIR, feas_json(SMOKE_DIR), SMOKE_DIR / "figs", a7.SMOKE_ITERS,
                  smoke=True)
        print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒")
        if all_ok():
            json.dump({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
            print("\n✅ 迷你全流程全部通過 → 先跑 --feas(正式的可行性檢查);go 才送 run_scan7c_train.sh")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    if a.task is not None:
        kind_, s = TASKS[a.task]
        need_passed(me)
        fz = need_feas(me, feas_json())
        md = model_dir(kind_, s)
        if (md / "final.pt").exists():
            raise SystemExit(f"❌ {md}/final.pt 已存在:這個模型已經訓練完。為避免覆蓋,先告訴 Claude")
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有訓練。把輸出貼給 Claude")
            sys.exit(1)
        train_7c(kind_, s, dev, fz)
        print("\n" + (f"✅ {md.name} 訓練完成、檢查全過" if all_ok() else f"❌ {md.name} 有檢查未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.final:
        out = final_json()
        if out.exists() and json.load(open(out)).get("complete"):
            raise SystemExit(f"❌ {out} 已存在且完整:正式結果已經有了。為避免覆蓋,先告訴 Claude")
        need_passed(me)
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有彙整。把輸出貼給 Claude")
            sys.exit(1)
        res = run_final(dev, RUN_ROOT, RUN_ROOT, RUN_ROOT, None, feas_json(), FIG_DIR, a7.ITERS, smoke=False)
        print("\n" + ("✅ 內建檢查全部通過" if all_ok() and res else "❌ 有項目未通過(判讀先不要採信)"))
        print("把完整輸出與 figs_scan7c/ 的圖傳給 Claude")
        sys.exit(0 if all_ok() and res else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
