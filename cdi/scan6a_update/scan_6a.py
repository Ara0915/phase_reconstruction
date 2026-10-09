#!/usr/bin/env python
"""階段六:更大的掃描 —— 整張影像的拼接(階段六協定)。不重新訓練,只用期中考凍結的 P6B4。

資料:驗證場(物體 seed 80,000,000;512 場)、7 × 7 掃描(步距 8)、雜訊 82,000,000 + 1000 × seed。
網路的整張流程(皆為固定計算量、一次前向):
  S25  25 塊各自跑 P6B4 → 各取 R0 → 相位同步 + |P|² 加權合併(主要)
  S9   只用步距 16 的 9 塊(探索性)
  G25  25 塊只跑前面網路(草稿)→ 合併成整張起點 → P6B4 的 4 級物理步 + 小 CNN 直接作用在整張(探索性)
  G9   同 G25,起點只用 9 塊(探索性)
對手:整張的迭代法包絡(4 演算法 × 6 起點 + K-HIO 合併),同一 job 重新計時。

用法(需在計算節點執行):
    python scan_6a.py --check   # 只跑內建檢查
    python scan_6a.py --smoke   # 迷你全流程(32 個場;dev 節點);通過才可正式執行
    python scan_6a.py           # 正式(run_scan6a.sh)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5h as h                                                  # noqa: E402  (載入 scan_5g、讓 P6B4 可建構)

g = h.g
s5, s5b, s5c, s5d, s5e = h.s5, h.s5b, h.s5c, h.s5d, h.s5e
RUN_ROOT = s5.RUN_ROOT
QUICK = h.QUICK
PROBE = h.PROBE
SEEDS = h.SEEDS
N_TRAIN = h.N_TRAIN
GROUP = "P6B4"
FIELD_SEED = s5e.VAL_FIELD_SEED              # 80,000,000(驗證場)
NOISE_BASE = 82_000_000
NPOS, STEP = 7, 8
CEN49 = (NPOS // 2) * NPOS + NPOS // 2       # 7 × 7 的中央位置 = 24
PATCH25 = [(i, j) for i in range(1, NPOS - 1) for j in range(1, NPOS - 1)]
PATCH9 = [(i, j) for i in (1, 3, 5) for j in (1, 3, 5)]
PIPES = ["S25", "S9", "G25", "G9"]
PRIMARY = "S25"
CHUNK = 1600                                 # 網路一次推論的塊數(= batch 64 張影像的 25 塊)
MCHUNK = 64                                  # 合併時一次處理的影像數(記憶體)
PCTS = [10, 50, 90]
RAND_SEED = 2027
N_RANDOM = 3
MARGIN = 4
AMP_MIN = 0.05
REPRO_TOL, PS_TOL, VIZ_TOL = 1e-5, 1e-6, 1e-4
SYNC_ITERS = 100                             # 冪次法的次數(5 × 5 / 7 × 7 網格的特徵值間距下,誤差約 1e-5 以下)
SUF = "_quick" if QUICK else ""
SMOKE_FIELDS = 8 if QUICK else 32
ENV_JSON = RUN_ROOT / f"scan6a_env{SUF}.json"
RAW_JSON = RUN_ROOT / f"scan6a_raw{SUF}.json"
OUT_JSON = RUN_ROOT / f"scan6a{SUF}.json"
SMOKE_DIR = RUN_ROOT / f"scan6a_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
P5_RAW = RUN_ROOT / f"scan5h_p5_raw{SUF}.json"
FIG_DIR = Path("figs_scan6a")
COL = {"iter_b64": "#9a9994", "iter_b512": "#0b0b0b", "S25": "#d6452a", "S9": "#eb6834", "G25": "#2a78d6",
       "G9": "#1baf7a", "S25w": "#8a5cd1"}

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and h.all_ok()


def md5(path):
    m = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


def absdiff(a, b):
    d = abs(float(a) - float(b))
    return d if np.isfinite(d) else float("inf")


def seeds_of(s):
    return {"field": FIELD_SEED, "noise": NOISE_BASE + 1000 * s, "init": NOISE_BASE + 1000 * s + 1}


# ============================================================================
# 幾何
# ============================================================================
class Geom:
    """7 × 7 掃描、25 / 9 塊、各塊的方框位置與 R0、評分範圍 U。"""

    def __init__(self, cfg, pr, dev):
        self.cfg, self.pr, self.dev = cfg, pr, dev
        self.F, self.W = s5.field_size(cfg), cfg.canvas
        base = self.F // 2 - self.W // 2
        o = [STEP * (k - NPOS // 2) for k in range(NPOS)]
        self.starts = [(base + dy, base + dx) for dy in o for dx in o]             # 49 個,列優先
        self.b0, self.S, self.rel = s5c.geometry(cfg)                              # 方框 80、9 個視窗的相對位置
        self.cen_rel = self.rel[s5c.CENTER]                                        # (8, 8)
        self.R0box = s5c.r0_box(cfg, pr, dev)                                      # [80, 80] bool
        wb = torch.zeros(self.S, self.S, device=dev)
        y, x = self.cen_rel
        wb[y:y + self.W, x:x + self.W] = pr["Pa"] ** 2
        self.Wbox = wb * self.R0box                                                # |P|² 權重,限 R0
        self.fp = pr["Pa"] >= s5.TAU * pr["Pa"].max()                              # 視窗內的探針範圍
        self.Wwin = (pr["Pa"] ** 2) * self.fp
        self.U = self.masks(PATCH25)[0].any(0)                                     # 25 個 R0 的聯集

    @staticmethod
    def pos(i, j):
        return i * NPOS + j

    def nbr(self, i, j):
        return [self.pos(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]

    def origin(self, i, j):
        y, x = self.starts[self.pos(i, j)]
        return (y - self.cen_rel[0], x - self.cen_rel[1])

    def masks(self, patches):
        """各塊在場座標的 R0 [P, F, F] bool 與權重 [P, F, F]。"""
        M = torch.zeros(len(patches), self.F, self.F, dtype=torch.bool, device=self.dev)
        Wt = torch.zeros(len(patches), self.F, self.F, device=self.dev)
        for p, (i, j) in enumerate(patches):
            y, x = self.origin(i, j)
            M[p, y:y + self.S, x:x + self.S] = self.R0box
            Wt[p, y:y + self.S, x:x + self.S] = self.Wbox
        return M, Wt


def measure49(fields, geo, cp, bs, seed):
    counts, _ = s5.measure_scan(fields, PROBE, geo.pr, cp, bs, geo.starts, seed=seed)
    return counts


def setup(s, dev, n=None):
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[PROBE]
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS["3x3s8"][2])
    bs = beamstop_mask(cfg, device=dev)
    geo = Geom(cfg, pr, dev)
    n = n or (s5b.N_FIELDS or cfg.eval_n)
    sd = seeds_of(s)
    fields = s5.make_fields(cfg, n, seed=sd["field"], device=dev)
    O = torch.polar(fields[:, 0], fields[:, 1])
    counts = measure49(fields, geo, cp, bs, sd["noise"])
    return dict(cfg=cfg, pr=pr, cp=cp, bs=bs, geo=geo, O=O, counts=counts, init=sd["init"], n=n, fields=fields)


# ============================================================================
# 合併(相位同步 + 加權)
# ============================================================================
@torch.no_grad()
def merge(est, origins, Mloc, Wloc, F, sync=True, weighted=True, want_C=False):
    """est:[P, B, S, S] complex(各塊在自己方框的估計);origins:各塊方框在場中的左上角;
    Mloc / Wloc:方框內的遮罩與權重 [S, S]。回傳 (整張 [B, F, F], 覆蓋 [F, F] bool, C 或 None)。"""
    P, B, S, _ = est.shape
    dev = est.device
    w = Wloc if weighted else Mloc.to(Wloc.dtype)
    out = torch.zeros(B, F, F, dtype=est.dtype, device=dev)
    Cs = []
    Wt = torch.zeros(P, F, F, device=dev)
    for p, (y, x) in enumerate(origins):
        Wt[p, y:y + S, x:x + S] = w
    den = Wt.sum(0)
    cover = den > 0
    for b0 in range(0, B, MCHUNK):
        e = est[:, b0:b0 + MCHUNK] * Mloc                                          # [P, b, S, S]
        b = e.shape[1]
        A = torch.zeros(b, P, F, F, dtype=est.dtype, device=dev)
        for p, (y, x) in enumerate(origins):
            A[:, p, y:y + S, x:x + S] = e[p]
        if sync or want_C:
            Z = (A * Wt).reshape(b, P, -1)
            G = Z @ Z.conj().transpose(1, 2)                                       # G_pq = Σ w_p w_q z_p conj(z_q)(含對角,半正定)
            G = 0.5 * (G + G.conj().transpose(1, 2))
            dg = torch.diagonal(G, dim1=1, dim2=2).real                            # Σ w_p² |z_p|²
            if want_C:
                Cs.append(G - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2)))
            if sync:
                u = sync_phases(G, dg)
                A = A * u[:, :, None, None]
        out[b0:b0 + b] = torch.where(cover, (A * Wt).sum(1) / den.clamp_min(1e-12), torch.zeros_like(out[b0:b0 + b]))
    return out, cover, (torch.cat(Cs) if want_C else None)


def sync_phases(G, dg, iters=None):
    """最大特徵向量(冪次法,由全 1 出發;兩邊的合併用同一個解法)→ 各塊的旋轉 u(|u| = 1)。
    特徵向量的整體相位本身不確定 → 再固定「能量加權的平均旋轉 = 0」,使已一致的塊不被整體轉動。"""
    iters = iters or SYNC_ITERS
    x = torch.ones(G.shape[0], G.shape[1], 1, dtype=G.dtype, device=G.device)
    for _ in range(iters):
        x = G @ x
        x = x / x.abs().amax(1, keepdim=True).clamp_min(1e-30)
    v = x[..., 0]
    u = torch.where(v.abs() > 1e-12, v.conj() / v.abs().clamp_min(1e-12), torch.ones_like(v))
    ref = (u * dg).sum(1)
    fix = torch.where(ref.abs() > 1e-30, ref.conj() / ref.abs().clamp_min(1e-30), torch.ones_like(ref))
    return u * fix[:, None]


# ============================================================================
# 網路的整張流程
# ============================================================================
def load_p6b4(s, cfg, pr, dev):
    md = s5c.run_dir(GROUP, PROBE, s, N_TRAIN)
    meta = json.load(open(md / "result.json"))
    net = h.build_model(GROUP, cfg, pr).to(dev)
    net.load_state_dict(torch.load(md / "final.pt", map_location=dev))
    net.eval()
    return net, meta["norm"], meta


def patch_counts(counts, geo, patches):
    """9 個列表,每個 [P·B, W, W](塊為外層)。"""
    return [torch.cat([counts[geo.nbr(i, j)[k]] for (i, j) in patches]) for k in range(9)]


@torch.no_grad()
def net_patches(net, norm, counts, geo, cp, bs, patches, stages=False):
    """各塊跑完整 P6B4。回傳最終輸出 [P, B, S, S] complex(stages=True 另回傳起點與各級)。"""
    c9 = patch_counts(counts, geo, patches)
    B = counts[0].shape[0]
    if stages:
        O0, outs = g.predict(net, c9, bs, norm, GROUP, cp, chunk=CHUNK, all_stages=True)
        fin = torch.polar(outs[-1][:, 0], outs[-1][:, 1])
        return fin.reshape(len(patches), B, geo.S, geo.S), O0.reshape(len(patches), B, geo.S, geo.S)
    out = g.predict(net, c9, bs, norm, GROUP, cp, chunk=CHUNK)
    return torch.polar(out[:, 0], out[:, 1]).reshape(len(patches), B, geo.S, geo.S), None


@torch.no_grad()
def net_starts(net, norm, counts, geo, cp, bs, patches):
    """各塊只跑 P6B4 的前面網路,回傳起點(R0 內 = 網路、R0 外 = 常數)[P, B, S, S]。"""
    c9 = patch_counts(counts, geo, patches)
    n = c9[0].shape[0]
    outs = [net.initial(g.prep([c[i:i + CHUNK] for c in c9], bs, norm, GROUP, cp)) for i in range(0, n, CHUNK)]
    B = counts[0].shape[0]
    return torch.cat(outs).reshape(len(patches), B, geo.S, geo.S)


class GlobalStages:
    """P6B4 的 K 級「物理步 + 小 CNN + 投影」作用在任意大小的物體與任意視窗位置(權重取自訓練好的網路)。"""

    def __init__(self, net, starts, F, pr, bs, dev):
        self.net, self.starts, self.F = net, starts, F
        self.W = net.W
        self.P = pr["P"].to(dev)
        self.Pc = pr["P"].conj().resolve_conj().to(dev)
        wsum = torch.zeros(F, F, device=dev)
        for y0, x0 in starts:
            wsum[y0:y0 + self.W, x0:x0 + self.W] += pr["Pa"] ** 2
        self.wsum = wsum
        self.wn = (wsum / wsum.max())[None, None]
        self.delta = s5b.AP_DELTA * float(wsum.max())
        self.bs = bs

    @classmethod
    def from_box(cls, net):
        """用 P6B4 自己的方框幾何(單元測試:應完全重現 P6B4 的前向)。"""
        self = cls.__new__(cls)
        self.net, self.starts, self.F, self.W = net, net.starts, net.wsum.shape[-1], net.W
        self.P, self.Pc, self.wsum, self.wn, self.delta, self.bs = net.P, net.Pc, net.wsum, net.wn, net.delta, net.bs
        return self

    def ap_step(self, O, meas):
        W = self.W
        Oj = torch.stack([O[:, y0:y0 + W, x0:x0 + W] for y0, x0 in self.starts], 1)
        E = torch.fft.fftshift(torch.fft.fft2(self.P * Oj, norm="ortho"), dim=(-2, -1))
        mag = torch.sqrt(E.real * E.real + E.imag * E.imag + self.net.eps_e)
        E2 = torch.where(self.bs > 0, E * (meas / mag), E)
        psi2 = torch.fft.ifft2(torch.fft.ifftshift(E2, dim=(-2, -1)), norm="ortho")
        gg = self.Pc * psi2
        num = self.delta * O
        for j, (y0, x0) in enumerate(self.starts):
            num[:, y0:y0 + W, x0:x0 + W] = num[:, y0:y0 + W, x0:x0 + W] + gg[:, j]
        return num / (self.wsum + self.delta)

    def stage(self, k, O, meas, alpha=None, cnn=True):
        net = self.net
        if net.phys:
            al = torch.sigmoid(net.a[k]) if alpha is None else alpha
            O = O + al * (self.ap_step(O, meas) - O)
        if cnn:
            d = net.refine[k](torch.cat([O.real[:, None], O.imag[:, None], self.wn.expand(O.shape[0], -1, -1, -1)], 1))
            O = O + torch.complex(d[:, 0], d[:, 1])
        amp, ph = net.project(O)
        return torch.polar(amp, ph)

    @torch.no_grad()
    def run(self, O, meas):
        for k in range(self.net.K):
            O = self.stage(k, O, meas)
        return O


def measured(counts, cp, dev):
    from src.hio import _measured_amp
    return torch.stack([_measured_amp(c, cp).to(dev) for c in counts], 1)             # [B, J, W, W]


@torch.no_grad()
def run_pipe(pipe, net, norm, d, gs=None, extras=False):
    """一個網路流程:量測 → 整張影像 [B, F, F]。extras=True 另回傳診斷用的東西(不計時)。"""
    geo, cp, bs, counts = d["geo"], d["cp"], d["bs"], d["counts"]
    patches = PATCH25 if pipe.endswith("25") else PATCH9
    origins = [geo.origin(i, j) for (i, j) in patches]
    if pipe.startswith("S"):
        fin, _ = net_patches(net, norm, counts, geo, cp, bs, patches)
        out, cover, C = merge(fin, origins, geo.R0box, geo.Wbox, geo.F, sync=True, want_C=extras)
        if not extras:
            return out
        return out, {"fin": fin, "cover": cover, "C": C}
    st = net_starts(net, norm, counts, geo, cp, bs, patches)
    O0, cover, _ = merge(st, origins, geo.R0box, geo.Wbox, geo.F, sync=True)
    a0 = s5b.const_amp(counts[CEN49], bs, cp, geo.pr)
    c = torch.polar(a0[:, None, None].expand(O0.shape).contiguous(), torch.full(O0.shape, net.phase_max / 2, device=O0.device))
    O0 = torch.where(cover, O0, c)
    gs = gs or GlobalStages(net, geo.starts, geo.F, geo.pr, bs, O0.device)
    meas = measured(counts, cp, O0.device)
    outs = [gs.run(O0[i:i + MCHUNK].clone(), meas[i:i + MCHUNK]) for i in range(0, O0.shape[0], MCHUNK)]
    out = torch.cat(outs)
    if not extras:
        return out
    return out, {"cover": cover, "start": O0}


# ============================================================================
# 整張的迭代法
# ============================================================================
def kit_list():
    return sorted(set(s5b.ITERS) | set(s5b.KHIO_N.values()))


@torch.no_grad()
def khio_windows(d, dev):
    """49 個位置各跑 K-HIO;回傳 {次數: 各視窗探針範圍內的物體估計 [49, B, W, W] complex}。"""
    import probe_4_2b as pb
    geo, cp, bs, counts = d["geo"], d["cp"], d["bs"], d["counts"]
    W = geo.W
    ones = torch.ones(W, W, device=dev)
    kit = kit_list()
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


def merge_windows(win, geo):
    return merge(win, geo.starts, geo.fp, geo.Wwin, geo.F, sync=True)


def const_field(d, n, dev):
    a0 = s5b.const_amp(d["counts"][CEN49], d["bs"], d["cp"], d["geo"].pr)
    F = d["geo"].F
    return torch.polar(a0[:, None, None].expand(n, F, F).contiguous(), torch.full((n, F, F), d["cfg"].phase_max / 2, device=dev))


def make_start(kind, d, kw, dev):
    n, F = d["n"], d["geo"].F
    if kind == "rand":
        return s5.epie_init(n, F, d["cfg"].phase_max, d["init"], dev)
    if kind == "one":
        return torch.ones(n, F, F, dtype=torch.complex64, device=dev)
    if kind == "zero":
        return torch.zeros(n, F, F, dtype=torch.complex64, device=dev)
    c = const_field(d, n, dev)
    if kind == "const":
        return c
    Of, cover = kw[s5b.KHIO_N[kind]]
    return torch.where(cover, Of, c)


def per_sample(est, O, M):
    m = M.to(est.real.dtype)
    a, t = est * m, O * m
    E = (t.abs() ** 2).sum((1, 2))
    cross = (a * t.conj()).sum((1, 2)).abs()
    v = ((a.abs() ** 2).sum((1, 2)) + E - 2 * cross) / E.clamp_min(1e-12)
    return torch.where(E > 1e-12, v, torch.full_like(v, float("nan")))


def metrics(est, O, U):
    m = s5.field_metrics(est, O, U)
    m["ps"] = per_sample(est, O, U).cpu().numpy().tolist()
    return m


@torch.no_grad()
def measure_envelope(dev, n=None, seeds=None):
    res = {PROBE: []}
    for s in (seeds or SEEDS):
        d = setup(s, dev, n)
        geo, O = d["geo"], d["O"]
        win = khio_windows(d, dev)
        kw = {}
        rec = {"K-HIO": {}}
        for k, wv in win.items():
            Of, cover, _ = merge_windows(wv, geo)
            kw[k] = (Of, cover)
            if k in s5b.ITERS:
                rec["K-HIO"][str(k)] = metrics(torch.where(cover, Of, torch.zeros_like(Of)), O, geo.U)
        del win
        for kind in s5b.INITS:
            O0 = make_start(kind, d, kw, dev)
            m0 = metrics(O0, O, geo.U)
            for meth in s5b.METHODS:
                outs = s5b.run_method(meth, d["counts"], d["bs"], d["cp"], geo.pr, geo.starts, s5b.ITERS, O0, d["init"])
                rec[f"{meth}|{kind}"] = {str(it): metrics(outs[it], O, geo.U) for it in s5b.ITERS}
                rec[f"{meth}|{kind}"]["0"] = m0
                del outs
        res[PROBE].append({"run": f"{s5.BASE}_s{s}", "rec": rec})
        del kw
        print(f"  [seed {s}] 整張迭代法完成", flush=True)
    return res


def configs():
    return [f"{m}|{i}" for m in s5b.METHODS for i in s5b.INITS] + ["K-HIO"]


def stops(conf):
    return list(s5b.ITERS) if conf == "K-HIO" else [0] + list(s5b.ITERS)


def cfg_time(conf, it, tB):
    tK, tM = tB["K-HIO49"], tB["merge49"]
    if conf == "K-HIO":
        return it * tK + tM
    meth, kind = conf.split("|")
    t0 = s5b.KHIO_N[kind] * tK + tM if kind in s5b.KHIO_N else 0.0
    return t0 + it * tB[meth]


def env_vals(env, conf, it, key="nerr_ph"):
    return np.array([sd["rec"][conf][str(it)][key] for sd in env[PROBE]], float)


def envelope(env, T, tB):
    best = None
    for conf in configs():
        for it in stops(conf):
            t = cfg_time(conf, it, tB)
            if t > T * (1 + 1e-9):
                continue
            v = env_vals(env, conf, it)
            if best is None or v.mean() < best[2].mean():
                best = (conf, it, v, t)
    return best


# ============================================================================
# 計時
# ============================================================================
@torch.no_grad()
def timing(dev, nets):
    import probe_4_2b as pb
    torch.backends.cudnn.benchmark = True
    cfg = s5.load_cfg(0)
    res = {}
    for Bl in ["64", "512"]:
        B = int(Bl) if not QUICK else {"64": 4, "512": 8}[Bl]
        d = setup(0, dev, B)
        d["init"] = 0
        geo, cp, bs, counts = d["geo"], d["cp"], d["bs"], d["counts"]
        row = {}
        O0 = s5.epie_init(B, geo.F, cfg.phase_max, 0, dev)
        for meth in s5b.METHODS:
            s5b.run_method(meth, counts, bs, cp, geo.pr, geo.starts, [2], O0, 0)
            ts = []
            for _ in range(s5b.TIME_REPS):
                s5._sync(dev)
                t0 = time.perf_counter()
                s5b.run_method(meth, counts, bs, cp, geo.pr, geo.starts, [10], O0, 0)
                s5._sync(dev)
                ts.append((time.perf_counter() - t0) / 10 / B * 1e3)
            row[meth] = float(np.median(ts))
        ones = torch.ones(geo.W, geo.W, device=dev)
        init = torch.cat([s5.k_init(counts[j], cp, j, dev) for j in range(len(geo.starts))])
        cc = torch.cat(counts)
        pb.hio_probe(init, cc, bs, cp, [3], "K", geo.pr, ones)
        ts = []
        for _ in range(s5b.TIME_REPS):
            s5._sync(dev)
            t0 = time.perf_counter()
            pb.hio_probe(init, cc, bs, cp, [20], "K", geo.pr, ones)
            s5._sync(dev)
            ts.append((time.perf_counter() - t0) / 20 / B * 1e3)
        row["K-HIO49"] = float(np.median(ts))
        win = torch.polar(torch.rand(len(geo.starts), B, geo.W, geo.W, device=dev),
                          torch.rand(len(geo.starts), B, geo.W, geo.W, device=dev))
        merge_windows(win, geo)
        ts = []
        for _ in range(s5b.TIME_REPS):
            s5._sync(dev)
            t0 = time.perf_counter()
            merge_windows(win, geo)
            s5._sync(dev)
            ts.append((time.perf_counter() - t0) / B * 1e3)
        row["merge49"] = float(np.median(ts))
        net, norm = nets
        gs = GlobalStages(net, geo.starts, geo.F, geo.pr, bs, dev)
        for pipe in PIPES:
            run_pipe(pipe, net, norm, d, gs)
            run_pipe(pipe, net, norm, d, gs)
            ts = []
            for _ in range(s5b.TIME_REPS):
                s5._sync(dev)
                t0 = time.perf_counter()
                run_pipe(pipe, net, norm, d, gs)
                s5._sync(dev)
                ts.append((time.perf_counter() - t0) / B * 1e3)
            row[f"net:{pipe}"] = float(np.median(ts))
        res[Bl] = row
        del d, win
    return res


# ============================================================================
# 網路的量測與診斷
# ============================================================================
@torch.no_grad()
def residual(fin, d, patches):
    """逐塊的資料殘差:用塊的最終估計算出 9 張繞射圖的振幅,與量測比(beamstop 外)。回傳 [P, B]。"""
    geo, cp, bs, counts = d["geo"], d["cp"], d["bs"], d["counts"]
    W = geo.W
    P = geo.pr["P"]
    out = []
    for p, (i, j) in enumerate(patches):
        meas = measured([counts[k] for k in geo.nbr(i, j)], cp, fin.device)                  # [B, 9, W, W]
        Oj = torch.stack([fin[p][:, y:y + W, x:x + W] for (y, x) in geo.rel], 1)
        E = torch.fft.fftshift(torch.fft.fft2(P * Oj, norm="ortho"), dim=(-2, -1)).abs()
        m = bs > 0
        num = (((E - meas) ** 2) * m).sum((1, 2, 3))
        den = ((meas ** 2) * m).sum((1, 2, 3)).clamp_min(1e-12)
        out.append(num / den)
    return torch.stack(out)


@torch.no_grad()
def measure_nets(dev, env, tim, n=None):
    """各 seed:4 個流程的整張誤差、S25 的診斷、視覺化用的迭代法;seed 0 另留影像。"""
    res = {p: [] for p in PIPES + ["S25w", "S25eq"]}
    diag = {"patch": [], "pooled": [], "ring": [], "phase": [], "resid_rho": [], "capture": [], "center": [],
            "cover": {}}
    ps = {k: [] for k in PIPES + ["S25w", "iter_b64", "iter_b512"]}
    keep = {}
    iconf = {B: envelope(env, tim[B][f"net:{PRIMARY}"], tim[B])[:2] for B in ["64", "512"]}
    worst_ps, worst_it = 0.0, 0.0
    for s in SEEDS:
        d = setup(s, dev, n)
        geo, O = d["geo"], d["O"]
        net, norm, meta = load_p6b4(s, d["cfg"], geo.pr, dev)
        gs = GlobalStages(net, geo.starts, geo.F, geo.pr, d["bs"], dev)
        for pipe in PIPES:
            out, ex = run_pipe(pipe, net, norm, d, gs, extras=True)
            if pipe == PRIMARY:
                fin, C = ex["fin"], ex["C"]
            m = metrics(out, O, geo.U)
            m["cover_frac"] = float((ex["cover"] & geo.U).sum() / geo.U.sum())
            res[pipe].append(m)
            ps[pipe].append(np.array(m["ps"]))
            worst_ps = max(worst_ps, absdiff(np.nanmean(m["ps"]), m["nerr_ph"]) / max(1.0, abs(m["nerr_ph"])))
            if s == SEEDS[0]:
                keep[pipe] = out.cpu()
        origins = [geo.origin(i, j) for (i, j) in PATCH25]
        for tag, kw in [("S25w", dict(sync=False, weighted=True)), ("S25eq", dict(sync=False, weighted=False))]:
            out, _, _ = merge(fin, origins, geo.R0box, geo.Wbox, geo.F, **kw)
            m = metrics(out, O, geo.U)
            res[tag].append(m)
            if tag == "S25w":
                ps[tag].append(np.array(m["ps"]))
                if s == SEEDS[0]:
                    keep[tag] = out.cpu()
        # 逐塊(各自對齊)與合併誤差、分圈
        M, _ = geo.masks(PATCH25)
        Ef = torch.zeros(len(PATCH25), d["n"], device=dev)
        Nf = torch.zeros_like(Ef)
        for p, (y, x) in enumerate(origins):
            ef = torch.zeros(d["n"], geo.F, geo.F, dtype=fin.dtype, device=dev)
            ef[:, y:y + geo.S, x:x + geo.S] = fin[p]
            v = per_sample(ef, O, M[p])
            E = ((O.abs() ** 2) * M[p]).sum((1, 2))
            Ef[p] = torch.nan_to_num(v, nan=0.0) * E
            Nf[p] = v
        Etot = torch.stack([((O.abs() ** 2) * M[p]).sum((1, 2)) for p in range(len(PATCH25))])
        et = Etot.sum(0)
        pooled = torch.where(et > 1e-12, Ef.sum(0) / et.clamp_min(1e-12), torch.full_like(et, float("nan"))).cpu().numpy()
        diag["pooled"].append(float(np.nanmean(pooled)))
        rings = [max(abs(i - 3), abs(j - 3)) for (i, j) in PATCH25]
        diag["ring"].append([float((Ef[[k for k, r in enumerate(rings) if r == q]].sum() /
                                    Etot[[k for k, r in enumerate(rings) if r == q]].sum()).item()) for q in range(3)])
        cen = PATCH25.index((3, 3))
        diag["center"].append(float(torch.nanmean(Nf[cen]).item()))
        diag["patch"].append(float(torch.nanmean(Nf).item()))
        # 相鄰塊的整體相位差(同步前)
        adj = [(PATCH25.index((i, j)), PATCH25.index((i, j + 1))) for (i, j) in PATCH25 if (i, j + 1) in PATCH25] + \
              [(PATCH25.index((i, j)), PATCH25.index((i + 1, j))) for (i, j) in PATCH25 if (i + 1, j) in PATCH25]
        pi_ = torch.tensor([p for p, _ in adj], device=dev)
        qi_ = torch.tensor([q for _, q in adj], device=dev)
        Cpq = C[:, pi_, qi_]
        ph = torch.angle(Cpq).abs()[Cpq.abs() > 1e-12].cpu().numpy()
        diag["phase"].append({"median": float(np.median(ph)), "p90": float(np.percentile(ph, 90)),
                              "frac_gt_0.1": float((ph > 0.1).mean()), "n": int(ph.size),
                              "hist": np.histogram(ph, bins=np.linspace(0, np.pi, 61))[0].tolist() if s == SEEDS[0] else None})
        # 資料殘差 vs 逐塊誤差
        r = residual(fin, d, PATCH25).cpu().numpy().ravel()
        e = Nf.cpu().numpy().ravel()
        ok = np.isfinite(e) & np.isfinite(r)
        rho = spearman(r[ok], e[ok])
        top = r[ok] >= np.percentile(r[ok], 95)
        fail = e[ok] > 0.1
        diag["resid_rho"].append(rho)
        diag["capture"].append({"n_fail": int(fail.sum()),
                                "caught": float((top & fail).sum() / fail.sum()) if fail.sum() else float("nan"),
                                "precision": float((top & fail).sum() / max(1, top.sum()))})
        # 視覺化用的迭代法(整張包絡在 S25 的時間所選的設定;逐 seed 重算並核對)
        need = sorted({c for c, _ in iconf.values()})
        kw = None
        if any(c != "K-HIO" and c.split("|")[1] in s5b.KHIO_N or c == "K-HIO" for c in need):
            win = khio_windows(d, dev)
            kw = {k: merge_windows(wv, geo)[:2] for k, wv in win.items()}
            del win
        for B, (conf, it) in iconf.items():
            if conf == "K-HIO":
                Of, cover = kw[it]
                est = torch.where(cover, Of, torch.zeros_like(Of))
            else:
                meth, kind = conf.split("|")
                O0 = make_start(kind, d, kw, dev)
                est = O0 if it == 0 else s5b.run_method(meth, d["counts"], d["bs"], d["cp"], geo.pr, geo.starts, [it], O0,
                                                         d["init"])[it]
            v = per_sample(est, O, geo.U).cpu().numpy()
            ps[f"iter_b{B}"].append(v)
            worst_it = max(worst_it, absdiff(np.nanmean(v), env_vals(env, conf, it)[SEEDS.index(s)]))
            if s == SEEDS[0]:
                keep[f"iter_b{B}"] = est.cpu()
        if s == SEEDS[0]:
            keep.update({"O": O.cpu(), "geo": geo, "cfg": d["cfg"]})
        del net, gs, fin, C
        print(f"  [seed {s}] 網路完成", flush=True)
    check(f"逐樣本計算一致(整張):逐樣本平均 = field_metrics(差 / max(1, 值) < {PS_TOL:.0e})", worst_ps < PS_TOL,
          f"最大差 {worst_ps:.1e}")
    check(f"視覺化的迭代法 = 包絡檔(逐 seed,< {VIZ_TOL:.0e})", worst_it < VIZ_TOL,
          " / ".join(f"b{B}: {s5b.fmt_conf(c)} ×{i}" for B, (c, i) in iconf.items()) + f";最大差 {worst_it:.1e}")
    return res, diag, ps, keep, iconf


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ============================================================================
# 內建檢查
# ============================================================================
@torch.no_grad()
def checks(dev):
    torch.backends.cudnn.benchmark = True
    print("=" * 70)
    print("階段六:內建檢查")
    print("=" * 70)
    d = setup(0, dev, 8)
    geo, cfg, pr, cp, bs, O = d["geo"], d["cfg"], d["pr"], d["cp"], d["bs"], d["O"]
    # (1) 幾何
    inside = all(0 <= y and y + geo.W <= geo.F and 0 <= x and x + geo.W <= geo.F for y, x in geo.starts)
    cen_nbr = [geo.starts[k] for k in geo.nbr(3, 3)]
    boxes_in = all(0 <= geo.origin(i, j)[0] and geo.origin(i, j)[0] + geo.S <= geo.F and
                   0 <= geo.origin(i, j)[1] and geo.origin(i, j)[1] + geo.S <= geo.F for (i, j) in PATCH25)
    check("幾何:49 個視窗在場內;中央塊的 9 個視窗 = 階段五 3x3s8;中央塊的方框位置 = 階段五;25 塊的方框皆在場內",
          len(geo.starts) == 49 and inside and cen_nbr == s5.window_starts(cfg, "3x3s8")
          and geo.origin(3, 3) == (geo.b0, geo.b0) and boxes_in,
          f"方框 {geo.S};中央方框 {geo.origin(3, 3)};U 面積 {int(geo.U.sum())} px")
    # (2) 等變性
    import probe_4_2b as pb
    worst = 0.0
    for (i, j) in [(1, 1), (2, 4), (5, 5), (3, 3)]:
        y, x = geo.starts[geo.pos(i, j)]
        cy, cx = geo.starts[CEN49]
        rolled = torch.roll(d["fields"], (cy - y, cx - x), dims=(-2, -1))
        ref = [pb.make_psi(s5.crop(rolled, yy, xx, geo.W), PROBE, pr, cp) for yy, xx in s5.window_starts(cfg, "3x3s8")]
        got = [pb.make_psi(s5.crop(d["fields"], *geo.starts[k], geo.W), PROBE, pr, cp) for k in geo.nbr(i, j)]
        worst = max(worst, max(float((a - b).abs().max()) for a, b in zip(ref, got)))
    check("塊的取法(等變性):塊的 9 個出射波 = 場平移到該塊置中後的 3x3s8(逐位元)", worst == 0.0, f"最大差 {worst:.1e}")
    # (3) 網路路徑
    net, norm, _ = load_p6b4(0, cfg, pr, dev)
    fin, _ = net_patches(net, norm, d["counts"], geo, cp, bs, [(3, 3)])
    direct = g.predict(net, [d["counts"][k] for k in geo.nbr(3, 3)], bs, norm, GROUP, cp)
    e3 = float((fin[0] - torch.polar(direct[:, 0], direct[:, 1])).abs().max())
    old = json.load(open(P5_RAW))["results"][GROUP]
    diffs = []
    for s in SEEDS:
        dd = s5d.setup_seed(s, s5e.val_seeds, dev)
        nt, nm, _ = load_p6b4(s, dd["cfg"], dd["pr"], dev)
        pred = g.predict(nt, dd["counts"], dd["bs"], nm, GROUP, dd["cp"])
        v = s5.field_metrics(s5c.to_field(torch.polar(pred[:, 0], pred[:, 1]), dd["cfg"], dd["F"]), dd["O"], dd["R0"])["nerr_ph"]
        diffs.append(absdiff(v, old[SEEDS.index(s)]["net"]["nerr_ph"]))
        del nt, dd
    check(f"網路路徑:中央塊經本程式 = 直接推論(差 0);P6B4 在驗證場的 3 × 3 量測 = §19.11(< {REPRO_TOL:.0e})",
          e3 == 0.0 and max(diffs) < REPRO_TOL, f"中央塊差 {e3:.1e};重現差 {' / '.join(f'{x:.1e}' for x in diffs)}")
    # (4) 合併
    origins = [geo.origin(i, j) for (i, j) in PATCH25]
    tp = torch.stack([O[:, y:y + geo.S, x:x + geo.S] for (y, x) in origins])                     # [25, B, S, S]
    errs = {}
    for tag, kw in [("等權", dict(sync=False, weighted=False)), ("加權", dict(sync=False)), ("同步", dict())]:
        out, _, _ = merge(tp, origins, geo.R0box, geo.Wbox, geo.F, **kw)
        errs[tag] = float(np.nanmax(per_sample(out, O, geo.U).cpu().numpy()))
    gen = torch.Generator().manual_seed(5)
    rph = torch.exp(1j * (torch.rand(len(origins), O.shape[0], generator=gen) * 2 * np.pi)).to(dev)[:, :, None, None]
    out_s, _, _ = merge(tp * rph, origins, geo.R0box, geo.Wbox, geo.F)
    out_n, _, _ = merge(tp * rph, origins, geo.R0box, geo.Wbox, geo.F, sync=False)
    e_s = float(np.nanmax(per_sample(out_s, O, geo.U).cpu().numpy()))
    e_n = float(np.nanmin(per_sample(out_n, O, geo.U).cpu().numpy()))
    out_t, _, _ = merge(tp, origins, geo.R0box, geo.Wbox, geo.F)                                  # 已一致的塊:同步不得整體轉動
    e_t = float(((out_t - O) * geo.U).abs().max())
    win = torch.stack([O[:, y:y + geo.W, x:x + geo.W] * geo.fp for (y, x) in geo.starts])
    kw_t, kc, _ = merge_windows(win, geo)
    e_k = float(((kw_t - O) * kc).abs().max())
    check("合併:真值的 25 塊 → 三種合併皆還原真值(< 1e-6);各塊乘上隨機整體相位 → 同步後還原(< 1e-6)、不同步 > 0.1;"
          "已一致的塊(網路的 25 塊、K-HIO 的 49 個視窗)同步後**不需對齊**即 = 真值(< 1e-5)",
          max(errs.values()) < 1e-6 and e_s < 1e-6 and e_n > 0.1 and e_t < 1e-5 and e_k < 1e-5,
          "、".join(f"{k} {v:.1e}" for k, v in errs.items()) + f";隨機相位:同步 {e_s:.1e}、不同步(最小){e_n:.2f};"
          f"不對齊:25 塊 {e_t:.1e}、49 窗 {e_k:.1e}")
    # (5) 整張物理級
    b0, S = geo.b0, geo.S
    c9 = [d["counts"][k] for k in geo.nbr(3, 3)]
    inp = g.prep(c9, bs, norm, GROUP, cp)
    O0, outs = net(inp, all_stages=True)
    gb = GlobalStages.from_box(net)
    x = O0
    for k in range(net.K):
        x = gb.stage(k, x, inp["meas"])
    ref = torch.polar(outs[-1][:, 0], outs[-1][:, 1])
    e5a = float((x - ref).abs().max())
    gc = GlobalStages(net, geo.rel, geo.S, pr, bs, dev)                                      # G 實際用的建構方式
    x = O0
    for k in range(net.K):
        x = gc.stage(k, x, inp["meas"])
    e5a = max(e5a, float((x - ref).abs().max()))
    gs = GlobalStages(net, geo.starts, geo.F, pr, bs, dev)
    Ot = s5.epie_init(O.shape[0], geo.F, cfg.phase_max, 3, dev)
    meas = measured(d["counts"], cp, dev)
    eps0, net.eps_e = net.eps_e, 0.0                                           # 同 scan_5g 的單元測試:ε = 0 時應完全等於 AP
    one = gs.ap_step(Ot.clone(), meas)
    net.eps_e = eps0
    ap1 = s5b.ap(d["counts"], bs, cp, pr, geo.starts, [1], False, Ot)[1]
    e5b = float((one - ap1).abs().max())
    check("整張物理級:通用物理級(兩種建構方式)在 80 × 80、9 個視窗上 = P6B4 前向(< 1e-5);49 個視窗的一次物理步(ε = 0)= scan_5b.ap(< 1e-5)",
          e5a < 1e-5 and e5b < 1e-5, f"{e5a:.1e} / {e5b:.1e}")
    # (6) K-HIO 的物體轉換
    Psafe = torch.where(geo.fp, pr["P"], torch.ones_like(pr["P"]))
    worst = 0.0
    for k in [0, CEN49, 48]:
        y, x = geo.starts[k]
        ps_ = pb.make_psi(s5.crop(d["fields"], y, x, geo.W), PROBE, pr, cp)
        psi = torch.polar(ps_[:, 0], ps_[:, 1])
        o = torch.where(geo.fp, psi / Psafe, torch.zeros_like(psi))
        worst = max(worst, float(((o - O[:, y:y + geo.W, x:x + geo.W]) * geo.fp).abs().max()))
    check("K-HIO 的物體轉換:真值的出射波 → 各視窗探針範圍內 = 真值物體(< 1e-6)", worst < 1e-6, f"最大差 {worst:.1e}")
    # (7) 指標
    est = O * torch.exp(torch.tensor(0.7j, device=dev)) + 0.05 * torch.randn_like(O.real)
    fm = s5.field_metrics(est, O, geo.U)["nerr_ph"]
    pm = float(np.nanmean(per_sample(est, O, geo.U).cpu().numpy()))
    check("指標:逐樣本平均 = field_metrics(U 上;差 / max(1, 值) < 1e-6)", absdiff(pm, fm) / max(1.0, fm) < PS_TOL,
          f"{pm:.6f} vs {fm:.6f}")
    print(f"\n     裝置:{dev}")


# ============================================================================
# 判讀
# ============================================================================
def vals(res, x, key="nerr_ph"):
    return np.array([r[key] for r in res[x]], float)


def report(env, res, diag, ps, tim, iconf, smoke):
    v = {}
    Bs = ["64", "512"]
    if smoke:
        print("\n" + "!" * 100)
        print("迷你流程:場數少 → 數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    print("\n" + "=" * 100)
    print("計時(ms / 整張影像;本 job 重新量)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {tim[B][k]:.4f}" for k in tim[B]))
    print("\n" + "=" * 100)
    print("整張影像(U 上的 nerr_ph,只對齊一個整體相位;3 seeds)與 vs 整張包絡")
    print("=" * 100)
    for x in PIPES:
        a = vals(res, x)
        cov = np.mean([r["cover_frac"] for r in res[x]])
        tag = "(主要)" if x == PRIMARY else "(探索性)"
        print(f"  {x}{tag}:{a.mean():.4f}({' '.join(f'{y:.4f}' for y in a)})  U 的覆蓋 {100 * cov:.1f}%")
        gv = {"nerr": list(a), "cover": cov}
        for B in Bs:
            tn = tim[B][f"net:{x}"]
            e = envelope(env, tn, tim[B])
            r = s5b.ratio(a, e[2])
            gv[B] = {"t_net": tn, "opp": [e[0], e[1], list(e[2]), e[3]], "ratio": list(r)}
            print(f"     batch {B}:{tn:.3f} ms vs 對手 {s5b.fmt_conf(e[0])}({e[1]} 次){e[2].mean():.4f} → 比值 {r[1]:.3f}"
                  f"(z {r[2]:+.1f})→ {r[0]}")
        gv["useful"] = gv["64"]["ratio"][0] == "較準"
        gv["robust"] = gv["useful"] and gv["512"]["ratio"][0] == "較準"
        print(f"     ▶ {'穩健有用' if gv['robust'] else ('有用(只在 batch 64)' if gv['useful'] else '未達有用')}")
        v[x] = gv
    print("\n" + "=" * 100)
    print("判讀(階段六協定 §七,結果出來前寫定)")
    print("=" * 100)
    print(f"  (w1) S25 vs 整張包絡:{'穩健有用' if v['S25']['robust'] else ('只在 b64 有用' if v['S25']['useful'] else '未達有用')}")
    pooled = np.array(diag["pooled"])
    loss = s5b.ratio(vals(res, "S25"), pooled)
    lab = ("幾乎無損(合併反而較準)" if loss[1] < 0.90 else "幾乎無損") if loss[1] <= 1.10 else ("小" if loss[1] <= 1.25 else "明顯")
    print(f"  (w2) 拼接損失:S25 整張 {vals(res, 'S25').mean():.4f} / 逐塊合併 {pooled.mean():.4f} = {loss[1]:.3f}(z {loss[2]:+.1f})→ {lab}")
    v["w2"] = {"ratio": list(loss), "label": lab, "pooled": list(pooled)}
    r3 = s5b.ratio(vals(res, "S25"), vals(res, "S25w"))
    r3e = s5b.ratio(vals(res, "S25w"), vals(res, "S25eq"))
    ph = diag["phase"]
    print(f"  (w3) 相位同步:S25 vs 無同步 {r3[1]:.3f}(z {r3[2]:+.1f});加權 vs 等權 {r3e[1]:.3f}(z {r3e[2]:+.1f});"
          f"相鄰塊相位差(同步前)中位數 {np.mean([p['median'] for p in ph]):.3f} rad、第 90 百分位 {np.mean([p['p90'] for p in ph]):.3f}、"
          f"> 0.1 rad {100 * np.mean([p['frac_gt_0.1'] for p in ph]):.1f}%")
    v["w3"] = {"sync": list(r3), "weight": list(r3e), "phase": ph}
    for x in ["S9", "G25", "G9"]:
        r = s5b.ratio(vals(res, x), vals(res, "S25"))
        t64, t512 = tim["64"][f"net:{x}"] / tim["64"]["net:S25"], tim["512"][f"net:{x}"] / tim["512"]["net:S25"]
        print(f"  (w4) {x} vs S25:誤差比值 {r[1]:.3f}(z {r[2]:+.1f});時間 × {t64:.2f}(b64)/ × {t512:.2f}(b512);"
              f"vs 包絡:{'穩健有用' if v[x]['robust'] else ('只在 b64 有用' if v[x]['useful'] else '未達有用')}")
        v[f"w4_{x}"] = {"ratio": list(r), "t64": t64, "t512": t512}
    ring = np.array(diag["ring"])
    print(f"  (w5) 分圈的逐塊合併誤差(各圈所有樣本的能量加總;中心 / 第 1 圈 / 第 2 圈):" + " / ".join(f"{x:.4f}" for x in ring.mean(0))
          + f";第 2 圈 / 中心 = {ring.mean(0)[2] / max(ring.mean(0)[0], 1e-12):.2f}")
    v["w5"] = ring.tolist()
    cap = diag["capture"]
    print(f"  (w6) 資料殘差 vs 逐塊誤差:Spearman {np.nanmean(diag['resid_rho']):.2f};殘差最高 5% 抓到失敗塊(> 0.1)"
          f" {100 * np.nanmean([c['caught'] for c in cap]):.0f}%(失敗塊平均 {np.mean([c['n_fail'] for c in cap]):.0f} 個、"
          f"命中率 {100 * np.mean([c['precision'] for c in cap]):.0f}%)")
    v["w6"] = {"rho": diag["resid_rho"], "capture": cap}
    print(f"  (w7) 中央塊的誤差 {np.mean(diag['center']):.4f}(逐 seed {' / '.join(f'{x:.4f}' for x in diag['center'])});"
          f"階段五驗證場 P6B4 0.0195(雜訊不同,描述);全部 25 塊平均 {np.mean(diag['patch']):.4f}")
    v["w7"] = {"center": diag["center"], "patch_mean": diag["patch"]}
    rob = [x for x in PIPES if v[x]["robust"]]
    choice = None
    if rob:
        best = min(rob, key=lambda x: vals(res, x).mean())
        close = [x for x in rob if x != best and s5b.ratio(vals(res, x), vals(res, best))[0] == "相近"]
        choice = min([best] + close, key=lambda x: tim["64"][f"net:{x}"])
    v["choice"] = choice
    if choice:
        print("     (相近時以 batch 64 的時間決定較快者)")
    print(f"\n  ▶ 帶到階段七的整張流程(事先寫定的選法):{choice if choice else '無(沒有穩健有用的流程)→ 照實回報,依 §八 討論'}"
          + ("  [smoke:無意義]" if smoke else ""))
    print("\n逐樣本(整張誤差;3 seeds 平均)")
    for k in ["iter_b64", "iter_b512", "S25w"] + PIPES:
        rows = [[np.nanmean(e), np.nanmedian(e), np.nanpercentile(e, 90), np.nanmean(e < 0.05), np.nanmean(e < 0.1),
                 np.nanmax(e)] for e in ps[k]]
        m = np.mean(rows, 0)
        name = {"iter_b64": f"迭代法 b64({s5b.fmt_conf(iconf['64'][0])} ×{iconf['64'][1]})",
                "iter_b512": f"迭代法 b512({s5b.fmt_conf(iconf['512'][0])} ×{iconf['512'][1]})",
                "S25w": "S25(無同步)"}.get(k, k)
        print(f"  {name:<44} 平均 {m[0]:.4f}  中位數 {m[1]:.4f}  第 90 百分位 {m[2]:.4f}  < 0.05 {100 * m[3]:.0f}%"
              f"  < 0.1 {100 * m[4]:.0f}%  最差 {m[5]:.3f}")
    for ref in ["iter_b512", "iter_b64"]:
        w = [float(np.nanmean(a < b)) for a, b in zip(ps[PRIMARY], ps[ref])]
        print(f"  S25 逐樣本比「{ref}」準的比例:" + " / ".join(f"{100 * x:.1f}%" for x in w))
        v[f"s25_beats_{ref}"] = w
    return v


# ============================================================================
# 圖
# ============================================================================
def figures(env, res, diag, ps, keep, tim, iconf, out_dir, where):
    plt = s5._plt()
    if plt is None:
        check("matplotlib 可用(畫圖需要)", False)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    geo, O, cfg = keep["geo"], keep["O"], keep["cfg"]
    U = geo.U.cpu()
    pm = cfg.phase_max
    ys, xs = torch.where(U)
    y0, y1 = max(0, int(ys.min()) - MARGIN), min(geo.F, int(ys.max()) + 1 + MARGIN)
    x0, x1 = max(0, int(xs.min()) - MARGIN), min(geo.F, int(xs.max()) + 1 + MARGIN)
    Uc = U[y0:y1, x0:x1].numpy()
    M, _ = geo.masks(PATCH25)
    Mc = M[:, y0:y1, x0:x1].cpu().numpy()

    def align(est, i):
        m = U.to(est.real.dtype)
        c = ((O[i] * m) * (est[i] * m).conj()).sum()
        return est[i] * torch.exp(1j * torch.angle(c))

    def show(ax, im, cm, lo, hi):
        cmap = plt.get_cmap(cm).copy()
        cmap.set_bad("#d9d9d9")
        hnd = ax.imshow(im, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.imshow(np.where(Uc, np.nan, 0.0), cmap="Greys", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        return hnd

    def cbar(fig, hnd, axrow, label):
        cb = fig.colorbar(hnd, ax=list(axrow), fraction=0.012, pad=0.01)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    err0 = {k: ps[k][0] for k in ps}
    lab = {"iter_b64": f"iterative b64: {s5b.fmt_conf(iconf['64'][0])} x{iconf['64'][1]}\n(= S25's time at batch 64)",
           "iter_b512": f"iterative b512: {s5b.fmt_conf(iconf['512'][0])} x{iconf['512'][1]}\n(= S25's time at batch 512)",
           "S25w": "S25 without phase sync", "S25": "S25 (sync + weighted)", "G25": "G25 (whole-image stages)"}
    for k, tk in [("S25w", "S25"), ("S25", "S25"), ("G25", "G25")]:
        lab[k] += f"\n(b64 {tim['64'][f'net:{tk}']:.2f} ms)"
    cols = [("truth", None)] + [(k, keep[k]) for k in ["iter_b64", "iter_b512", "S25w", "S25", "G25"]]
    valid = np.isfinite(err0[PRIMARY])
    idx = np.where(valid)[0][np.argsort(err0[PRIMARY][valid], kind="stable")]
    picks = {f"p{p}": int(idx[min(len(idx) - 1, int(round(p / 100 * (len(idx) - 1))))]) for p in PCTS}
    picks["worst"] = int(idx[-1])
    rng = np.random.default_rng(RAND_SEED)
    rand = sorted(int(i) for i in rng.choice(O.shape[0], size=min(N_RANDOM, O.shape[0]), replace=False))
    for name, ids in list((k, [v]) for k, v in picks.items()) + [("random", rand)]:
        nr = 3 if len(ids) == 1 else 2 * len(ids)
        fig, axs = plt.subplots(nr, len(cols), figsize=(2.4 * len(cols) + 0.6, 2.5 * nr), squeeze=False, constrained_layout=True)
        for r, i in enumerate(ids):
            t = O[i, y0:y1, x0:x1].numpy()
            for c, (k, f) in enumerate(cols):
                z = t if f is None else align(f, i)[y0:y1, x0:x1].numpy()
                rr = 3 * r if len(ids) == 1 else 2 * r
                ha = show(axs[rr, c], np.abs(z), "gray", 0, 1)
                hp = show(axs[rr + 1, c], np.where(np.abs(t) > AMP_MIN, np.angle(z), np.nan), "viridis", 0, pm)
                if len(ids) == 1:
                    if f is None:
                        axs[rr + 2, c].axis("off")
                    else:
                        he = show(axs[rr + 2, c], np.abs(z - t), "magma", 0, 0.5)
                ttl = "truth" if f is None else f"{lab[k]}\nnerr {err0[k][i]:.4f}"
                axs[rr, c].set_title(ttl, fontsize=7)
            axs[rr, 0].set_ylabel(f"#{i}\namplitude", fontsize=8)
            axs[rr + 1, 0].set_ylabel(f"#{i}\nphase", fontsize=8)
            cbar(fig, ha, axs[rr], "amplitude")
            cbar(fig, hp, axs[rr + 1], "phase (rad)")
            if len(ids) == 1:
                axs[rr + 2, 0].set_ylabel("|error|", fontsize=8)
                cbar(fig, he, axs[rr + 2], "|error|")
        what = {"worst": "worst sample", "random": f"random samples (numpy seed {RAND_SEED})"}.get(name, f"{name[1:]}th percentile of S25")
        fig.suptitle(f"{where}: whole image (union of 25 R0 = scored region U; outside dimmed). {what}; seed-0 networks; "
                     f"one global phase aligned over U", fontsize=8.5)
        save(fig, f"w_{name}.png")

    # 平均誤差圖(接縫)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, k in zip(axs, ["iter_b512", "S25", "G25"]):
        est = keep[k]
        m = U.to(est.real.dtype)
        c = ((O * m) * (est * m).conj()).sum((1, 2))
        al = est * torch.exp(1j * torch.angle(c))[:, None, None]
        num = ((al - O).abs() ** 2).mean(0)[y0:y1, x0:x1].numpy()
        den = float((O.abs() ** 2)[:, y0:y1, x0:x1][:, torch.from_numpy(Uc)].mean())
        hnd = ax.imshow(np.where(Uc, num / den, np.nan), cmap="magma", vmin=0, vmax=None, interpolation="nearest")
        ax.contour(Mc[PATCH25.index((3, 3))].astype(float), levels=[0.5], colors="#2a78d6", linewidths=0.8)
        for (i, j) in PATCH25:
            yy, xx = geo.starts[geo.pos(i, j)]
            ax.plot(xx + geo.W / 2 - 0.5 - x0, yy + geo.W / 2 - 0.5 - y0, ".", color="#2a78d6", ms=3)
        ax.set_title(lab[k].replace("\n", " "), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(hnd, ax=ax, fraction=0.045)
    fig.suptitle(f"{where}: mean |error|² / mean |truth|² per pixel (seed-0 networks, all samples; each panel has its own "
                 "colour scale). Blue dots = the 25 patch centres, blue line = one R0 (seams would show as a grid)", fontsize=9)
    save(fig, "w_mean_error_map.png")

    # 時間 vs 誤差
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, B in zip(axs, ["64", "512"]):
        tt = sorted({cfg_time(c, it, tim[B]) for c in configs() for it in stops(c)} - {0.0})
        ax.plot(tt, [envelope(env, T, tim[B])[2].mean() for T in tt], color="#0b0b0b", lw=2, drawstyle="steps-post",
                label="whole-image iterative envelope")
        for x in PIPES:
            a = vals(res, x)
            tn = tim[B][f"net:{x}"]
            e = envelope(env, tn, tim[B])
            ax.plot([tn], [a.mean()], "o", ms=8, color=COL[x], zorder=4,
                    label=f"{x}: {a.mean():.4f} at {tn:.2f} ms (ratio {s5b.ratio(a, e[2])[1]:.3f})")
            ax.plot([tn], [e[2].mean()], "x", ms=7, color=COL[x], zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{PROBE}, whole image, batch {B} ({where})", fontsize=10)
        ax.set_xlabel("time per whole image (ms)")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    axs[0].set_ylabel("nerr on U (one global phase)")
    fig.tight_layout()
    save(fig, "w_vs_envelope.png")

    hist = diag["phase"][0]["hist"]
    edges = np.linspace(0, np.pi, 61)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge", color="#2a78d6", edgecolor="white")
    ax.axvline(0.1, color="#0b0b0b", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("|global-phase difference| between adjacent patches before sync (rad)")
    ax.set_ylabel("adjacent pairs")
    ax.set_title(f"{where}: are the 25 patches' global phases consistent? (seed-0 networks; dotted = 0.1 rad)", fontsize=9)
    fig.tight_layout()
    save(fig, "w_phase_diff.png")


# ============================================================================
# 流程
# ============================================================================
def dump_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, default=float)
    os.replace(tmp, path)


def flow(dev, n, out_dir, fig_dir, smoke):
    where = "validation fields (smoke)" if smoke else "validation fields, 7x7 scan"
    cfg0 = s5.load_cfg(0)
    import probe_4_2b as pb
    pr0 = pb.build_probes(cfg0, dev)[PROBE]
    net0, norm0, _ = load_p6b4(0, cfg0, pr0, dev)
    t_start = time.time()
    print("\n  計時(ms / 整張影像)", flush=True)
    tim = timing(dev, (net0, norm0))
    print(f"  (計時完成,經過 {time.time() - t_start:.0f} 秒)", flush=True)
    del net0
    env_path = out_dir / f"scan6a_env{SUF}.json"
    meta = {"n": n or (s5b.N_FIELDS or cfg0.eval_n), "field_seed": FIELD_SEED, "noise_base": NOISE_BASE, "quick": QUICK,
            "iters": list(s5b.ITERS), "script_md5": md5(Path(__file__).resolve())}
    old = json.load(open(env_path)) if env_path.exists() else {}
    if old.get("meta") == meta:
        print(f"\n  整張包絡已存在(先前中斷時留下),沿用:{env_path}")
        env = old["envelope"]
    else:
        print("\n  整張迭代法包絡(4 演算法 × 6 起點 + K-HIO)", flush=True)
        env = measure_envelope(dev, n)
        dump_json({"envelope": env, "meta": meta}, env_path)
        print(f"  存檔:{env_path}(經過 {time.time() - t_start:.0f} 秒)")
    print("\n  網路的整張流程與診斷", flush=True)
    res, diag, ps, keep, iconf = measure_nets(dev, env, tim, n)
    print(f"  (網路完成,經過 {time.time() - t_start:.0f} 秒)", flush=True)
    raw = out_dir / f"scan6a_raw{SUF}.json"
    dump_json({"results": res, "diag": diag, "timing": tim, "iconf": iconf, "checks_ok": all_ok(), "smoke": smoke,
               "per_sample": {k: [np.asarray(e).tolist() for e in v] for k, v in ps.items()}}, raw)
    print(f"  原始結果先存檔:{raw}")
    if not all_ok():
        print("\n⚠️ 上面有檢查未通過 —— 以下的判讀先不要採信,把完整輸出貼給 Claude")
    verdict = report(env, res, diag, ps, tim, iconf, smoke)
    final = out_dir / f"scan6a{SUF}.json"
    dump_json({"results": res, "diag": diag, "timing": tim, "verdict": verdict, "checks_ok": all_ok(), "smoke": smoke}, final)
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        figures(env, res, diag, ps, keep, tim, iconf, fig_dir, where)
    except Exception as e:                                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"畫圖完成(判讀不受影響,已存於 {final.name})", False, f"{type(e).__name__}: {e}")
    dump_json({"results": res, "diag": diag, "timing": tim, "verdict": verdict, "checks_ok": all_ok(), "smoke": smoke}, final)


def check_files():
    need = [Path(f) for f in ("scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py", "scan_5e.py", "scan_5g.py", "scan_5h.py",
                              "probe_4_2b.py", "probe_4_1.py", "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    need += [s5c.run_dir(GROUP, PROBE, s, N_TRAIN) / f for s in SEEDS for f in ("final.pt", "result.json")]
    need += [P5_RAW]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在(含 P6B4 的 3 個模型)", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = md5(Path(__file__).resolve())
    print(f"scan_6a.py md5 {me}")
    if not check_files():
        print("\n❌ 缺檔案,停下來")
        sys.exit(1)
    if a.check:
        checks(dev)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.smoke:
        print("#" * 70)
        print(f"迷你全流程(--smoke):{SMOKE_FIELDS} 個場;輸出到 {SMOKE_DIR}")
        print("#" * 70)
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)                                      # 只刪這個 smoke 專用資料夾
        SMOKE_DIR.mkdir(parents=True)
        t0 = time.time()
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
        flow(dev, SMOKE_FIELDS, SMOKE_DIR, SMOKE_DIR / "figs", smoke=True)
        print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒")
        if all_ok():
            json.dump({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
            print("\n✅ 迷你全流程全部通過 → 可以送出正式的 job(sbatch run_scan6a.sh)")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    for p in (RAW_JSON, OUT_JSON):
        if p.exists():
            raise SystemExit(f"❌ {p} 已存在:正式結果已經有了。為避免覆蓋,先告訴 Claude")
    if not PASSED.exists():
        raise SystemExit(f"❌ 找不到 {PASSED}:先在 dev 節點跑 python scan_6a.py --smoke,全部通過才可送件")
    pm = json.load(open(PASSED))["script_md5"]
    if pm != me:
        raise SystemExit(f"❌ smoke 通過時的 scan_6a.py(md5 {pm})與現在的({me})不同:用現在的版本重跑 --smoke")
    print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")
    checks(dev)
    if not all_ok():
        print("\n❌ 內建檢查未通過 → 沒有量測。把輸出貼給 Claude")
        sys.exit(1)
    flow(dev, None, RUN_ROOT, FIG_DIR, smoke=False)
    print("\n" + ("✅ 內建檢查全部通過" if all_ok() else "❌ 有項目未通過(判讀先不要採信)"))
    print("把完整輸出與 figs_scan6a/ 的圖傳給 Claude")
    sys.exit(0 if all_ok() else 1)


if __name__ == "__main__":
    main()
