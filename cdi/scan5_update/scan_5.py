#!/usr/bin/env python
"""階段五 5-1 + 5-2:重疊掃描的模擬與迭代法基準(階段五協定 §二、§三;不需訓練)。

問:有了重疊掃描(ptychography),迭代法能比單張(4-2b 的 K-HIO)好多少、快多少?
   這也定出 5-3 網路要贏過的對手。

物體場:112×112,由 4×4 張 28×28 的程序生成物體拼貼(與先前各階段相同的產生器與特徵尺度)。
視窗:每個掃描位置取 64×64(= 先前的畫布與偵測器),出射波 ψ_j = P · O(視窗)。
探針:DEF2(4-2b 選定)與 DISK(對照),沿用 probe_4_2b.py 的探針(逐位元相同)。
掃描(以場中心為對稱中心,位移為整數 px):
  1x1      單一位置(= 單張,基準)
  2x2s8    2×2,步距 8 px
  3x3s8    3×3,步距 8 px(高重疊)
  3x3s16   3×3,步距 16 px(低重疊)
  3x3s8d9  同 3x3s8,每張劑量 ÷ 9(總劑量 = 1x1;分開「資訊」與「劑量」)
方法:
  ePIE-C   ePIE(已知探針、不更新探針、α = 1),每個位置更新後把該視窗的物體投影到 |O| ≤ 1、相位 ∈ [0, φmax](與 K-HIO 相同的先驗)
  ePIE     同上,不投影(標準 ePIE,參考)
  K-HIO    1x1 單張、知道整個探針的 HIO(probe_4_2b.hio_probe;support = 整個視窗,物體不再只在方框內)
評分:物體 O(不是 ψ),主區域 R0 = 場中央探針的範圍(|P| ≥ 0.1,約 448 px)—— 所有掃描都涵蓋、可共同比較;
     另報各掃描自己的照明範圍(各位置 |P| ≥ 0.1 的聯集)。
主指標:只對齊整體相位的正規化誤差 nerr_ph = Σ_R |e^{iθ}Ô − O|² / Σ_R |O|²(實驗上做得到的對齊);
       另報完整對齊(平移 + 翻轉 + 整體相位)的 nerr_al、選翻轉 / 有平移比例。

用法(需在計算節點執行;需 probe_4_2b.py、probe_4_1.py、ambiguity_check.py):
    python scan_5.py --check   # 幾何 + 單元測試(約 1–3 分鐘)
    python scan_5.py           # 完整量測(用 run_scan5.sh 送 job)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Cfg                                          # noqa: E402

RUN_ROOT = Path(os.environ.get("CDI_RUN_ROOT", "/work/elviss0915/runs"))
OUT_JSON = RUN_ROOT / "scan5.json"
FIG_DIR = Path("figs_scan5")
BASE = "ideal_base"
SEEDS = [0, 1, 2]
PROBES = ["DEF2", "DISK"]
TILES = 4                                   # 場 = 4×4 張 28×28 的物體
SCANS = {                                   # 名稱: (n, 步距 px, 每張劑量倍率)
    "1x1": (1, 0, 1.0),
    "2x2s8": (2, 8, 1.0),
    "3x3s8": (3, 8, 1.0),
    "3x3s16": (3, 16, 1.0),
    "3x3s8d9": (3, 8, 1.0 / 9),
}
METHODS = ["ePIE-C", "ePIE"]
EP_ITERS = [1, 2, 5, 10, 20, 50, 100, 200, 500]          # ePIE:完整掃過一次所有位置 = 1 次
K_ITERS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]     # K-HIO(1x1)
BUDGET_K = [100, 1000]                      # 同時間比較的預算:K-HIO 100 / 1000 次的時間
MAIN_SWEEPS = 100
TAU = 0.1
FIELD_SEED_OFFSET = 777                     # 測試場的物體 seed = test_seed + 777(與先前測試集不同)
CALIB_FIELDS = 256
N_FIELDS = None                             # None = cfg.eval_n(512)
TIME_REPS = 5
GOOD_NERR = 0.25

QUICK = os.environ.get("CDI_QUICK") == "1"
if QUICK:
    EP_ITERS, K_ITERS, BUDGET_K, MAIN_SWEEPS = [1, 5, 20], [5, 20, 60], [20, 60], 20
    CALIB_FIELDS, N_FIELDS, TIME_REPS = 8, 16, 2

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


# ============================================================================
# 物體場與掃描幾何
# ============================================================================
def tile_size(cfg):
    return cfg.digit


def field_size(cfg):
    return TILES * tile_size(cfg)


def make_fields(cfg, n, seed, device=None, roll=True):
    """n 個物體場 [n, 2, F, F]:每場由 TILES² 張獨立的程序生成物體(28×28,原本的中央區域)拼貼,
    再整場循環平移一個隨機量(每場各自,seed 固定)。
    平移的理由:產生器的物體偏向 28×28 的中央,不平移時場的統計隨位置改變(拼接處較空),
    各掃描位置照到的物體量會不同;隨機平移後各位置在期望上相同(劑量校準才適用於所有位置)。"""
    import probe_4_1 as p41
    d = tile_size(cfg)
    s0 = (cfg.canvas - d) // 2
    objs = p41.make_home(cfg, n * TILES * TILES, seed=seed)[:, :, s0:s0 + d, s0:s0 + d]
    f = objs.reshape(n, TILES, TILES, 2, d, d).permute(0, 3, 1, 4, 2, 5).reshape(n, 2, TILES * d, TILES * d)
    f = f.contiguous()
    if roll:
        g = torch.Generator().manual_seed(seed + 1)
        sh = torch.randint(0, d, (n, 2), generator=g)
        f = torch.stack([torch.roll(f[i], (int(sh[i, 0]), int(sh[i, 1])), dims=(-2, -1)) for i in range(n)])
    return f.to(device) if device is not None else f


def offsets(n, step):
    if n == 1:
        v = [0]
    elif n == 2:
        v = [-step // 2, step // 2]
    elif n == 3:
        v = [-step, 0, step]
    else:
        raise ValueError(n)
    return [(dy, dx) for dy in v for dx in v]


def window_starts(cfg, scan):
    """各位置視窗(W×W)左上角在場中的座標。場中心 (F−1)/2 = 視窗中心 (W−1)/2 + 起點,位移對稱。"""
    n, step, _ = SCANS[scan]
    F, W = field_size(cfg), cfg.canvas
    base = F // 2 - W // 2
    return [(base + dy, base + dx) for dy, dx in offsets(n, step)]


def footprint_mask(cfg, pr, starts, device):
    """各位置 |P| ≥ 0.1 的聯集(場座標)。"""
    F, W = field_size(cfg), cfg.canvas
    m = torch.zeros(F, F, dtype=torch.bool, device=device)
    fp = pr["Pa"] >= TAU * pr["Pa"].max()
    for y0, x0 in starts:
        m[y0:y0 + W, x0:x0 + W] |= fp
    return m


def illumination(cfg, pr, starts, device):
    F, W = field_size(cfg), cfg.canvas
    w = torch.zeros(F, F, device=device)
    for y0, x0 in starts:
        w[y0:y0 + W, x0:x0 + W] += pr["Pa"] ** 2
    return w


def crop(fields, y0, x0, W):
    return fields[..., y0:y0 + W, x0:x0 + W]


# ============================================================================
# 劑量
# ============================================================================
_RATIO = {}


def scan_cfg(cfg, name, pr, dose=1.0):
    """每張繞射圖的平均總光子數 = 平面波慣例(4-1 起)× dose。
    能量比在校準場(≠ 測試場)上計算:3x3s16 九個視窗的 Σ|P·O_視窗|² 平均 / 標準校準集的平均 Σ|O|²。"""
    import probe_4_1 as p41
    import probe_4_2b as pb
    from src.physics import calibrate_flux
    dev = pr["P"].device
    if name not in _RATIO:
        cal = make_fields(cfg, CALIB_FIELDS, seed=p41.CALIB_SEED + 1, device=dev)
        psi = torch.cat([pb.make_psi(crop(cal, y0, x0, cfg.canvas), name, pr, cfg)
                         for y0, x0 in window_starts(cfg, "3x3s16")])
        ref = p41.make_home(cfg, p41.CALIB_N, seed=p41.CALIB_SEED)
        _RATIO[name] = calibrate_flux(psi) / calibrate_flux(ref)
    c = Cfg.from_dict(cfg.to_dict())
    c.ref_energy = cfg.ref_energy * _RATIO[name]
    c.photons_per_pix = cfg.photons_per_pix * dose
    return c


def measure_scan(fields, name, pr, cfg_p, bs, starts, seed):
    """每個位置一張繞射圖(依位置順序抽 Poisson;seed 固定)。回傳 counts 清單與 ψ 清單。"""
    import probe_4_2b as pb
    from src.physics import forward_measure
    torch.manual_seed(seed)
    psis, counts = [], []
    for y0, x0 in starts:
        psi = pb.make_psi(crop(fields, y0, x0, cfg_p.canvas), name, pr, cfg_p)
        psis.append(psi)
        counts.append(forward_measure(psi, bs, cfg_p))
    return counts, psis


# ============================================================================
# 迭代法
# ============================================================================
@torch.no_grad()
def epie(counts, bs, cfg_p, pr, starts, iters, constrain, O0, seed):
    """ePIE(Maiden & Rodenburg 2009),已知探針、不更新探針、α = 1。
    每次(= 掃過所有位置一次)的位置順序為隨機排列(seed 固定)。
    傅立葉約束與 hio_sw 相同(beamstop 內保留模型的振幅)。回傳 {次數: 物體場 complex [N, F, F]}。"""
    from src.hio import _measured_amp
    W = cfg_p.canvas
    meas = [_measured_amp(c, cfg_p).to(O0.device) for c in counts]
    P = pr["P"]
    Pc = P.conj() / (pr["Pa"] ** 2).max()
    phmax = cfg_p.phase_max
    O = O0.clone()
    gen = torch.Generator().manual_seed(seed)
    want, outs = set(iters), {}
    for k in range(max(iters)):
        for j in torch.randperm(len(starts), generator=gen).tolist():
            y0, x0 = starts[j]
            Oj = O[:, y0:y0 + W, x0:x0 + W]
            psi = P * Oj
            E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
            new_mag = torch.where(bs > 0, meas[j], E.abs())
            psi2 = torch.fft.ifft2(torch.fft.ifftshift(torch.polar(new_mag, torch.angle(E)), dim=(-2, -1)),
                                   norm="ortho")
            On = Oj + Pc * (psi2 - psi)
            if constrain:
                On = torch.polar(On.abs().clamp(max=1.0), torch.angle(On).clamp(0, phmax))
            O[:, y0:y0 + W, x0:x0 + W] = On
        if (k + 1) in want:
            outs[k + 1] = O.clone()
    return outs


def epie_init(n, F, phmax, seed, device):
    """ePIE 的隨機起點:振幅 U(0, 1)、相位 U(0, φmax)(seed 固定)。"""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, F, F, generator=g)
    p = torch.rand(n, F, F, generator=g) * phmax
    return torch.polar(a, p).to(device)


def k_init(counts, cfg_p, seed, device):
    """K-HIO 的隨機起點:量測振幅 × 隨機相位 → 反轉換(同 hio.random_init,但**不套方框**:物體填滿視窗)。"""
    from src.hio import _measured_amp
    g = torch.Generator().manual_seed(seed)
    amp = _measured_amp(counts, cfg_p).to(device)
    pha = (torch.rand(amp.shape, generator=g).to(device) * 2 - 1) * math.pi
    x = torch.fft.ifft2(torch.fft.ifftshift(torch.polar(amp, pha), dim=(-2, -1)), norm="ortho")
    return torch.stack([x.abs().clamp(0, 1), torch.angle(x).clamp(0, cfg_p.phase_max)], dim=1)


def k_to_field(out, pr, cfg, start, F):
    """K-HIO 輸出的 ψ̂(視窗)→ 物體 Ô = ψ̂ / P(只在 |P| ≥ 0.1 處),放回場座標。"""
    psi = torch.polar(out[:, 0], out[:, 1])
    fp = pr["Pa"] >= TAU * pr["Pa"].max()
    o = torch.where(fp, psi / torch.where(fp, pr["P"], torch.ones_like(pr["P"])), torch.zeros_like(psi))
    f = torch.zeros(out.shape[0], F, F, dtype=psi.dtype, device=psi.device)
    y0, x0 = start
    f[:, y0:y0 + cfg.canvas, x0:x0 + cfg.canvas] = o
    return f


# ============================================================================
# 指標
# ============================================================================
@torch.no_grad()
def field_metrics(Oh, O, M):
    """在區域 M 內比較估計 Oh 與真值 O(皆 complex [N, F, F])。空樣本(M 內無物體)不列入。"""
    from ambiguity_check import align
    m = M.to(Oh.real.dtype)
    a, t = Oh * m, O * m
    E = (t.abs() ** 2).sum((1, 2))
    ok = E > 1e-12
    cross = (a * t.conj()).sum((1, 2)).abs()
    nerr_ph = ((a.abs() ** 2).sum((1, 2)) + E - 2 * cross) / E.clamp_min(1e-12)
    al, tw, sh = align(a, t)
    nerr_al = ((al * m - t).abs() ** 2).sum((1, 2)) / E.clamp_min(1e-12)
    good = ok & (nerr_al < GOOD_NERR)
    return {"nerr_ph": float(nerr_ph[ok].mean()), "nerr_ph_median": float(nerr_ph[ok].median()),
            "nerr_al": float(nerr_al[ok].mean()), "nerr_al_median": float(nerr_al[ok].median()),
            "twin_frac": float(tw[ok].float().mean()), "shift_frac": float((sh.abs().sum(1) > 0)[ok].float().mean()),
            "good_frac": float(good.float().sum() / ok.float().sum().clamp_min(1)),
            "twin_good": float(tw[good].float().mean()) if bool(good.any()) else float("nan"),
            "n_empty": int((~ok).sum())}


# ============================================================================
# 計時
# ============================================================================
def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def timing(cfg, probes, dev):
    """同一程序、同一 GPU:ePIE 每次(掃過所有位置;以 DEF2 計,兩個探針的運算相同)與
    K-HIO 每次迭代(各探針分別計)的時間(ms / 張),batch 64 與 512。"""
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    bs = beamstop_mask(cfg, device=dev)
    F, W = field_size(cfg), cfg.canvas
    pr = probes[PROBES[0]]
    ones = torch.ones(W, W, device=dev)
    res = {}
    for B in [64, 512]:
        fields = make_fields(cfg, B, seed=cfg.test_seed + 99, device=dev)
        cp = scan_cfg(cfg, PROBES[0], pr)
        row = {}
        for scan in SCANS:
            if SCANS[scan][2] != 1.0:
                continue
            st = window_starts(cfg, scan)
            counts, _ = measure_scan(fields, PROBES[0], pr, cp, bs, st, seed=0)
            O0 = epie_init(B, F, cfg.phase_max, 0, dev)
            epie(counts, bs, cp, pr, st, [2], True, O0, 0)                      # 暖機
            ts = []
            for _ in range(TIME_REPS):
                _sync(dev)
                t0 = time.perf_counter()
                epie(counts, bs, cp, pr, st, [10], True, O0, 0)
                _sync(dev)
                ts.append((time.perf_counter() - t0) / 10 / B * 1e3)
            row[scan] = float(np.median(ts))
        for name in PROBES:                                                     # K-HIO 的成本隨探針略有不同
            prk = probes[name]
            cpk = scan_cfg(cfg, name, prk)
            c1, _ = measure_scan(fields, name, prk, cpk, bs, window_starts(cfg, "1x1"), seed=0)
            init = k_init(c1[0], cpk, 0, dev)
            pb.hio_probe(init, c1[0], bs, cpk, [5], "K", prk, ones)
            ts = []
            for _ in range(TIME_REPS):
                _sync(dev)
                t0 = time.perf_counter()
                pb.hio_probe(init, c1[0], bs, cpk, [50], "K", prk, ones)
                _sync(dev)
                ts.append((time.perf_counter() - t0) / 50 / B * 1e3)
            row[f"K-HIO:{name}"] = float(np.median(ts))
        res[str(B)] = row
    return res


# ============================================================================
# 檔案、幾何與單元測試
# ============================================================================
def check_files():
    print("=" * 70)
    print("檔案")
    print("=" * 70)
    need = [Path("probe_4_2b.py"), Path("probe_4_1.py"), Path("ambiguity_check.py"), Path("realign_eval.py")]
    for s in SEEDS:
        need += [RUN_ROOT / f"{BASE}_s{s}" / "config_used.json"]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def load_cfg(s=0):
    return Cfg.from_dict(json.load(open(RUN_ROOT / f"{BASE}_s{s}" / "config_used.json")))


@torch.no_grad()
def unit_tests(dev):
    import probe_4_1 as p41
    import probe_4_2b as pb
    from src.physics import beamstop_mask, forward_measure

    print("\n" + "=" * 70)
    print("幾何與單元測試")
    print("=" * 70)
    cfg = load_cfg(0)
    allp = pb.build_probes(cfg, dev)
    probes = {k: allp[k] for k in PROBES}
    F, W = field_size(cfg), cfg.canvas
    bs = beamstop_mask(cfg, device=dev)
    phmax = cfg.phase_max
    geo = {}

    # (1) 掃描幾何:視窗在場內、對稱、重疊率
    ok_in, ok_sym = True, True
    for scan in SCANS:
        st = window_starts(cfg, scan)
        ok_in &= all(0 <= y and y + W <= F and 0 <= x and x + W <= F for y, x in st)
        flipped = sorted((F - W - y, F - W - x) for y, x in st)
        ok_sym &= flipped == sorted(st)
    check(f"所有視窗都在 {F}×{F} 的場內、掃描格點對場中心對稱", ok_in and ok_sym)
    for name, pr in probes.items():
        fp = (pr["Pa"] >= TAU * pr["Pa"].max()).float()
        g = {"footprint_px": int(fp.sum())}
        for scan in SCANS:
            st = window_starts(cfg, scan)
            g[scan] = {"positions": len(st), "region_px": int(footprint_mask(cfg, pr, st, dev).sum())}
            n, step, _ = SCANS[scan]
            if n > 1:
                sh = torch.roll(fp, step, dims=1)
                g[scan]["overlap_adjacent"] = float((fp * sh).sum() / fp.sum())
        geo[name] = g
    for name, g in geo.items():
        print(f"     {name}:範圍 {g['footprint_px']} px;" + ";".join(
            f"{s} 區域 {g[s]['region_px']} px" + (f"、相鄰重疊 {g[s]['overlap_adjacent']:.0%}" if 'overlap_adjacent' in g[s] else "")
            for s in SCANS))

    # (2) 物體場:拼貼的每一塊 = 原產生器的中央 28×28(逐位元)
    fl = make_fields(cfg, 2, seed=5, roll=False)
    d = tile_size(cfg)
    s0 = (cfg.canvas - d) // 2
    ref = p41.make_home(cfg, 2 * TILES * TILES, seed=5)[:, :, s0:s0 + d, s0:s0 + d]
    same = all(torch.equal(fl[i, :, a * d:(a + 1) * d, b * d:(b + 1) * d], ref[i * TILES * TILES + a * TILES + b])
               for i in range(2) for a in range(TILES) for b in range(TILES))
    check("物體場的每一塊 = 產生器的中央 28×28(逐位元)", same)
    ncheck = N_FIELDS or cfg.eval_n
    test = make_fields(cfg, ncheck, seed=cfg.test_seed + FIELD_SEED_OFFSET, device=dev)
    for name, pr in probes.items():
        R0 = footprint_mask(cfg, pr, window_starts(cfg, "1x1"), dev)
        E = ((torch.polar(test[:, 0], test[:, 1]).abs() ** 2) * R0).sum((1, 2))
        geo[name]["R0_empty_frac"] = float((E == 0).float().mean())
        geo[name]["R0_object_frac"] = float(((test[:, 0] > 0) & R0).float().sum((1, 2)).mean() / R0.sum())
    print("     R0 內物體像素比例 / 空樣本:" + " / ".join(
        f"{k} {g['R0_object_frac']:.1%} / {g['R0_empty_frac']:.1%}" for k, g in geo.items()))

    # (3) 前向模型:把 4-2b 的物體放進場的中央視窗,1x1 的 ψ 與 counts = 4-2b(逐位元)
    home = p41.make_home(cfg, 32).to(dev)
    y0, x0 = window_starts(cfg, "1x1")[0]
    fld = torch.zeros(32, 2, F, F, device=dev)
    fld[:, :, y0:y0 + W, x0:x0 + W] = home
    worst = 0.0
    for name, pr in probes.items():
        cp = pb.probe_cfg(cfg, name, pr)
        c_new, p_new = measure_scan(fld, name, pr, cp, bs, [(y0, x0)], seed=123)
        p_old = pb.make_psi(home, name, pr, cfg)
        torch.manual_seed(123)
        c_old = forward_measure(p_old, bs, cp)
        worst = max(worst, float((p_new[0] - p_old).abs().max()), float((c_new[0] - c_old).abs().max()))
    check("單一位置的前向 = 4-2b(ψ 與光子數逐位元相同;兩個探針)", worst == 0.0, f"最大差 {worst:.1e}")

    # (4) 真值為固定點(無雜訊,移到約束邊界內):ePIE-C / ePIE(每種掃描)、K-HIO(1x1)
    small = make_fields(cfg, 8, seed=11, device=dev)
    Oc = torch.polar(small[:, 0], small[:, 1])
    on = Oc.abs() > 0
    Om = torch.polar(Oc.abs() * 0.95, torch.where(on, 0.05 * phmax + 0.9 * torch.angle(Oc).clamp(0, phmax),
                                                   torch.zeros_like(Oc.real)))
    fm = torch.stack([Om.abs(), torch.angle(Om)], 1)
    errs = {}
    for name, pr in probes.items():
        c0 = Cfg.from_dict(scan_cfg(cfg, name, pr).to_dict())
        c0.add_poisson = False
        e = 0.0
        for scan in SCANS:
            st = window_starts(cfg, scan)
            cnt, _ = measure_scan(fm, name, pr, c0, bs, st, seed=0)
            reg = footprint_mask(cfg, pr, st, dev)
            for con in [True, False]:
                out = epie(cnt, bs, c0, pr, st, [20], con, Om, 0)[20]
                e = max(e, float(((out - Om).abs() * reg).max()))
        cnt, psi = measure_scan(fm, name, pr, c0, bs, window_starts(cfg, "1x1"), seed=0)
        ones = torch.ones(W, W, device=dev)
        out = pb.hio_probe(psi[0], cnt[0], bs, c0, [100], "K", pr, ones)[100]
        e = max(e, float((torch.polar(out[:, 0], out[:, 1]) - torch.polar(psi[0][:, 0], psi[0][:, 1])).abs().max()))
        errs[name] = e
    check("真值為固定點(ePIE-C / ePIE × 5 種掃描 20 次、K-HIO 100 次;無雜訊)", all(v < 1e-3 for v in errs.values()),
          " / ".join(f"{k} {v:.1e}" for k, v in errs.items()))

    # (5) 翻轉的機制(無雜訊):場的翻轉解 e^{iφmax}·conj(O(−r)) 是否與所有量測相符
    #     DISK(實數對稱探針)+ 對稱掃描格點:翻轉解在位置 −d 的繞射圖 = 真值在位置 d 的繞射圖(整組量測相同)
    #     → 預期 ePIE 打不破翻轉
    #     DEF2(複數探針):不同 → 被量測排除
    tw = torch.conj(torch.flip(Om, dims=(-2, -1))) * complex(math.cos(phmax), math.sin(phmax))
    twm = torch.stack([tw.abs(), torch.angle(tw)], 1)
    diff = {}
    for name, pr in probes.items():
        c0 = Cfg.from_dict(scan_cfg(cfg, name, pr).to_dict())
        c0.add_poisson = False
        st = window_starts(cfg, "3x3s8")
        a, _ = measure_scan(fm, name, pr, c0, bs, st, seed=0)
        b, _ = measure_scan(twm, name, pr, c0, bs, st, seed=0)
        # 場的翻轉把位置 d 對應到 −d(格點依列優先排列且對稱 → 第 j 個對應第 len−1−j 個)
        diff[name] = max(float(((a[j] - b[len(st) - 1 - j]).abs().sum() / a[j].abs().sum())) for j in range(len(st)))
    print("     翻轉解與真值的繞射圖相對差(3x3s8,無雜訊):" + " / ".join(f"{k} {v:.2e}" for k, v in diff.items()))
    check("DISK + 對稱格點:翻轉解的繞射圖與真值相同(< 1e-4);DEF2:不同(> 1e-2)",
          diff["DISK"] < 1e-4 and diff["DEF2"] > 1e-2)

    # (6) 指標:真值 → 誤差 0、翻轉解 → 判為翻轉;與真值無關的估計 → 選翻轉約 0.5
    R0 = footprint_mask(cfg, probes["DEF2"], window_starts(cfg, "1x1"), dev)
    m1 = field_metrics(Om, Om, R0)
    m2 = field_metrics(tw, Om, R0)
    check("指標:真值的 nerr 為 0、場的翻轉解判為翻轉", m1["nerr_ph"] < 1e-6 and m1["twin_frac"] == 0.0
          and m2["twin_frac"] == 1.0, f"真值 nerr {m1['nerr_ph']:.1e};翻轉解選翻轉 {m2['twin_frac']:.2f}")
    other = torch.polar(test[:, 0], test[:, 1])
    null = field_metrics(torch.roll(other, 1, 0), other, R0)["twin_frac"]
    check("指標對「與真值無關的估計」無偏:選翻轉比例 0.35–0.65" + (";QUICK 張數少,只列出" if QUICK else ""),
          QUICK or 0.35 <= null <= 0.65, f"{null:.2f}")

    # (7) 劑量:ψ 的平均總光子數 / 平面波(不含 beamstop)
    bs0 = beamstop_mask(cfg, radius=-1, device=dev)
    ref = p41.make_home(cfg, ncheck).to(dev)
    torch.manual_seed(0)
    cpl = forward_measure(ref, bs0, cfg)
    ratios = {}
    for name, pr in probes.items():
        cp = scan_cfg(cfg, name, pr)
        cnt, _ = measure_scan(test, name, pr, cp, bs0, window_starts(cfg, "3x3s8"), seed=0)
        ratios[name] = float(torch.stack([c.sum((1, 2)) for c in cnt]).mean() / cpl.sum((1, 2)).mean())
    check(f"每張的平均總光子數 / 平面波 在 0.9–1.1(3x3s8,{ncheck} 場)" + (";QUICK 只列出" if QUICK else ""),
          QUICK or all(0.9 <= v <= 1.1 for v in ratios.values()), " / ".join(f"{k} {v:.3f}" for k, v in ratios.items()))

    make_geometry_figure(cfg, probes, test[:2].cpu())
    print(f"\n     裝置:{dev}")
    return probes, geo


# ============================================================================
# 量測
# ============================================================================
@torch.no_grad()
def measure(dev):
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    res = {p: [] for p in PROBES}
    for s in SEEDS:
        cfg = load_cfg(s)
        probes = {k: v for k, v in pb.build_probes(cfg, dev).items() if k in PROBES}
        n = N_FIELDS or cfg.eval_n
        F, W = field_size(cfg), cfg.canvas
        bs = beamstop_mask(cfg, device=dev)
        fields = make_fields(cfg, n, seed=cfg.test_seed + FIELD_SEED_OFFSET, device=dev)
        O = torch.polar(fields[:, 0], fields[:, 1])
        ones = torch.ones(W, W, device=dev)
        for pi_, name in enumerate(PROBES):
            pr = probes[name]
            R0 = footprint_mask(cfg, pr, window_starts(cfg, "1x1"), dev)
            rec = {"scans": {}}
            for si, scan in enumerate(SCANS):
                st = window_starts(cfg, scan)
                Rs = footprint_mask(cfg, pr, st, dev)
                cp = scan_cfg(cfg, name, pr, SCANS[scan][2])
                counts, _ = measure_scan(fields, name, pr, cp, bs, st,
                                         seed=cfg.test_seed + s + 20_000 + 1000 * pi_ + 100 * si)
                O0 = epie_init(n, F, cfg.phase_max, cfg.test_seed + s, dev)
                rs = {}
                for meth in METHODS:
                    outs = epie(counts, bs, cp, pr, st, EP_ITERS, meth == "ePIE-C", O0, cfg.test_seed + s)
                    rs[meth] = {str(it): {"R0": field_metrics(outs[it], O, R0), "Rscan": field_metrics(outs[it], O, Rs)}
                                for it in EP_ITERS}
                    del outs
                if scan == "1x1":
                    init = k_init(counts[0], cp, cfg.test_seed + s, dev)
                    outs = pb.hio_probe(init, counts[0], bs, cp, K_ITERS, "K", pr, ones)
                    rs["K-HIO"] = {str(it): {"R0": field_metrics(k_to_field(outs[it], pr, cfg, st[0], F), O, R0)}
                                   for it in K_ITERS}
                    del outs
                rec["scans"][scan] = rs
            res[name].append({"run": f"{BASE}_s{s}", "rec": rec})
            print(f"  [{BASE}_s{s}] {name} 完成", flush=True)
    return res


# ============================================================================
# 判讀(階段五協定 §3.2–3.3,結果出來前寫定)
# ============================================================================
def vals(res, p, scan, meth, it, field="nerr_ph", reg="R0"):
    return np.array([sd["rec"]["scans"][scan][meth][str(it)][reg][field] for sd in res[p]], float)


def nmean(v):
    v = np.asarray(v, float)
    return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float("nan")


def ratio(a, b):
    """逐 seed 比值的幾何平均與 log 比值的 z。"""
    r = np.log(np.asarray(a) / np.asarray(b))
    m = r.mean()
    s = r.std(ddof=1) if len(r) > 1 else 0.0
    z = m / (s / math.sqrt(len(r))) if s > 0 else float("inf") * np.sign(m)
    mr = float(np.exp(m))
    v = "較準" if (mr <= 0.8 and z < -2) else ("較差" if (mr >= 1.25 and z > 2) else "相近")
    return v, mr, float(z)


def best_in_budget(res, p, scan, meth, t_budget, t_per_iter, iters):
    """時間預算內(次數 × 每次時間 ≤ 預算)平均 nerr_ph 最佳的停止點,回傳該點各 seed 的值(對迭代法最有利)。"""
    allowed = [it for it in iters if it * t_per_iter <= t_budget * (1 + 1e-9)]
    if not allowed:
        return None, None
    arr = np.stack([vals(res, p, scan, meth, it) for it in allowed])     # [n_it, seeds]
    i = int(np.argmin(arr.mean(1)))                                      # 平均最佳的停止點(同 4-2a 的對手定義)
    return arr[i], allowed[i]


def report(res, tim):
    v = {}
    print("\n" + "=" * 100)
    print("計時(ms / 張;ePIE 為掃過所有位置一次,K-HIO 為一次迭代)")
    print("=" * 100)
    for B, row in tim.items():
        print(f"  batch {B}:" + "  ".join(f"{k} {t:.4f}" for k, t in row.items()))
    for p in PROBES:
        print("\n" + "=" * 100)
        print(f"探針 {p}:主區域 R0(場中央探針範圍)的 nerr_ph(只對齊整體相位)/ nerr_al(完整對齊)/ 選翻轉,3 seeds 平均")
        print("=" * 100)
        print(f"  {'掃描 / 方法':<18}" + "".join(f"{it:>18}" for it in EP_ITERS))
        for scan in SCANS:
            for meth in METHODS:
                print(f"  {scan + ' ' + meth:<18}" + "".join(
                    f"{vals(res, p, scan, meth, it).mean():>7.3f}/{vals(res, p, scan, meth, it, 'nerr_al').mean():.3f}/"
                    f"{vals(res, p, scan, meth, it, 'twin_frac').mean():.2f}" for it in EP_ITERS))
        print(f"  {'1x1 K-HIO':<18}" + "".join(
            f"{it:>6}:{vals(res, p, '1x1', 'K-HIO', it).mean():.3f}/{vals(res, p, '1x1', 'K-HIO', it, 'nerr_al').mean():.3f}/"
            f"{vals(res, p, '1x1', 'K-HIO', it, 'twin_frac').mean():.2f}" for it in K_ITERS))
        print(f"  各掃描自己的照明範圍(ePIE-C,{MAIN_SWEEPS} 次)nerr_ph / 有平移:" + "  ".join(
            f"{scan} {vals(res, p, scan, 'ePIE-C', MAIN_SWEEPS, reg='Rscan').mean():.3f}/"
            f"{vals(res, p, scan, 'ePIE-C', MAIN_SWEEPS, 'shift_frac', 'Rscan').mean():.2f}" for scan in SCANS))

        # (1) 重疊的效益:ePIE-C 3x3s8 vs K-HIO 1x1,同時間
        rows = []
        for B in tim:
            for kb in BUDGET_K:
                tb = kb * tim[B][f"K-HIO:{p}"]
                a, ia = best_in_budget(res, p, "3x3s8", "ePIE-C", tb, tim[B]["3x3s8"], EP_ITERS)
                b, ib = best_in_budget(res, p, "1x1", "K-HIO", tb, tim[B][f"K-HIO:{p}"], K_ITERS)
                if a is None or b is None:
                    rows.append((B, kb, None))
                    continue
                rows.append((B, kb, ratio(a, b) + (ia, ib)))
        main = next((r for r in rows if r[0] == "64" and r[1] == BUDGET_K[0]), None)
        v1 = main[2][0] if main and main[2] else "無法判定(預算內沒有可比的點)"
        print(f"  (1) 重疊的效益(ePIE-C 3x3s8 vs K-HIO 1x1,同時間;主判:batch 64、預算 = K-HIO {BUDGET_K[0]} 次):{v1}")
        for B, kb, r in rows:
            print(f"      batch {B}、預算 K-HIO {kb} 次:" + (f"比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}"
                  f"(ePIE-C 最佳在 {r[3]} 次、K-HIO 在 {r[4]} 次)" if r else "預算內沒有可比的點"))
        # (2) 重疊率、(3) 位置數、(4) 劑量:同次數(每次掃過全部位置)
        v2 = ratio(vals(res, p, "3x3s16", "ePIE-C", MAIN_SWEEPS), vals(res, p, "3x3s8", "ePIE-C", MAIN_SWEEPS))
        v3 = ratio(vals(res, p, "2x2s8", "ePIE-C", MAIN_SWEEPS), vals(res, p, "3x3s8", "ePIE-C", MAIN_SWEEPS))
        v4 = ratio(vals(res, p, "3x3s8d9", "ePIE-C", MAIN_SWEEPS), vals(res, p, "3x3s8", "ePIE-C", MAIN_SWEEPS))
        v7 = ratio(vals(res, p, "3x3s8", "ePIE-C", MAIN_SWEEPS), vals(res, p, "1x1", "ePIE-C", MAIN_SWEEPS))
        v5 = ratio(vals(res, p, "3x3s8", "ePIE-C", MAIN_SWEEPS), vals(res, p, "3x3s8", "ePIE", MAIN_SWEEPS))
        print(f"  (2) 重疊率({MAIN_SWEEPS} 次):3x3s16 vs 3x3s8 比值 {v2[1]:.3f}(z {v2[2]:+.1f})→ 低重疊{v2[0]}")
        print(f"  (3) 位置數({MAIN_SWEEPS} 次):2x2s8 vs 3x3s8 比值 {v3[1]:.3f}(z {v3[2]:+.1f})→ 2×2 {v3[0]}")
        print(f"  (4) 劑量({MAIN_SWEEPS} 次):3x3s8d9 vs 3x3s8 比值 {v4[1]:.3f}(z {v4[2]:+.1f})→ 劑量 ÷9 {v4[0]}")
        print(f"  (5) 物體約束({MAIN_SWEEPS} 次):ePIE-C vs ePIE(3x3s8)比值 {v5[1]:.3f}(z {v5[2]:+.1f})→ ePIE-C {v5[0]}")
        print(f"  (7) 同一方法下的重疊({MAIN_SWEEPS} 次):ePIE-C 3x3s8 vs 1x1 比值 {v7[1]:.3f}(z {v7[2]:+.1f})→ 3x3s8 {v7[0]}")
        tw = vals(res, p, "3x3s8", "ePIE-C", MAIN_SWEEPS, "twin_frac").mean()
        print(f"  (6) 翻轉(ePIE-C 3x3s8,{MAIN_SWEEPS} 次):選翻轉 {tw:.2f};解出的樣本比例 "
              f"{vals(res, p, '3x3s8', 'ePIE-C', MAIN_SWEEPS, 'good_frac').mean():.2f}、其中選翻轉 "
              f"{nmean(vals(res, p, '3x3s8', 'ePIE-C', MAIN_SWEEPS, 'twin_good')):.2f}")
        v[p] = {"overlap_benefit": v1, "overlap_rows": [(B, kb, list(r) if r else None) for B, kb, r in rows],
                "low_overlap": v2, "positions_2x2": v3, "dose_div9": v4, "constraint": v5, "same_method_overlap": v7,
                "twin_3x3s8": float(tw)}

    # 探針比較與 5-3 的主條件(規則見協定 §3.3)
    vp = ratio(vals(res, "DEF2", "3x3s8", "ePIE-C", MAIN_SWEEPS), vals(res, "DISK", "3x3s8", "ePIE-C", MAIN_SWEEPS))
    print("\n" + "=" * 100)
    print(f"(8) 探針(ePIE-C 3x3s8,{MAIN_SWEEPS} 次,R0 的 nerr_ph):DEF2 vs DISK 比值 {vp[1]:.3f}(z {vp[2]:+.1f})→ DEF2 {vp[0]}")
    probe_pick = "DISK" if vp[0] == "較差" else "DEF2"
    lo = v[probe_pick]["low_overlap"][0]
    scan_pick = "3x3s16" if lo in ("相近", "較準") else "3x3s8"
    print(f"結論:5-3 的主條件 = 探針 {probe_pick}、掃描 {scan_pick}(規則:DEF2 除非較差;低重疊不較差即取低重疊)")
    print(f"      5-3 的對手 = 同條件的 ePIE-C,在網路的實測時間內可選的最佳 nerr_ph(batch 64 與 512)")
    if v[probe_pick]["overlap_benefit"] != "較準":
        print("      ⚠️ 依 §3.3:主探針的重疊效益未達「較準」→ 進 5-3 前先停下來討論(檢查模擬與迭代法)")
    print("判讀準則見階段五協定 §3.2–3.3(結果出來前已寫定)")
    v["probe_cmp"], v["pick"] = vp, {"probe": probe_pick, "scan": scan_pick}
    return v


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


def make_geometry_figure(cfg, probes, fields):
    plt = _plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    scans = [s for s in SCANS if SCANS[s][2] == 1.0]
    fig, axs = plt.subplots(len(probes), len(scans) + 1, figsize=(2.6 * (len(scans) + 1), 2.7 * len(probes)))
    for i, (name, pr) in enumerate(probes.items()):
        prc = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in pr.items()}
        ax = axs[i, 0]
        ax.imshow(fields[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title("object field (amp)" if i == 0 else "", fontsize=8)
        ax.set_ylabel(name)
        for j, scan in enumerate(scans):
            ax = axs[i, j + 1]
            w = illumination(cfg, prc, window_starts(cfg, scan), "cpu")
            ax.imshow(w.numpy(), cmap="magma")
            m = footprint_mask(cfg, prc, window_starts(cfg, "1x1"), "cpu")
            ax.contour(m.numpy(), levels=[0.5], colors="#1baf7a", linewidths=0.8)
            ax.set_title(f"{scan}: Σ|P_j|²" if i == 0 else "", fontsize=8)
        for a in axs[i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Stage 5 geometry (no reconstruction). Green: R0 = central probe footprint (common scoring region)",
                 fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / "scan5_geometry.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


def make_figures(res):
    plt = _plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    col = {"1x1": "#9a9994", "2x2s8": "#2a78d6", "3x3s8": "#eb6834", "3x3s16": "#1baf7a", "3x3s8d9": "#8a5cd1"}
    fig, axs = plt.subplots(1, len(PROBES), figsize=(5.2 * len(PROBES), 4), sharey=True)
    for ax, p in zip(np.atleast_1d(axs), PROBES):
        for scan in SCANS:
            ax.plot(EP_ITERS, [vals(res, p, scan, "ePIE-C", it).mean() for it in EP_ITERS], color=col[scan],
                    marker="o", ms=3, lw=1.8, label=f"ePIE-C {scan}")
        ax.plot(K_ITERS, [vals(res, p, "1x1", "K-HIO", it).mean() for it in K_ITERS], color="#0b0b0b", ls="--",
                marker="s", ms=3, lw=1.4, label="K-HIO 1x1")
        ax.set_xscale("log")
        ax.set_title(p, fontsize=10)
        ax.set_xlabel("iterations (ePIE: sweeps over all positions)")
        ax.grid(True, color="#e4e3df")
    np.atleast_1d(axs)[0].set_ylabel("nerr on R0 (global phase only)")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Stage 5-2: overlapped scanning vs single pattern (3 seeds, mean)", fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "scan5_curves.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑檔案檢查、幾何與單元測試")
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
    print("\n計時")
    tim = timing(load_cfg(0), probes, dev)
    print("\n量測")
    res = measure(dev)
    raw = RUN_ROOT / ("scan5_raw.json" if not QUICK else "scan5_raw_quick.json")
    json.dump({"results": res, "timing": tim, "geometry": geo, "quick": QUICK}, open(raw, "w"))
    print(f"  原始結果先存檔:{raw}")
    verdict = report(res, tim)
    json.dump({"results": res, "timing": tim, "geometry": geo, "verdict": verdict, "scans": list(SCANS),
               "ep_iters": EP_ITERS, "k_iters": K_ITERS, "quick": QUICK},
              open(OUT_JSON if not QUICK else RUN_ROOT / "scan5_quick.json", "w"))
    make_figures(res)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))
    print("把完整輸出貼給 Claude")


if __name__ == "__main__":
    main()
