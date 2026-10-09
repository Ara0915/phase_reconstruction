#!/usr/bin/env python
"""階段五 5-2b:對手定案(階段五協定 §十;不需訓練)。

問:在很短的時間內(網路只做一次運算的時間),迭代法最強能做到多好?
   5-2 的 ePIE 最後很準、但起步慢;短時間的強弱主要取決於「起點」與「每次的成本」。
   本程式把起點與演算法都試過,畫出「時間 vs 誤差」的最強包絡,作為 5-3 網路的對手。

資料:與 5-2 完全相同的 3x3s8 量測(同一批測試場、同樣的 Poisson seed;scan_5.py 的函式,不修改)。
起點(6):rand(= 5-2)、one、zero、const(由量測能量決定的常數)、khio20 / khio100(中央那張先跑 K-HIO)
演算法(4):ePIE-C、ePIE(= 5-2)、AP-C、AP(同時投影:9 個位置一起做傅立葉約束,再合併)
另加:K-HIO 單張(中央那張)。
計時:同一程序、同一 GPU,batch 64(主)與 512(參考);另量未訓練 U-Net(9 通道輸入)的推論時間作參考。

用法(需在計算節點執行;需 scan_5.py、probe_4_2b.py、probe_4_1.py、ambiguity_check.py):
    python scan_5b.py --check   # 單元測試(約 1–3 分鐘)
    python scan_5b.py           # 完整量測(用 run_scan5b.sh 送 job)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5 as s5                                                  # noqa: E402
from src.config import Cfg                                          # noqa: E402

RUN_ROOT = s5.RUN_ROOT
OUT_JSON = RUN_ROOT / "scan5b.json"
FIG_DIR = Path("figs_scan5b")
SEEDS = s5.SEEDS
PROBES = s5.PROBES
SCAN = "3x3s8"
CENTER = 4                                   # 3x3 依列優先排列,第 4 個 = 中央位置
METHODS = ["ePIE-C", "ePIE", "AP-C", "AP"]
INITS = ["rand", "one", "zero", "const", "khio20", "khio100"]
KHIO_N = {"khio20": 20, "khio100": 100}
ITERS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500]
K_ITERS = ITERS + [1000, 2000]
B1_STOPS = [1, 5, 20, 100]                   # (b1) 起點效果的比較點
BUDGET_K = [0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500]   # 預算 = k × ePIE 3x3 掃一次的時間
KHIO_REF = [100, 1000]                       # (b4) 預算 = K-HIO 100 / 1000 次
AP_DELTA = 1e-3
TIME_REPS = 5
N_FIELDS = None                              # None = cfg.eval_n(512)
REPRO_TOL = 1e-5

QUICK = os.environ.get("CDI_QUICK") == "1"
if QUICK:
    ITERS = [1, 2, 5, 20]
    K_ITERS = [1, 5, 20, 60]
    KHIO_N = {"khio20": 20, "khio100": 60}
    B1_STOPS = [1, 5, 20]
    BUDGET_K = [0.5, 1, 2, 5, 20]
    KHIO_REF = [20, 60]
    TIME_REPS, N_FIELDS = 2, s5.N_FIELDS   # 16 場(與 scan_5.py 的 QUICK 相同,才能比對重現)

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


# ============================================================================
# 同時投影(AP)
# ============================================================================
def illum_sum(cfg, pr, starts, device):
    return s5.illumination(cfg, pr, starts, device)


@torch.no_grad()
def ap(counts, bs, cfg_p, pr, starts, iters, constrain, O0):
    """同時投影(重疊投影,Thibault et al. 2008 的 ER 型更新),已知探針。
    每次:所有位置同時做傅立葉約束(一次批次 FFT),再合併
        O ← (Σ_j P* ψ′_j + δ·O) / (Σ_j |P|² + δ),δ = AP_DELTA · max Σ_j |P|²(無照明處保持原值)。
    constrain=True:合併後整場投影到 |O| ≤ 1、相位 ∈ [0, φmax]。回傳 {次數: 物體場 complex [N, F, F]}。"""
    from src.hio import _measured_amp
    W = cfg_p.canvas
    meas = torch.stack([_measured_amp(c, cfg_p).to(O0.device) for c in counts], 1)     # [N, J, W, W]
    P, Pc = pr["P"], pr["P"].conj()
    wsum = illum_sum(cfg_p, pr, starts, O0.device)                                       # [F, F]
    delta = AP_DELTA * float(wsum.max())
    den = wsum + delta
    phmax = cfg_p.phase_max
    O = O0.clone()
    want, outs = set(iters), {}
    for k in range(max(iters)):
        Oj = torch.stack([O[:, y0:y0 + W, x0:x0 + W] for y0, x0 in starts], 1)         # [N, J, W, W]
        psi = P * Oj
        E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
        new_mag = torch.where(bs > 0, meas, E.abs())
        psi2 = torch.fft.ifft2(torch.fft.ifftshift(torch.polar(new_mag, torch.angle(E)), dim=(-2, -1)),
                               norm="ortho")
        g = Pc * psi2
        num = delta * O
        for j, (y0, x0) in enumerate(starts):
            num[:, y0:y0 + W, x0:x0 + W] += g[:, j]
        O = num / den
        if constrain:
            O = torch.polar(O.abs().clamp(max=1.0), torch.angle(O).clamp(0, phmax))
        if (k + 1) in want:
            outs[k + 1] = O.clone()
    return outs


def run_method(meth, counts, bs, cp, pr, starts, iters, O0, seed):
    if meth in ("ePIE-C", "ePIE"):
        return s5.epie(counts, bs, cp, pr, starts, iters, meth == "ePIE-C", O0, seed)
    return ap(counts, bs, cp, pr, starts, iters, meth == "AP-C", O0)


# ============================================================================
# 起點
# ============================================================================
def const_amp(counts_c, bs, cfg_p, pr):
    """a₀² Σ|P|² = 中央繞射圖在 beamstop 外的量測能量(不需真值),a₀ ≤ 1。回傳 [N]。"""
    from src.hio import _measured_amp
    m = _measured_amp(counts_c, cfg_p).to(bs.device)
    E = ((m ** 2) * bs).sum((-2, -1))
    return (E / (pr["Pa"] ** 2).sum()).sqrt().clamp(max=1.0)


def make_init(kind, n, F, cfg, cp, pr, bs, counts, starts, seed, dev, k_out=None):
    """起點(物體場 complex [n, F, F])。k_out:{次數: K-HIO 輸出}(khio 起點用,與 K-HIO 單張共用同一次執行)。"""
    phmax = cfg.phase_max
    if kind == "rand":
        return s5.epie_init(n, F, phmax, seed, dev)                     # = 5-2(逐位元)
    if kind == "one":
        return torch.ones(n, F, F, dtype=torch.complex64, device=dev)
    if kind == "zero":
        return torch.zeros(n, F, F, dtype=torch.complex64, device=dev)
    a0 = const_amp(counts[CENTER], bs, cp, pr)
    c = torch.polar(a0[:, None, None].expand(n, F, F).contiguous(),
                    torch.full((n, F, F), phmax / 2, device=dev))
    if kind == "const":
        return c
    nk = KHIO_N[kind]
    Of = s5.k_to_field(k_out[nk], pr, cfg, starts[CENTER], F)
    fp = s5.footprint_mask(cfg, pr, [starts[CENTER]], dev)
    return torch.where(fp, Of, c)


def init_cost(kind, tK):
    return KHIO_N[kind] * tK if kind in KHIO_N else 0.0


# ============================================================================
# 計時
# ============================================================================
@torch.no_grad()
def timing(cfg, probes, dev):
    """ms / 張。ePIE / AP:每次(掃過 9 個位置);K-HIO:每次迭代(各探針);U-Net:一次推論(參考)。"""
    import probe_4_2b as pb
    from src.model import UNet
    from src.physics import beamstop_mask
    bs = beamstop_mask(cfg, device=dev)
    F, W = s5.field_size(cfg), cfg.canvas
    st = s5.window_starts(cfg, SCAN)
    ones = torch.ones(W, W, device=dev)
    res = {}
    net = UNet(cfg, cin=len(st)).to(dev).eval()
    res["unet_params"] = int(sum(p.numel() for p in net.parameters()))
    for B in [64, 512]:
        fields = s5.make_fields(cfg, B, seed=cfg.test_seed + 99, device=dev)
        row = {}
        pr = probes[PROBES[0]]
        cp = s5.scan_cfg(cfg, PROBES[0], pr)
        counts, _ = s5.measure_scan(fields, PROBES[0], pr, cp, bs, st, seed=0)
        O0 = s5.epie_init(B, F, cfg.phase_max, 0, dev)
        for meth in METHODS:
            run_method(meth, counts, bs, cp, pr, st, [2], O0, 0)                        # 暖機
            ts = []
            for _ in range(TIME_REPS):
                _sync(dev)
                t0 = time.perf_counter()
                run_method(meth, counts, bs, cp, pr, st, [10], O0, 0)
                _sync(dev)
                ts.append((time.perf_counter() - t0) / 10 / B * 1e3)
            row[meth] = float(np.median(ts))
        for name in PROBES:
            prk = probes[name]
            cpk = s5.scan_cfg(cfg, name, prk)
            ck, _ = s5.measure_scan(fields, name, prk, cpk, bs, [st[CENTER]], seed=0)
            init = s5.k_init(ck[0], cpk, 0, dev)
            pb.hio_probe(init, ck[0], bs, cpk, [5], "K", prk, ones)
            ts = []
            for _ in range(TIME_REPS):
                _sync(dev)
                t0 = time.perf_counter()
                pb.hio_probe(init, ck[0], bs, cpk, [50], "K", prk, ones)
                _sync(dev)
                ts.append((time.perf_counter() - t0) / 50 / B * 1e3)
            row[f"K-HIO:{name}"] = float(np.median(ts))
        x = torch.randn(B, len(st), W, W, device=dev)
        net(x)
        ts = []
        for _ in range(TIME_REPS):
            _sync(dev)
            t0 = time.perf_counter()
            net(x)
            _sync(dev)
            ts.append((time.perf_counter() - t0) / B * 1e3)
        row["UNet(ref)"] = float(np.median(ts))
        res[str(B)] = row
    return res


def cfg_time(conf, it, tim_B, probe):
    """設定 conf 在第 it 次的時間(ms / 張)= 起點的時間 + it × 每次的時間。"""
    tK = tim_B[f"K-HIO:{probe}"]
    if conf == "K-HIO":
        return it * tK
    meth, init = conf.split("|")
    return init_cost(init, tK) + it * tim_B[meth]


# ============================================================================
# 單元測試
# ============================================================================
@torch.no_grad()
def unit_tests(dev):
    import probe_4_2b as pb
    from src.hio import _measured_amp
    from src.physics import beamstop_mask
    print("=" * 70)
    print("5-2b 單元測試")
    print("=" * 70)
    cfg = s5.load_cfg(0)
    allp = pb.build_probes(cfg, dev)
    probes = {k: allp[k] for k in PROBES}
    F, W = s5.field_size(cfg), cfg.canvas
    bs = beamstop_mask(cfg, device=dev)
    phmax = cfg.phase_max
    st = s5.window_starts(cfg, SCAN)

    # (1) 位置
    check("3x3s8 的中央視窗 = 1x1 的視窗", st[CENTER] == s5.window_starts(cfg, "1x1")[0],
          f"{st[CENTER]} vs {s5.window_starts(cfg, '1x1')[0]}")

    # 真值(無雜訊、移到約束邊界內;同 scan_5.py)
    small = s5.make_fields(cfg, 8, seed=11, device=dev)
    Oc = torch.polar(small[:, 0], small[:, 1])
    on = Oc.abs() > 0
    Om = torch.polar(Oc.abs() * 0.95, torch.where(on, 0.05 * phmax + 0.9 * torch.angle(Oc).clamp(0, phmax),
                                                   torch.zeros_like(Oc.real)))
    fm = torch.stack([Om.abs(), torch.angle(Om)], 1)
    ones = torch.ones(W, W, device=dev)
    e_ap, e_en, e_k = {}, {}, {}
    for name, pr in probes.items():
        c0 = Cfg.from_dict(s5.scan_cfg(cfg, name, pr).to_dict())
        c0.add_poisson = False
        cnt, psi = s5.measure_scan(fm, name, pr, c0, bs, st, seed=0)
        reg = s5.footprint_mask(cfg, pr, st, dev)
        # (2) AP 固定點
        e = 0.0
        for con in [True, False]:
            out = ap(cnt, bs, c0, pr, st, [20], con, Om)[20]
            e = max(e, float(((out - Om).abs() * reg).max()))
        e_ap[name] = e
        # (3) 能量尺度:量測能量(beamstop 外)= Σ|ψ|²(beamstop 外)
        m = _measured_amp(cnt[CENTER], c0)
        Epsi = torch.fft.fftshift(torch.fft.fft2(torch.polar(psi[CENTER][:, 0], psi[CENTER][:, 1]), norm="ortho"),
                                  dim=(-2, -1))
        a, b = ((m ** 2) * bs).sum((-2, -1)), ((Epsi.abs() ** 2) * bs).sum((-2, -1))
        e_en[name] = float(((a - b).abs() / b).max())
        # (4) khio 起點:以真值 ψ 起始的 K-HIO(無雜訊)→ 起點在中央探針範圍內 = 真值
        k_out = pb.hio_probe(psi[CENTER], cnt[CENTER], bs, c0, [KHIO_N["khio20"]], "K", pr, ones)
        O0 = make_init("khio20", 8, F, cfg, c0, pr, bs, cnt, st, 0, dev, k_out=k_out)
        fp = s5.footprint_mask(cfg, pr, [st[CENTER]], dev)
        e_k[name] = float(((O0 - Om).abs() * fp).max())
    check("真值為 AP-C / AP 的固定點(20 次,無雜訊;< 1e-3)", all(v < 1e-3 for v in e_ap.values()),
          " / ".join(f"{k} {v:.1e}" for k, v in e_ap.items()))
    check("量測能量(beamstop 外)= Σ|ψ|²(beamstop 外)(相對差 < 1e-4;const 起點的依據)",
          all(v < 1e-4 for v in e_en.values()), " / ".join(f"{k} {v:.1e}" for k, v in e_en.items()))
    check("khio 起點:以真值起始時,中央探針範圍內 = 真值(< 1e-3)", all(v < 1e-3 for v in e_k.values()),
          " / ".join(f"{k} {v:.1e}" for k, v in e_k.items()))
    print(f"\n     裝置:{dev}")
    return probes


# ============================================================================
# 量測
# ============================================================================
@torch.no_grad()
def measure(dev):
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    res = {p: [] for p in PROBES}
    si = list(s5.SCANS).index(SCAN)
    for s in SEEDS:
        cfg = s5.load_cfg(s)
        probes = {k: v for k, v in pb.build_probes(cfg, dev).items() if k in PROBES}
        n = N_FIELDS or cfg.eval_n
        F, W = s5.field_size(cfg), cfg.canvas
        bs = beamstop_mask(cfg, device=dev)
        fields = s5.make_fields(cfg, n, seed=cfg.test_seed + s5.FIELD_SEED_OFFSET, device=dev)
        O = torch.polar(fields[:, 0], fields[:, 1])
        ones = torch.ones(W, W, device=dev)
        st = s5.window_starts(cfg, SCAN)
        for name in PROBES:
            pi_ = s5.PROBES.index(name)
            pr = probes[name]
            R0 = s5.footprint_mask(cfg, pr, s5.window_starts(cfg, "1x1"), dev)
            cp = s5.scan_cfg(cfg, name, pr, s5.SCANS[SCAN][2])
            counts, _ = s5.measure_scan(fields, name, pr, cp, bs, st,
                                        seed=cfg.test_seed + s + 20_000 + 1000 * pi_ + 100 * si)   # = 5-2
            rec = {}
            # K-HIO 單張(中央那張);khio 起點共用同一次執行
            kit = sorted(set(K_ITERS) | set(KHIO_N.values()))
            init = s5.k_init(counts[CENTER], cp, cfg.test_seed + s, dev)
            k_out = pb.hio_probe(init, counts[CENTER], bs, cp, kit, "K", pr, ones)
            rec["K-HIO"] = {str(it): s5.field_metrics(s5.k_to_field(k_out[it], pr, cfg, st[CENTER], F), O, R0)
                            for it in K_ITERS}
            for kind in INITS:
                O0 = make_init(kind, n, F, cfg, cp, pr, bs, counts, st, cfg.test_seed + s, dev, k_out=k_out)
                rec[f"init|{kind}"] = s5.field_metrics(O0, O, R0)                       # 第 0 次(起點本身)
                for meth in METHODS:
                    outs = run_method(meth, counts, bs, cp, pr, st, ITERS, O0, cfg.test_seed + s)
                    rec[f"{meth}|{kind}"] = {str(it): s5.field_metrics(outs[it], O, R0) for it in ITERS}
                    rec[f"{meth}|{kind}"]["0"] = rec[f"init|{kind}"]                # 停止點 0 = 起點本身
                    del outs
            del k_out
            res[name].append({"run": f"{s5.BASE}_s{s}", "rec": rec})
            print(f"  [{s5.BASE}_s{s}] {name} 完成", flush=True)
        json.dump({"results": res, "partial_through_seed": s, "quick": QUICK},
                  open(RUN_ROOT / ("scan5b_partial.json" if not QUICK else "scan5b_partial_quick.json"), "w"))
    return res


# ============================================================================
# 判讀(階段五協定 §10.4–10.6,結果出來前寫定)
# ============================================================================
def configs():
    return [f"{m}|{i}" for m in METHODS for i in INITS] + ["K-HIO"]


def stops(conf):
    """可選的停止點;非 K-HIO 的設定含 0(起點本身,時間 = 起點的時間),讓包絡不會漏掉「直接輸出起點」。"""
    return K_ITERS if conf == "K-HIO" else [0] + ITERS


def vals(res, p, conf, it, field="nerr_ph"):
    return np.array([sd["rec"][conf][str(it)][field] for sd in res[p]], float)


def ratio(a, b):
    """逐 seed 比值的幾何平均與 log 比值的 z(同 scan_5.ratio;值夾在 ≥ 1e-12 以免 log 0)。"""
    a = np.maximum(np.asarray(a, float), 1e-12)
    b = np.maximum(np.asarray(b, float), 1e-12)
    if np.array_equal(a, b):                                   # 同一個點(例:兩邊的包絡都選到同一個起點本身)
        return "相近", 1.0, 0.0
    return s5.ratio(a, b)


def envelope(res, p, T, tim_B, subset=None, min_stop=0):
    """時間 ≤ T 的所有(設定, 停止點 ≥ min_stop)中,3 seeds 平均 nerr_ph 最小者。
    回傳 (設定, 次數, 逐 seed 值, 時間) 或 None。"""
    best = None
    for conf in (subset or configs()):
        for it in stops(conf):
            if it < min_stop:
                continue
            t = cfg_time(conf, it, tim_B, p)
            if t > T * (1 + 1e-9):
                continue
            v = vals(res, p, conf, it)
            if best is None or v.mean() < best[2].mean():
                best = (conf, it, v, t)
    return best


def is_3x3(conf):
    return conf != "K-HIO"


def env3(res, p, T, tim_B):
    """「只用 3×3」的包絡:非 K-HIO 的設定、且至少跑 1 次 3×3 的演算法(停止點 0 = 起點本身不算,
    因為 khio 起點的第 0 次在 R0 內就等於 K-HIO 單張的輸出)。"""
    return envelope(res, p, T, tim_B, [c for c in configs() if is_3x3(c)], min_stop=1)


def repro_check(res):
    """量測後的一致性檢查(§10.3):rand 起點的 ePIE-C / ePIE 與 5-2 的逐 seed nerr_ph 相同。"""
    path = RUN_ROOT / ("scan5_raw.json" if not QUICK else "scan5_raw_quick.json")
    if not path.exists():
        print(f"  (b0) 5-2 重現:{path} 不存在,略過(只註明)")
        return {"status": "no_file"}
    old = json.load(open(path))["results"]
    worst, n = 0.0, 0
    for p in PROBES:
        if p not in old or len(old[p]) != len(res[p]):
            print(f"  ⚠️ (b0) {path} 的 {p} seed 數與本次不同,無法比對")
            return {"status": "mismatch_seeds"}
        for k, sd in enumerate(res[p]):
            osd = old[p][k]
            if osd["run"] != sd["run"]:
                print(f"  ⚠️ seed 對不上:{osd['run']} vs {sd['run']}")
                return {"status": "mismatch_run"}
            for meth in ["ePIE-C", "ePIE"]:
                for it in ITERS:
                    o = osd["rec"]["scans"][SCAN][meth].get(str(it))
                    if o is None:
                        continue
                    worst = max(worst, abs(o["R0"]["nerr_ph"] - sd["rec"][f"{meth}|rand"][str(it)]["nerr_ph"]))
                    n += 1
    ok = n > 0 and worst < REPRO_TOL
    print(f"  {'✅' if ok else '❌'} (b0) 5-2 重現:rand 起點的 ePIE-C / ePIE,共同的 {n} 個(探針 × seed × 方法 × 次數)"
          f"最大差 {worst:.1e}(標準 < {REPRO_TOL:.0e})")
    return {"status": "ok" if ok else "diff", "max_diff": worst, "n": n}


def fmt_conf(c):
    return c.replace("|", "+")


def report(res, tim):
    v = {}
    Bs = ["64", "512"]
    print("\n" + "=" * 100)
    print("計時(ms / 張;ePIE / AP 為掃過 9 個位置一次,K-HIO 為一次迭代,U-Net 為一次推論,未訓練、只作參考)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {t:.4f}" for k, t in tim[B].items()))
    print(f"  U-Net 參數量 {tim['unet_params']:,}")
    for B in Bs:
        te = tim[B]["ePIE-C"]
        print(f"  batch {B}:U-Net 推論 ≈ ePIE 3x3 {tim[B]['UNet(ref)'] / te:.2f} 次、"
              f"AP {tim[B]['UNet(ref)'] / tim[B]['AP-C']:.2f} 次、K-HIO {tim[B]['UNet(ref)'] / tim[B]['K-HIO:DEF2']:.1f} 次的時間")

    print()
    v["repro"] = repro_check(res)

    show = [it for it in [1, 2, 5, 10, 20, 50, 100, 500] if it in ITERS]
    print("\n  參考:nerr_ph = 1 等於「輸出全零」;> 1 表示比全零還差(常見於能量過大的起點)")
    for p in PROBES:
        print("\n" + "=" * 100)
        print(f"探針 {p}:R0 的 nerr_ph(3 seeds 平均);第 0 欄 = 起點本身")
        print("=" * 100)
        print(f"  {'方法 + 起點':<18}{'0':>8}" + "".join(f"{it:>8}" for it in show))
        for meth in METHODS:
            for kind in INITS:
                c = f"{meth}|{kind}"
                z = np.mean([sd["rec"][f"init|{kind}"]["nerr_ph"] for sd in res[p]])
                print(f"  {fmt_conf(c):<18}{z:>8.3f}" + "".join(f"{vals(res, p, c, it).mean():>8.3f}" for it in show))
        ks = [it for it in K_ITERS if it in (1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000)]
        print(f"  {'K-HIO 單張':<16}" + "  ".join(f"{it}:{vals(res, p, 'K-HIO', it).mean():.3f}" for it in ks))

        # (b1) 起點的效果
        print("\n  (b1) 起點 vs rand(同方法、同次數;比值 < 1 = 比 rand 準)")
        b1 = {}
        for meth in METHODS:
            for kind in INITS[1:]:
                row = []
                for it in B1_STOPS:
                    r = ratio(vals(res, p, f"{meth}|{kind}", it), vals(res, p, f"{meth}|rand", it))
                    row.append(r)
                    b1[f"{meth}|{kind}|{it}"] = r
                print(f"      {meth:<7}{kind:<8}" + "  ".join(f"第{it}次 {r[1]:.3f} {r[0]}" for it, r in zip(B1_STOPS, row)))

        # (b2)、(b3)、(b4)
        b3 = {}
        for B in Bs:
            te = tim[B]["ePIE-C"]
            ep = [c for c in configs() if c.startswith("ePIE")]
            apc = [c for c in configs() if c.startswith("AP")]
            print(f"\n  (b2)(b3) batch {B}:預算 = k × ePIE 3x3 掃一次({te:.4f} ms)"
                  "(「只用 3×3」與 (b2) 至少跑 1 次 3×3 的演算法;最強設定可含第 0 次 = 起點本身)")
            print(f"      {'k':>6} {'最強設定(次數)':<26}{'nerr_ph':>9}   {'只用 3×3':<24}{'nerr':>7}   "
                  f"{'只用 K-HIO':>12}   AP 家族 vs ePIE 家族")
            rows = []
            for k in BUDGET_K:
                T = k * te
                a = envelope(res, p, T, tim[B])
                a3 = env3(res, p, T, tim[B])
                ak = envelope(res, p, T, tim[B], ["K-HIO"])
                e_ep = envelope(res, p, T, tim[B], ep, min_stop=1)                   # (b2) 比演算法:至少跑 1 次
                e_ap = envelope(res, p, T, tim[B], apc, min_stop=1)
                cmp_ = ratio(e_ap[2], e_ep[2]) if (e_ep and e_ap) else None
                rows.append({"k": k, "T": T, "all": a and [a[0], a[1], list(a[2]), a[3]],
                             "only3x3": a3 and [a3[0], a3[1], list(a3[2]), a3[3]],
                             "onlyK": ak and [ak[0], ak[1], list(ak[2]), ak[3]], "ap_vs_epie": cmp_})
                s_all = f"{fmt_conf(a[0])}({a[1]})" if a else "—"
                s3 = f"{fmt_conf(a3[0])}({a3[1]})" if a3 else "—"
                print(f"      {k:>6} {s_all:<26}{(a[2].mean() if a else float('nan')):>9.3f}   {s3:<24}"
                      f"{(a3[2].mean() if a3 else float('nan')):>7.3f}   {(ak[2].mean() if ak else float('nan')):>12.3f}   "
                      + (f"{cmp_[1]:.3f} {cmp_[0]}" if cmp_ else "(預算內其中一方沒有點)")
                      + (f"   [最強設定:完整對齊 {vals(res, p, a[0], a[1], 'nerr_al').mean():.3f}、"
                         f"選翻轉 {vals(res, p, a[0], a[1], 'twin_frac').mean():.2f}]" if a else ""))
            b3[B] = rows
            # (b4) 3×3 包絡 vs K-HIO,預算 = K-HIO 100 / 1000 次
            for kb in KHIO_REF:
                T = kb * tim[B][f"K-HIO:{p}"]
                a3 = env3(res, p, T, tim[B])
                ak = envelope(res, p, T, tim[B], ["K-HIO"])
                if a3 and ak:
                    r = ratio(a3[2], ak[2])
                    print(f"  (b4) batch {B}、預算 K-HIO {kb} 次:3×3 最強 {fmt_conf(a3[0])}({a3[1]})"
                          f" vs K-HIO({ak[1]})比值 {r[1]:.3f}(z {r[2]:+.1f})→ 3×3 {r[0]}")
                    v.setdefault(p, {}).setdefault("b4", []).append([B, kb, list(r), a3[0], a3[1], ak[1]])
                else:
                    print(f"  (b4) batch {B}、預算 K-HIO {kb} 次:預算內 3×3 沒有可比的點")
                    v.setdefault(p, {}).setdefault("b4", []).append([B, kb, None])
            # 網路時間的參考值
            Tn = tim[B]["UNet(ref)"]
            a = envelope(res, p, Tn, tim[B])
            if a:
                print(f"  ▶ batch {B}:未訓練 U-Net 的推論時間 {Tn:.4f} ms → 預覽對手 = {fmt_conf(a[0])}({a[1]} 次)"
                      f" nerr_ph {a[2].mean():.3f}(正式對手以 5-3 實測的網路時間計算)")
            else:
                print(f"  ▶ batch {B}:未訓練 U-Net 的推論時間 {Tn:.4f} ms 比所有設定的最短時間還短"
                      f"(依 §10.6 取最短時間的包絡,並註明)")
            v.setdefault(p, {})[f"preview_{B}"] = a and [a[0], a[1], list(a[2]), a[3], Tn]
        v.setdefault(p, {}).update({"b1": {k: list(r) for k, r in b1.items()}, "b3": b3})

    # (b5) 探針
    print("\n" + "=" * 100)
    print("(b5) 探針:DEF2 vs DISK 的包絡(k ≥ 1 的每個預算、兩種 batch;比值 < 1 = DEF2 較準)")
    print("=" * 100)
    verdicts = []
    for B in Bs:
        te = tim[B]["ePIE-C"]
        line = []
        for k in BUDGET_K:
            if k < 1:
                continue
            a, b = envelope(res, "DEF2", k * te, tim[B]), envelope(res, "DISK", k * te, tim[B])
            if a is None or b is None:
                line.append(f"k={k}:—")
                verdicts.append("無")
                continue
            r = ratio(a[2], b[2])
            verdicts.append(r[0])
            line.append(f"k={k}:{r[1]:.2f} {r[0]}")
        print(f"  batch {B}:" + "  ".join(line))
    if verdicts and all(x == "較準" for x in verdicts):
        keep = ["DEF2"]
    elif verdicts and all(x == "較差" for x in verdicts):
        keep = ["DISK"]
    else:
        keep = ["DEF2", "DISK"]
    print(f"結論:5-3 帶入的探針 = {' 與 '.join(keep)}"
          f"(規則:在每個預算、兩種 batch 下都較準才算絕對優勢;否則兩個都保留)")
    print("      5-3 的對手 = 同探針、同一組 3x3s8 量測下,包絡在網路實測時間的值(§10.6)")
    print("判讀準則見階段五協定 §10.4–10.6(結果出來前已寫定)")
    v["b5"] = {"verdicts": verdicts, "keep": keep}
    return v


# ============================================================================
# 圖
# ============================================================================
def make_figures(res, tim):
    plt = s5._plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    col = {"ePIE-C": "#eb6834", "ePIE": "#2a78d6", "AP-C": "#1baf7a", "AP": "#8a5cd1"}
    ls = {"rand": ":", "one": (0, (1, 3)), "zero": (0, (5, 2)), "const": "-.", "khio20": "--", "khio100": "-"}
    B = "64"
    fig, axs = plt.subplots(1, len(PROBES), figsize=(6.2 * len(PROBES), 4.6), sharey=True)
    for ax, p in zip(np.atleast_1d(axs), PROBES):
        for conf in configs():
            its = stops(conf)
            its = [it for it in its if cfg_time(conf, it, tim[B], p) > 0]          # 對數軸畫不出時間 0
            t = [cfg_time(conf, it, tim[B], p) for it in its]
            y = [vals(res, p, conf, it).mean() for it in its]
            if conf == "K-HIO":
                ax.plot(t, y, color="#0b0b0b", lw=1.4, ls="--", label="K-HIO 1x1" if p == PROBES[0] else None)
            else:
                meth, init = conf.split("|")
                ax.plot(t, y, color=col[meth], lw=0.9, ls=ls[init], alpha=0.75,
                        label=conf.replace("|", "+") if p == PROBES[0] else None)
        tt = sorted({cfg_time(c, it, tim[B], p) for c in configs() for it in stops(c)} - {0.0})
        env = [envelope(res, p, T, tim[B]) for T in tt]
        ax.plot(tt, [e[2].mean() for e in env], color="#c2410c", lw=3, alpha=0.35,
                label="envelope (best at time)" if p == PROBES[0] else None)
        ax.axvline(tim[B]["UNet(ref)"], color="#555", lw=1, ls=":")
        ax.text(tim[B]["UNet(ref)"], 0.98, " U-Net (untrained, ref)", fontsize=7, rotation=90, va="top",
                color="#555", transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{p} (3x3s8, batch {B})", fontsize=10)
        ax.set_xlabel("time per sample (ms)")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    np.atleast_1d(axs)[0].set_ylabel("nerr on R0 (global phase only)")
    fig.legend(loc="center right", fontsize=6.5, frameon=False)
    fig.suptitle("Stage 5-2b: time vs error for every init x algorithm (3 seeds, mean)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.83, 1))
    path = FIG_DIR / "scan5b_envelope.png"
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{path}")


# ============================================================================
def check_files():
    need = [Path("scan_5.py"), Path("probe_4_2b.py"), Path("probe_4_1.py"), Path("ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--check", action="store_true", help="只跑檔案檢查與單元測試")
    a = ap_.parse_args()
    if not check_files():
        print("\n❌ 缺檔案,停下來")
        sys.exit(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    probes = unit_tests(dev)
    if a.check or not ok_all:
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過,先不要跑完整量測"))
        sys.exit(0 if ok_all else 1)
    print("\n計時")
    tim = timing(s5.load_cfg(0), probes, dev)
    print("\n量測")
    res = measure(dev)
    raw = RUN_ROOT / ("scan5b_raw.json" if not QUICK else "scan5b_raw_quick.json")
    json.dump({"results": res, "timing": tim, "quick": QUICK}, open(raw, "w"))
    print(f"  原始結果先存檔:{raw}")
    verdict = report(res, tim)
    json.dump({"results": res, "timing": tim, "verdict": verdict, "methods": METHODS, "inits": INITS,
               "iters": ITERS, "k_iters": K_ITERS, "budget_k": BUDGET_K, "quick": QUICK},
              open(OUT_JSON if not QUICK else RUN_ROOT / "scan5b_quick.json", "w"))
    make_figures(res, tim)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))
    print("把完整輸出貼給 Claude")


if __name__ == "__main__":
    main()
