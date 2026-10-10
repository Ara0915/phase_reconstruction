#!/usr/bin/env python
"""階段八 8d:P6B4e8-L 的「精度天花板」診斷(只評估、不訓練;階段八實驗設計協定 §十九,v1.8)。

現象(§18):L-ideal 的物體上,劑量 > 0.01 後 P6B4e8-L 的 nerr_c 停在約 0.0064(CRLB 繼續下降),同時 S 單空缺的漏判增加。
測試:T0 各級的誤差、T1 關掉 CNN 修正、T2 位置步、T3 多接物理步(只當診斷)、T4 無雜訊、T5 次像素平移、T6 連續的精細對齊、
      T7 誤差的頻率與位置分布、T8 投影本身的偏差。不修改既有檔案;重用 scan_8 / scan_8c / scan_7c 的函式。

用法(計算節點):
    python scan_8d.py --check     # 內建檢查(§19.3)
    python scan_8d.py --smoke     # 迷你全流程(階段八 smoke 的模型);通過才可送件
    python scan_8d.py --run       # 正式(run_scan8d.sh)
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
import scan_8c as c8                                                  # noqa: E402  (8c;不修改)

d7, c7, b7, a7, a6, s5 = s8.d7, s8.c7, s8.b7, s8.a7, s8.a6, s8.s5
RUN_ROOT, QUICK, SEEDS, SUF = s8.RUN_ROOT, s8.QUICK, s8.SEEDS, s8.SUF
S8_MD5 = "4821d2607f6e8aa381f0e6b5c5e72513"
S8C_MD5 = "3d0915fffaff52a5b4b2b749a9e102dd"
check, all_ok, dump_json = s8.check, s8.all_ok, s8.dump_json
NEW_L, BASE = s8.NEW_L, s8.BASE
tg = s8.tg

# ============================================================================
# 設定(協定 §十九)
# ============================================================================
N_FIELDS = 4 if QUICK else 256               # L-ideal 的前 n 個場
DOSES = [0.001, 0.01, 0.1, 1.0]
FREE = "free"                                # 無雜訊(期望強度,劑量 1 的尺度)
EXTRA_REC = [8, 24, 56]                      # T3:基準輸出之後再接的物理步數(記錄點)
SHIFTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]      # T5:x 方向的整體平移(px)
SHIFT_DOSES = [1.0, 0.01]
SUB_BINS = [0.0, 0.18, 0.35, 0.53, 1.0]      # T5:次像素偏移的分組(px)
HALF, KEEP_08, STAGE_DESC, RECALL_OK = 0.5, 0.8, 0.9, 0.9
POS_RMS_MIN = 0.05                           # T2:δ 的 RMS 門檻(px)
SUB_X, SUB_D = 2.0, 0.10                     # T5:漏判率 ≥ 最低組 × 2 且差 ≥ 10 個百分點
ATOM_PH = 0.2                                # T7:「原子處」= 真值相位 > 0.2 rad
FINE_ITERS = 60

SEED_NOISE, SEED_SHIFT, SMOKE_OFF = 88_920_000, 88_955_000, 40_000
MY_RANGES_8D = [(88_920_000, 88_999_999)]

OUT_JSON = RUN_ROOT / f"scan8d{SUF}.json"
SMOKE_DIR = RUN_ROOT / f"scan8d_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
FIG_DIR = Path("figs_scan8d")


def me_md5():
    return s8.md5(Path(__file__).resolve())


def ctx_of(smoke):
    off = SMOKE_OFF if smoke else 0
    base = {"smoke": smoke, "roots": s8.roots_of(smoke), "noise": SEED_NOISE + off, "shift": SEED_SHIFT + off}
    if smoke:
        return {**base, "out": SMOKE_DIR / f"scan8d{SUF}.json", "fig": SMOKE_DIR / "figs", "n": 4 if QUICK else 8,
                "extra": [8], "shifts": [0.0, 0.5]}
    return {**base, "out": OUT_JSON, "fig": FIG_DIR, "n": N_FIELDS, "extra": EXTRA_REC, "shifts": SHIFTS}


# ============================================================================
# 量測、網路流程的變體
# ============================================================================
def remeasure(d0, dose, ns):
    """同一組物體,指定劑量(或 FREE = 無雜訊)重新量測。回傳新的考卷 dict(物體、判讀與 d0 相同)。"""
    n = d0["n"]
    d = dict(d0)
    if dose == FREE:
        dv = torch.ones(n)
        d["counts"] = s8.measure_L(d0["fields"], d0["geo"], d0["cp"], d0["bs"], 0, d0["spec"], d0["z"], d0["sign"], d0["ops"], dv,
                                   poisson=False)
    else:
        s8.guard([ns])
        dv = torch.full((n,), float(dose))
        d["counts"] = s8.measure_L(d0["fields"], d0["geo"], d0["cp"], d0["bs"], ns, d0["spec"], d0["z"], d0["sign"], d0["ops"], dv)
    d["dose"] = dv
    return d


@torch.no_grad()
def run_custom(model, norm, d, cnn=True, beta=None, extra=()):
    """= scan_7c.run_g9e,另回傳草稿、各級輸出、最後的位置 δ;cnn = False → 各級不做 CNN 修正;beta → 位置步的步長(0 = 固定位置);
    extra:基準輸出之後再接的物理步數(記錄點;不用 CNN;步長 α、β 同第 8 級)。"""
    geo, bs = d["geo"], d["bs"]
    dev = bs.device
    net = model.net
    K = model.K
    O0 = c7.g9_draft(net, norm, d)
    sg = c7.field_geo(net, geo, bs, dev)
    meas = a6.measured(d["counts"], d["cp"], dev)
    al_last = torch.sigmoid(net.a[K - 1]) if net.alpha_override is None else net.alpha_override
    b_last = model.beta(K - 1) if beta is None else beta
    st = [[] for _ in range(K)]
    dl_all, ex = [], {r: [] for r in extra}
    for i in range(0, O0.shape[0], a6.MCHUNK):
        o, ds, _ = model.stages(O0[i:i + a6.MCHUNK].clone(), meas[i:i + a6.MCHUNK], sg, bs, cnn=cnn, beta=beta)
        for k in range(K):
            st[k].append(torch.polar(o[k][:, 0], o[k][:, 1]))
        dl_all.append(ds[-1])
        O, dl, done = torch.polar(o[-1][:, 0], o[-1][:, 1]), ds[-1], 0
        for r in sorted(extra):
            oo, dd, _ = model.stages(O, meas[i:i + a6.MCHUNK], sg, bs, delta=dl, n_stages=r - done, cnn=False, alpha=al_last, beta=b_last)
            O, dl, done = torch.polar(oo[-1][:, 0], oo[-1][:, 1]), dd[-1], r
            ex[r].append(O)
    return {"draft": O0, "stages": [torch.cat(x) for x in st], "final": torch.cat(st[-1]), "delta": torch.cat(dl_all),
            "extra": {r: torch.cat(v) for r, v in ex.items()}}


def nerr_grid(est, d):
    """階段八的 nerr_c(align_full)逐場。"""
    v0, _, _, _, _, eal = s8.align_full(est, d["O"], d["geo"].U)
    m = d["geo"].U.to(d["O"].real.dtype)
    c = ((eal - d["O"]).abs() ** 2 * m).sum((1, 2)) / d["Ec"].clamp_min(1e-30)
    return torch.where(torch.isfinite(v0), c, torch.full_like(c, float("nan"))).cpu()


def fine_align(est, d, iters=FINE_ITERS):
    """T6:連續的精細對齊。從 align_full 的格點解(平移 t、斜坡 q)出發,以 L-BFGS 精修 t、q(整體相位用閉式解),
    回傳 (逐場 nerr_c(取精細與格點的較小者), 精修後的 t、q, 對齊後的估計)。"""
    O, U = d["O"], d["geo"].U
    F = O.shape[-1]
    dev = O.device
    v0, v1, v2, t, q, _ = s8.align_full(est, O, U)
    k = torch.fft.fftfreq(F, dtype=torch.float64, device=dev)
    ky, kx = torch.meshgrid(k, k, indexing="ij")
    yc = torch.arange(F, dtype=torch.float64, device=dev) - (F - 1) / 2
    m = U.to(torch.float64)
    E = est.to(torch.complex128)
    Ot = O.to(torch.complex128)
    B = torch.fft.fft2(E)
    Ec = d["Ec"].double()
    out_c, out_p, out_e = [], [], []
    for i0 in range(0, E.shape[0], 32):
        sl = slice(i0, i0 + 32)
        p = torch.cat([t[sl].double(), q[sl].double()], 1).clone().requires_grad_(True)

        def shifted(p_):
            ph = 2 * math.pi * (ky[None] * p_[:, 0, None, None] + kx[None] * p_[:, 1, None, None])
            es = torch.fft.ifft2(B[sl] * torch.polar(torch.ones_like(ph), ph))
            ey = torch.polar(torch.ones_like(yc).expand(p_.shape[0], -1), -2 * math.pi * p_[:, 2, None] * yc[None])
            exx = torch.polar(torch.ones_like(yc).expand(p_.shape[0], -1), -2 * math.pi * p_[:, 3, None] * yc[None])
            er = es * ey[:, :, None] * exx[:, None, :]
            cc = ((Ot[sl] * m) * (er * m).conj()).sum((1, 2))
            eal = er * torch.exp(1j * torch.angle(cc))[:, None, None]
            return ((eal - Ot[sl]).abs() ** 2 * m).sum((1, 2)), eal

        opt = torch.optim.LBFGS([p], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-16)

        def closure():
            opt.zero_grad()
            f = shifted(p)[0].sum()
            f.backward()
            return f

        with torch.enable_grad():
            opt.step(closure)
        with torch.no_grad():
            err, eal = shifted(p)
            cf = err / Ec[sl]
        out_c.append(cf.detach())
        out_p.append(p.detach())
        out_e.append(eal.detach())
    cfine = torch.cat(out_c).cpu()
    cgrid = nerr_grid(est, d).double()
    fin = torch.isfinite(cgrid)
    return torch.where(fin, torch.minimum(cfine, cgrid), cgrid), torch.cat(out_p).cpu(), torch.cat(out_e)


@torch.no_grad()
def spectrum(eal, d):
    """T7:誤差的徑向功率(48 × 48 方塊、Tukey 窗;合併所有場)與真值的功率;誤差能量在「原子處」的比例。"""
    O, U = d["O"].to(torch.complex128), d["geo"].U
    y0, x0, _ = s8.frc_block(U)
    win, ring = s8._frc_win(O.device)
    nr = int(ring.max()) + 1
    rr = ring.flatten()

    def pw(z):
        z = z[:, y0:y0 + s8.FRC_N, x0:x0 + s8.FRC_N]
        z = z - z.mean((1, 2), keepdim=True)
        A = torch.fft.fft2(z * win.double())
        return torch.zeros(nr, dtype=torch.float64, device=O.device).index_add_(0, rr, (A.abs() ** 2).sum(0).flatten())

    e = eal.to(torch.complex128) - O
    m = U.to(torch.float64)
    atom = (torch.angle(O) > ATOM_PH).double() * m
    ee = (e.abs() ** 2)
    frac_atom = float((ee * atom).sum() / (ee * m).sum().clamp_min(1e-300))
    area_atom = float(atom.sum() / m.sum())
    return {"err_pow": pw(e).cpu().tolist(), "true_pow": pw(O).cpu().tolist(), "frac_atom": frac_atom, "area_atom": area_atom}


def eval_out(est, d, fine=True):
    """一組輸出:格點 nerr_c、精細 nerr_c、SV 判讀(合併的混淆矩陣)。"""
    p = c8.predict(est, d)
    r = {"c": float(np.nanmean(p["c"].numpy())), "conf": p["conf"].sum(0).numpy().astype(int).tolist()}
    if fine:
        cf, _, eal = fine_align(est, d)
        r["c_fine"] = float(np.nanmean(cf.numpy()))
        r["_eal"] = eal
    return r


# ============================================================================
# T5:平移後的晶格(重新產生,精確)
# ============================================================================
def shifted_exam(d0, s_px, dose, ns):
    """同一組物體 seed、晶格平移 (0, s_px) px(改 draw_params 的平移 t;其他亂數不變 → 同一片晶格、同樣的空缺)。重新量測、建判讀。"""
    geo, dx, dev = d0["geo"], d0["dx"], d0["O"].device
    F = geo.F
    G = F + 2 * s8.HALO
    idx, _ = s8.index_grid(G, dx)
    fields, cols = [], []
    for i, sd in enumerate(d0["obj_seeds"]):
        mat = s8.MATS[int(d0["mats"][i])]
        prm = s8.draw_params(sd, idx.shape[0], mat)
        a = s8.A_LAT[mat] / dx
        Lm = torch.tensor([[0.0, a], [a * math.sqrt(3) / 2, a / 2]], dtype=torch.float64)
        _, _, _, M = s8.geom_of(prm, idx, F, dx)
        dt = torch.tensor([0.0, float(s_px)], dtype=torch.float64) @ torch.linalg.inv(Lm @ M.T)
        fx = {"t": (prm["t"] + dt).tolist()}
        f, meta = s8.make_lattice([sd], F, dev, materials=[mat], U=geo.U.cpu(), eta=s8.ETA, info=True, fixed=fx, dx=dx)
        fields.append(f)
        cols.append(meta["cols"][0])
    fields = torch.cat(fields)
    n = fields.shape[0]
    d = dict(d0)
    d["fields"] = fields
    d["O"] = torch.polar(fields[:, 0], fields[:, 1])
    d["meta"] = dict(d0["meta"], cols=cols)
    s8.guard([ns])
    dv = torch.full((n,), float(dose))
    d["counts"] = s8.measure_L(fields, geo, d0["cp"], d0["bs"], ns, d0["spec"], d0["z"], d0["sign"], d0["ops"], dv)
    d["dose"] = dv
    d["vac"] = s8.Vac(cols, F, geo.U.cpu(), fields[:, 1].cpu(), norm="all")
    mU = geo.U.to(d["O"].real.dtype)
    cs = (d["O"] * mU).sum((1, 2)) / mU.sum()
    d["Ec"] = (((d["O"] - cs[:, None, None]).abs() ** 2) * mU).sum((1, 2))
    d["E"] = ((d["O"].abs() ** 2) * mU).sum((1, 2))
    return d


def subpix(pos):
    return (pos.double() - torch.round(pos.double())).norm(dim=1)


# ============================================================================
# 各部分
# ============================================================================
def dose_list():
    return DOSES + [FREE]


def ns_of(ctx, di, s):
    return ctx["noise"] + 10_000 * di + 1000 * s


def part_base(dev, ctx, log):
    """T0 / T4 / T6 / T7 / T8:基準流程(P6B4e8-L 與 P6B4e8)的各級、草稿、精細對齊、誤差分布、投影偏差、位置 δ(T2a)。"""
    n = ctx["n"]
    out = {nm: {str(x): {"c": [], "c_fine": [], "conf": [], "stage_c": [], "draft_c": [], "delta_rms": [], "spec": []} for x in dose_list()}
           for nm in (NEW_L, BASE)}
    out["proj"] = []
    t0 = time.time()
    for s in SEEDS:
        d0 = s8.setup_exam(s, dev, "L-ideal", n, ctx["smoke"])
        for nm in (NEW_L, BASE):
            kind, model, norm = s8.get_net(nm, s, d0["cfg"], d0["geo"].pr, dev, ctx["roots"])
            if nm == NEW_L:
                amp, ph = model.net.project(d0["O"].clone())
                out["proj"].append(float(np.nanmean(nerr_grid(torch.polar(amp, ph), d0).numpy())))
            for di, x in enumerate(dose_list()):
                d = remeasure(d0, x, ns_of(ctx, di, s) if x != FREE else None)
                r = run_custom(model, norm, d)
                ev = eval_out(r["final"], d, fine=True)
                rec = out[nm][str(x)]
                rec["c"].append(ev["c"])
                rec["c_fine"].append(ev["c_fine"])
                rec["conf"].append(ev["conf"])
                rec["stage_c"].append([float(np.nanmean(nerr_grid(o, d).numpy())) for o in r["stages"]])
                rec["draft_c"].append(float(np.nanmean(nerr_grid(r["draft"], d).numpy())))
                rec["delta_rms"].append(float(r["delta"].pow(2).sum(-1).mean(-1).sqrt().mean()))
                if nm == NEW_L:
                    rec["spec"].append(spectrum(ev["_eal"], d))
                del d, r, ev
            del model
        del d0
        log(f"  [base] seed {s} 完成(經過 {time.time() - t0:.0f} 秒)")
    return out


def part_variants(dev, ctx, log):
    """T1(無 CNN)、T2b(β = 0)、T3(多接物理步):P6B4e8-L,4 個劑量 + 無雜訊。"""
    n = ctx["n"]
    out = {v: {str(x): {"c": [], "c_fine": [], "conf": []} for x in dose_list()} for v in ("nocnn", "beta0")}
    out["extra"] = {str(x): {str(r): {"c": [], "c_fine": [], "conf": []} for r in ctx["extra"]} for x in dose_list()}
    out["beta0_delta_rms"] = {str(x): [] for x in dose_list()}
    t0 = time.time()
    for s in SEEDS:
        d0 = s8.setup_exam(s, dev, "L-ideal", n, ctx["smoke"])
        kind, model, norm = s8.get_net(NEW_L, s, d0["cfg"], d0["geo"].pr, dev, ctx["roots"])
        for di, x in enumerate(dose_list()):
            d = remeasure(d0, x, ns_of(ctx, di, s) if x != FREE else None)
            for v, kw in (("nocnn", {"cnn": False}), ("beta0", {"beta": 0.0})):
                r = run_custom(model, norm, d, **kw)
                ev = eval_out(r["final"], d)
                for k_ in ("c", "c_fine", "conf"):
                    out[v][str(x)][k_].append(ev[k_])
                if v == "beta0":
                    out["beta0_delta_rms"][str(x)].append(float(r["delta"].pow(2).sum(-1).mean(-1).sqrt().mean()))
                del r, ev
            r = run_custom(model, norm, d, extra=ctx["extra"])
            for rr, est in r["extra"].items():
                ev = eval_out(est, d)
                for k_ in ("c", "c_fine", "conf"):
                    out["extra"][str(x)][str(rr)][k_].append(ev[k_])
                del ev
            del r, d
        del model, d0
        log(f"  [variants] seed {s} 完成(經過 {time.time() - t0:.0f} 秒)")
    return out


def part_shift(dev, ctx, log):
    """T5:晶格整體平移 s(x 方向),劑量 1 與 0.01;每個 SV 柱的次像素偏移與是否漏判。"""
    n = ctx["n"]
    out = {str(x): {str(sh): {"c": [], "conf": []} for sh in ctx["shifts"]} for x in SHIFT_DOSES}
    cols = {str(x): {"sub": [], "miss": [], "cid": []} for x in SHIFT_DOSES}
    t0 = time.time()
    for s in SEEDS:
        d0 = s8.setup_exam(s, dev, "L-ideal", n, ctx["smoke"])
        kind, model, norm = s8.get_net(NEW_L, s, d0["cfg"], d0["geo"].pr, dev, ctx["roots"])
        for k, sh in enumerate(ctx["shifts"]):
            for dj, x in enumerate(SHIFT_DOSES):
                ns = ctx["shift"] + 1000 * s + 100 * (2 * k + dj)
                d = shifted_exam(d0, sh, x, ns)
                p = c8.predict(s8.run_net(kind, model, norm, d), d)
                out[str(x)][str(sh)]["c"].append(float(np.nanmean(p["c"].numpy())))
                out[str(x)][str(sh)]["conf"].append(p["conf"].sum(0).numpy().astype(int).tolist())
                vac = d["vac"]
                sv = torch.where(vac.lab == 1)[0]
                cols[str(x)]["sub"] += subpix(vac.pos[sv]).tolist()
                cols[str(x)]["miss"] += (p["pred"][sv] == 0).int().tolist()
                # 柱的身分:(seed, 場, 平移前的位置四捨五入到 0.1 px)
                pos0 = vac.pos[sv].double() - torch.tensor([0.0, float(sh)], dtype=torch.float64)
                cols[str(x)]["cid"] += [f"{s}|{int(f_)}|{float(py):.1f}|{float(px):.1f}" for f_, (py, px) in zip(vac.fid[sv].tolist(), pos0.tolist())]
                del d, p
        del model, d0
        log(f"  [shift] seed {s} 完成(經過 {time.time() - t0:.0f} 秒)")
    out["cols"] = cols
    return out


# ============================================================================
# 報表與判讀(§19.2,事先寫定)
# ============================================================================
def mean3(v):
    return float(np.mean(v))


def recall(confs):
    return s8.vac_stats(np.sum(np.array(confs, float), 0))["sv_r"]


def report(R, ctx):
    out = {}
    if ctx["smoke"]:
        print("\n" + "!" * 100)
        print("迷你流程:場數極少、模型是 smoke 的 → 數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    B, V, S = R["base"], R["variants"], R["shift"]
    L = B[NEW_L]
    c0 = mean3(L["1.0"]["c"])
    c0f = mean3(L["1.0"]["c_fine"])
    rec0 = recall(L["1.0"]["conf"])
    print("\n" + "=" * 100)
    print(f"8d:精度天花板的診斷(L-ideal 的物體,{ctx['n']} 個場 × 3 seeds;基準 c₀ = 劑量 1 的 P6B4e8-L nerr_c = {c0:.5f},精細對齊 {c0f:.5f};"
          f"SV 召回率 {rec0:.3f})")
    print("=" * 100)
    print("  T0 / T4 / T6 基準(nerr_c 格點 / 精細;SV 召回率;草稿 → 第 1…8 級;位置 δ 的 RMS)")
    for nm in (NEW_L, BASE):
        for x in dose_list():
            r = B[nm][str(x)]
            stg = np.mean(np.array(r["stage_c"]), 0)
            print(f"    {tg(nm):<9} 劑量 {str(x):<6}:{mean3(r['c']):.5f} / {mean3(r['c_fine']):.5f};召回率 {recall(r['conf']):.3f};"
                  f"草稿 {mean3(r['draft_c']):.4f} → " + " ".join(f"{v:.4f}" for v in stg) + f";δ RMS {mean3(r['delta_rms']):.3f} px")
    # T6
    lab6 = "天花板主要是評分的假象" if c0f <= HALF * c0 else "不是評分的問題"
    print(f"  ▶ T6:劑量 1 精細對齊 {c0f:.5f} vs 格點 {c0:.5f}(比 {c0f / c0:.3f};≤ {HALF} → 評分的假象):{lab6}")
    # T4
    cfree = mean3(L[FREE]["c"])
    lab4 = "天花板與雜訊無關(網路固有)" if cfree >= KEEP_08 * c0 else "天花板與雜訊有關"
    print(f"  ▶ T4:無雜訊 {cfree:.5f}(精細 {mean3(L[FREE]['c_fine']):.5f})vs c₀ {c0:.5f}(比 {cfree / c0:.3f};≥ {KEEP_08} → 網路固有):{lab4}")
    # T0
    stg1 = np.mean(np.array(L["1.0"]["stage_c"]), 0)
    desc0 = stg1[-1] / stg1[-2] < STAGE_DESC
    print(f"  ▶ T0:劑量 1 第 8 級 / 第 7 級 = {stg1[-1] / stg1[-2]:.3f} → {'到第 8 級仍在下降' if desc0 else '第 8 級已不再明顯下降'}")
    # T8
    print(f"  ▶ T8:真值經過網路的投影後的 nerr_c = {mean3(B['proj']):.2e}(投影本身造成的下限;描述)")
    # T7
    sp = L["1.0"]["spec"]
    ep = np.sum([np.array(x["err_pow"]) for x in sp], 0)
    tp = np.sum([np.array(x["true_pow"]) for x in sp], 0)
    nr = len(ep)
    hi = slice(nr // 2, nr)
    print(f"  T7(劑量 1,精細對齊後):誤差能量在高頻(環 ≥ 一半)的比例 {ep[hi].sum() / ep.sum():.2f}(真值 {tp[hi].sum() / tp.sum():.2f});"
          f"原子處(相位 > {ATOM_PH} rad,占 U 的 {np.mean([x['area_atom'] for x in sp]):.0%})的誤差能量比例 {np.mean([x['frac_atom'] for x in sp]):.2f}")
    out.update({"c0": c0, "c0_fine": c0f, "T6": lab6, "T4": lab4, "T0_desc": bool(desc0)})
    # T1、T2、T3
    print("\n  T1 / T2 / T3(P6B4e8-L 的變體;nerr_c 格點 / 精細;SV 召回率)")
    for x in dose_list():
        cells = [f"基準 {mean3(L[str(x)]['c']):.5f}"]
        for v, nmv in (("nocnn", "無 CNN"), ("beta0", "β = 0")):
            r = V[v][str(x)]
            cells.append(f"{nmv} {mean3(r['c']):.5f} / {mean3(r['c_fine']):.5f}(召回率 {recall(r['conf']):.3f})")
        for rr in ctx["extra"]:
            r = V["extra"][str(x)][str(rr)]
            cells.append(f"+{rr} 步 {mean3(r['c']):.5f}(召回率 {recall(r['conf']):.3f})")
        print(f"    劑量 {str(x):<6}:" + ";".join(cells))
    labs = {}
    c1 = mean3(V["nocnn"]["1.0"]["c"])
    labs["T1"] = "CNN 修正造成天花板" if c1 <= HALF * c0 else ("CNN 不是原因" if c1 >= c0 else "部分")
    drms = mean3(L["1.0"]["delta_rms"])
    c2 = mean3(V["beta0"]["1.0"]["c"])
    labs["T2"] = "位置步造成天花板" if (drms >= POS_RMS_MIN and c2 <= HALF * c0) else "不是位置步"
    rmax = max(ctx["extra"])
    c3 = mean3(V["extra"]["1.0"][str(rmax)]["c"])
    labs["T3"] = "級數不足(8 級還沒收斂)" if c3 <= HALF * c0 else "不是級數不足"
    print(f"  ▶ T1:無 CNN {c1:.5f} vs c₀ {c0:.5f}(比 {c1 / c0:.3f}):{labs['T1']}")
    print(f"  ▶ T2:位置 δ 的 RMS {drms:.3f} px(門檻 {POS_RMS_MIN});β = 0 {c2:.5f}(比 {c2 / c0:.3f};β = 0 時 δ 的 RMS "
          f"{mean3(V['beta0_delta_rms']['1.0']):.1e}):{labs['T2']}")
    print(f"  ▶ T3:+{rmax} 步 {c3:.5f}(比 {c3 / c0:.3f}):{labs['T3']}")
    same = []
    for nmv, cval, rv in (("無 CNN", c1, recall(V["nocnn"]["1.0"]["conf"])), ("β = 0", c2, recall(V["beta0"]["1.0"]["conf"])),
                          (f"+{rmax} 步", c3, recall(V["extra"]["1.0"][str(rmax)]["conf"]))):
        if cval <= HALF * c0 and rv >= RECALL_OK:
            same.append(nmv)
    print(f"  ▶ 與漏判同源:{'、'.join(same) + ' 同時解釋漏判(nerr_c ≤ 0.5 c₀ 且召回率 ≥ 0.9)' if same else '沒有任何變體同時滿足 nerr_c ≤ 0.5 c₀ 與召回率 ≥ 0.9'}")
    out.update(labs)
    out["same_source"] = same
    # T5
    print("\n  T5 次像素平移(晶格整體平移 s;nerr_c / SV 召回率)")
    for x in SHIFT_DOSES:
        cells = []
        for sh in ctx["shifts"]:
            r = S[str(x)][str(sh)]
            cells.append(f"s = {sh:g}:{mean3(r['c']):.5f} / {recall(r['conf']):.3f}")
        print(f"    劑量 {x:g}:" + ";".join(cells))
    t5 = {}
    for x in SHIFT_DOSES:
        cc = S["cols"][str(x)]
        sub, miss = np.array(cc["sub"]), np.array(cc["miss"], bool)
        rates, ns_ = [], []
        for a_, b_ in zip(SUB_BINS[:-1], SUB_BINS[1:]):
            k = (sub >= a_) & (sub < b_)
            ns_.append(int(k.sum()))
            rates.append(float(miss[k].mean()) if k.any() else float("nan"))
        fin = [r_ for r_ in rates if np.isfinite(r_)]
        sens = bool(fin) and max(fin) >= SUB_X * min(fin) and max(fin) - min(fin) >= SUB_D
        ids = {}
        for cid, m_ in zip(cc["cid"], miss):
            ids.setdefault(cid, set()).add(bool(m_))
        flip = float(np.mean([len(v) > 1 for v in ids.values()])) if ids else float("nan")
        t5[str(x)] = {"rates": rates, "n": ns_, "sensitive": sens, "flip": flip}
        print(f"    劑量 {x:g}:次像素偏移 " + "、".join(f"{a_:.2f}–{b_:.2f} px {r_:.1%}(n {n_})" for a_, b_, r_, n_ in zip(SUB_BINS[:-1], SUB_BINS[1:], rates, ns_))
              + f";同一個柱在不同平移下判定會改變的比例 {flip:.1%}")
    out["T5"] = t5
    lab5 = "漏判對次像素位置敏感" if t5["1.0"]["sensitive"] else "漏判對次像素位置不敏感"
    print(f"  ▶ T5(劑量 1;某組 ≥ 最低組 × {SUB_X:g} 且差 ≥ {SUB_D * 100:.0f} 個百分點):{lab5}")
    out["T5_label"] = lab5
    return out


def figures(R, ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = ctx["fig"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    L = R["base"][NEW_L]
    ds = np.array(DOSES)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ds, [np.mean(L[str(x)]["c"]) for x in DOSES], "o-", color="#1baf7a", label="P6B4e8-L (grid align)")
    ax[0].plot(ds, [np.mean(L[str(x)]["c_fine"]) for x in DOSES], "s--", color="#1baf7a", label="P6B4e8-L (fine align)")
    for v, col, lab in (("nocnn", "#d6452a", "no CNN"), ("beta0", "#8a5cd1", "beta = 0")):
        ax[0].plot(ds, [np.mean(R["variants"][v][str(x)]["c"]) for x in DOSES], "o-", color=col, label=lab)
    rmax = str(max(ctx["extra"]))
    ax[0].plot(ds, [np.mean(R["variants"]["extra"][str(x)][rmax]["c"]) for x in DOSES], "o-", color="#0b0b0b", label=f"+{rmax} AP steps")
    ax[0].axhline(np.mean(L[FREE]["c"]), color="#999", ls=":", lw=0.8, label="noise-free (base)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("dose (relative to base)")
    ax[0].set_ylabel("nerr_c (L-ideal objects)")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, which="both", lw=0.4, color="#e4e3df")
    for x, col in zip(DOSES, ("#c9e4f5", "#7fb8e0", "#2a78d6", "#0b3d6b")):
        stg = np.mean(np.array(L[str(x)]["stage_c"]), 0)
        ax[1].plot(range(len(stg) + 1), [np.mean(L[str(x)]["draft_c"])] + list(stg), "o-", color=col, label=f"dose {x:g}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("stage (0 = draft)")
    ax[1].set_ylabel("nerr_c")
    ax[1].legend(fontsize=7)
    ax[1].grid(True, which="both", lw=0.4, color="#e4e3df")
    fig.suptitle("8d: precision ceiling of P6B4e8-L (lattice, 7x7 scan)", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "d8_ceiling.png"
    fig.savefig(p, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


# ============================================================================
# 內建檢查(§19.3)
# ============================================================================
@torch.no_grad()
def checks(dev, ctx):
    print("=" * 70)
    print("階段八 8d:內建檢查")
    print("=" * 70)
    m8, m8c = s8.me_md5(), c8.me_md5()
    check("輸入:scan_8.py、scan_8c.py = 階段八 / 8c 的版本", m8 == S8_MD5 and m8c == S8C_MD5, f"{m8[:8]}… / {m8c[:8]}…")
    d0 = s8.setup_exam(0, dev, "L-ideal", 4, smoke=True)
    kind, model, norm = s8.get_net(NEW_L, 0, d0["cfg"], d0["geo"].pr, dev, s8.roots_of(True))
    ref = c7.run_g9e(model, norm, d0)[0]
    r = run_custom(model, norm, d0, extra=[2])
    e1 = float((r["final"] - ref).abs().max() / ref.abs().max())
    r0 = run_custom(model, norm, d0, beta=0.0)
    e2 = float(r0["delta"].abs().max())
    r1 = run_custom(model, norm, d0, cnn=False)
    differs = float((r1["final"] - ref).abs().max())
    check("自訂的網路流程:預設 = scan_7c.run_g9e(< 1e-6);β = 0 時 δ 恆為 0;關掉 CNN 的輸出確實不同;多接的物理步可執行",
          e1 < 1e-6 and e2 == 0.0 and differs > 0 and len(r["extra"][2]) == 4, f"{e1:.1e}、δ 最大 {e2:.1e}、無 CNN 的差 {differs:.2e}")
    # 平移
    ns = SEED_SHIFT + SMOKE_OFF + 999
    d_s0 = shifted_exam(d0, 0.0, 1.0, ns)
    same0 = float((d_s0["O"] - d0["O"]).abs().max())
    d_s = shifted_exam(d0, 0.3, 1.0, ns)
    # 平移後的柱位置 = 原本 + (0, 0.3):平移後離場邊 ≥ 10 px 的柱,對照原本所有的柱(避開場邊的柱進出)
    Fq = d0["geo"].F
    dpos = 0.0
    for i in range(len(d0["meta"]["cols"])):
        p0 = d0["meta"]["cols"][i]["pos"].double()
        p1 = d_s["meta"]["cols"][i]["pos"].double()
        p1 = p1[((p1 >= 10) & (p1 <= Fq - 11)).all(1)] - torch.tensor([0.0, 0.3], dtype=torch.float64)
        if len(p1):
            dpos = max(dpos, float(torch.cdist(p1, p0).min(1).values.max()))
    pr_, ok_, _ = d_s["vac"].read(d_s["fields"][:, 1].cpu())
    C_, _ = d_s["vac"].confusion(pr_)
    Ct = C_.sum(0).numpy()
    diag = bool((Ct - np.diag(np.diag(Ct))).sum() == 0)
    check("平移:s = 0 時物體與原本相同(逐位元);s = 0.3 時柱的位置 = 原本 + 0.3 px(< 1e-4;位置是 float32);平移後的真值判讀為對角",
          same0 == 0.0 and dpos < 1e-4 and diag, f"{same0:.1e}、{dpos:.1e}、{Ct.astype(int).tolist()}")
    # 精細對齊
    gen = torch.Generator().manual_seed(SEED_SHIFT + SMOKE_OFF + 998)
    tt = (torch.rand(4, 2, generator=gen, dtype=torch.float64) - 0.5) * 0.06 + 0.013
    qq = (torch.rand(4, 2, generator=gen, dtype=torch.float64) - 0.5) * 2e-4 + 3.3e-5
    F = d0["geo"].F
    kk = torch.fft.fftfreq(F, dtype=torch.float64, device=dev)
    ky, kx = torch.meshgrid(kk, kk, indexing="ij")
    yc = torch.arange(F, dtype=torch.float64, device=dev) - (F - 1) / 2
    O = d0["O"].to(torch.complex128)
    ph = -2 * math.pi * (ky[None] * tt[:, 0, None, None].to(dev) + kx[None] * tt[:, 1, None, None].to(dev))
    ramp = torch.polar(torch.ones(4, F, F, dtype=torch.float64, device=dev),
                       2 * math.pi * (qq[:, 0, None, None].to(dev) * yc[None, :, None] + qq[:, 1, None, None].to(dev) * yc[None, None, :]))
    # 先乘斜坡、再平移(對齊時先平移回來、再去斜坡 → 精確的逆運算)
    est = (torch.fft.ifft2(torch.fft.fft2(O * ramp) * torch.polar(torch.ones_like(ph), ph)) * np.exp(0.7j)).to(torch.complex64)
    cg = nerr_grid(est, d0)
    cf, _, _ = fine_align(est, d0)
    rnd = torch.polar(torch.rand(4, F, F, device=dev), torch.rand(4, F, F, device=dev))
    cfr, _, _ = fine_align(rnd, d0)
    cgr = nerr_grid(rnd, d0)
    check("精細對齊:已知的次像素平移 + 斜坡 + 整體相位 → 精細 nerr_c < 1e-6(格點版做不到);隨機輸入時精細 ≤ 格點",
          float(cf.max()) < 1e-6 and float(cg.min()) > 1e-6 and bool((cfr <= cgr + 1e-9).all()),
          f"精細 最大 {float(cf.max()):.1e};格點 最小 {float(cg.min()):.1e};隨機 精細 ≤ 格點 {bool((cfr <= cgr + 1e-9).all())}")
    # 無雜訊
    dfree = remeasure(d0, FREE, None)
    refc = s8.measure_L(d0["fields"], d0["geo"], d0["cp"], d0["bs"], 0, d0["spec"], d0["z"], d0["sign"], d0["ops"], torch.ones(4), poisson=False)
    ef = max(float((a - b).abs().max()) for a, b in zip(dfree["counts"], refc))
    check("無雜訊的量測 = measure_L(poisson = False)(逐位元)", ef == 0.0, f"{ef:.1e}")
    # seed
    found = s8.scan_seed_constants({"scan_8d.py"})
    clash = sorted(v for v in found if any(lo <= v <= hi for lo, hi in MY_RANGES_8D))
    mx = max(SEED_NOISE + 10_000 * (len(DOSES) - 1) + 1000 * max(SEEDS), SEED_SHIFT + 1000 * max(SEEDS) + 100 * (2 * (len(SHIFTS) - 1) + 1) + 999) + SMOKE_OFF
    over = [x for x in (SEED_NOISE, SEED_SHIFT) if any(lo <= x <= hi for lo, hi in s8.MY_RANGES + c8.MY_RANGES_8C)]
    check("seed:8d 的區間 88,920,000–88,999,999 不與既有腳本的常數、階段八、8c 重疊;所有派生的 seed 在區間內",
          not clash and not over and mx <= MY_RANGES_8D[0][1], f"重疊 {clash} {over};最大 {mx:,}")
    del d0, model


# ============================================================================
# 主程式
# ============================================================================
def check_models(ctx):
    mb = []
    for s in SEEDS:
        for md in (s8.model_dir("L", s, ctx["roots"]["m"]), d7.model_dir("e", s, ctx["roots"]["e"])):
            if not ((md / "final.pt").exists() and (md / "result.json").exists()):
                mb.append(md.name)
    check(f"模型:P6B4e8-L、P6B4e8 各 3 seeds({'smoke' if ctx['smoke'] else '正式'})", not mb, "缺 " + "、".join(mb) if mb else "")
    return not mb


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return o.tolist()
    raise TypeError(str(type(o)))


def strip(o):
    """存檔前去掉以 _ 開頭的鍵(大型張量)。"""
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


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
    print(f"scan_8d.py md5 {me};GPU {a7.gpu_name(dev)}")
    smoke = bool(a.smoke)
    ctx = ctx_of(smoke)
    if a.check:
        checks(dev, ctx)
        check_models(ctx_of(False))
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if smoke:
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)
        SMOKE_DIR.mkdir(parents=True)
        checks(dev, ctx)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
    elif a.run:
        if not PASSED.exists() or json.load(open(PASSED)).get("script_md5") != me:
            raise SystemExit(f"❌ 找不到 {PASSED} 或 smoke 時的版本不同:先在 dev 節點用現在的版本跑 python scan_8d.py --smoke")
        print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")
        m8, m8c = s8.me_md5(), c8.me_md5()
        check("輸入:scan_8.py、scan_8c.py = 階段八 / 8c 的版本", m8 == S8_MD5 and m8c == S8C_MD5, f"{m8[:8]}… / {m8c[:8]}…")
    else:
        ap.print_help()
        sys.exit(2)
    if not check_models(ctx) or not all_ok():
        print("\n❌ 輸入不完整,停下來")
        sys.exit(1)
    part_path = Path(str(ctx["out"]) + ".partial")
    meta = {"md5": me, "smoke": smoke, "n": ctx["n"], "quick": QUICK}
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
    for name, fn in (("base", lambda: part_base(dev, ctx, log)), ("variants", lambda: part_variants(dev, ctx, log)),
                     ("shift", lambda: part_shift(dev, ctx, log))):
        if name in R:
            continue
        t1 = time.time()
        print(f"\n  [{name}] 開始", flush=True)
        R[name] = json.loads(json.dumps(strip(fn()), default=_js))
        Rok[name] = bool(all_ok())
        tp[name] = time.time() - t1
        dump_json({"meta": meta, "parts": R, "parts_ok": Rok}, part_path)
        print(f"  [{name}] 完成(經過 {tp[name]:.0f} 秒)", flush=True)
    rep = report(R, ctx)
    res = {"meta": meta, "parts": R, "report": json.loads(json.dumps(rep, default=_js)), "checks_ok": all_ok(), "complete": False,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0, "part_seconds": tp}
    dump_json(res, ctx["out"])
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        figures(R, ctx)
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
            print("\n✅ 迷你全流程全部通過 → 可以送出 run_scan8d.sh")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    print("\n" + ("✅ 內建檢查全部通過" if all_ok() else "❌ 有檢查未通過"))
    print("把完整輸出與 figs_scan8d/ 的圖傳給 Claude")


if __name__ == "__main__":
    main()
