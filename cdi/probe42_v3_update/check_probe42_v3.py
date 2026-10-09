#!/usr/bin/env python
"""階段四 4-2a 展開版 v3:平頂探針(r = 12 px)下訓練網路 —— 送件前檢查、關卡與結果判讀(階段四協定 v2 §四、§8.10)。

與 check_probe42_ds.py 相同,只差:
  - 展開版改為 probe_v3_unroll(= probe_ds_unroll + 自然讀出 + 修正上限 0.1 + 輸出殘差上限 0.1 + 分段 K = 5)。
  - U-Net、Oracle-psi 沿用首批 probe_unet_s*、probe_unet_oracle_s*。
  - 送件前多兩項:未訓練輸出 = 自己內部狀態的自然讀出(差 0);未訓練 ≈ 同起點 hio.py 50 次(差 ≤ 0.05 且 ≥ 0.35)。
  - 訓練失敗多一項「退化」:訓練後 F < 同 seed 未訓練模型的 F − 0.02。
  - 另報 |修正| / |h| 與輸出頭殘差大小。

用法(需在計算節點執行):
    python check_probe42_v3.py --configs-only   # 送件前
    python check_probe42_v3.py --gate           # 關卡:只看 seed 0 的訓練失敗判定與退化(通過才訓練 seed 1、2)
    python check_probe42_v3.py                  # 三個 seed 跑完後:完整評分與 §4 判讀
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
OUT_JSON = RUN_ROOT / "probe42_v3.json"
FIG_DIR = Path("figs_probe42_v3")
NETS = {"unet": "probe_unet", "unroll": "probe_v3_unroll", "oracle": "probe_unet_oracle"}
NET_LABEL = {"unet": "U-Net", "unroll": "物理內嵌 U-Net(展開 v3)", "oracle": "U-Net + 相位(上限)"}
SEEDS = [0, 1, 2]
ITERS = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
UNSEEN = ["mnist_test", "fashion_mnist", "random_shapes", "random_texture"]
SHORT = {"mnist_test": "mnist", "fashion_mnist": "fashion",
         "random_shapes": "shapes", "random_texture": "texture"}
TIME_BATCHES = [64, 512]
MAIN_BATCH = 64
TIME_REPS = 5
SANITY_TOL = 0.01
REPRO_TOL = 0.005
N_OBJ = None
QUICK = os.environ.get("CDI_QUICK") == "1"     # 只供本地測程式流程
if QUICK:
    ITERS, TIME_BATCHES, MAIN_BATCH, TIME_REPS, N_OBJ = [5, 20, 50], [8, 32], 8, 1, 32

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def raw_home(cfg, n):
    """未套探針的物體(與 generalization_suites 的 procedural 同一批)。"""
    from src.procedural import make_procedural
    return make_procedural(n, cfg, seed=cfg.test_seed, kinds=tuple(cfg.proc_kinds),
                           weights=tuple(cfg.proc_weights),
                           target_support=cfg.match_support_to,
                           contrast_gamma=cfg.proc_contrast_gamma)


def box_cfg(cfg):
    """同一份設定但不啟用探針 —— 只用於 A-HIO 的方框 support(量測尺度相同)。"""
    return Cfg.from_dict({**cfg.to_dict(), "probe": "none"})


# ============================================================================
# 送件前
# ============================================================================
def configs_and_unit_tests():
    from src.data import build_pool, generalization_suites
    from src.hio import hio, random_init
    from src.model import build_model
    from src.physics import (apply_probe, beamstop_mask, build_input, calibrate_flux,
                             calibrate_input_norm, forward_measure, probe_amplitude,
                             support_mask)

    print("=" * 70)
    print("變因控制")
    print("=" * 70)
    if not Path("configs/probe_v3_unroll.yaml").exists():
        check("configs/probe_v3_unroll.yaml 存在", False)
        return
    base = Cfg.load("configs/amb_base.yaml").to_dict()
    cu = {a: Cfg.load(f"configs/{n}.yaml").to_dict() for a, n in NETS.items()}

    def diff(a, b):
        return sorted(k for k in a if a[k] != b[k])
    d0 = diff(base, cu["unet"])
    # probe_r 的預設值即 12.0,故差異可能只有 probe 一欄
    check("amb_base vs probe_unet 只差 probe(、probe_r)",
          "probe" in d0 and set(d0) <= {"probe", "probe_r"} and cu["unet"]["probe_r"] == 12.0,
          f"實際差異: {d0};probe_r = {cu['unet']['probe_r']}")
    old = Cfg.load("configs/probe_ds_unroll.yaml").to_dict()
    d3 = diff(old, cu["unroll"])
    check("probe_ds_unroll vs probe_v3_unroll 只差 unroll_readout = natural、unroll_fix_bound = 0.1、"
          "unroll_out_bound = 0.1、unroll_trunc = 5",
          d3 == ["unroll_fix_bound", "unroll_out_bound", "unroll_readout", "unroll_trunc"]
          and cu["unroll"]["unroll_readout"] == "natural" and cu["unroll"]["unroll_fix_bound"] == 0.1
          and cu["unroll"]["unroll_out_bound"] == 0.1 and cu["unroll"]["unroll_trunc"] == 5,
          f"實際差異: {d3}")
    d1 = diff(cu["unet"], cu["oracle"])
    check("probe_unet vs probe_unet_oracle 只差 oracle_phase", d1 == ["oracle_phase"], f"實際差異: {d1}")
    d2 = diff(cu["unet"], cu["unroll"])
    check("probe_unet vs probe_v3_unroll 只差 arch、unroll_*、grad_clip",
          "arch" in d2 and set(d2) <= {"arch", "unroll_t", "unroll_refine", "unroll_every",
                                       "unroll_trunc", "grad_clip", "unroll_readout",
                                       "unroll_fix_bound", "unroll_out_bound"},
          f"實際差異: {d2}")
    sel = RUN_ROOT / ("probe42_select_quick.json" if QUICK else "probe42_select.json")
    if sel.exists():
        pk = json.load(open(sel))["pick"]
        same = all(cu["unroll"][k] == pk[k] for k in ["unroll_t", "unroll_refine", "unroll_every"])
        check("probe_v3_unroll.yaml 與 select_unroll.py 的選定一致", same,
              f"T = {pk['unroll_t']}、{pk['unroll_refine']}、每 {pk['unroll_every']} 輪修正")
    else:
        check(f"{sel.name} 存在", False)

    print("\n" + "=" * 70)
    print("探針管線")
    print("=" * 70)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Cfg.load("configs/probe_unet.yaml")
    c0 = box_cfg(cfg)
    D = probe_amplitude(cfg)
    box = support_mask(c0)
    check("support_mask:探針設定回傳圓盤、原設定仍為方框",
          torch.equal(support_mask(cfg), (D > 0).float()) and int(box.sum()) == 1024
          and bool(((D > 0) & (box <= 0)).sum() == 0),
          f"圓盤 {int(D.sum())} px(方框 {int(box.sum())} px)")
    raw = raw_home(cfg, 32)
    psi = apply_probe(raw, cfg)
    check("apply_probe:圓盤內 = 物體、圓盤外 = 0",
          torch.equal(psi[:, 0], raw[:, 0] * D) and torch.equal(psi[:, 1], raw[:, 1] * D))
    check("apply_probe:probe = none 時原樣回傳", apply_probe(raw, c0) is raw)
    try:
        g_p = generalization_suites(cfg, 16)
        g_0 = generalization_suites(c0, 16)
        ok_suite = all(bool((g_p[k] * (1 - D)).abs().sum() == 0) for k in g_p) and \
            torch.equal(g_p["procedural"], apply_probe(g_0["procedural"], cfg)) and \
            not torch.equal(g_p["procedural"], g_0["procedural"])
        check("eval.py 的測試集(generalization_suites)全部套上探針", ok_suite,
              "五個測試集圓盤外皆為 0;procedural = 平面波版 × 圓盤")
    except RuntimeError as e:
        check("eval.py 的測試集(generalization_suites)全部套上探針", False, f"無法載入資料集:{e}")
    pool = build_pool(cfg, 64, seed=0)
    check("訓練池(build_pool)套上探針", bool((pool * (1 - D)).abs().sum() == 0))

    print("\n" + "=" * 70)
    print("模型單元測試")
    print("=" * 70)
    for a in NETS:
        c = Cfg.load(f"configs/{NETS[a]}.yaml")
        ps = apply_probe(raw_home(c, 32), c)
        c.ref_energy = calibrate_flux(ps)
        ps = ps.to(dev)
        bs = beamstop_mask(c, device=dev)
        calibrate_input_norm(ps, bs, c)
        counts = forward_measure(ps, bs, c)
        m = build_model(c).to(dev)
        y = m(build_input(ps, counts, bs, c))
        (y ** 2).sum().backward()
        good = (y.shape == ps.shape and bool(torch.isfinite(y).all())
                and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                        for p in m.parameters()))
        outside = float((y.detach() * (1 - probe_amplitude(c, device=dev))).abs().max())
        check(f"{NETS[a]}:前向 / 反向正常,輸出在探針外為 0", good and outside == 0,
              f"參數量 {sum(p.numel() for p in m.parameters()):,}")

    # 分段反傳(協定 §8.5):最後一段 = forward;分段後初始化梯度範數 < 10
    from src.losses import build_loss
    c = Cfg.load("configs/probe_v3_unroll.yaml")
    ps = apply_probe(raw_home(c, 64), c)
    c.ref_energy = calibrate_flux(ps)
    ps = ps.to(dev)
    bs = beamstop_mask(c, device=dev)
    calibrate_input_norm(ps, bs, c)
    torch.manual_seed(0)
    counts = forward_measure(ps, bs, c)
    x = build_input(ps, counts, bs, c)
    lf = build_loss(c, dev)
    gn = {}
    for K in [c.unroll_trunc, 0]:
        torch.manual_seed(0)
        m = build_model(Cfg.from_dict({**c.to_dict(), "unroll_trunc": K})).to(dev).train()
        if K:
            outs = m.forward_segments(x)
            (sum(lf(o, ps, counts, bs)[0] for o in outs) / len(outs)).backward()
        else:
            lf(m(x), ps, counts, bs)[0].backward()
        gn[K] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None)))
        if K:
            seg_d = 0.0
            with torch.no_grad():
                for mode in ("train", "eval"):
                    getattr(m, mode)()
                    seg_d = max(seg_d, float((m(x) - m.forward_segments(x)[-1]).abs().max()))
            n_seg = len(outs)
    check(f"分段輸出:{n_seg} 段,最後一段 = forward(推論不變)", seg_d == 0 and n_seg == c.unroll_t // c.unroll_trunc,
          f"最大差 {seg_d}")
    check(f"分段後初始化總梯度範數 < 10(64 張)", math.isfinite(gn[c.unroll_trunc]) and gn[c.unroll_trunc] < 10,
          f"分段 {gn[c.unroll_trunc]:.3g};(參考)不分段 {gn[0]:.3g}")

    # v3(協定 §8.10):未訓練時輸出 = 自己內部狀態的自然讀出;且 ≈ 同起點的 hio.py 50 次
    from realign_eval import score_all
    c = Cfg.load("configs/probe_v3_unroll.yaml")
    nv3 = N_OBJ or 512
    raw3 = raw_home(c, nv3)
    ps = apply_probe(raw3, c)
    c.ref_energy = calibrate_flux(ps)
    ps = ps.to(dev)
    bs = beamstop_mask(c, device=dev)
    calibrate_input_norm(ps, bs, c)
    torch.manual_seed(c.test_seed)
    counts = forward_measure(ps, bs, c)
    x = build_input(ps, counts, bs, c)
    torch.manual_seed(0)
    m = build_model(c).to(dev).eval()
    with torch.no_grad():
        y, ynat, yref = [], [], []
        for i in range(0, nv3, 128):
            xs = x[i:i + 128]
            g0, A, bsm = m.start(xs)
            y.append(m(xs))
            ynat.append(m.natural(m.rounds(g0, A, bsm)) * m.sup)
            init = torch.stack([g0.abs(), torch.angle(g0)], dim=1)
            yref.append(hio(init, counts[i:i + 128], bs, c, n_iter=c.unroll_t, beta=c.hio_beta))
        y, ynat, yref = torch.cat(y), torch.cat(ynat), torch.cat(yref)
    d_nat = float((y - ynat).abs().max())
    check("v3 未訓練:輸出 = 自己 50 輪內部狀態的自然讀出(端到端)", d_nat == 0.0, f"最大差 {d_nat}")
    f_m = score_all(m, ps, bs, c, y)[0]["aligned"]["frc_gain"]
    f_h = score_all(m, ps, bs, c, yref)[0]["aligned"]["frc_gain"]
    ok3 = abs(f_m - f_h) <= 0.05 and (f_m >= 0.35 or QUICK)
    check(f"v3 未訓練 ≈ 同起點 hio.py {c.unroll_t} 次(差 ≤ 0.05{'' if QUICK else '、且 ≥ 0.35'})", ok3,
          f"未訓練模型 {f_m:+.4f};hio.py {f_h:+.4f}(對齊後 F,{nv3} 張)")

    # 展開版:未訓練時的物理步驟 = hio()(探針 support)。
    # 展開版的傅立葉約束以 sqrt(|G|² + 1e-12) 取振幅(為了梯度穩定),hio.py 用 |G|;
    # 兩者只在 |G| 極小的頻率上有微小差異(float32 / float64 皆約 1e-5),
    # 但隨機起點下 HIO 的軌跡對微小差異極敏感,數輪後少數樣本即分岔(實測 3 輪 4e-5、5 輪起個別樣本 > 1e-3)。
    # 故逐步等價以 T = every + 1 輪檢驗(同時涵蓋「有修正」與「只跑物理」兩種輪次);較多輪只作參考。
    c = Cfg.load("configs/probe_v3_unroll.yaml")
    every = int(c.unroll_every)
    ps = apply_probe(raw_home(c, 32), c)
    c.ref_energy = calibrate_flux(ps)
    ps = ps.to(dev)
    bs = beamstop_mask(c, device=dev)
    calibrate_input_norm(ps, bs, c)
    counts = forward_measure(ps, bs, c)
    x = build_input(ps, counts, bs, c)
    init = random_init(counts, c, seed=1, device=dev)
    sup = support_mask(c, device=dev)
    from src.hio import _measured_amp
    errs = {}
    for T in [every + 1, 5 * every]:
        ct = Cfg.from_dict({**c.to_dict(), "unroll_t": T, "unroll_trunc": 0})   # 物理步驟檢驗,與分段無關
        m = build_model(ct).to(dev).eval()
        with torch.no_grad():
            if T == every + 1:
                # 以「相對於最大振幅」衡量:零光子的像素 A = 0,反推時 expm1 的捨入會留下 ~1e-6 的殘值,
                # 逐點相對誤差在該處沒有意義(國網 GPU 上曾因此得到 1.6 的假警報)。
                A_in, A_ct = m.measured_amp(x), _measured_amp(counts, ct)
                a_err = float((A_in - A_ct).abs().max() / A_ct.abs().max())
                lit = counts >= 1
                a_rel = float(((A_in - A_ct).abs()[lit] / A_ct[lit]).max())
            g = m.rounds(torch.polar(init[:, 0], init[:, 1]), m.measured_amp(x), x[:, -1])
            ref = hio(init, counts, bs, ct, n_iter=T, beta=ct.hio_beta)
        mine = torch.polar(g.abs().clamp(0, 1) * sup, torch.angle(g).clamp(0, ct.phase_max) * sup)
        d = (mine - torch.polar(ref[:, 0], ref[:, 1])).abs().amax(dim=(-2, -1))
        errs[T] = (float(d.max()), int((d > 1e-3).sum()))
    check("展開版由輸入反推的量測振幅 = hio.py 的量測振幅", a_err < 1e-4 and a_rel < 1e-3,
          f"最大差 / 最大振幅 {a_err:.1e};有光子的像素逐點相對差最大 {a_rel:.1e}")
    T1 = every + 1
    check(f"未訓練的展開 {T1} 輪 = hio.py(探針 support){T1} 次", errs[T1][0] < 1e-3,
          f"最大差 {errs[T1][0]:.1e}")
    T5 = 5 * every
    print(f"     (參考){T5} 輪:最大差 {errs[T5][0]:.1e},差 > 1e-3 的樣本 {errs[T5][1]}/32"
          f"(HIO 對微小差異敏感,見上方註解)")

    # 真值(移到約束邊界內、無雜訊)為展開的固定點
    t = ps.clone()
    on = t[:, 0] > 0
    t[:, 0] = t[:, 0] * 0.95
    t[:, 1] = torch.where(on, t.new_tensor(c.phase_max) * 0.05 + 0.9 * t[:, 1], t[:, 1])
    cn = Cfg.from_dict(c.to_dict())
    cn.add_poisson = False
    cnt0 = forward_measure(t, bs, cn)
    x0 = build_input(t, cnt0, bs, cn)
    tt = torch.polar(t[:, 0], t[:, 1])
    with torch.no_grad():
        g = m.rounds(tt, m.measured_amp(x0), x0[:, -1])
    err = float((g - tt).abs().max())
    check("真值是展開(探針版)的固定點", err < 1e-3, f"最大差 {err:.1e}")

    # 既有 checkpoint 在新 model.py 下不變(預設值 = 原版)
    from realign_eval import load, net_out, score
    for name in ["amb_base", "arch_unroll_amb"]:
        rd = RUN_ROOT / f"{name}_s0"
        if not (rd / "final.pt").exists():
            check(f"既有 checkpoint {name}_s0 在新 model.py 下重現 metrics.json", False, "找不到檔案")
            continue
        cfg0, mdl = load(rd, dev)
        n = N_OBJ or cfg0.eval_n
        home = raw_home(cfg0, n).to(dev)
        bs0 = beamstop_mask(cfg0, device=dev)
        torch.manual_seed(cfg0.test_seed)
        cnt = forward_measure(home, bs0, cfg0)
        r = score(mdl, home, bs0, cfg0, net_out(mdl, home, cnt, bs0, cfg0))["frc_gain"]
        mm = json.load(open(rd / "metrics.json"))["test"]["frc_gain"]
        check(f"既有 checkpoint {name}_s0 在新 model.py 下重現 metrics.json", abs(r - mm) <= SANITY_TOL,
              f"{r:+.4f} vs {mm:+.4f}")
    print(f"\n     裝置:{dev}")


# ============================================================================
# 跑完後
# ============================================================================
def time_call(fn, dev, reps=TIME_REPS):
    fn()
    sync(dev)
    t0 = time.perf_counter()
    fn()
    sync(dev)
    one = max(time.perf_counter() - t0, 1e-6)
    inner = max(1, int(math.ceil(0.02 / one)))
    ts = []
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        sync(dev)
        ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts))


TRAIN_FAIL = {}
DEGRADE = {}          # seed -> (訓練後 F, 未訓練 F)


def untrained_F(cfg, s, psi, bs, counts, dev):
    """同 seed 的未訓練 v3 模型(同一設定、torch.manual_seed(s) 初始化)在同一批物體與量測上的對齊後 F。"""
    from realign_eval import net_out, score_all
    from src.model import build_model
    torch.manual_seed(s)
    m0 = build_model(Cfg.from_dict(cfg.to_dict())).to(dev).eval()
    return score_all(m0, psi, bs, cfg, net_out(m0, psi, counts, bs, cfg))[0]["aligned"]["frc_gain"]


@torch.no_grad()
def fix_stats(model, x):
    """訓練後的 |修正| / |h| 平均與輸出頭殘差平均(振幅、相位 / phase_max)。"""
    ratios, res_a, res_p = [], [], []
    for i in range(0, len(x), 128):
        xs = x[i:i + 128]
        g, A, bsm = model.start(xs)
        k = 0
        for t in range(model.t):
            g_new = model.fourier_step(g, A, bsm)
            h = model.hio_step(g, g_new)
            if (t + 1) % model.every == 0:
                k += 1
                feat = torch.cat([model._c2r(g_new), model._c2r(g), model._c2r(h)], dim=1)
                r = model._fix(model.refine[k](feat))
                ratios.append(float(r.abs().mean() / h.abs().mean().clamp_min(1e-12)))
                g = h + r
            else:
                g = h
        z = model.head(torch.cat([model._c2r(g), g.abs()[:, None]], dim=1))
        b = float(model.out_bound)
        sup = model.sup > 0
        res_a.append(float((b * torch.tanh(z[:, 0])).abs()[:, sup].mean()))
        res_p.append(float((b * torch.tanh(z[:, 1])).abs()[:, sup].mean()))
    return float(np.mean(ratios)), float(np.mean(res_a)), float(np.mean(res_p))


def degrade_check(s, dev):
    """退化判定(協定 §8.10):訓練後 F < 同 seed 未訓練 F − 0.02 → 訓練失敗。回傳 True = 退化。"""
    from realign_eval import load, net_out, score_all
    from src.physics import apply_probe, beamstop_mask, build_input, forward_measure
    rd = RUN_ROOT / f"{NETS['unroll']}_s{s}"
    cfg, model = load(rd, dev)
    n = N_OBJ or cfg.eval_n
    psi = apply_probe(raw_home(cfg, n), cfg).to(dev)
    bs = beamstop_mask(cfg, device=dev)
    torch.manual_seed(cfg.test_seed + s)                   # 與評分相同的量測
    counts = forward_measure(psi, bs, cfg)
    f_t = score_all(model, psi, bs, cfg, net_out(model, psi, counts, bs, cfg))[0]["aligned"]["frc_gain"]
    f_0 = untrained_F(cfg, s, psi, bs, counts, dev)
    ratio, ra, rp = fix_stats(model, build_input(psi, counts, bs, cfg))
    DEGRADE[s] = (f_t, f_0)
    bad = f_t < f_0 - 0.02
    print(f"    {rd.name}:訓練後 F {f_t:+.4f};未訓練 F {f_0:+.4f};差 {f_t - f_0:+.4f}"
          f"  |修正|/|h| {ratio:.3f};輸出殘差 振幅 {ra:.4f}、相位 {rp:.4f}(上限 0.1)"
          + ("  ❌ 退化(訓練失敗)" if bad else "  ✅"), flush=True)
    return bad


def train_fail_reasons(name, s):
    """§8.5 的三項訓練失敗判定;回傳 (原因 list, 說明字串)。"""
    h = json.load(open(RUN_ROOT / f"{name}_s{s}" / "history.json"))
    tot = [e["total"] for e in h]
    nonfin = not all(np.isfinite(tot))
    skipped = sum(e.get("skipped", 0) for e in h)
    n_steps = len(h) * (20000 // 128)
    collapse = np.isfinite(tot[-1]) and tot[-1] > 1.10 * np.nanmin(tot)
    why = [w for w, c in [("loss 非有限", nonfin), (f"跳過 {skipped} 步", skipped > 0.01 * n_steps),
                          (f"最終 {tot[-1]:.3f} > 最低 {np.nanmin(tot):.3f} × 1.10", collapse)] if c]
    if "grad_norm_max" in h[0]:
        gmax = np.nanmax([e.get("grad_norm_max", np.nan) for e in h])
        clip = np.nanmean([e.get("clip_frac", np.nan) for e in h])
        info = f"跳過 {skipped} 步;梯度最大 {gmax:.3g};裁切比例 {100 * clip:.1f}%"
        if clip > 0.05:
            info += "(> 5%:安全網經常觸發,結果中須註明)"
    else:
        info = "首批(無梯度紀錄)"
    if "seg_loss" in h[-1]:
        info += ";末 epoch 各段 loss " + " ".join(f"{v:.3f}" for v in h[-1]["seg_loss"])
    info += f";loss 首 {tot[0]:.3f} → 末 {tot[-1]:.3f}"
    return why, info


def gate(dev):
    """關卡(協定 §8.10):只看 seed 0 的展開版 v3。"""
    print("\n" + "=" * 70)
    print("關卡:probe_v3_unroll_s0 的訓練失敗判定與退化(通過才訓練 seed 1、2)")
    print("=" * 70)
    why, info = train_fail_reasons(NETS["unroll"], 0)
    print(f"    {NETS['unroll']}_s0:{info}" + (f"  ❌ 訓練失敗({'、'.join(why)})" if why else "  ✅"))
    bad = degrade_check(0, dev)
    passed = not why and not bad
    print("\n" + ("✅ 關卡通過:可以訓練 seed 1、2" if passed else "❌ 關卡未通過:不要訓練 seed 1、2,把輸出貼給 Claude"))
    return passed


def convergence():
    """收斂與訓練失敗判定(協定 §8.5,結果出來前寫定):
       任一 epoch 的 loss 非有限、或跳過的 step 超過總數 1%、或最終 loss > 全程最低 loss × 1.10(暴衝後停滯)
       → 該組訓練失敗,不依 §4 判讀。"""
    print("\n" + "=" * 70)
    print("收斂與訓練失敗判定(末 5 epoch 平均每步降幅佔 loss 比例;須相近才可比較)")
    print("=" * 70)
    out = {}
    for a, name in NETS.items():
        fr, fl, sec, lm, fails = [], [], [], [], []
        for s in SEEDS:
            h = json.load(open(RUN_ROOT / f"{name}_s{s}" / "history.json"))
            tot = [e["total"] for e in h]
            k = min(5, len(tot) - 1)
            fr.append((tot[-1 - k] - tot[-1]) / k / tot[-1])
            fl.append(tot[-1])
            sec.append(np.mean([e["sec"] for e in h]))
            lm.append(json.load(open(RUN_ROOT / f"{name}_s{s}" / "config_used.json"))["log_mean"])
            why, info = train_fail_reasons(name, s)
            fails.append(bool(why))
            print(f"    {name}_s{s}:{info}" + (f"  ❌ 訓練失敗({'、'.join(why)})" if why else "  ✅"))
        out[a] = float(np.mean(fr))
        TRAIN_FAIL[a] = bool(any(fails))
        print(f"  {NET_LABEL[a]:<22} {100 * np.mean(fr):.2f}%   最終 loss {np.mean(fl):.4f}   "
              f"單 epoch {np.mean(sec):.1f} s   log_mean {np.mean(lm):.4f}")
    lm0 = json.load(open(RUN_ROOT / "amb_base_s0" / "config_used.json"))["log_mean"]
    lm1 = json.load(open(RUN_ROOT / "probe_unet_s0" / "config_used.json"))["log_mean"]
    check("正規化常數在探針條件下重新校正(log_mean 與平面波 amb_base 不同)", abs(lm1 - lm0) > 1e-3,
          f"amb_base {lm0:.4f} → probe_unet {lm1:.4f}")
    return out


@torch.no_grad()
def timing(dev):
    """網路與 HIO 在同一程序、同一 GPU、相同 batch 下計時(日誌 §20.3)。"""
    from realign_eval import load
    from src.hio import hio, random_init
    from src.physics import apply_probe, beamstop_mask, build_input, forward_measure
    print("\n" + "=" * 70)
    print("計時(ms/sample;中位數)")
    print("=" * 70)
    cfg, _ = load(RUN_ROOT / "probe_unet_s0", dev)
    bs = beamstop_mask(cfg, device=dev)
    psi = apply_probe(raw_home(cfg, max(TIME_BATCHES)), cfg).to(dev)
    torch.manual_seed(cfg.test_seed)
    counts_all = forward_measure(psi, bs, cfg)
    cb = box_cfg(cfg)
    out = {"hio": {}, "fit": {}, "net": {}}
    for B in TIME_BATCHES + ([1] if not QUICK else []):
        counts = counts_all[:B]
        if B != 1:
            for key, c in [("P", cfg), ("A", cb)]:
                init = random_init(counts, c, seed=0, device=dev)
                for attempt in range(2):
                    ms = {n: 1e3 * time_call(lambda: hio(init, counts, bs, c, n_iter=n,
                                                         beta=c.hio_beta), dev) / B for n in ITERS}
                    ns = np.array(ITERS, float)
                    ts = np.array([ms[n] for n in ITERS])
                    b, a = np.polyfit(ns, ts, 1)
                    r2 = 1 - ((ts - (a + b * ns)) ** 2).sum() / ((ts - ts.mean()) ** 2).sum()
                    if r2 >= 0.99:
                        break
                out["hio"].setdefault(str(B), {})[key] = {str(n): ms[n] for n in ITERS}
                out["fit"].setdefault(str(B), {})[key] = {"a": float(a), "b": float(b), "r2": float(r2)}
        for arch, name in NETS.items():
            ncfg, model = load(RUN_ROOT / f"{name}_s0", dev)
            x = build_input(psi[:B], counts, bs, ncfg)
            out["net"].setdefault(str(B), {})[arch] = 1e3 * time_call(lambda: model(x), dev) / B
    for B in TIME_BATCHES:
        for key in ["P", "A"]:
            f = out["fit"][str(B)][key]
            check(f"batch {B} {key}-HIO 的時間與迭代數成線性(R² ≥ 0.99)", f["r2"] >= 0.99,
                  f"R² {f['r2']:.4f};每次迭代 {f['b'] * 1e3:.3f} µs/sample")
    return out


def n_eq(tim, B, arch, key="P"):
    f = tim["fit"][str(B)][key]
    return max(1, int(round((tim["net"][str(B)][arch] - f["a"]) / f["b"])))


@torch.no_grad()
def evaluate_all(dev, tim):
    from ambiguity_check import align, to_c
    from realign_eval import load, net_out, proto_labels, score, score_all
    from src.data import generalization_suites
    from src.hio import hio, random_init
    from src.hio_sw import hio_ext
    from src.physics import apply_probe, beamstop_mask, forward_measure

    eq = sorted({n_eq(tim, B, a) for B in TIME_BATCHES for a in ["unet", "unroll"]})
    n_list = {"P": sorted(set(ITERS) | set(eq)), "A": ITERS, "S": ITERS}
    res = {"nets": {a: [] for a in NETS}, "hio": [], "eq_n": eq}
    kinds, figs = None, None

    for s in SEEDS:
        # ---- 網路 ----
        preds = {}
        for a, name in NETS.items():
            rd = RUN_ROOT / f"{name}_s{s}"
            cfg, model = load(rd, dev)
            assert cfg.probe == "disk" and float(cfg.probe_r) == 12.0, f"{rd.name} 的探針設定不符"
            n = N_OBJ or cfg.eval_n
            raw = raw_home(cfg, n)
            labels, kinds = proto_labels(n, cfg, raw.to(dev))
            psi = apply_probe(raw, cfg).to(dev)
            bs = beamstop_mask(cfg, device=dev)
            torch.manual_seed(cfg.test_seed + s)
            counts = forward_measure(psi, bs, cfg)
            pred = net_out(model, psi, counts, bs, cfg)
            r, al = score_all(model, psi, bs, cfg, pred)
            m = json.load(open(rd / "metrics.json"))["test"]["frc_gain"]
            # eval.py 的量測雜訊未固定 seed;v3 的輸出經過 50 輪 HIO,對雜訊的實現值敏感(其他網路不敏感),
            # 故 v3 的容許差放寬為 0.05(512 張;QUICK 的 32 張為 0.10)。本檢查只確認載入的模型與訓練時一致;
            # 所有方法之間的比較都用本腳本同一份量測,不受影響(協定 §8.10)。
            tol = SANITY_TOL if a != "unroll" else (0.10 if QUICK else 0.05)
            if abs(r["raw"]["frc_gain"] - m) > tol:
                raise SystemExit(f"{rd.name}:未對齊 {r['raw']['frc_gain']:+.4f} 與 metrics.json "
                                 f"{m:+.4f} 對不上,停下來查")
            _, _, sh = align(to_c(pred), to_c(psi))
            r["shift_frac"] = float((sh.abs().sum(1) > 0).float().mean())
            r["proto"] = {}
            for i, k in enumerate(kinds):
                msk = labels.to(dev) == i
                p = score(model, psi[msk], bs, cfg, al[msk])
                r["proto"][k] = {"frc_gain": p["frc_gain"], "material_mae": p["material_mae"]}
            r["unseen"] = {}
            gen = generalization_suites(cfg, cfg.gen_n)
            for j, u in enumerate(UNSEEN):
                objs = gen[u].to(dev)
                torch.manual_seed(cfg.test_seed + s + 1000 * (j + 1))
                c = forward_measure(objs, bs, cfg)
                ru, _ = score_all(model, objs, bs, cfg, net_out(model, objs, c, bs, cfg))
                r["unseen"][u] = {"aligned": ru["aligned"]}
            res["nets"][a].append(r)
            preds[a] = al.cpu()
            print(f"  [{rd.name}] 完成(未對齊 {r['raw']['frc_gain']:+.4f},metrics.json {m:+.4f})",
                  flush=True)

        # ---- HIO 對照(同一批物體、同一份量測;以 probe_unet 的設定)----
        cfg, model = load(RUN_ROOT / f"probe_unet_s{s}", dev)
        n = N_OBJ or cfg.eval_n
        raw = raw_home(cfg, n)
        psi = apply_probe(raw, cfg).to(dev)
        bs = beamstop_mask(cfg, device=dev)
        torch.manual_seed(cfg.test_seed + s)
        counts = forward_measure(psi, bs, cfg)
        cb = box_cfg(cfg)
        rec = {"P": {}, "A": {}, "S": {}}
        keep = None
        for key in ["P", "A", "S"]:
            c = cb if key == "A" else cfg
            init = random_init(counts, c, seed=cfg.test_seed + s, device=dev)
            for it in n_list[key]:
                if key == "S":
                    out = hio_ext(init, counts, bs, cfg, it, beta=cfg.hio_beta,
                                  support=(psi[:, 0] > 0).float())
                else:
                    out = hio(init, counts, bs, c, n_iter=it, beta=c.hio_beta)
                r, al = score_all(model, psi, bs, cfg, out)
                r["proto"] = {}
                for i, k in enumerate(kinds):
                    msk = labels.to(dev) == i
                    r["proto"][k] = {"frc_gain": score(model, psi[msk], bs, cfg, al[msk])["frc_gain"]}
                rec[key][str(it)] = r
                if key == "P" and it == n_eq(tim, MAIN_BATCH, "unroll"):
                    keep = al.cpu()
            if key == "P":
                # P-HIO 的未見分布(與網路相同的量測種子)
                gen = generalization_suites(cfg, cfg.gen_n)
                rec["P_unseen"] = {}
                for j, u in enumerate(UNSEEN):
                    objs = gen[u].to(dev)
                    torch.manual_seed(cfg.test_seed + s + 1000 * (j + 1))
                    cu = forward_measure(objs, bs, cfg)
                    it = 100
                    ou = hio(random_init(cu, cfg, seed=cfg.test_seed + s, device=dev), cu, bs, cfg,
                             n_iter=it, beta=cfg.hio_beta)
                    rec["P_unseen"][u] = {"aligned": score_all(model, objs, bs, cfg, ou)[0]["aligned"]}
        res["hio"].append(rec)
        if s == SEEDS[0]:
            figs = (psi.cpu(), labels, kinds, preds, keep)
        print(f"  [HIO 對照 s{s}] 完成", flush=True)
    return res, kinds, figs


def repro_4_1(dev):
    """P-HIO 重現 probe_4_1.json(r = 12、seed 0、50 次;4-1 的原始量測設定)。"""
    import probe_4_1 as p41
    from realign_eval import load, score_all
    from src.hio import hio, random_init
    from src.hio_sw import hio_ext
    from src.physics import beamstop_mask, forward_measure
    ref_file = RUN_ROOT / ("probe_4_1_quick.json" if QUICK else "probe_4_1.json")
    ref = json.load(open(ref_file))
    radii = ref["radii"]
    r, ri = 12, radii.index(12) if 12 in radii else None
    if ri is None:
        check("P-HIO 重現 probe_4_1.json", False, "4-1 的結果沒有 r = 12")
        return
    cfg, model = load(RUN_ROOT / "ideal_base_s0", dev)
    n = N_OBJ or cfg.eval_n
    raw = raw_home(cfg, n).to(dev)
    D = p41.disk(r, cfg, dev)
    psi = p41.to_psi(raw, D)
    cp = p41.psi_cfg(cfg, r)
    bs = beamstop_mask(cfg, device=dev)
    torch.manual_seed(cfg.test_seed + 0 + 1000 * (ri + 1))
    counts = forward_measure(psi, bs, cp)
    init = random_init(counts, cp, seed=cfg.test_seed, device=dev)
    a = hio_ext(init, counts, bs, cp, 50, beta=cp.hio_beta, support=D.expand(len(psi), *D.shape))
    # 新管線:同一份量測,以 probe 設定的 support_mask(圓盤)跑 hio()
    cnew = Cfg.from_dict({**cp.to_dict(), "probe": "disk", "probe_r": float(r)})
    b = hio(init, counts, bs, cnew, n_iter=50, beta=cnew.hio_beta)
    same = float((torch.polar(a[:, 0], a[:, 1]) - torch.polar(b[:, 0], b[:, 1])).abs().max())
    check("新管線的 hio()(support = 圓盤)= 4-1 的 hio_ext(support = 圓盤)", same < 1e-5, f"最大差 {same:.1e}")
    got = score_all(model, psi, bs, cp, a)[0]["aligned"]["frc_gain"]
    exp = ref["results"][str(r)][0]["rec"]["P"]["50"]["aligned"]["frc_gain"]
    check("P-HIO 重現 probe_4_1.json(r = 12、seed 0、50 次)", abs(got - exp) <= REPRO_TOL,
          f"{got:+.4f} vs {exp:+.4f}")


# ============================================================================
# 彙整與判讀
# ============================================================================
def ms_(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def zpair(d):
    m, s = ms_(d)
    return m, s, (m / (s / math.sqrt(len(d))) if s > 0 else float("inf") * np.sign(m))


def nv(res, a, *path):
    out = []
    for r in res["nets"][a]:
        x = r
        for k in path:
            x = x[k]
        out.append(x)
    return np.array(out, float)


def hv(res, key, it, *path):
    out = []
    for rec in res["hio"]:
        x = rec[key][str(it)]
        for k in path:
            x = x[k]
        out.append(x)
    return np.array(out, float)


def report(res, kinds, tim, conv):
    print("\n" + "=" * 100)
    print("訓練分布(psi,512 張)對齊後 FRC gain(mean ± std,3 seeds);括號為材料 MAE")
    print("=" * 100)
    for a in NETS:
        f = ms_(nv(res, a, "aligned", "frc_gain"))
        mat = nv(res, a, "aligned", "material_mae").mean()
        print(f"  {NET_LABEL[a]:<22} {f[0]:+.4f} ± {f[1]:.4f}({mat:.3f})   "
              f"選翻轉 {nv(res, a, 'twin_frac').mean():.2f}  有平移 {nv(res, a, 'shift_frac').mean():.2f}")
    print(f"  機率水準(U-Net 錯配對齊):{nv(res, 'unet', 'mismatch', 'frc_gain').mean():+.4f}")
    print(f"\n  {'HIO 迭代':>8}" + "".join(f"{x:>18}" for x in ["P-HIO(探針)", "A-HIO(方框)", "S*ψ(上限)"]))
    for it in ITERS:
        row = f"  {it:>8}"
        for key in ["P", "A", "S"]:
            m = hv(res, key, it, "aligned", "frc_gain").mean()
            mat = hv(res, key, it, "aligned", "material_mae").mean()
            row += f"{m:>+11.4f}({mat:.3f})"
        print(row)

    print("\n  各原型(對齊後 FRC gain;HIO 為 50 次)")
    for k in kinds:
        row = f"  {k:<10}" + "".join(
            f"{NET_LABEL[a][:6]:>8} {nv(res, a, 'proto', k, 'frc_gain').mean():+.3f}" for a in NETS)
        row += f"   P-HIO {hv(res, 'P', 50, 'proto', k, 'frc_gain').mean():+.3f}"
        print(row)

    print("\n  未見分布(相對增益 dB = 對齊後 PSNR − 平庸基準;P-HIO 為 100 次)")
    def unseen_row(label, getter):
        vals = [getter(u) for u in UNSEEN]
        return f"  {label:<22}" + "".join(f"{SHORT[u]} {v:+6.2f}  " for u, v in zip(UNSEEN, vals)) + \
            f"贏 {sum(v > 0 for v in vals)}/4"
    for a in NETS:
        print(unseen_row(NET_LABEL[a], lambda u, a=a: np.mean(
            [r["unseen"][u]["aligned"]["amp_psnr"] - r["unseen"][u]["aligned"]["amp_psnr_trivial"]
             for r in res["nets"][a]])))
    print(unseen_row("P-HIO 100", lambda u: np.mean(
        [rec["P_unseen"][u]["aligned"]["amp_psnr"] - rec["P_unseen"][u]["aligned"]["amp_psnr_trivial"]
         for rec in res["hio"]])))

    print("\n  速度(ms/sample):" + "   ".join(
        f"{NET_LABEL[a]} " + " / ".join(f"b{B} {tim['net'][str(B)][a]:.4f}"
                                        for B in sorted(tim["net"], key=int))
        for a in NETS))

    # ---------------- §4.1 ----------------
    print("\n" + "=" * 100)
    print("§4.1 網路是否用上探針資訊")
    print("=" * 100)
    verdict = {}
    for a in ["unet", "unroll"]:
        F = nv(res, a, "aligned", "frc_gain").mean()
        tier = "達 P-HIO 水準" if F >= 0.60 else ("明顯受益" if F >= 0.35 else "未能用上")
        if TRAIN_FAIL.get(a):
            tier = "訓練失敗,無法判定(§8.5)"
        pg = nv(res, a, "proto", "polygon", "frc_gain").mean() if "polygon" in kinds else float("nan")
        bp = nv(res, a, "proto", "bandpass", "frc_gain").mean() if "bandpass" in kinds else float("nan")
        nonper = pg >= 0.30 and bp >= 0.30
        print(f"  {NET_LABEL[a]:<22} F {F:+.4f} → {tier};polygon {pg:+.3f} / bandpass {bp:+.3f}"
              f" → {'能' if nonper else '未能'}處理非週期結構")
        verdict[a] = {"F": float(F), "tier": tier, "nonperiodic": bool(nonper)}
    Fo = nv(res, "oracle", "aligned", "frc_gain").mean()
    po = nv(res, "oracle", "proto", "polygon", "frc_gain").mean()
    bo = nv(res, "oracle", "proto", "bandpass", "frc_gain").mean()
    print(f"  Oracle-ψ(上限)F {Fo:+.4f};polygon {po:+.3f} / bandpass {bo:+.3f}"
          + ("  → ⚠️ 給了相位仍 < 0.30:轉換問題在探針條件下依然存在" if (po < 0.30 or bo < 0.30) else ""))

    # ---------------- §4.2 ----------------
    print("\n" + "=" * 100)
    print("§4.2 網路是否「有用」:對手 = P-HIO 在時間預算內的最佳值(逐 seed 配對)")
    print("=" * 100)
    for a in ["unet", "unroll"]:
        verdict[a]["useful"] = {}
        for B in TIME_BATCHES:
            ne = n_eq(tim, B, a)
            grid = [it for it in sorted({*ITERS, *res["eq_n"]}) if it <= ne and
                    str(it) in res["hio"][0]["P"]]
            if not grid:
                grid = [min(int(k) for k in res["hio"][0]["P"])]
            best_it = max(grid, key=lambda it: hv(res, "P", it, "aligned", "frc_gain").mean())
            opp = hv(res, "P", best_it, "aligned", "frc_gain")
            net = nv(res, a, "aligned", "frc_gain")
            d = zpair(net - opp)
            ok = d[0] >= 0.05 and d[2] > 2
            verdict[a]["useful"][str(B)] = bool(ok)
            print(f"  {NET_LABEL[a]:<22} batch {B:>3}:{tim['net'][str(B)][a]:.4f} ms = P-HIO {ne} 次;"
                  f"對手取 {best_it} 次 {opp.mean():+.4f};網路 {net.mean():+.4f};"
                  f"差 {d[0]:+.4f} ± {d[1]:.4f}(z {d[2]:+.1f})→ {'✅ 有用' if ok else '未達'}")
        m, r = str(MAIN_BATCH), str([B for B in TIME_BATCHES if B != MAIN_BATCH][0])
        verdict[a]["useful_label"] = ("穩健有用" if verdict[a]["useful"][m] and verdict[a]["useful"][r]
                                      else "有用" if verdict[a]["useful"][m] else "未達")
        print(f"  → {NET_LABEL[a]}:{verdict[a]['useful_label']}")

    # ---------------- §4.4 ----------------
    print("\n" + "=" * 100)
    print("§4.4 結果與下一步")
    print("=" * 100)
    failed = [a for a in ["unet", "unroll"] if TRAIN_FAIL.get(a)]
    if failed:
        print("  ⚠️ 訓練失敗的組(不列入判定):" + "、".join(NET_LABEL[a] for a in failed)
              + " → 依 §8.10:停下來討論")
    cand = [a for a in ["unet", "unroll"] if verdict[a]["F"] >= 0.35 and not TRAIN_FAIL.get(a)]
    if cand:
        best = max(cand, key=lambda a: verdict[a]["F"])
        if len(cand) == 2 and abs(verdict["unet"]["F"] - verdict["unroll"]["F"]) < 0.02:
            best = min(cand, key=lambda a: tim["net"][str(MAIN_BATCH)][a])
        print(f"  至少一種網路 F ≥ 0.35 → 4-2b 的 HIO 部分照做;網路部分照做,採用 {NET_LABEL[best]}")
    else:
        best = None
        print("  兩種網路皆 F < 0.35 → 4-2b 的 HIO 部分照做;網路部分先停,檢討網路設計")
    verdict["next_net"] = best
    print(f"\n  收斂降幅:" + "  ".join(f"{NET_LABEL[a]} {100 * conv[a]:.2f}%" for a in NETS))
    print("判讀準則見 階段四實驗設計協定 v2 §四(結果出來前已寫定)")
    return verdict


def make_figures(res, tim, figs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ 無 matplotlib,略過圖")
        return
    FIG_DIR.mkdir(exist_ok=True)
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    col = {"P": "#2a78d6", "A": "#eb6834", "S": "#1baf7a"}
    name = {"P": "P-HIO (probe support)", "A": "A-HIO (box support)", "S": "S*ψ-HIO (true support)"}
    B = str(MAIN_BATCH)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=grid, lw=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for key in ["P", "A"]:
        t = np.array([tim["hio"][B][key][str(it)] for it in ITERS])
        mu = np.array([hv(res, key, it, "aligned", "frc_gain").mean() for it in ITERS])
        ax.plot(t, mu, "-", color=col[key], lw=2, marker="o", ms=5, label=name[key])
    marks = {"unet": "s", "unroll": "^", "oracle": "D"}
    lab = {"unet": "U-Net", "unroll": "Physics-embedded U-Net (unrolled)", "oracle": "U-Net + phase (ceiling)"}
    for a in NETS:
        v = nv(res, a, "aligned", "frc_gain")
        t = tim["net"][B][a]
        ax.errorbar([t], [v.mean()], yerr=[v.std(ddof=1)], fmt=marks[a], color=ink, ms=8,
                    mfc="white", mew=2, capsize=3)
        ax.annotate(lab[a], (t, v.mean()), xytext=(6, -12), textcoords="offset points",
                    fontsize=8, color=ink)
    ax.set_xscale("log")
    ax.set_xlabel(f"time per sample (ms, batch {MAIN_BATCH}, same GPU)", color=ink)
    ax.set_ylabel("aligned FRC gain on ψ (mean ± std, 3 seeds)", color=ink)
    ax.set_title("4-2a: networks vs HIO under a known flat-top probe (r = 12 px)", color=ink, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left")
    fig.tight_layout()
    p = FIG_DIR / "probe42_vs_time.png"
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")

    psi, labels, kinds, preds, keep = figs
    rows = []
    for i, k in enumerate(kinds):
        rows += [(k, j) for j in (labels == i).nonzero()[:, 0][:2].tolist()]
    cols = [("GT ψ", None), ("U-Net", "unet"), ("Unrolled", "unroll"), ("U-Net+phase", "oracle"),
            ("P-HIO (equal time)", "hio")]
    sl = slice(16, 48)
    for ch, nm in [(0, "amplitude"), (1, "phase")]:
        fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.1 * len(cols), 2.1 * len(rows)))
        for r_, (k, j) in enumerate(rows):
            for c_, (title, key) in enumerate(cols):
                img = psi[j, ch] if key is None else (keep[j, ch] if key == "hio" else preds[key][j, ch])
                if ch == 1:
                    amp = psi[j, 0] if key is None else (keep[j, 0] if key == "hio" else preds[key][j, 0])
                    img = torch.where(amp > 0.02, img, torch.full_like(img, float("nan")))
                a_ = axs[r_, c_]
                a_.imshow(img[sl, sl].numpy(), cmap="gray" if ch == 0 else "viridis",
                          vmin=0, vmax=(1.0 if ch == 0 else float(np.pi / 2)))
                a_.set_xticks([]); a_.set_yticks([])
                if r_ == 0:
                    a_.set_title(title, fontsize=9, color=ink)
                if c_ == 0:
                    a_.set_ylabel(f"{k} #{j}", fontsize=9, color=ink)
        fig.suptitle(f"4-2a reconstructions ({nm}, aligned), seed 0", fontsize=10, color=ink)
        fig.tight_layout()
        p = FIG_DIR / f"probe42_{nm}.png"
        fig.savefig(p, dpi=130, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-only", action="store_true")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    if a.gate:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sys.exit(0 if gate(dev) else 1)
    configs_and_unit_tests()
    if a.configs_only or not ok_all:
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過,先不要送件 / 分析"))
        sys.exit(0 if ok_all else 1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv = convergence()
    repro_4_1(dev)
    tim = timing(dev)
    if not ok_all:
        print("\n❌ 檢查未通過,停下來查")
        sys.exit(1)
    print("\n退化判定(協定 §8.10:訓練後 F < 同 seed 未訓練 F − 0.02 → 訓練失敗)")
    deg = [degrade_check(s, dev) for s in SEEDS]
    TRAIN_FAIL["unroll"] = bool(TRAIN_FAIL.get("unroll")) or any(deg)
    print("\n評分")
    res, kinds, figs = evaluate_all(dev, tim)
    verdict = report(res, kinds, tim, conv)
    json.dump({"results": res, "timing": tim, "verdict": verdict, "iters": ITERS, "quick": QUICK,
               "train_fail": TRAIN_FAIL, "degrade": {str(k): v for k, v in DEGRADE.items()}},
              open(OUT_JSON if not QUICK else RUN_ROOT / "probe42_v3_quick.json", "w"))
    make_figures(res, tim, figs)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))


if __name__ == "__main__":
    main()
