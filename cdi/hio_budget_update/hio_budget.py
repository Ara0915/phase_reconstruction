#!/usr/bin/env python
"""HIO 需要跑幾次?—— 迭代次數、時間與 support 的量測(實驗設計 §二十三,不需訓練)。

四個問題:
  ① 同樣的計算時間內,HIO 能拿到幾分?(網路「有用」的目標線)
  ② HIO 要跑幾次才到 0.35 / 0.60?(展開 v2 需要幾輪)
  ③ 若 support 是準的,HIO 會快多少?(S* vs A)
  ④ 實務上的 shrinkwrap 有沒有用?(B vs A;公平基準用哪一版)

方法(同一批 512 張、同一份量測,3 seeds;與 realign_eval.py 逐點相同):
  A   簡單版 HIO(src/hio.py,不修改),隨機起點
  S*  同 A,support = 真實物體範圍(診斷用上限)
  B   A + shrinkwrap + 最後 10% ER(src/hio_sw.py)
  D   A,起點為 U-Net(ideal_base)輸出(參考)

計時:同一程序、同一 GPU、相同 batch(主要 64,另報 512)量 HIO 與三個網路。

用法(需在計算節點執行;需 realign_eval.py、ambiguity_check.py 在同一資料夾,
      且 /work/elviss0915/runs/realign_eval.json 已存在):
    python hio_budget.py --check    # 單元測試 + 檔案檢查(約 1 分鐘)
    python hio_budget.py            # 完整量測(約 15-30 分鐘)
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
OUT_JSON = RUN_ROOT / "hio_budget.json"
FIG_DIR = Path("figs_hio_budget")
BASE = "ideal_base"
NETS = {"unet": "amb_base", "fft": "arch_fft_amb", "unroll": "arch_unroll_amb"}
NET_LABEL = {"unet": "U-Net", "fft": "內建轉換(單次)", "unroll": "展開 HIO"}
SEEDS = [0, 1, 2]
METHODS = ["A", "S", "B", "D"]
M_LABEL = {"A": "A 簡單 HIO", "S": "S* 真實 support", "B": "B shrinkwrap+ER", "D": "D U-Net 起點"}
ITERS = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
REF_ITERS = [50, 200, 500, 1000, 5000]          # realign_eval.json 有的點
SUPPORT_N = [20, 50, 100, 200]                  # ③ 看的迭代數
SW_N = [50, 100, 200, 500]                      # ④ 看的迭代數
REPORT_N = [20, 50, 200, 1000, 5000]            # 各原型表
TIME_BATCHES = [64, 512]
MAIN_BATCH = 64
TIME_REPS = 5
REPRO_TOL = 0.005
SANITY_TOL = 0.01
CALIB_SEED = 2026                               # 門檻校準集(≠ test_seed)
CALIB_N = 1024

# 本地測程式流程用(不在國網使用):小樣本、少迭代
QUICK = os.environ.get("CDI_QUICK") == "1"
if QUICK:
    ITERS, REF_ITERS, SUPPORT_N, SW_N, REPORT_N = [5, 20, 50], [50], [20, 50], [50], [20, 50]
    TIME_BATCHES, MAIN_BATCH, TIME_REPS, CALIB_N = [8, 32], 8, 3, 128

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


def make_home(cfg, n):
    """與 generalization_suites(cfg, n)["procedural"] 完全相同(不需載入 MNIST)。"""
    from src.procedural import make_procedural
    return make_procedural(n, cfg, seed=cfg.test_seed, kinds=tuple(cfg.proc_kinds),
                           weights=tuple(cfg.proc_weights),
                           target_support=cfg.match_support_to,
                           contrast_gamma=cfg.proc_contrast_gamma)


def run_method(key, cfg, init, counts, bs, n, home):
    """回傳 (輸出 [N,2,H,W], 最終 support 或 None)。"""
    from src.hio import hio
    from src.hio_sw import ER_FRAC, hio_ext, true_support
    if key in ("A", "D"):
        return hio(init, counts, bs, cfg, n_iter=n, beta=cfg.hio_beta), None
    if key == "S":
        return hio_ext(init, counts, bs, cfg, n, beta=cfg.hio_beta,
                       support=true_support(home)), None
    if key == "B":
        return hio_ext(init, counts, bs, cfg, n, beta=cfg.hio_beta, shrinkwrap=True,
                       er_frac=ER_FRAC, return_support=True)
    raise ValueError(key)


# ============================================================================
# 送件前 / 開跑前:檔案與單元測試
# ============================================================================
def check_files():
    print("=" * 70)
    print("檔案")
    print("=" * 70)
    need = [RUN_ROOT / "realign_eval.json"]
    for name in [BASE] + list(NETS.values()):
        for s in SEEDS:
            d = RUN_ROOT / f"{name}_s{s}"
            need += [d / "config_used.json", d / "final.pt", d / "metrics.json"]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing,
          "缺:" + ", ".join(missing[:5]) if missing else "")
    if not missing:
        from realign_eval import load
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bad = []
        for a, name in NETS.items():
            try:
                cfg, _ = load(RUN_ROOT / f"{name}_s0", dev)
                if getattr(cfg, "arch", "unet") != a:
                    bad.append(f"{name}: arch={cfg.arch}")
            except Exception as e:                                   # noqa: BLE001
                bad.append(f"{name}: {type(e).__name__} {e}")
        check("三個網路的 checkpoint 都能載入(src/model.py 支援 unroll)", not bad,
              "; ".join(bad))


def unit_tests():
    from src.hio import er, hio, random_init
    from src.hio_sw import (SIGMA0, SIGMA_MIN, THRESH, gaussian_blur, hio_ext,
                            shrinkwrap_support, true_support)
    from src.physics import beamstop_mask, calibrate_flux, forward_measure, support_mask
    from src.procedural import make_procedural

    print("\n" + "=" * 70)
    print("單元測試(src/hio_sw.py)")
    print("=" * 70)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(poisson=True):
        cfg = Cfg.load(f"configs/{BASE}.yaml")
        cfg.add_poisson = poisson
        obj = make_procedural(32, cfg, seed=7, kinds=tuple(cfg.proc_kinds),
                              weights=tuple(cfg.proc_weights),
                              target_support=cfg.match_support_to,
                              contrast_gamma=cfg.proc_contrast_gamma)
        cfg.ref_energy = calibrate_flux(obj)
        obj = obj.to(dev)
        bs = beamstop_mask(cfg, device=dev)
        return cfg, obj, bs

    def cdiff(a, b):
        return float((torch.polar(a[:, 0], a[:, 1]) - torch.polar(b[:, 0], b[:, 1])).abs().max())

    # (1) 關閉 shrinkwrap 與 ER = hio.py
    cfg, obj, bs = setup()
    counts = forward_measure(obj, bs, cfg)
    init = random_init(counts, cfg, seed=3, device=dev)
    e1 = cdiff(hio_ext(init, counts, bs, cfg, 50, beta=cfg.hio_beta),
               hio(init, counts, bs, cfg, n_iter=50, beta=cfg.hio_beta))
    check("hio_ext(關閉 shrinkwrap 與 ER)= hio.py 的 hio()", e1 < 1e-4, f"最大差 {e1:.1e}")

    # (2) 全程 ER = hio.py 的 er()
    e2 = cdiff(hio_ext(init, counts, bs, cfg, 50, beta=cfg.hio_beta, er_frac=1.0),
               er(init, counts, bs, cfg, n_iter=50))
    check("hio_ext(全程 ER)= hio.py 的 er()", e2 < 1e-4, f"最大差 {e2:.1e}")

    # (3) 真實 support 下,真值為固定點(無雜訊;相位縮到邊界內,見 §21.1 的數值註記)
    cfg, obj, bs = setup(poisson=False)
    obj = obj.clone()
    obj[:, 1] *= 0.95
    counts = forward_measure(obj, bs, cfg)
    e3 = cdiff(hio_ext(obj, counts, bs, cfg, 100, beta=cfg.hio_beta,
                       support=true_support(obj)), obj)
    check("真實 support 下真值為固定點(S* 的物理部分無誤)", e3 < 1e-3, f"最大差 {e3:.1e}")

    # (3b) 參考:B 從真值出發 100 次(含 4 次 support 更新)偏離多少
    outb, supb = hio_ext(obj, counts, bs, cfg, 100, beta=cfg.hio_beta, shrinkwrap=True,
                         er_frac=0.1, return_support=True)
    t = torch.polar(obj[:, 0], obj[:, 1])
    rel = float((torch.polar(outb[:, 0], outb[:, 1]) - t).norm() / t.norm())
    cov = float(((supb > 0) & (obj[:, 0] > 0)).float().sum() / (obj[:, 0] > 0).float().sum())
    print(f"     (參考)B 從真值出發 100 次:相對誤差 {rel:.2e},support 涵蓋真值 {100 * cov:.2f}%、"
          f"平均面積 {float(supb.sum((1, 2)).mean()):.0f} px")

    # (4) FFT 高斯模糊 = 空間卷積(離散高斯核、循環邊界)
    x = obj[:, 0]
    worst = 0.0
    for s in [SIGMA0, SIGMA_MIN]:
        r = int(math.ceil(6 * s))
        ax = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
        k1 = torch.exp(-ax ** 2 / (2 * s ** 2))
        k2 = (k1[:, None] * k1[None, :])
        k2 = k2 / k2.sum()
        xp = torch.nn.functional.pad(x[:, None], (r, r, r, r), mode="circular")
        ref = torch.nn.functional.conv2d(xp, k2[None, None])[:, 0]
        worst = max(worst, float((gaussian_blur(x, s) - ref).abs().max() / x.abs().max()))
    check("FFT 高斯模糊 = 空間卷積", worst < 1e-4, f"最大相對差 {worst:.1e}")

    # (5) 門檻校準規則:由真值算出的 support 須包住物體 ≥ 99.9%(校準集,seed ≠ test_seed)
    cfg = Cfg.load(f"configs/{BASE}.yaml")
    cal = make_procedural(CALIB_N, cfg, seed=CALIB_SEED, kinds=tuple(cfg.proc_kinds),
                          weights=tuple(cfg.proc_weights),
                          target_support=cfg.match_support_to,
                          contrast_gamma=cfg.proc_contrast_gamma).to(dev)
    A = cal[:, 0]
    box = support_mask(cfg, device=dev)
    ok = (A > 0).sum((1, 2)) > 0
    print("     門檻校準(校準集 %d 張;涵蓋率 / 平均面積 px):" % CALIB_N)
    rule_ok = True
    for thr in [0.20, 0.10, 0.04]:
        row = f"       門檻 {thr:.2f}:"
        for s in [3.0, 2.0, 1.5]:
            S = shrinkwrap_support(A, box.expand_as(A), box, s, thr)
            c = ((S > 0) & (A > 0)).float().sum((1, 2)) / (A > 0).float().sum((1, 2)).clamp_min(1)
            row += f"  σ{s}: {100 * float(c[ok].mean()):.2f}% / {float(S.sum((1, 2))[ok].mean()):.0f}"
            if abs(thr - THRESH) < 1e-9:
                rule_ok &= float(c[ok].mean()) >= 0.999
        print(row)
    print(f"       真實物體平均 {float((A > 0).float().sum((1, 2))[ok].mean()):.0f} px;方框 {int(box.sum())} px")
    check(f"使用的門檻 {THRESH} 符合校準規則(各 σ 平均涵蓋 ≥ 99.9%)", rule_ok)
    print(f"\n     裝置:{dev}")


# ============================================================================
# 計時
# ============================================================================
def _time_call(fn, dev, reps):
    """回傳單次呼叫的中位數秒數。單次太短時在內圈重複,使每次量測 ≥ 20 ms。"""
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


@torch.no_grad()
def timing(dev):
    from realign_eval import load
    from src.hio import random_init
    from src.physics import beamstop_mask, build_input, forward_measure

    print("\n" + "=" * 70)
    print("計時(ms/sample;同一 GPU、同一程序;中位數)")
    print("=" * 70)
    cfg, _ = load(RUN_ROOT / f"{BASE}_s0", dev)
    if QUICK:
        cfg.eval_n = 32
    bs = beamstop_mask(cfg, device=dev)
    home = make_home(cfg, max(TIME_BATCHES)).to(dev)
    torch.manual_seed(cfg.test_seed)
    counts_all = forward_measure(home, bs, cfg)
    out = {"hio": {}, "net": {}, "fit": {}}
    for B in TIME_BATCHES:
        counts = counts_all[:B]
        init = random_init(counts, cfg, seed=cfg.test_seed, device=dev)
        for key in ["A", "B"]:
            # 計時雜訊是暫時性的:線性度不足時整組重量一次(最多 2 次),仍不足才判不通過
            for attempt in range(2):
                ms = {}
                for n in ITERS:
                    ms[n] = 1e3 * _time_call(
                        lambda: run_method(key, cfg, init, counts, bs, n, home[:B]), dev,
                        TIME_REPS) / B
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
            x = build_input(home[:B], counts, bs, ncfg)
            out["net"].setdefault(str(B), {})[arch] = 1e3 * _time_call(
                lambda: model(x), dev, TIME_REPS) / B
    for B in TIME_BATCHES:
        for key in ["A", "B"]:
            f = out["fit"][str(B)][key]
            check(f"batch {B} {key} 的時間與迭代數成線性(R² ≥ 0.99)", f["r2"] >= 0.99,
                  f"R² {f['r2']:.4f};每次迭代 {f['b'] * 1e3:.3f} µs/sample,固定開銷 {f['a'] * 1e3:.2f} µs/sample")
    return out


def n_equal_time(tim, B, key, arch):
    f = tim["fit"][str(B)][key]
    return max(1, int(round((tim["net"][str(B)][arch] - f["a"]) / f["b"])))


# ============================================================================
# 評分
# ============================================================================
@torch.no_grad()
def evaluate_all(dev, tim):
    from realign_eval import load, net_out, proto_labels, score, score_all
    from src.hio import random_init
    from src.physics import beamstop_mask, forward_measure

    eq_n = {key: sorted({n_equal_time(tim, B, key, a) for B in TIME_BATCHES for a in NETS})
            for key in ["A", "B"]}
    n_list = {"A": sorted(set(ITERS) | set(eq_n["A"])), "S": ITERS,
              "B": sorted(set(ITERS) | set(eq_n["B"])), "D": [0] + ITERS}
    ref_all = json.load(open(RUN_ROOT / "realign_eval.json"))["proc"]["seeds"]

    res = {"seeds": [], "nets": {a: [] for a in NETS}}
    for si, s in enumerate(SEEDS):
        run_dir = RUN_ROOT / f"{BASE}_s{s}"
        cfg, model = load(run_dir, dev)
        if QUICK:
            cfg.eval_n = 32
        bs = beamstop_mask(cfg, device=dev)
        home = make_home(cfg, cfg.eval_n).to(dev)
        torch.manual_seed(cfg.test_seed + s)          # 與 realign_eval.py 相同 -> 同一份量測
        counts = forward_measure(home, bs, cfg)
        labels, kinds = proto_labels(cfg.eval_n, cfg, home)
        amp_t = home[:, 0]
        has = (amp_t > 0).sum((1, 2)) > 0
        inits = {"random": random_init(counts, cfg, seed=cfg.test_seed + s, device=dev),
                 "network": net_out(model, home, counts, bs, cfg)}
        m0 = json.load(open(run_dir / "metrics.json"))["test"]["frc_gain"]
        r0 = score(model, home, bs, cfg, inits["network"])["frc_gain"]
        if abs(r0 - m0) > SANITY_TOL:
            raise SystemExit(f"{run_dir.name}:網路未對齊 {r0:+.4f} 與 metrics.json {m0:+.4f} 對不上,停下來查")
        ref = ref_all[si]
        if ref["run"] != run_dir.name:
            raise SystemExit(f"realign_eval.json 的第 {si} 筆是 {ref['run']},不是 {run_dir.name}")

        rec = {m: {} for m in METHODS}
        for key in METHODS:
            init = inits["network" if key == "D" else "random"]
            for n in n_list[key]:
                pred, sup = run_method(key, cfg, init, counts, bs, n, home)
                r, al = score_all(model, home, bs, cfg, pred)
                r["proto"] = {}
                for i, k in enumerate(kinds):
                    m = labels.to(dev) == i
                    sc = score(model, home[m], bs, cfg, al[m])
                    r["proto"][k] = {"frc_gain": sc["frc_gain"], "material_mae": sc["material_mae"]}
                if sup is not None:
                    covd = ((al[:, 0] > 0) & (amp_t > 0)).float().sum((1, 2)) / \
                        (amp_t > 0).float().sum((1, 2)).clamp_min(1)
                    r["support_px"] = float(sup.sum((1, 2)).mean())
                    r["support_cover"] = float(covd[has].mean())
                rec[key][str(n)] = r
            # 重現 realign_eval.json(A = 隨機起點、D = 網路起點)
            if key in ("A", "D"):
                src = "network" if key == "D" else "random"
                for n in REF_ITERS:
                    a = rec[key][str(n)]["aligned"]["frc_gain"]
                    b = ref["home"][src][str(n)]["aligned"]["frc_gain"]
                    if abs(a - b) > REPRO_TOL:
                        raise SystemExit(f"{run_dir.name} {key} {n} 次:對齊後 {a:+.4f} 與 "
                                         f"realign_eval.json {b:+.4f} 對不上,停下來查")
        res["seeds"].append({"run": run_dir.name, "rec": rec})
        print(f"  [{run_dir.name}] 完成(網路未對齊 {r0:+.4f},metrics.json {m0:+.4f};"
              f"A、D 重現 realign_eval.json ✅)", flush=True)

    for arch, name in NETS.items():
        for s in SEEDS:
            run_dir = RUN_ROOT / f"{name}_s{s}"
            cfg, model = load(run_dir, dev)
            if QUICK:
                cfg.eval_n = 32
            bs = beamstop_mask(cfg, device=dev)
            home = make_home(cfg, cfg.eval_n).to(dev)
            torch.manual_seed(cfg.test_seed + s)
            counts = forward_measure(home, bs, cfg)
            pred = net_out(model, home, counts, bs, cfg)
            r, _ = score_all(model, home, bs, cfg, pred)
            m = json.load(open(run_dir / "metrics.json"))["test"]["frc_gain"]
            if abs(r["raw"]["frc_gain"] - m) > SANITY_TOL:
                raise SystemExit(f"{run_dir.name}:未對齊 {r['raw']['frc_gain']:+.4f} 與 "
                                 f"metrics.json {m:+.4f} 對不上,停下來查")
            res["nets"][arch].append(r)
        print(f"  [{name}] 完成", flush=True)
    res["eq_n"] = eq_n
    return res, kinds


# ============================================================================
# 彙整與判讀
# ============================================================================
def vals(res, key, n, field="frc_gain", kind="aligned"):
    return np.array([sd["rec"][key][str(n)][kind][field] for sd in res["seeds"]], float)


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def zpair(d):
    m, s = ms(d)
    return m, s, (m / (s / math.sqrt(len(d))) if s > 0 else float("inf") * np.sign(m))


def first_reach(res, key, q):
    for n in ITERS:
        if vals(res, key, n).mean() >= q:
            return n
    return None


def report(res, kinds, tim):
    S = res["seeds"]
    print("\n" + "=" * 100)
    print("對齊後 FRC gain(mean ± std,3 seeds);括號內為材料 MAE(平庸約 0.30)")
    print("=" * 100)
    print(f"  {'迭代':>6}" + "".join(f"{M_LABEL[m]:>24}" for m in METHODS))
    for n in [0] + ITERS:
        row = f"  {n:>6}"
        for m in METHODS:
            if str(n) in S[0]["rec"][m]:
                a = ms(vals(res, m, n))
                mat = vals(res, m, n, "material_mae").mean()
                row += f"{a[0]:>+11.4f}±{a[1]:.4f}({mat:.3f})"
            else:
                row += f"{'—':>24}"
        print(row)
    mis = vals(res, "A", ITERS[-1], kind="mismatch").mean()
    print(f"  機率水準(A {ITERS[-1]} 次的錯配對齊):{mis:+.4f}")

    print("\n  各原型(對齊後 FRC gain)")
    print(f"  {'迭代':>6}{'原型':>10}" + "".join(f"{M_LABEL[m]:>20}" for m in METHODS))
    for n in REPORT_N:
        for k in kinds:
            row = f"  {n:>6}{k:>10}"
            for m in METHODS:
                v = [sd["rec"][m][str(n)]["proto"][k]["frc_gain"] for sd in S]
                row += f"{np.mean(v):>+20.4f}"
            print(row)

    print("\n  B 的最終 support(平均面積 px / 涵蓋真實物體的比例;方框 1024 px,物體約 140 px)")
    print("  " + "   ".join(f"{n}: {np.mean([sd['rec']['B'][str(n)]['support_px'] for sd in S]):.0f} / "
                           f"{100 * np.mean([sd['rec']['B'][str(n)]['support_cover'] for sd in S]):.1f}%"
                           for n in ITERS))

    # ---------------- 計時 ----------------
    print("\n" + "=" * 100)
    print("計時(ms/sample)")
    print("=" * 100)
    for B in TIME_BATCHES:
        h = tim["hio"][str(B)]
        print(f"  batch {B}:" + "".join(f"  {n}次 A {h['A'][str(n)]:.4f} / B {h['B'][str(n)]:.4f}"
                                        for n in ITERS[::2]))
        print("    網路:" + "   ".join(f"{NET_LABEL[a]} {tim['net'][str(B)][a]:.4f}" for a in NETS))

    # ---------------- ① ----------------
    print("\n" + "=" * 100)
    print("① 同樣時間的目標線(對齊後 FRC gain;網路與 HIO 在同一批物體、同一份量測上逐 seed 配對)")
    print("=" * 100)
    for B in TIME_BATCHES:
        tag = "(主要)" if B == MAIN_BATCH else "(參考)"
        print(f"  batch {B} {tag}")
        for arch in NETS:
            net = np.array([r["aligned"]["frc_gain"] for r in res["nets"][arch]])
            nA, nB = n_equal_time(tim, B, "A", arch), n_equal_time(tim, B, "B", arch)
            fA, fB = vals(res, "A", nA), vals(res, "B", nB)
            best = fA if fA.mean() >= fB.mean() else fB
            d = zpair(net - best)
            print(f"    {NET_LABEL[arch]:<10} {tim['net'][str(B)][arch]:.4f} ms = A {nA} 次 "
                  f"({fA.mean():+.4f}) / B {nB} 次 ({fB.mean():+.4f});網路 {net.mean():+.4f};"
                  f" 網路 − 較強者 {d[0]:+.4f} ± {d[1]:.4f}(z {d[2]:+.1f})")
    print("  之後「網路有用」的判準:網路 − 較強者 ≥ +0.05 且 z > 2(§23.4 ①)")

    # ---------------- ② ----------------
    print("\n" + "=" * 100)
    print("② 需要幾次(格點中第一個達標的迭代數)")
    print("=" * 100)
    for m in METHODS:
        print(f"  {M_LABEL[m]:<18} 0.35:{first_reach(res, m, 0.35)}   0.60:{first_reach(res, m, 0.60)}")
    cand = [x for x in [first_reach(res, "A", 0.35), first_reach(res, "B", 0.35)] if x is not None]
    n35 = min(cand) if cand else None
    if n35 is not None and n35 <= 50:
        v2 = f"N_0.35 = {n35} ≤ 50 → 幾十次就夠:展開 v2 增加輪數(各輪共用修正網路)"
    elif n35 is not None and n35 <= 500:
        v2 = f"N_0.35 = {n35}(100–500)→ 需要網路加速數倍以上:展開 v2 須加入針對主要障礙的學習元件(依 ③)"
    else:
        v2 = f"N_0.35 = {n35}(> 500 或未達)→ 增加輪數不可行:先做階段四"
    print(f"  判定:{v2}")

    # ---------------- ③ ----------------
    print("\n" + "=" * 100)
    print("③ support 是不是主要障礙(S* − A,逐 seed 配對)")
    print("=" * 100)
    big, small = False, True
    for n in SUPPORT_N:
        d = zpair(vals(res, "S", n) - vals(res, "A", n))
        print(f"  {n:>5} 次:{d[0]:+.4f} ± {d[1]:.4f}(z {d[2]:+.1f})")
        big |= d[0] >= 0.10 and d[2] > 2
        small &= abs(d[0]) <= 0.02
    v3 = ("support 資訊是主要障礙" if big else
          "support 不是障礙,難度在相位本身" if small else "部分成立")
    print(f"  判定:{v3}")

    # ---------------- ④ ----------------
    print("\n" + "=" * 100)
    print("④ 實務 shrinkwrap 有沒有用(B − A,逐 seed 配對)")
    print("=" * 100)
    eff = False
    for n in SW_N:
        d = zpair(vals(res, "B", n) - vals(res, "A", n))
        print(f"  {n:>5} 次:{d[0]:+.4f} ± {d[1]:.4f}(z {d[2]:+.1f})")
        eff |= d[0] >= 0.05 and d[2] > 2
    print(f"  判定:{'shrinkwrap 有效' if eff else 'shrinkwrap 未達有效門檻'}"
          "(公平基準一律取 A、B 中較強者)")

    print("\n  參考:D − A(網路起點的加速)")
    print("  " + "   ".join(f"{n}: {(vals(res, 'D', n) - vals(res, 'A', n)).mean():+.4f}"
                           for n in ITERS))
    print("\n判讀準則見 實驗設計_1b6 §二十三(結果出來前已寫定)")
    return {"n35": n35, "v2": v2, "support": v3, "shrinkwrap_effective": bool(eff)}


def make_figure(res, tim):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ 無 matplotlib,略過圖")
        return
    FIG_DIR.mkdir(exist_ok=True)
    col = {"A": "#2a78d6", "B": "#eb6834", "S": "#1baf7a", "D": "#eda100"}
    sty = {"A": "-", "B": "-", "S": "--", "D": ":"}
    name = {"A": "A: simple HIO", "B": "B: shrinkwrap + ER", "S": "S*: true support (diagnostic)",
            "D": "D: HIO from U-Net"}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax in axs:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=grid, lw=0.8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(ink2)
        ax.tick_params(colors=ink2)
    ax = axs[0]
    for m in METHODS:
        ns = ITERS
        mu = np.array([vals(res, m, n).mean() for n in ns])
        sd = np.array([vals(res, m, n).std(ddof=1) for n in ns])
        ax.plot(ns, mu, sty[m], color=col[m], lw=2, marker="o", ms=5, label=name[m])
        ax.fill_between(ns, mu - sd, mu + sd, color=col[m], alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("iterations", color=ink)
    ax.set_ylabel("aligned FRC gain (mean ± std, 3 seeds)", color=ink)
    ax.set_title("Quality vs iterations", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink)

    ax = axs[1]
    B = str(MAIN_BATCH)
    for m in ["A", "B"]:
        t = np.array([tim["hio"][B][m][str(n)] for n in ITERS])
        mu = np.array([vals(res, m, n).mean() for n in ITERS])
        ax.plot(t, mu, "-", color=col[m], lw=2, marker="o", ms=5, label=name[m])
    marks = {"unet": "s", "fft": "D", "unroll": "^"}
    lab = {"unet": "U-Net", "fft": "FFT single-pass", "unroll": "Unrolled HIO (5)"}
    for arch in NETS:
        v = np.array([r["aligned"]["frc_gain"] for r in res["nets"][arch]])
        t = tim["net"][B][arch]
        ax.errorbar([t], [v.mean()], yerr=[v.std(ddof=1)], fmt=marks[arch], color=ink, ms=8,
                    mfc="white", mew=2, capsize=3)
        ax.annotate(lab[arch], (t, v.mean()), xytext=(6, -12), textcoords="offset points",
                    fontsize=8, color=ink)
    ax.set_xscale("log")
    ax.set_xlabel(f"time per sample (ms, batch {MAIN_BATCH}, same GPU)", color=ink)
    ax.set_ylabel("aligned FRC gain", color=ink)
    ax.set_title("Quality vs compute time (networks = open markers)", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left")
    fig.suptitle("HIO iteration budget, current conditions (bs=3, 1e3 photons/px), 512 objects",
                 color=ink, fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "hio_budget.png"
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑檔案檢查與單元測試")
    a = ap.parse_args()
    check_files()
    unit_tests()
    if a.check or not ok_all:
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過,先不要跑完整量測"))
        sys.exit(0 if ok_all else 1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tim = timing(dev)
    if not ok_all:
        print("\n❌ 計時檢查未通過,停下來查")
        sys.exit(1)
    print("\n評分")
    res, kinds = evaluate_all(dev, tim)
    verdict = report(res, kinds, tim)
    json.dump({"timing": tim, "results": res, "verdict": verdict,
               "iters": ITERS, "quick": QUICK},
              open(OUT_JSON if not QUICK else RUN_ROOT / "hio_budget_quick.json", "w"))
    make_figure(res, tim)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))


if __name__ == "__main__":
    main()
