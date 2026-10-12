#!/usr/bin/env python
"""期末考(期末考實驗設計協定 v0.4;cdi/claude_final/docs/期末考實驗設計協定.md)。

所有最終版本的網路(P6B4e8 主線、P6B4e8-L 晶格、P6B4 / P6B4r / P6B4e8-C 對照)在從未使用過的資料上考三部分:
  A 理想條件(程序生成物體 97,000,000)、B 貼近真實條件(同 A 的物體,階段七的其餘 12 個條件)、
  C 晶格(TMD 晶格 98,000,000;階段八的 5 份考卷)。
對手:85 個迭代法設定的包絡,在考試資料上整套重跑;次數以「時間上限」延長(T_H = AP-C 2000 次的時間,
  n_s = round(T_H / τ_s) 夾在 [500, 2000];τ_s 取自階段八的計時檔;協定 §三)。只在 T_c 以內下結論。
判讀 f1–f5(協定 §六,結果出來前寫定);可重現性 = 考試 / 驗證(驗證值在彙整 job 以同一套程式重算)。
不修改任何既有檔案:import scan_8c(→ scan_8 → scan_7d → … → scan_5),重用量測、網路流程、指標與 CRLB。

用法(需在計算節點執行):
    python scan_final.py --check          # 內建檢查(協定 §七;只用驗證資料;寫出 final_ns.json、final_models.json)
    python scan_final.py --smoke          # 迷你全流程(驗證資料代替考試資料、縮小版;dev 節點);通過才可正式執行
    python scan_final.py --run 0..16      # run_final_env.sh 的 array:0–11 = A / B 的 12 個條件、12–16 = C 的 5 份考卷(迭代法包絡)
    python scan_final.py --run final      # 彙整(run_final_agg.sh):計時、網路、CRLB、關鍵設定、判讀、圖
期末考資料(97,000,000–98,999,999)只有 --run 才會產生;--check / --smoke 一律拒絕。
"""
import argparse
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
import scan_8c as c8                                                 # noqa: E402  (8c;不修改。也載入 scan_8 與階段七 / 六 / 五)

s8 = c8.s8
d7, c7, b7, a7, a6, s5, s5b, s5c = s8.d7, s8.c7, s8.b7, s8.a7, s8.a6, s8.s5, s8.s5b, s8.s5c
h, g = s8.h, s8.g
RUN_ROOT, QUICK, PROBE, SEEDS, SUF = s8.RUN_ROOT, s8.QUICK, s8.PROBE, s8.SEEDS, s8.SUF
S8_MD5 = "4821d2607f6e8aa381f0e6b5c5e72513"   # 階段八正式結果所用的 scan_8.py
C8_MD5 = "3d0915fffaff52a5b4b2b749a9e102dd"   # 8c 正式結果所用的 scan_8c.py
D7_MD5 = s8.D7_MD5
import probe_4_1 as p41                                              # noqa: E402  (物體產生器;只包一層保護)

# ============================================================================
# 陣容、資料、判讀的常數(協定 §一、§二、§五、§六)
# ============================================================================
NETS = list(s8.NETS)                         # P6B4、P6B4r、P6B4e8、P6B4e8C、P6B4e8L
BASE, NEW_L, NEW_C = s8.BASE, s8.NEW_L, s8.NEW_C
MAIN_AB, MAIN_C = BASE, NEW_L                # 主判的網路:A / B = P6B4e8;C = P6B4e8-L(P6B4e8 並列)
tg = s8.tg
AB_ENV = list(c7.CONDS)                      # 12 個有包絡的條件(ideal = A;其餘 = B)
AB_EVAL = list(c7.EVAL_CONDS)                # 13 個條件(+ comboNP:只評估網路)
COND_A = "ideal"
EXAMS = list(s8.EXAMS)                       # 5 份考卷
MAIN_EXAM = s8.MAIN_EXAM                     # L-paper
TASKS = [("ab", c) for c in AB_ENV] + [("c", e) for e in EXAMS]
QS, Q_MAIN, SPEED_MIN = list(b7.QS), b7.Q_MAIN, b7.SPEED_MIN       # A / B:Q = 0.05 / 0.02 / 0.01(0.02 為主);穩健 = 下界 ≥ 1.25
F2_COND, F2_Q = "combo", 0.05                # f2:combo、Q = 0.05 的確認性檢定
K_MAIN = 1.5                                 # f3:Q = 1.5 × CRLB(主判)
K_CURVE = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]   # 加速倍數 vs k(描述;含 k = 2)
QC_FIXED = [0.05, 0.02]                      # L-paper 的固定門檻(nerr_c;描述,與階段八 h1 對照)
REP_LO, REP_HI = 0.8, 1.25                   # f4:可重現性的等價範圍
HELDOUT, OUTRANGE = list(b7.HELDOUT), dict(b7.OUTRANGE)
FIG_AB = ["ideal", F2_COND]                  # 視覺化:A(ideal)、B(combo)、C(L-paper)
N_RAND_FIG = 6
FIG_SEED = 2026
BOOT_N, BOOT_SEED = s8.BOOT_N, s8.BOOT_SEED
REPRO_TOL = 1e-4                             # 重現:相對差(GPU 上不要求逐位元;協定 §七 1)

# ============================================================================
# seed(協定 §二;皆未使用)
# ============================================================================
EXAM_AB = dict(field=97_000_000, noise=97_100_000, pos=97_200_000, df=97_300_000)
VAL_AB = dict(field=a6.FIELD_SEED, noise=a6.NOISE_BASE, pos=a7.POS_BASE, df=a7.DF_BASE)      # 驗證場(80M、82M、83M、84M)
EXAM_C = dict(obj=98_000_000, noise=98_100_000, pos=98_200_000, df=98_300_000, newdef=98_400_000, init=98_500_000)
VAL_C = dict(s8.SEED_VAL)                    # 晶格驗證場(87M)
EXAM_LO, EXAM_HI = 97_000_000, 98_999_999    # 期末考保留的區間:只有 --run 才允許
RESERVED = {97_000_000: {"scan_5i.py", "scan_8.py"},          # 保留的宣告與保護的測試值:只豁免這些檔案裡的這個值
            98_000_000: {"scan_8.py"}, 98_999_999: {"scan_8.py"},  # scan_5i.FINAL_FIELD_SEED;scan_8.FINAL_LO / HI、派生範圍的清單
            98_123_456: {"scan_8.py"}, 98_500_000: {"scan_8.py"}}  # scan_8.checks 的「保護會拒絕」測試(不產生資料)

# ============================================================================
# 迭代法的延長(協定 §三;v0.4)
# ============================================================================
EXT_MULT = (1.4, 2.0, 2.8, 4.0)              # 延長的停止點 = 基本最多次數 × {1.4, 2, 2.8, 4}(500 → 700、1000、1400、2000)
NS_X = 4                                     # 次數上限 = 基本最多次數 × 4(500 → 2000);T_H = AP-C 跑上限次數的時間(b512)

DEPEND = {                                   # 依賴的既有程式(= install_final.sh;smoke 通過時連同輸入檔一起凍結)
    "scan_8c.py": "3d0915fffaff52a5b4b2b749a9e102dd", "scan_8.py": "4821d2607f6e8aa381f0e6b5c5e72513",
    "scan_7d.py": "b6e14b742fbf45615f87046f5618bb4c", "scan_7c.py": "34575ffb68a7b5a17ce451ca89cdf0f7",
    "scan_7b.py": "e030d9e42da69a1848e28854a5bffe8a", "scan_7a.py": "44c350438076dc75ce875206d5d1c97f",
    "scan_6a.py": "7ea8e8ee2fcdc78eece76744dd61b172", "scan_5h.py": "f9f209f7c15c1c96f4c627b982958550",
    "scan_5g.py": "18238c4ebda1da7bae86b051f35b9587", "scan_5e.py": "cecd24eb06ec1d22abc7f6f56adc6e8a",
    "scan_5d.py": "f2f9d1acba473926831537c8c4f5fb49", "scan_5c.py": "161df412d2771c611ec75490a1ea019f",
    "scan_5b.py": "f3ce06c0b08c1775dad81e5635c7ed0e", "scan_5.py": "68a6924fc6f4eb589ea15372cd18f761",
    "probe_4_2b.py": "5c98526596de2d854efaab0d0c5da190", "probe_4_1.py": "b5ebcef9ce546b480c27ca38ba22443f",
    "src/hio_sw.py": "3b2a58b9f487279cc78b2556dd18949c", "src/hio.py": "a5eac4662f326b269ad1f122760f607b",
    "src/physics.py": "f7d6902e4cb27345f643c7b3e11bd507", "src/metrics.py": "796c05881ba0a09b96a8aa497bf92ae4",
    "src/procedural.py": "adca05ecfa98ef63c75bfe952dc76194", "src/config.py": "d9d63f6b0c8c46033c1a21b189c3a724",
    "src/model.py": "5187ec38cf0c49fae3267e90026aca7a", "src/data.py": "e6f66a5e669bf6a5381649ee52a3c7de",
    "realign_eval.py": "f826432269d473b39f3e71bae2643e5f", "ambiguity_check.py": "9e9aafc246e43788ba34369621bece4e"}

# ============================================================================
# 檔案
# ============================================================================
SMOKE_DIR = RUN_ROOT / f"final_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
S8_JSON = s8.final_json()                    # 階段八的結果(計時檔 = 延長次數的依據;驗證的數字)
C8_JSON = c8.OUT_JSON                        # 8c 的結果(驗證資料的 CRLB)
FIG_DIR = Path("figs_final")

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and c8.all_ok()


md5, dump_json, absdiff = a7.md5, a7.dump_json, a7.absdiff


def me_md5():
    return md5(Path(__file__).resolve())


def reldiff(a, b):
    """|a − b| / max(|b|, 1e-12);非有限值 → inf。"""
    d = abs(float(a) - float(b)) / max(abs(float(b)), 1e-12)
    return d if np.isfinite(d) else float("inf")


def env_json(task, root=None):
    return (root or RUN_ROOT) / f"final_env_{task}{SUF}.json"


def final_json(root=None):
    return (root or RUN_ROOT) / f"final{SUF}.json"


def ns_json(root=None):
    return (root or RUN_ROOT) / f"final_ns{SUF}.json"


def models_json(root=None):
    return (root or RUN_ROOT) / f"final_models{SUF}.json"


def cache_dir(root=None):
    return (root or RUN_ROOT) / f"final_cache{SUF}"


def task_name(k):
    return TASKS[k][1]


# ============================================================================
# 期末考資料的保護(協定 §二):階段八的 s8.guard 一律拒絕 98M → 換成本程式的保護;97M 包住物體產生器
# ============================================================================
_EXAM = {"ok": False}


def in_exam(v):
    return EXAM_LO <= int(v) <= EXAM_HI


def guard(seeds):
    """97,000,000–98,999,999:只有 --run(且最終結果檔不存在、smoke 已通過)才允許。"""
    bad = [int(x) for x in seeds if in_exam(x)]
    if bad and not _EXAM["ok"]:
        raise ValueError(f"seed {bad[:3]} 落在期末考保留的區間 {EXAM_LO:,}–{EXAM_HI:,}:只有 --run 才允許 → 拒絕")


s8.guard = guard                             # scan_8 的晶格產生器、量測在呼叫時查這個名稱
_make_home0 = p41.make_home


def _make_home_guarded(cfg, n, seed=None):
    if seed is not None:
        guard([seed])
    return _make_home0(cfg, n, seed=seed)


p41.make_home = _make_home_guarded           # s5.make_fields 在函式內 import probe_4_1 → 也經過這層


def enable_exam():
    _EXAM["ok"] = True
    print("  ⚠️ 期末考資料已開放(--run)")


# ============================================================================
# 資料:A / B(同 scan_7a.setup_cond,seed 基底由參數給)、C(同 scan_8.setup_exam)
# ============================================================================
def seeds_ab(s, base):
    """= scan_6a.seeds_of + scan_7a.draw_errors 的 seed(基底換成 base)。"""
    return {"field": base["field"], "noise": base["noise"] + 1000 * s, "init": base["noise"] + 1000 * s + 1,
            "pos": base["pos"] + 1000 * s, "df": base["df"] + 1000 * s}


def seeds_c(s, base):
    """= scan_8.lattice_seeds(基底換成 base):物體與新缺陷 3 seeds 共用,其餘 + 1000 × seed。"""
    return {k: (v if k in ("obj", "newdef") else v + 1000 * s) for k, v in base.items()}


def draw_errors_ab(sd, n, dev):
    """= scan_7a.draw_errors(同抽法;seed 由 sd 給)。"""
    gz = torch.Generator().manual_seed(sd["pos"])
    z = torch.randn(n, a6.NPOS * a6.NPOS, 2, generator=gz).clamp(-a7.Z_CLIP, a7.Z_CLIP)
    gd = torch.Generator().manual_seed(sd["df"])
    sign = torch.where(torch.rand(n, generator=gd) < 0.5, -1.0, 1.0)
    return z.to(dev), sign.to(dev)


def setup_ab(s, dev, cond, n, base):
    """A / B 的一個條件(介面同 scan_7a.setup_cond;重建用的 pr / cp / geo 都是名目的)。量測 = scan_7a.measure_cond
    (已由 scan_7b 擴充部分同調、scan_7c 加入 comboNP)。"""
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    sd = seeds_ab(s, base)
    guard(list(sd.values()))
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[PROBE]
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS["3x3s8"][2])
    bs = beamstop_mask(cfg, device=dev)
    geo = a6.Geom(cfg, pr, dev)
    fields = s5.make_fields(cfg, n, seed=sd["field"], device=dev)
    O = torch.polar(fields[:, 0], fields[:, 1])
    z, sign = draw_errors_ab(sd, n, dev)
    ops = a7.Ops(pr, geo.W, dev)
    spec = a7.SPEC[cond]
    counts = a7.measure_cond(fields, geo, cp, bs, sd["noise"], spec, z, sign, ops)
    return dict(cfg=cfg, pr=pr, cp=cp, bs=bs, geo=geo, O=O, counts=counts, init=sd["init"], n=n, fields=fields,
                z=z, sign=sign, ops=ops, spec=spec, cond=cond, noise=sd["noise"], sd=sd)


def setup_c(s, dev, exam, n, base, eta=s8.ETA):
    """C 的一份考卷(= scan_8.setup_exam 的內容,seed 由 base 給;介面相同 → 迭代法、網路流程、lat_metrics 直接使用)。"""
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    sd = seeds_c(s, base)
    guard(list(sd.values()))
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[PROBE]
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS["3x3s8"][2])
    bs = beamstop_mask(cfg, device=dev)
    geo = a6.Geom(cfg, pr, dev)
    ops = a7.Ops(pr, geo.W, dev)
    dx = s8.pixel_size(pr, geo.W)
    spec = dict(s8.ESPEC[exam])
    obj_seeds = [sd["obj"] + i for i in range(n)]
    mats = [s8.MATS[i % 2] for i in range(n)]                                         # 分層:偶數 WS₂、奇數 MoS₂
    nds = [sd["newdef"] + i for i in range(n)] if spec["newdef"] else None
    Ucpu = geo.U.cpu()
    fields, meta = s8.make_lattice(obj_seeds, geo.F, dev, materials=mats, newdef_seeds=nds, U=Ucpu, eta=eta, info=True, dx=dx)
    O = torch.polar(fields[:, 0], fields[:, 1])
    z, sign = s8.draw_errors_L(sd, n, dev)
    mt = torch.tensor(meta["mats"])
    dv = s8.dose_vec(s, spec, meta["mats"])
    counts = s8.measure_L(fields, geo, cp, bs, sd["noise"], spec, z, sign, ops, dv)
    vac = s8.Vac(meta["cols"], geo.F, Ucpu, fields[:, 1].cpu(), norm=s8.NEWDEF_NORM if spec["newdef"] else "all")
    m = geo.U.to(O.real.dtype)
    cs = (O * m).sum((1, 2)) / m.sum()
    Ec = (((O - cs[:, None, None]).abs() ** 2) * m).sum((1, 2))
    E = ((O.abs() ** 2) * m).sum((1, 2))
    return dict(cfg=cfg, pr=pr, cp=cp, bs=bs, geo=geo, O=O, counts=counts, init=sd["init"], n=n, fields=fields, z=z, sign=sign,
                ops=ops, spec=spec, cond=exam, mats=mt, dose=dv, vac=vac, meta=meta, Ec=Ec, E=E, dx=dx, noise=sd["noise"],
                obj_seeds=obj_seeds, sd=sd)


# ============================================================================
# 延長的次數與停止點(協定 §三)
# ============================================================================
def meth_of(conf):
    return "K-HIO" if conf == "K-HIO" else conf.split("|")[0]


def tau_of(meth, T):
    """τ_s:每次的時間(ms / 整張,b512)。平常版 = 每次;修正版 = 修正進行中(|on);退火版 = 退火結束後的每次時間
    (= 同一演算法非退火版的 |on;協定 v0.4,使用者 10/12:以實驗公平性為準);K-HIO = 49 張的每次。"""
    if meth == "K-HIO":
        return T["K-HIO49"]
    if ":" not in meth:
        return T[meth]
    base, _, mode = meth.partition(":")
    mode = mode[:-1] if mode.endswith("A") else mode
    return T[f"{base}:{mode}|on"]


def ns_table(T, base_max):
    """n_s = round(T_H / τ_s) 夾在 [base_max, NS_X × base_max];T_H = AP-C 跑 NS_X × base_max 次的時間(b512)。"""
    hi = NS_X * base_max
    TH = T["AP-C"] * hi
    ns, raw = {}, {}
    for meth in list(a7.METHODS7) + ["K-HIO"]:
        r = TH / tau_of(meth, T)
        raw[meth] = r
        ns[meth] = int(min(hi, max(base_max, math.floor(r + 0.5))))
    return {"T_H": TH, "ns": ns, "raw": raw, "base_max": base_max, "hi": hi}


def stops_table(ns, base_iters):
    """每個設定的停止點:基本的停止點(0–500)+ 延長的停止點(< n_s)+ n_s;K-HIO 沒有 0。"""
    lo = max(base_iters)
    ext = [int(round(lo * m)) for m in EXT_MULT]
    S = {}
    for conf in a7.configs7():
        n = ns[meth_of(conf)]
        st = sorted(set(int(i) for i in base_iters) | {e for e in ext if lo < e < n} | {int(n)})
        S[conf] = st if conf == "K-HIO" else [0] + st
    return S


def stops_val(iters):
    """驗證的包絡檔(階段七 / 八):每個設定同一組停止點。"""
    return {c: list(a7.stops7(c, iters)) for c in a7.configs7()}


def t_c(S, tB):
    """T_c = 各設定最後一個停止點的時間的最小值(包絡在 T_c 以內完整;協定 §三)。"""
    best = (float("inf"), None)
    for c, st in S.items():
        t = a7.cfg_time7(c, st[-1], tB)
        if t < best[0]:
            best = (t, c)
    return best


def plan_of(smoke, T512):
    """正式:基本停止點 = scan_7a.ITERS(0–500)、上限 2000;smoke:≤ 20、上限 80(同一套公式,縮小)。"""
    base_iters = list(a7.SMOKE_ITERS) if smoke else list(a7.ITERS)
    nt = ns_table(T512, max(base_iters))
    return {"base_iters": base_iters, **nt, "S": stops_table(nt["ns"], base_iters)}


# ============================================================================
# 包絡、達到門檻的時間、加速倍數(停止點依設定;只在 T_c 以內下結論)
# ============================================================================
def env_vals(env, conf, it, key):
    return np.array(env[conf][str(it)][key], float)


def env_points(env, S, tB, key):
    """所有 (時間, 3 seeds 平均, 設定, 次數),依時間排序。"""
    pts = [(a7.cfg_time7(c, it, tB), float(env_vals(env, c, it, key).mean()), c, it) for c in S for it in S[c]]
    return sorted(pts, key=lambda x: x[0])


def envelope(env, T, tB, S, key):
    """時間 ≤ T 的所有(設定, 停止點)中,3 seeds 平均最小者:(設定, 次數, 逐 seed 值, 時間);沒有 → None。"""
    best = None
    for c in S:
        for it in S[c]:
            t = a7.cfg_time7(c, it, tB)
            if t > T * (1 + 1e-9):
                continue
            v = env_vals(env, c, it, key)
            if best is None or v.mean() < best[2].mean():
                best = (c, it, v, t)
    return best


def t_first(env, key, Q, tB, S, Tc):
    """包絡最早達到 Q(3 seeds 平均 ≤ Q)的 (時間, 設定, 次數),只看 T_c 以內;與下界時間:
    各設定第一個 ≤ Q 的停止點的「前一個停止點」的時間取最小(第一個停止點就 ≤ Q:起點 = 該點的時間;K-HIO = 0 次的時間;
    同 scan_7b.t_match_lo)。下界也包含第一個達到點在 T_c 之後的設定:真正越過 Q 的時刻可能在前一個停止點(< T_c)之後就發生。"""
    best = (float("inf"), None, None)
    lo = float("inf")
    for c in S:
        prev = None
        for it in S[c]:
            if float(env_vals(env, c, it, key).mean()) <= Q:
                t = a7.cfg_time7(c, it, tB)
                lo = min(lo, a7.cfg_time7(c, prev if prev is not None else (it if it == 0 else 0), tB))
                if t <= Tc * (1 + 1e-9) and t < best[0]:
                    best = (t, c, it)
                break
            prev = it
    return best, lo


def speed(a_mean, env, tB, t_net, S, Tc, key, qs, higher=False):
    """加速倍數(同 scan_7b.speed_row,加上 T_c):迭代法在 T_c 內達到 → [x_lo, x] = [下界時間, 達到時間] / 網路時間;
    T_c 內未達 → 下界 x_lo = min(T_c, 下界時間) / 網路時間(上界不存在,x = inf)。
    higher=True:指標越高越好(SV-F1;包絡檔的鍵存 1 − 值,門檻同樣換算)。"""
    out = {}
    for Q in qs:
        Qe = 1.0 - Q if higher else Q
        (t_it, conf, it), lo = t_first(env, key, Qe, tB, S, Tc)
        reach = bool(a_mean >= Q) if higher else bool(a_mean <= Q)
        in_tc = bool(np.isfinite(t_it))
        t_lo = lo if in_tc else min(Tc, lo)
        out[str(Q)] = {"Q": float(Q), "reach": reach, "iter_in_tc": in_tc, "t_iter": t_it, "t_iter_lo": t_lo, "conf": conf, "it": it,
                       "Tc": Tc, "x": (t_it / t_net if in_tc else float("inf")) if reach else None,
                       "x_lo": (t_lo / t_net) if reach else None}
    return out


def speed_label(s64, s512):
    """協定 §五:兩種 batch 的下界都 ≥ 1.25 → 穩健加速(迭代法 T_c 內未達時,下界 = T_c / 網路時間)。"""
    if not s64["reach"]:
        return "網路未達"
    tail = "(迭代法 T_c 內未達 → 下界)" if (not s64["iter_in_tc"] and not s512["iter_in_tc"]) else ""
    if all(x["x_lo"] >= SPEED_MIN for x in (s64, s512)):
        return "穩健加速" + tail
    if any(x["x_lo"] >= SPEED_MIN for x in (s64, s512)):
        return "達到、只在 " + "、".join(f"b{B}" for B, x in (("64", s64), ("512", s512)) if x["x_lo"] >= SPEED_MIN) + " 穩健加速" + tail
    if any(x["x"] >= SPEED_MIN for x in (s64, s512)):
        return "達到、加速不確定(只有上界 ≥ 1.25)" + tail
    return "達到、無明顯加速" + tail


def fmt_x(r):
    if r is None or not r["reach"]:
        return "網路未達"
    if not r["iter_in_tc"]:
        return f"迭代法 T_c({r['Tc']:.1f} ms)內未達 → ≥ × {r['x_lo']:.2f}"
    return f"× {r['x_lo']:.2f}–{r['x']:.2f}"


def f3_text(r):
    """f3 的寫法(協定 §六):網路達到、迭代法 T_c 內達到 → 加速區間;T_c 內未達 → 網路至少快 T_c / 網路時間 倍(下界);網路未達 → 未達。"""
    if not r["reach"]:
        return "網路未達"
    if r["iter_in_tc"]:
        return f"加速 × {r['x_lo']:.2f}–{r['x']:.2f}"
    return f"網路至少快 × {r['x_lo']:.2f}(下界;迭代法 T_c = {r['Tc']:.1f} ms 內未達)"


# ============================================================================
# 迭代法包絡:A / B 的一個條件(同 scan_7a.measure_condition 的設定與指標;停止點依設定延長)
# ============================================================================
def _resume(partial, meta, log):
    if partial is None or not partial.exists():
        return None
    old = json.load(open(partial))
    if old.get("meta") != meta:
        raise SystemExit(f"❌ {partial} 的設定或程式版本不同:不續跑。先告訴 Claude")
    log(f"  續跑:暫存檔已有 seed {old['done']}({partial.name})")
    return old


@torch.no_grad()
def measure_ab(dev, cond, n, base, S, log=print, partial=None, meta=None):
    kit = sorted(set(S["K-HIO"]) | set(a7.KHIO_N.values()))
    env = {c: {str(it): {q: [] for q in a7.MKEYS} for it in S[c]} for c in S}
    for c in S:
        if ":" in c:
            env[c]["diag"] = {"pos_rms": [], "prb_err": []}
    stats, done = [], []
    old = _resume(partial, meta, log)
    if old is not None:
        env, stats, done = old["env"], old["stats"], old["done"]
    t0 = time.time()
    for s in SEEDS:
        if s in done:
            continue
        d = setup_ab(s, dev, cond, n, base)
        geo, O, U, spec = d["geo"], d["O"], d["geo"].U, d["spec"]
        Pt = a7.true_probes(d)
        want = spec["pos"] * d["z"]
        want = want - want.mean(1, keepdim=True)
        st = {"pos_std_px": float((spec["pos"] * d["z"]).std()) if spec["pos"] > 0 else 0.0,
              "pos_max_px": float((spec["pos"] * d["z"]).abs().max()) if spec["pos"] > 0 else 0.0,
              "df_plus_frac": float((d["sign"] > 0).float().mean()), "n": d["n"],
              "pos_rms0": float(want.norm(dim=-1).pow(2).mean(1).sqrt().mean()),
              "prb_err0": float(a7.probe_err(geo.pr["P"], Pt).mean()),
              "mean_counts": float(torch.stack([c.sum((-2, -1)) for c in d["counts"]]).mean())}
        if not b7.is_ideal(spec):
            ideal = a6.measure49(d["fields"], geo, d["cp"], d["bs"], d["noise"])
            mi = float(torch.stack([c.sum((-2, -1)) for c in ideal]).mean())
            st["raw_count_ratio"] = st["mean_counts"] * spec["dose"] / mi
            st["counts_differ"] = bool(max(float((a - b).abs().max()) for a, b in zip(d["counts"], ideal)) > 0)
            del ideal
        stats.append(st)
        win = a7.khio_windows7(d, dev, kit)
        kw = {}
        for k, wv in win.items():
            Of, cover, _ = a6.merge_windows(wv, geo)
            kw[k] = (Of, cover)
            if k in S["K-HIO"]:
                m = a7.metrics7(torch.where(cover, Of, torch.zeros_like(Of)), O, U)
                for q in a7.MKEYS:
                    env["K-HIO"][str(k)][q].append(m[q])
        del win
        for kind in s5b.INITS:
            O0 = a6.make_start(kind, d, kw, dev)
            m0 = a7.metrics7(O0, O, U)
            for meth in a7.METHODS7:
                conf = f"{meth}|{kind}"
                its = [i for i in S[conf] if i > 0]
                outs, stt = a7.run_method7(meth, d, its, O0, state=True)
                for it in its:
                    m = a7.metrics7(outs[it], O, U)
                    for q in a7.MKEYS:
                        env[conf][str(it)][q].append(m[q])
                for q in a7.MKEYS:
                    env[conf]["0"][q].append(m0[q])
                if stt is not None:                                                  # 修正後(最後一個停止點 n_s)的位置 / 探針誤差
                    env[conf]["diag"]["pos_rms"].append(float((stt["r"] - want).norm(dim=-1).pow(2).mean(1).sqrt().mean()))
                    env[conf]["diag"]["prb_err"].append(float(a7.probe_err(stt["P"] if stt["P"] is not None else geo.pr["P"],
                                                                           Pt).mean()))
                del outs
        del kw, d
        done.append(s)
        if partial is not None:
            dump_json({"meta": meta, "env": env, "stats": stats, "done": done}, partial)
        log(f"  [{cond} seed {s}] 完成(經過 {time.time() - t0:.0f} 秒)")
    return env, stats


# ============================================================================
# 迭代法包絡:C 的一份考卷(同 scan_8.measure_exam;每個停止點也做空缺判讀)
# ============================================================================
@torch.no_grad()
def measure_c(dev, exam, n, base, S, log=print, partial=None, meta=None):
    kit = sorted(set(S["K-HIO"]) | set(a7.KHIO_N.values()))
    env = {c: {str(it): {} for it in S[c]} for c in S}
    stats, done = [], []
    old = _resume(partial, meta, log)
    if old is not None:
        env, stats, done = old["env"], old["stats"], old["done"]

    def put(conf, it, r, mats):
        for k, v in s8.summarize(r, mats).items():
            env[conf][str(it)].setdefault(k, []).append(v)

    t0 = time.time()
    for s in SEEDS:
        if s in done:
            continue
        d = setup_c(s, dev, exam, n, base)
        mats = d["mats"]
        stats.append(s8.exam_stats(d))
        win = a7.khio_windows7(d, dev, kit)
        kw = {}
        for k, wv in win.items():
            Of, cover, _ = a6.merge_windows(wv, d["geo"])
            kw[k] = (Of, cover)
            if k in S["K-HIO"]:
                put("K-HIO", k, s8.lat_metrics(torch.where(cover, Of, torch.zeros_like(Of)), d), mats)
        del win
        for kind in s5b.INITS:
            O0 = a6.make_start(kind, d, kw, dev)
            r0 = s8.lat_metrics(O0, d)
            for meth in a7.METHODS7:
                conf = f"{meth}|{kind}"
                its = [i for i in S[conf] if i > 0]
                outs = a7.run_method7(meth, d, its, O0)
                put(conf, 0, r0, mats)
                for it in its:
                    put(conf, it, s8.lat_metrics(outs[it], d), mats)
                del outs
        del kw, d
        done.append(s)
        if partial is not None:
            dump_json({"meta": meta, "env": env, "stats": stats, "done": done}, partial)
        log(f"  [{exam} seed {s}] 完成(經過 {time.time() - t0:.0f} 秒)")
    return env, stats


def run_task(dev, k, n, base, plan, root, smoke):
    """一個 array job:A / B 的一個條件或 C 的一份考卷 → final_env_{名稱}.json(每個 seed 完成即存暫存檔,中斷可續跑)。"""
    part, name = TASKS[k]
    me = me_md5()
    t0 = time.time()
    S = plan["S"]
    path = env_json(name, root)
    partial = Path(str(path) + ".partial")
    meta = {"task": k, "name": name, "part": part, "n": n, "S": S, "md5": me, "smoke": smoke, "base": base}
    print(f"\n  {'A / B 條件' if part == 'ab' else 'C 考卷'} {name}({n} 個場;延長的次數:"
          + "、".join(f"{m} {v}" for m, v in plan["ns"].items()) + ")", flush=True)
    lg = (lambda x: print(x, flush=True))
    if part == "ab":
        env, stats = measure_ab(dev, name, n, base, S, lg, partial, meta)
        sp = a7.SPEC[name]
        if sp["pos"] > 0:
            r = np.mean([x["pos_std_px"] for x in stats]) / sp["pos"]
            check(f"[{name}] 位置誤差的標準差 = 設定 × (1 ± 5%)", abs(r - 1) <= 0.05, f"實際 / 設定 = {r:.3f}")
        if sp["df"] > 0:
            f = np.mean([x["df_plus_frac"] for x in stats])
            check(f"[{name}] 離焦符號約各半(± 3 個標準誤)", abs(f - 0.5) <= 3 * 0.5 / math.sqrt(n), f"+ 的比例 {f:.3f}")
        if sp.get("coh", 0) > 0:
            rr = np.mean([x["raw_count_ratio"] for x in stats])
            check(f"[{name}] 平均總計數 / 理想 在 1 ± {b7.COH_COUNT_TOL}(劑量不變)", abs(rr - 1) <= b7.COH_COUNT_TOL, f"{rr:.4f}")
        elif sp["dose"] != 1 and sp["pos"] == 0 and sp["df"] == 0:                  # 只有劑量誤差的條件(探針誤差會改變 beamstop 擋掉的比例)
            rr = np.mean([x["raw_count_ratio"] for x in stats]) / sp["dose"]
            check(f"[{name}] 原始計數 / 理想 = 劑量比 × (1 ± 2%)", abs(rr - 1) <= 0.02, f"{rr:.4f}")
        if not b7.is_ideal(sp):
            check(f"[{name}] 量測與理想不同", all(x.get("counts_differ", False) for x in stats))
    else:
        env, stats = measure_c(dev, name, n, base, S, lg, partial, meta)
        sp = s8.ESPEC[name]
        if sp["pos"] > 0:
            r = np.mean([x["pos_std_px"] for x in stats]) / sp["pos"]
            check(f"[{name}] 位置誤差的標準差 = 設定 × (1 ± 5%)", abs(r - 1) <= 0.05, f"實際 / 設定 = {r:.3f}")
        if sp["df"] > 0:
            f = np.mean([x["df_plus_frac"] for x in stats])
            check(f"[{name}] 離焦符號約各半(± 3 個標準誤)", abs(f - 0.5) <= 3 * 0.5 / math.sqrt(n), f"+ 的比例 {f:.3f}")
        rr = np.mean([x["raw_vs_expected"] for x in stats])
        check(f"[{name}] 平均偵測計數 / 期望 = 1 ± 2%(劑量)", abs(rr - 1) <= 0.02, f"{rr:.4f}")
        pm = max(x["phase_max"] for x in stats)
        check(f"[{name}] 所有場的最大相位 < π/2", pm < math.pi / 2, f"最大 {pm:.3f} rad")
    out = {"meta": {**meta, "seeds": list(SEEDS), "script_md5": me, "quick": QUICK, "gpu": a7.gpu_name(dev),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0},
           "env": env, "stats": stats, "checks_ok": all_ok()}
    dump_json(out, path)
    partial.unlink(missing_ok=True)
    print(f"  存檔:{path}(經過 {time.time() - t0:.0f} 秒)")
    return out


# ============================================================================
# 彙整:網路評估(A / B:13 個條件;C:5 份考卷)—— 考試資料與驗證資料用同一套程式
# ============================================================================
def roots_real():
    return {"r": None, "e": None, "m": None}


@torch.no_grad()
def eval_nets_ab(dev, base, n, roots, keep=None, log=print):
    """13 個條件 × 3 seeds × 5 個網路(G9):逐場的 nerr_sr / nerr_ph、SSIM(振幅、相位)、有效場(真值在 U 內不是空的)。
    keep(dict):seed 0、FIG_AB 的條件另存真值與主判網路的輸出(畫圖用)。"""
    R = {c: {nm: {"sr": [], "ph": [], "ssim_amp": [], "ssim_ph": [], "nerr_sr": [], "nerr_ph": [], "n_fail": []} for nm in NETS}
         for c in AB_EVAL}
    valid = {}
    t0 = time.time()
    for c in AB_EVAL:
        for s in SEEDS:
            d = setup_ab(s, dev, c, n, base)
            O, U, phmax = d["O"], d["geo"].U, d["cfg"].phase_max
            mU = U.to(O.real.dtype)
            vm = (((O.abs() ** 2) * mU).sum((1, 2)) > 1e-12).cpu().numpy()
            valid.setdefault(c, vm)
            for nm in NETS:
                kind, model, norm = s8.get_net(nm, s, d["cfg"], d["geo"].pr, dev, roots)
                est = s8.run_net(kind, model, norm, d)
                m = a7.metrics7(est, O, U, keep_ps=True)
                sa, sp = b7.ssim_eval(est, O, U, phmax)
                r = R[c][nm]
                r["sr"].append(np.array(m["ps_sr"], float))
                r["ph"].append(np.array(m["ps_ph"], float))
                r["ssim_amp"].append(sa.cpu().numpy())
                r["ssim_ph"].append(sp.cpu().numpy())
                r["nerr_sr"].append(m["nerr_sr"])
                r["nerr_ph"].append(m["nerr_ph"])
                r["n_fail"].append(m["n_fail"])
                if keep is not None and s == SEEDS[0] and c in FIG_AB and nm == MAIN_AB:
                    keep.setdefault(c, {"img": {}})["O"] = O.cpu()
                    keep[c]["img"][tg(nm)] = est.cpu()
                    keep[c]["U"] = d["geo"].U.cpu()
                del model, est
            del d
        log(f"  [網路] {c} 完成(經過 {time.time() - t0:.0f} 秒)")
    return R, valid


def ab_field_mean(R, c, nm, key, valid):
    """每個有效場的 3 seeds 平均(物體在 seed 之間共用 → 場是獨立的單位)。"""
    X = np.stack(R[c][nm][key])                                                       # [S, n]
    return X[:, valid[c]].mean(0)


@torch.no_grad()
def eval_nets_c(dev, base, n, roots, keep=None, log=print):
    """5 份考卷 × 3 seeds × 5 個網路(同 scan_8.eval_exams;資料由 setup_c 產生)。"""
    R = {e: {nm: {k: [] for k in s8.PER_KEYS + ("conf", "conf_cat", "frc")} for nm in NETS} for e in EXAMS}
    info = {}
    t0 = time.time()
    for e in EXAMS:
        for s in SEEDS:
            d = setup_c(s, dev, e, n, base)
            if s == SEEDS[0]:
                info[e] = {"mats": d["mats"].numpy(), "theta": np.array(d["meta"]["theta"]), "stats": s8.exam_stats(d),
                           "m_true": d["vac"].m_true.numpy()}
                if keep is not None and e == MAIN_EXAM:
                    keep[e] = {"O": d["O"].cpu(), "vac": d["vac"], "img": {}, "pred": {}, "U": d["geo"].U.cpu()}
            for nm in NETS:
                kind, model, norm = s8.get_net(nm, s, d["cfg"], d["geo"].pr, dev, roots)
                est = s8.run_net(kind, model, norm, d)
                r = s8.lat_metrics(est, d, extra=True)
                for k in s8.PER_KEYS:
                    R[e][nm][k].append(r[k].numpy())
                R[e][nm]["conf"].append(r["conf"].numpy())
                R[e][nm]["conf_cat"].append(r["conf_cat"].numpy())
                R[e][nm]["frc"].append({k: [x.numpy().tolist() for x in v] for k, v in r["frc"].items()})
                if keep is not None and s == SEEDS[0] and e == MAIN_EXAM and nm in (MAIN_C, BASE):
                    keep[e]["img"][tg(nm)] = est.cpu()
                    ph = torch.where(r["fin"][:, None, None].to(est.device), torch.angle(s8.align_full(est, d["O"], d["geo"].U)[5]),
                                     torch.full_like(est.real, float("nan")))
                    keep[e]["pred"][tg(nm)] = d["vac"].read(ph.cpu(), r["fin"])[0]
                del model, est
            del d
        log(f"  [網路] {e} 完成(經過 {time.time() - t0:.0f} 秒)")
    return R, info


# ============================================================================
# 彙整:考試資料的 CRLB(協定 §五 C;同 8c 的 part_A,不做 (a2);全部 n 個物體)
# ============================================================================
def crlb_part(dev, base, n, log=print):
    """3 seeds 共用同一組物體與探針,只差劑量 1 的換算 c → 在 seed 0 算全部 n 個物體,其他 seed 以 T₁ ∝ 1/c 換算(並直接算第 1 個場核對)。"""
    out = {"T1": {}, "c": {}, "scale": {}}
    t0 = time.time()
    s0 = SEEDS[0]
    for grp, exam in (("ideal", "L-ideal"), ("newdef", "L-newdef")):
        d = setup_c(s0, dev, exam, n, base)
        ph = c8.Phys.of(d)
        out["c"][str(s0)] = ph.c
        Ec = d["Ec"].double().cpu()
        rows = []
        for i in range(n):
            r = c8.crlb_field(ph, d["O"][i])
            if not r["ok"]:                                                          # 失敗 → 停止(不回傳 → 不會存進暫存)
                raise SystemExit(f"❌ [CRLB] {exam} 場 {i}:Cholesky 失敗(info {r['info']}):停止,把輸出貼給 Claude")
            rows.append({"i": i, "mat": int(d["mats"][i]), "T": r["T"], "T5": r["T5"], "trC": r["trC"], "Ec": float(Ec[i]),
                         "null": r["null"], "n_par": r["n_par"], "n_U": r["n_U"]})
            if (i + 1) % 64 == 0:
                log(f"  [CRLB] {exam}:{i + 1} / {n} 個場(經過 {time.time() - t0:.0f} 秒)")
        out["T1"][grp] = rows
        sc = []
        for s in SEEDS[1:]:
            ds = setup_c(s, dev, exam, 1, base)
            phs = c8.Phys.of(ds)
            out["c"][str(s)] = phs.c
            dO = float((ds["O"][0] - d["O"][0]).abs().max() / d["O"][0].abs().max())
            dP = float((phs.P - ph.P).abs().max() / ph.P.abs().max())
            dB = float((phs.bs - ph.bs).abs().max())
            Ts = c8.crlb_field(phs, ds["O"][0], extra=False)["T"]
            ratio = Ts / (rows[0]["T"] * ph.c / phs.c)
            sc.append({"seed": s, "same_obj_probe": bool(dO < 1e-6 and dP < 1e-6 and dB == 0.0), "ratio": ratio, "dO": dO, "dP": dP, "dB": dB})
            del ds, phs
        out["scale"][grp] = sc
        if not all(x["same_obj_probe"] and abs(x["ratio"] - 1) < 1e-6 for x in sc):
            raise SystemExit(f"❌ [CRLB] {exam}:其他 seed 的物體 / 探針不同,或 T₁ 的換算不符({sc}):停止,把輸出貼給 Claude")
        del d, ph
        log(f"  [CRLB] {exam}:{len(rows)} 個物體完成(經過 {time.time() - t0:.0f} 秒)")
    return out


def validate_crlb(A, n):
    """每次使用(含讀回暫存)都核對:兩組各 n 個場、索引 0..n−1、數值有限且 > 0、其他 seed 的換算與物體相同。"""
    bad = []
    for grp in ("ideal", "newdef"):
        rows = A["T1"].get(grp, [])
        if [r["i"] for r in rows] != list(range(n)):
            bad.append(f"{grp}:場數 / 索引不符({len(rows)} / {n})")
        if not all(np.isfinite(r["T"]) and r["T"] > 0 and np.isfinite(r["Ec"]) and r["Ec"] > 0 for r in rows):
            bad.append(f"{grp}:T 或 Ec 非有限或 ≤ 0")
        sc = A["scale"].get(grp, [])
        if len(sc) != len(SEEDS) - 1 or not all(x["same_obj_probe"] and abs(x["ratio"] - 1) < 1e-6 for x in sc):
            bad.append(f"{grp}:其他 seed 的換算不符")
    if set(A["c"]) != {str(s) for s in SEEDS}:
        bad.append("缺 seed 的劑量換算 c")
    nul = max((r["null"] for g_ in ("ideal", "newdef") for r in A["T1"].get(g_, [])), default=float("nan"))
    check(f"[CRLB] 兩組各 {n} 個物體的 Cholesky 成功、數值有效;其他 seed 的物體與探針 = seed 0、T₁ 依 c 換算(< 1e-6)", not bad,
          ";".join(bad) + f"(零空間 ‖I_F ĝ‖/‖I_F‖ 最大 {nul:.1e})")
    return not bad


def crlb_table(A, dose):
    """各考卷的 CRLB(nerr_c 單位;去掉整體相位;∝ 1 / 劑量;同 8c 的 report):每個物體在 3 個 seed 的平均,再對物體平均。"""
    crl = {}
    for grp, exs in (("ideal", ("L-ideal", "L-paper", "L-combo", "L-coh")), ("newdef", ("L-newdef",))):
        rows = A["T1"][grp]
        T1 = np.array([r["T"] for r in rows])
        Ec = np.array([r["Ec"] for r in rows])
        mat = np.array([r["mat"] for r in rows])
        c0 = A["c"][str(SEEDS[0])]
        for e in exs:
            spec = s8.ESPEC[e]
            per_obj = np.zeros(len(rows))
            for sd_ in SEEDS:
                Dd = dose["seeds"][str(sd_)]
                dd = np.ones(len(rows)) if spec["dose"] == "one" else np.where(mat == 0, Dd["f_W"], Dd["f_Mo"])
                per_obj += T1 * (c0 / A["c"][str(sd_)]) / (dd * Ec) / len(SEEDS)
            rng = np.random.default_rng(BOOT_SEED)
            bi = rng.integers(0, len(per_obj), (BOOT_N, len(per_obj)))
            row = {"crlb_c": float(per_obj.mean()), "ci": [float(np.percentile(per_obj[bi].mean(1), 2.5)),
                                                          float(np.percentile(per_obj[bi].mean(1), 97.5))], "n_obj": int(len(rows))}
            for tagm, mm in (("_W", 0), ("_Mo", 1)):
                row["crlb_c" + tagm] = float(per_obj[mat == mm].mean()) if (mat == mm).any() else float("nan")
            crl[e] = row
    return crl


# ============================================================================
# 彙整:迭代法的關鍵設定(重算,3 seeds;同階段七 iter_ssim):SSIM、核對包絡檔、重建圖
# ============================================================================
@torch.no_grad()
def iter_key_ab(dev, E, V, n, base, want_img, log=print):
    out, worst, imgs = {}, 0.0, {}
    t0 = time.time()
    for c in AB_ENV:
        env = E[c]
        want = {}
        for B in ("64", "512"):
            row = V["ab"][c]["nets"][MAIN_AB][B]
            if row["opp"] is not None:
                want.setdefault((row["opp"][0], int(row["opp"][1])), []).append(f"T_net b{B}")
            for q, r in list(row["speed"].items()) + [("own", row["own"])]:
                if r["conf"] is not None:
                    want.setdefault((r["conf"], int(r["it"])), []).append(f"Q {q} b{B}")
        res = {k: {"sa": [], "sp": [], "sr": []} for k in want}
        for s in SEEDS:
            d = setup_ab(s, dev, c, n, base)
            O, U = d["O"], d["geo"].U
            for (conf, it), tags in want.items():
                est = b7.iter_est(d, conf, it, dev)
                sa, sp = b7.ssim_mean(est, O, U, d["cfg"].phase_max)
                v = float(a7.metrics7(est, O, U)["nerr_sr"])
                res[(conf, it)]["sa"].append(sa)
                res[(conf, it)]["sp"].append(sp)
                res[(conf, it)]["sr"].append(v)
                worst = max(worst, reldiff(v, env_vals(env, conf, it, "nerr_sr")[SEEDS.index(s)]))
                if s == SEEDS[0] and c in want_img and any(t.startswith("T_net") for t in tags):
                    imgs.setdefault(c, {})["iter " + "/".join(t for t in tags if t.startswith("T_net")) + f" {s5b.fmt_conf(conf)} x{it}"] = est.cpu()
                del est
            del d
        out[c] = [{"conf": conf, "it": it, "tags": tags, "ssim_amp": res[(conf, it)]["sa"], "ssim_ph": res[(conf, it)]["sp"],
                   "nerr_sr": res[(conf, it)]["sr"]} for (conf, it), tags in want.items()]
        log(f"  [迭代法的關鍵設定] {c}:{len(want)} 個(經過 {time.time() - t0:.0f} 秒)")
    return out, worst, imgs


@torch.no_grad()
def iter_key_c(dev, E, V, fc, n, base, want_img, log=print):
    out, worst, imgs = {}, 0.0, {}
    t0 = time.time()
    for e in EXAMS:
        env = E[e]
        want = {}
        for B in ("64", "512"):
            row = V["c"][e]["nets"][MAIN_C][B]
            if row["opp"] is not None:
                want.setdefault((row["opp"][0], int(row["opp"][1])), []).append(f"T_net b{B}")
            recs = [(f"k{k}", row["speed_k"][str(k)]) for k in K_CURVE] + [(f"Q {q}", row["speed_fixed"][str(q)]) for q in QC_FIXED]
            for tag, r in recs + [("own", row["own"])]:                              # 所有報告到的門檻的達到設定(去重)
                if r["conf"] is not None:
                    want.setdefault((r["conf"], int(r["it"])), []).append(f"{tag} b{B}")
        res = {k: {"sa": [], "sp": [], "c": []} for k in want}
        for s in SEEDS:
            d = setup_c(s, dev, e, n, base)
            for (conf, it), tags in want.items():
                est = s8.iter_est_L(d, conf, it, dev)
                r = s8.lat_metrics(est, d, extra=True)
                cv = r["c"].numpy()
                v = float(np.where(np.isfinite(cv), cv, fc[e]).mean())
                res[(conf, it)]["sa"].append(float(r["ssim_amp"].mean()))
                res[(conf, it)]["sp"].append(float(r["ssim_ph"].mean()))
                res[(conf, it)]["c"].append(v)
                worst = max(worst, reldiff(v, env_vals(env, conf, it, "nerr_c")[SEEDS.index(s)]))
                if s == SEEDS[0] and e in want_img and any(t.startswith("T_net") for t in tags):
                    imgs.setdefault(e, {})["iter " + "/".join(t for t in tags if t.startswith("T_net")) + f" {s5b.fmt_conf(conf)} x{it}"] = est.cpu()
                del est
            del d
        out[e] = [{"conf": conf, "it": it, "tags": tags, "ssim_amp": res[(conf, it)]["sa"], "ssim_ph": res[(conf, it)]["sp"],
                   "nerr_c": res[(conf, it)]["c"]} for (conf, it), tags in want.items()]
        log(f"  [迭代法的關鍵設定] {e}:{len(want)} 個(經過 {time.time() - t0:.0f} 秒)")
    return out, worst, imgs


# ============================================================================
# 彙整:各部分的數字(網路 vs 包絡、加速倍數)
# ============================================================================
def rows_ab(R, valid, E, tim, S, Tc):
    """A / B:每個條件 × 網路的 nerr_sr、SSIM、同時間包絡的比值與 z、加速倍數(Q = 0.05 / 0.02 / 0.01、網路自己的品質)。"""
    out = {}
    for c in AB_EVAL:
        out[c] = {"nets": {}}
        env = E.get(c)
        for nm in NETS:
            r = R[c][nm]
            a = np.array(r["nerr_sr"], float)
            vm = valid[c]
            row = {"nerr_sr": a.tolist(), "nerr_ph": list(map(float, r["nerr_ph"])), "n_fail": int(np.sum(r["n_fail"])),
                   "ssim": [float(np.mean([x[vm].mean() for x in r["ssim_amp"]])), float(np.mean([x[vm].mean() for x in r["ssim_ph"]]))]}
            if env is not None:
                for B in ("64", "512"):
                    tn = tim["net"][nm][B]
                    ev = envelope(env, tn, tim[B], S, "nerr_sr")
                    rr = s5b.ratio(a, ev[2]) if ev is not None else ("-", float("nan"), float("nan"))
                    sp = speed(a.mean(), env, tim[B], tn, S, Tc[B][0], "nerr_sr", QS)
                    own = speed(a.mean(), env, tim[B], tn, S, Tc[B][0], "nerr_sr", [float(a.mean())])[str(float(a.mean()))]
                    row[B] = {"t": tn, "opp": None if ev is None else [ev[0], int(ev[1]), list(map(float, ev[2])), ev[3]],
                              "ratio": list(rr), "speed": sp, "own": own}
            else:                                                                     # comboNP:沒有包絡,只記網路的時間
                for B in ("64", "512"):
                    row[B] = {"t": tim["net"][nm][B]}
            out[c]["nets"][nm] = row
        if env is not None:
            bo = envelope(env, Tc["512"][0], tim["512"], S, "nerr_sr")
            out[c]["iter_best_tc512"] = [bo[0], int(bo[1]), float(bo[2].mean())]
            out[c]["iter_fail"] = iter_fail(env, "ab")
    return out


def iter_fail(env, part):
    """迭代法的發散率(描述):有發散的(設定, 停止點)個數與最高的比例。"""
    fl = []
    for c, cr in env.items():
        for it, rec in cr.items():
            if it == "diag":
                continue
            if part == "ab":
                nf = float(np.sum(rec["n_fail"]))
                nn = None
            else:
                nf, nn = float(np.sum(rec["c_fail"])), float(np.sum(rec["c_n"]))
            fl.append((c, it, nf, nn))
    bad = [x for x in fl if x[2] > 0]
    if part == "ab":
        worst = max(fl, key=lambda x: x[2])
        return {"n_settings_with_fail": len(bad), "worst": [worst[0], worst[1], worst[2]]}
    worst = max(fl, key=lambda x: x[2] / max(x[3], 1))
    return {"n_settings_with_fail": len(bad), "worst": [worst[0], worst[1], worst[2] / max(worst[3], 1)]}


def rows_c(R, info, E, tim, S, Tc, fc, crl):
    """C:每份考卷 × 網路的 nerr_c(WS₂ / MoS₂)、SV 判讀、同時間包絡、加速倍數(k × CRLB 的 k 曲線、網路自己的品質、固定門檻)。"""
    out = {}
    rng = np.random.default_rng(BOOT_SEED)
    for e in EXAMS:
        env = E[e]
        mats = info[e]["mats"]
        out[e] = {"fail_c": fc[e], "nets": {}, "crlb": crl[e], "refl": s8.reflections(info[e]["stats"]["dx"])}
        for nm in NETS:
            A = s8.net_c(R, e, nm, fc[e])
            row = {"nerr_c": A.mean(1).tolist(), "nerr_c_W": s8.net_c(R, e, nm, fc[e], mats == 0).mean(1).tolist(),
                   "nerr_c_Mo": s8.net_c(R, e, nm, fc[e], mats == 1).mean(1).tolist(),
                   "nerr_sr": [float(np.where(f, x, s8.FAIL_SR).mean()) for x, f in zip(R[e][nm]["sr"], R[e][nm]["fin"])],
                   "ssim": [float(np.mean([x.mean() for x in R[e][nm]["ssim_amp"]])), float(np.mean([x.mean() for x in R[e][nm]["ssim_ph"]]))],
                   "n_fail": int(sum((~f).sum() for f in R[e][nm]["fin"])),
                   "unread": float(np.mean([(~x).mean() for x in R[e][nm]["readable"]])),
                   "vac": {k: s8.vac_stats(s8.pooled_conf(R, e, nm, sel)) for k, sel in (("all", None), ("W", mats == 0), ("Mo", mats == 1))},
                   "svf1_seed": {k: s8.seed_svf1(R, e, nm, sel) for k, sel in (("all", None), ("W", mats == 0), ("Mo", mats == 1))}}
            row["svf1"] = {k: float(np.mean(v)) for k, v in row["svf1_seed"].items()}
            row["frc"] = {}                                                           # pooled FRC(同階段八 h11;描述)
            for mk in ("W", "Mo"):
                curves = [s8.frc_curve(*[np.array(x) for x in fr[mk]]) for fr in R[e][nm]["frc"]]
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        row["frc"][mk] = np.nanmean(np.stack(curves), 0).tolist()
            if nm == MAIN_C and e == MAIN_EXAM:
                row["svf1_ci"] = {k: s8.boot_svf1(R[e][nm]["conf"], rng, sel) for k, sel in (("all", None), ("W", mats == 0), ("Mo", mats == 1))}
            for B in ("64", "512"):
                tn = tim["net"][nm][B]
                ev = envelope(env, tn, tim[B], S, "nerr_c")
                rr = s5b.ratio(np.array(row["nerr_c"]), ev[2]) if ev is not None else ("-", float("nan"), float("nan"))
                am = float(np.mean(row["nerr_c"]))
                rb = {"t": tn, "opp": None if ev is None else [ev[0], int(ev[1]), list(map(float, ev[2])), ev[3]], "ratio": list(rr),
                      "speed_k": {}, "speed_k_mat": {}, "own": speed(am, env, tim[B], tn, S, Tc[B][0], "nerr_c", [am])[str(am)],
                      "speed_fixed": speed(am, env, tim[B], tn, S, Tc[B][0], "nerr_c", QC_FIXED)}
                for k in K_CURVE:
                    rb["speed_k"][str(k)] = speed(am, env, tim[B], tn, S, Tc[B][0], "nerr_c", [k * crl[e]["crlb_c"]])[str(k * crl[e]["crlb_c"])]
                    if e == MAIN_EXAM:
                        rb["speed_k_mat"][str(k)] = {}
                        for mk in ("W", "Mo"):
                            amk = float(np.mean(row[f"nerr_c_{mk}"]))
                            Qm = k * crl[e][f"crlb_c_{mk}"]
                            rb["speed_k_mat"][str(k)][mk] = speed(amk, env, tim[B], tn, S, Tc[B][0], f"nerr_c_{mk}", [Qm])[str(Qm)]
                if e == MAIN_EXAM and nm == MAIN_C:                                       # 任務版的加速(描述;同階段八 h3):SV-F1
                    f1n = row["svf1"]["all"]
                    rb["task"] = speed(f1n, env, tim[B], tn, S, Tc[B][0], "sv_miss", [f1n, s8.SVF1_GOOD], higher=True)
                    evs = envelope(env, tn, tim[B], S, "sv_miss")
                    rb["iter_svf1_at_t"] = None if evs is None else [evs[0], int(evs[1]), 1 - float(np.mean(evs[2]))]
                row[B] = rb
            out[e]["nets"][nm] = row
        bo = envelope(env, Tc["512"][0], tim["512"], S, "nerr_c")
        out[e]["iter_best_tc512"] = [bo[0], int(bo[1]), float(bo[2].mean())]
        bo64 = envelope(env, Tc["64"][0], tim["64"], S, "nerr_c")
        out[e]["iter_best_tc64"] = [bo64[0], int(bo64[1]), float(bo64[2].mean())]
        out[e]["iter_fail"] = iter_fail(env, "c")
    return out


# ============================================================================
# 彙整:可重現性(f4)
# ============================================================================
def boot_two(xe, xv, rng):
    """比值 = 考試 / 驗證的平均;場為單位、兩邊各自獨立重抽(協定 §六 f4)。回傳 (比值, 90% 區間, 95% 區間)。"""
    xe, xv = np.asarray(xe, float), np.asarray(xv, float)
    r = float(xe.mean() / max(xv.mean(), 1e-30))
    ie = rng.integers(0, len(xe), (BOOT_N, len(xe)))
    iv = rng.integers(0, len(xv), (BOOT_N, len(xv)))
    rb = xe[ie].mean(1) / np.maximum(xv[iv].mean(1), 1e-30)
    return r, [float(np.percentile(rb, 5)), float(np.percentile(rb, 95))], [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]


def rep_label(r, ci90, ci95):
    """f4(v0.3 的等價性寫法):90% 區間整個在 [0.8, 1.25] → 重現;比值 ≥ 1.25 且 95% 區間不含 1 → 較差;≤ 0.8 且 95% 區間不含 1 → 較好;其餘 → 不確定。"""
    if REP_LO <= ci90[0] and ci90[1] <= REP_HI:
        return "重現"
    if r >= REP_HI and ci95[0] > 1:
        return "較差(驗證資料偏樂觀)"
    if r <= REP_LO and ci95[1] < 1:
        return "較好"
    return "不確定"


def c_field_mean(R, e, nm, fc):
    """C 的逐場 nerr_c(3 seeds 平均;失敗 = fc)。"""
    return s8.net_c(R, e, nm, fc).mean(0)


# ============================================================================
# 彙整:報表與判讀(協定 §五、§六;結果出來前寫定)
# ============================================================================
def report(P, AB, C, smoke):
    """P:彙整需要的全部輸入。回傳判讀 V(另印出報表)。"""
    Bs = ["64", "512"]
    tim, Tc, Tcv, plan = P["tim"], P["Tc"], P["Tc_val"], P["plan"]
    rng = np.random.default_rng(BOOT_SEED)
    V = {"plan": {"T_H": plan["T_H"], "ns": plan["ns"], "base_max": plan["base_max"], "hi": plan["hi"]},
         "Tc": {B: list(Tc[B]) for B in Bs}, "Tc_val": {B: list(Tcv[B]) for B in Bs}}
    if smoke:
        print("\n" + "!" * 100)
        print("迷你流程:驗證資料代替考試資料、場數少、停止點 ≤ 80 → 數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    print("\n" + "=" * 100)
    print(f"計時(ms / 整張;本 job 重新量;GPU {tim['gpu']})與迭代法的延長(協定 §三)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:G9 " + "、".join(f"{tg(nm)} {tim['net'][nm][B]:.4f}" for nm in NETS) + f";AP-C 每次 {tim[B]['AP-C']:.4f};"
              f"K-HIO49 {tim[B]['K-HIO49']:.4f}")
    print(f"  T_H = {plan['T_H']:.3f} ms(階段八計時檔 b512 的 AP-C × {plan['hi']});n_s:" + "、".join(f"{m} {v}" for m, v in plan["ns"].items()))
    for B in Bs:
        print(f"  T_c(b{B})= {Tc[B][0]:.2f} ms(最早結束的設定 {Tc[B][1]});驗證的包絡檔(500 次)的 T_c = {Tcv[B][0]:.2f} ms({Tcv[B][1]})")
    # ---------------- A / B ----------------
    V["ab"] = AB
    print("\n" + "=" * 100)
    print("A 理想條件、B 貼近真實條件(G9;主指標 nerr_sr;3 seeds;比值 = 網路 / 同時間包絡,≤ 0.8 且 z < −2 = 較準;"
          "加速 = 迭代法包絡最早達到 Q 的時間 / 網路時間,[下界–上界];只在 T_c 以內)")
    print("=" * 100)
    for c in AB_EVAL:
        tag = "(只測試;迭代法無混合態修正 → 加速只作描述)" if c in HELDOUT else ("(超出訓練範圍)" if c in OUTRANGE else "")
        tag += "(只評估網路,沒有包絡)" if c not in AB_ENV else ""
        print(f"\n  ■ {c}({a7.CLABEL[c]}){tag}")
        for nm in NETS:
            row = AB[c]["nets"][nm]
            line = (f"    {tg(nm):<9}:nerr_sr {np.mean(row['nerr_sr']):.5f}({' '.join(f'{x:.5f}' for x in row['nerr_sr'])});SSIM "
                    f"{row['ssim'][0]:.3f} / {row['ssim'][1]:.3f}" + (f";發散 {row['n_fail']} 個" if row["n_fail"] else ""))
            if c in AB_ENV:
                for B in Bs:
                    rb = row[B]
                    op = rb["opp"]
                    line += (f"\n       b{B}:{rb['t']:.3f} ms;同時間包絡 {s5b.fmt_conf(op[0])} ×{op[1]} {np.mean(op[2]):.5f} → 比值 {rb['ratio'][1]:.3f}"
                             f"(z {rb['ratio'][2]:+.1f});加速 " + "、".join(f"Q {q}:{fmt_x(rb['speed'][str(q)])}" for q in QS)
                             + f";自己的品質:{fmt_x(rb['own'])}")
            print(line)
        if c in AB_ENV:
            ib, fl = AB[c]["iter_best_tc512"], AB[c]["iter_fail"]
            print(f"    迭代法:T_c(b512)以內最佳 {s5b.fmt_conf(ib[0])} ×{ib[1]} {ib[2]:.2e};有發散的(設定, 停止點){fl['n_settings_with_fail']} 個")
    # ---------------- C ----------------
    V["c"] = C
    V["crlb"] = P["crlb"]
    print("\n" + "=" * 100)
    print("C 晶格(G9;主指標 nerr_c;Q = k × CRLB(考試資料的全部物體重算);SV 判讀;只在 T_c 以內)")
    print("=" * 100)
    for e in EXAMS:
        cr = P["crlb"][e]
        print(f"\n  ■ {e}({s8.ELABEL[e]});nerr_c 的失敗值 {P['fc'][e]:.3g};CRLB {cr['crlb_c']:.5f}(95% 區間 {cr['ci'][0]:.5f}–{cr['ci'][1]:.5f};"
              f"WS₂ {cr['crlb_c_W']:.5f} / MoS₂ {cr['crlb_c_Mo']:.5f};{cr['n_obj']} 個物體)"
              + {"L-combo": "(CRLB 假設位置 / 探針已知 → 只作描述)", "L-coh": "(CRLB 假設完全同調 → 只作描述)"}.get(e, ""))
        for nm in NETS:
            row = C[e]["nets"][nm]
            am = float(np.mean(row["nerr_c"]))
            line = (f"    {tg(nm):<9}:nerr_c {am:.4f}(= CRLB × {am / max(cr['crlb_c'], 1e-30):.2f};{' '.join(f'{x:.4f}' for x in row['nerr_c'])});"
                    f"WS₂ {np.mean(row['nerr_c_W']):.4f} / MoS₂ {np.mean(row['nerr_c_Mo']):.4f};SSIM {row['ssim'][0]:.3f} / {row['ssim'][1]:.3f};"
                    f"SV-F1 {row['svf1']['all']:.3f}(召回率 {row['vac']['all']['sv_r']:.3f}、精確率 {row['vac']['all']['sv_p']:.3f});"
                    f"無法判讀 {row['unread']:.1%}" + (f";發散 {row['n_fail']} 個" if row["n_fail"] else ""))
            for B in Bs:
                rb = row[B]
                op = rb["opp"]
                line += (f"\n       b{B}:{rb['t']:.3f} ms;同時間包絡 {s5b.fmt_conf(op[0])} ×{op[1]} {np.mean(op[2]):.4f} → 比值 {rb['ratio'][1]:.3f}"
                         f"(z {rb['ratio'][2]:+.1f});k = {K_MAIN}:{fmt_x(rb['speed_k'][str(K_MAIN)])};k = 2:{fmt_x(rb['speed_k']['2.0'])};"
                         f"自己的品質:{fmt_x(rb['own'])}")
            print(line)
        ib, fl = C[e]["iter_best_tc512"], C[e]["iter_fail"]
        print(f"    迭代法:T_c(b512)以內最佳 {s5b.fmt_conf(ib[0])} ×{ib[1]} nerr_c {ib[2]:.4f};有發散的(設定, 停止點){fl['n_settings_with_fail']} 個、"
              f"最高 {s5b.fmt_conf(fl['worst'][0])} ×{fl['worst'][1]} {fl['worst'][2]:.1%}")
    # ---------------- 判讀 ----------------
    print("\n" + "=" * 100)
    print("判讀(協定 §六,結果出來前寫定)")
    print("=" * 100)
    # f1
    ra = AB[COND_A]["nets"][MAIN_AB]
    V["f1"] = {"nerr_sr": float(np.mean(ra["nerr_sr"])),
               "by_q": {str(q): {"label": speed_label(ra["64"]["speed"][str(q)], ra["512"]["speed"][str(q)]),
                                 "b64": ra["64"]["speed"][str(q)], "b512": ra["512"]["speed"][str(q)]} for q in QS}}
    V["f1"]["label"] = V["f1"]["by_q"][str(Q_MAIN)]["label"]
    print(f"  (f1)A:P6B4e8 在 ideal 的 nerr_sr {V['f1']['nerr_sr']:.5f}(驗證資料 0.0037):"
          + ";".join(f"Q {q}:{V['f1']['by_q'][str(q)]['label']}(b64 {fmt_x(V['f1']['by_q'][str(q)]['b64'])}、b512 {fmt_x(V['f1']['by_q'][str(q)]['b512'])})"
                     for q in QS))
    # f2
    rc = AB[F2_COND]["nets"][MAIN_AB]
    s64, s512 = rc["64"]["speed"][str(F2_Q)], rc["512"]["speed"][str(F2_Q)]
    lab2 = speed_label(s64, s512)
    rob = s64["reach"] and all(x["x_lo"] >= SPEED_MIN for x in (s64, s512))
    V["f2"] = {"nerr_sr": float(np.mean(rc["nerr_sr"])), "label": lab2, "reproduced": bool(rob), "b64": s64, "b512": s512,
               "own": {B: rc[B]["own"] for B in Bs}}
    print(f"  (f2 主判)B:P6B4e8 在 combo、Q = {F2_Q}:nerr_sr {V['f2']['nerr_sr']:.5f}(驗證資料 0.0479,× 1.7–2.8)→ {lab2}"
          f"(b64 {fmt_x(s64)}、b512 {fmt_x(s512)})→ " + ("「combo 的穩健加速在期末考重現」" if rob else "未重現穩健加速(照實記錄)")
          + ("  [smoke:無意義]" if smoke else ""))
    print(f"       並列:網路自己的品質為門檻 b64 {fmt_x(rc['64']['own'])}、b512 {fmt_x(rc['512']['own'])}")
    V["f2p"] = {}
    for c in AB_ENV:
        rw = AB[c]["nets"][MAIN_AB]
        V["f2p"][c] = {str(q): speed_label(rw["64"]["speed"][str(q)], rw["512"]["speed"][str(q)]) for q in QS}
    print("  (f2′)B:其餘條件(P6B4e8;Q = 0.02 為主門檻,逐條件記錄;描述)")
    for c in AB_ENV:
        print(f"    {c:<7}:" + "、".join(f"Q {q} {V['f2p'][c][str(q)]}" for q in QS)
              + ("(描述:迭代法無混合態修正)" if c in HELDOUT else "") + ("(超出訓練範圍)" if c in OUTRANGE else ""))
    print(f"    comboNP:P6B4e8 nerr_sr {np.mean(AB['comboNP']['nets'][MAIN_AB]['nerr_sr']):.5f}(只有網路的誤差)")
    # f3
    rl = C[MAIN_EXAM]["nets"][MAIN_C]
    k3 = str(K_MAIN)
    V["f3"] = {"nerr_c": float(np.mean(rl["nerr_c"])), "crlb": P["crlb"][MAIN_EXAM]["crlb_c"], "Q": K_MAIN * P["crlb"][MAIN_EXAM]["crlb_c"],
               "label": speed_label(rl["64"]["speed_k"][k3], rl["512"]["speed_k"][k3]),
               "text": {B: f3_text(rl[B]["speed_k"][k3]) for B in Bs}, "b64": rl["64"]["speed_k"][k3], "b512": rl["512"]["speed_k"][k3],
               "k2": {B: rl[B]["speed_k"]["2.0"] for B in Bs}, "own": {B: rl[B]["own"] for B in Bs},
               "k_curve": {str(k): {B: rl[B]["speed_k"][str(k)] for B in Bs} for k in K_CURVE},
               "mat": {mk: {B: rl[B]["speed_k_mat"][k3][mk] for B in Bs} for mk in ("W", "Mo")},
               "fixed": {str(q): {B: rl[B]["speed_fixed"][str(q)] for B in Bs} for q in QC_FIXED},
               "P6B4e8": {B: C[MAIN_EXAM]["nets"][BASE][B]["speed_k"][k3] for B in Bs}}
    print(f"  (f3 主判)C:P6B4e8-L 在 L-paper 的 nerr_c {V['f3']['nerr_c']:.4f}(= CRLB × {V['f3']['nerr_c'] / max(V['f3']['crlb'], 1e-30):.2f});"
          f"Q = {K_MAIN} × CRLB = {V['f3']['Q']:.5f}:b64 {V['f3']['text']['64']};b512 {V['f3']['text']['512']}"
          + ("  [smoke:無意義]" if smoke else ""))
    print("       並列:k = 2 " + "、".join(f"b{B} {fmt_x(V['f3']['k2'][B])}" for B in Bs) + ";網路自己的品質 "
          + "、".join(f"b{B} {fmt_x(V['f3']['own'][B])}" for B in Bs) + ";P6B4e8(k = 1.5)"
          + "、".join(f"b{B} {fmt_x(V['f3']['P6B4e8'][B])}" for B in Bs))
    print("       加速倍數 vs k(描述):" + ";".join(f"k {k}:" + "、".join(f"b{B} {fmt_x(V['f3']['k_curve'][str(k)][B])}" for B in Bs) for k in K_CURVE))
    print("       材料(另列):" + ";".join(f"{'WS₂' if mk == 'W' else 'MoS₂'} " + "、".join(f"b{B} {fmt_x(V['f3']['mat'][mk][B])}" for B in Bs)
                                     for mk in ("W", "Mo")))
    print("       固定門檻(nerr_c;描述,與階段八 h1 對照):" + ";".join(f"Q {q}:" + "、".join(f"b{B} {fmt_x(V['f3']['fixed'][str(q)][B])}" for B in Bs)
                                                         for q in QC_FIXED))
    for e in EXAMS:
        if e == MAIN_EXAM:
            continue
        rw = C[e]["nets"][MAIN_C]
        print(f"       {e}(描述):k = {K_MAIN}:" + "、".join(f"b{B} {fmt_x(rw[B]['speed_k'][k3])}" for B in Bs))
    # f3'
    vs = rl["vac"]["all"]
    V["f3p"] = {"svf1": rl["svf1"], "ci": rl.get("svf1_ci"), "recall": vs["sv_r"], "precision": vs["sv_p"], "conf": vs["conf"],
                "task": {B: rl[B]["task"] for B in Bs}, "iter_svf1_at_t": {B: rl[B]["iter_svf1_at_t"] for B in Bs}}
    print(f"  (f3′)P6B4e8-L 在 L-paper 的 SV-F1 {rl['svf1']['all']:.3f}(95% 區間 {rl['svf1_ci']['all'][0]:.3f}–{rl['svf1_ci']['all'][1]:.3f};"
          f"WS₂ {rl['svf1']['W']:.3f} / MoS₂ {rl['svf1']['Mo']:.3f})、召回率 {vs['sv_r']:.3f}、精確率 {vs['sv_p']:.3f}(驗證資料 0.983 / 0.967 / 1.000);"
          "迭代法在網路時間的 SV-F1 " + "、".join(f"b{B} {V['f3p']['iter_svf1_at_t'][B][2]:.3f}" for B in Bs)
          + ";任務版的加速 " + ";".join(f"b{B} " + "、".join(f"SV-F1 {float(q):.3f}:{fmt_x(r)}" for q, r in rl[B]["task"].items()) for B in Bs))
    refl = C[MAIN_EXAM]["refl"]
    print("  (描述)L-paper 的 pooled FRC(各布拉格反射處;不把第一次越過 0.5 稱為解析度):")
    for nm in NETS:
        fr = C[MAIN_EXAM]["nets"][nm]["frc"]
        print(f"    {tg(nm):<9}:" + ";".join(f"{'WS₂' if mk == 'W' else 'MoS₂'} " + "、".join(
            f"{x['hk']} {fr[mk][int(round(x['ring']))]:.3f}" for x in refl if int(round(x["ring"])) < len(fr[mk])) for mk in ("W", "Mo")))
    # f4
    V["f4"] = f4_report(P, AB, C, rng)
    # f5
    V["f5"] = f5_report(P, AB, C)
    V["table"] = summary_table(AB, C)
    return V


def table_note(part, k):
    """總結果表的描述性註記(協定 §五、§六:只作描述的條件 / 考卷)。"""
    if part == "ab":
        if k not in AB_ENV:
            return "描述:只評估網路(沒有包絡)"
        if k in HELDOUT:
            return "描述:只測試的部分同調,迭代法無混合態修正"
        if k in OUTRANGE:
            return "超出訓練範圍"
        return "主判(f1)" if k == COND_A else ("主判(f2,Q = 0.05)" if k == F2_COND else "")
    if k == MAIN_EXAM:
        return "主判(f3)"
    return {"L-combo": "描述:CRLB 假設位置 / 探針已知", "L-coh": "描述:CRLB 假設完全同調"}.get(k, "描述")


def summary_table(AB, C):
    """總結果表(協定 §〇 2):每個網路 × 三部分 × 每個條件的誤差、SSIM、網路時間、主門檻下迭代法的達到時間與加速(晶格另加 SV 判讀)。"""
    rows = []
    Bs = ("64", "512")
    print("\n" + "=" * 100)
    print(f"總結果表(主門檻:A / B = nerr_sr ≤ {Q_MAIN};C = nerr_c ≤ {K_MAIN} × CRLB;時間 = ms / 整張;加速 = [下界–上界];b64 / b512)")
    print("=" * 100)

    def times(r, sp):
        out = {}
        for B in Bs:
            x = sp[B] if sp else None
            out[f"t_net_b{B}"] = r[B]["t"] if B in r else None
            if x is None:
                out[f"t_iter_b{B}"] = None
            elif x["iter_in_tc"]:
                out[f"t_iter_b{B}"] = [x["t_iter_lo"], x["t_iter"]]
            else:
                out[f"t_iter_b{B}"] = [x["t_iter_lo"], None]                         # T_c 內未達:只有下界
        return out

    for c in AB_EVAL:
        part = "A" if c == COND_A else "B"
        for nm in NETS:
            r = AB[c]["nets"][nm]
            sp = {B: r[B]["speed"][str(Q_MAIN)] for B in Bs} if c in AB_ENV else None
            rows.append({"part": part, "cond": c, "net": tg(nm), "err": float(np.mean(r["nerr_sr"])), "ssim_amp": r["ssim"][0],
                         "ssim_ph": r["ssim"][1], "label": speed_label(sp["64"], sp["512"]) if sp else "-",
                         "b64": fmt_x(sp["64"]) if sp else "-", "b512": fmt_x(sp["512"]) if sp else "-", "note": table_note("ab", c), **times(r, sp)})
    for e in EXAMS:
        for nm in NETS:
            r = C[e]["nets"][nm]
            sp = {B: r[B]["speed_k"][str(K_MAIN)] for B in Bs}
            rows.append({"part": "C", "cond": e, "net": tg(nm), "err": float(np.mean(r["nerr_c"])), "ssim_amp": r["ssim"][0], "ssim_ph": r["ssim"][1],
                         "svf1": r["svf1"]["all"], "label": speed_label(sp["64"], sp["512"]), "b64": fmt_x(sp["64"]), "b512": fmt_x(sp["512"]),
                         "note": table_note("c", e), **times(r, sp)})

    def ft(v):
        if v is None:
            return "-"
        return f"{v[0]:.2f}–{v[1]:.2f}" if v[1] is not None else f"> {v[0]:.2f}"

    for x in rows:
        tn = f"{x['t_net_b64']:.3f} / {x['t_net_b512']:.3f}" if x["t_net_b64"] is not None else "-"
        print(f"  {x['part']} {x['cond']:<8} {x['net']:<9} 誤差 {x['err']:.5f}  SSIM {x['ssim_amp']:.3f} / {x['ssim_ph']:.3f}"
              + (f"  SV-F1 {x['svf1']:.3f}" if "svf1" in x else "") + f"  網路 {tn} ms;迭代法達到 {ft(x['t_iter_b64'])} / {ft(x['t_iter_b512'])} ms"
              + f"  → {x['label']}(b64 {x['b64']};b512 {x['b512']})" + (f"  [{x['note']}]" if x["note"] else ""))
    return rows


def f4_report(P, AB, C, rng):
    """可重現性:主要的比值(P6B4e8 × 13 個條件、P6B4e8-L × L-paper)與標籤;其餘網路 / 考卷描述;包絡的考試 / 驗證比值(對照);同一方向。"""
    Bs = ["64", "512"]
    tim = P["tim"]
    out = {"main": {}, "desc": {}, "env": {}, "same": {}}
    print("  (f4)可重現性:比值 = 考試 / 驗證的平均誤差(場為單位、兩邊各自重抽;90% 區間整個在 [0.8, 1.25] → 重現;"
          "≥ 1.25 且 95% 區間不含 1 → 較差;≤ 0.8 且 95% 區間不含 1 → 較好;其餘 → 不確定)")
    for c in AB_EVAL:
        for nm in NETS:
            xe = ab_field_mean(P["Rab"], c, nm, "sr", P["valid_ab"])
            xv = ab_field_mean(P["Rab_val"], c, nm, "sr", P["valid_ab_val"])
            r, c90, c95 = boot_two(xe, xv, rng)
            rec = {"ratio": r, "ci90": c90, "ci95": c95, "label": rep_label(r, c90, c95), "exam": float(xe.mean()), "val": float(xv.mean())}
            (out["main"] if nm == MAIN_AB else out["desc"]).setdefault(c, {})[nm] = rec
    for e in EXAMS:
        for nm in NETS:
            fcm = max(P["fc"][e], P["fc_val"][e])                                         # 兩邊用同一個失敗值(取大者)
            xe = c_field_mean(P["Rc"], e, nm, fcm)
            xv = c_field_mean(P["Rc_val"], e, nm, fcm)
            r, c90, c95 = boot_two(xe, xv, rng)
            rec = {"ratio": r, "ci90": c90, "ci95": c95, "label": rep_label(r, c90, c95), "exam": float(xe.mean()), "val": float(xv.mean()),
                   "fail_c": fcm}
            (out["main"] if (nm == MAIN_C and e == MAIN_EXAM) else out["desc"]).setdefault(e, {})[nm] = rec
    for c in AB_EVAL:
        m = out["main"][c][MAIN_AB]
        print(f"    P6B4e8 {c:<7}:考試 {m['exam']:.5f} / 驗證 {m['val']:.5f} = {m['ratio']:.3f}(90% {m['ci90'][0]:.3f}–{m['ci90'][1]:.3f}、"
              f"95% {m['ci95'][0]:.3f}–{m['ci95'][1]:.3f})→ {m['label']}")
    m = out["main"][MAIN_EXAM][MAIN_C]
    print(f"    P6B4e8-L L-paper:考試 {m['exam']:.4f} / 驗證 {m['val']:.4f} = {m['ratio']:.3f}(90% {m['ci90'][0]:.3f}–{m['ci90'][1]:.3f}、"
          f"95% {m['ci95'][0]:.3f}–{m['ci95'][1]:.3f})→ {m['label']}")
    print("    其餘網路 / 考卷(描述):" + ";".join(f"{k}:" + "、".join(f"{tg(nm)} {v['ratio']:.2f}({v['label']})" for nm, v in d_.items())
                                       for k, d_ in out["desc"].items()))
    # 包絡的考試 / 驗證比值(同一時間點 = 主判網路的時間)
    for c in AB_ENV:
        out["env"][c] = {}
        for B in Bs:
            tn = tim["net"][MAIN_AB][B]
            ee = envelope(P["Eab"][c], tn, tim[B], P["S"], "nerr_sr")
            ev = envelope(P["Eab_val"][c], tn, tim[B], P["S_val"], "nerr_sr")
            out["env"][c][B] = {"exam": float(ee[2].mean()), "val": float(ev[2].mean()), "ratio": float(ee[2].mean() / max(ev[2].mean(), 1e-30))}
    for e in EXAMS:
        out["env"][e] = {}
        for B in Bs:
            tn = tim["net"][MAIN_C][B]
            ee = envelope(P["Ec"][e], tn, tim[B], P["S"], "nerr_c")
            ev = envelope(P["Ec_val"][e], tn, tim[B], P["S_val"], "nerr_c")
            out["env"][e][B] = {"exam": float(ee[2].mean()), "val": float(ev[2].mean()), "ratio": float(ee[2].mean() / max(ev[2].mean(), 1e-30))}
    print("    對照(描述):迭代法包絡在主判網路時間的考試 / 驗證比值 —— "
          + ";".join(f"{k} " + "/".join(f"{v[B]['ratio']:.2f}" for B in Bs) for k, v in out["env"].items()) + "(b64 / b512)")
    out["same"] = same_direction(P, AB, C)
    return out


def labels_val(P):
    """驗證資料上的 f1–f3 標籤(同一套判讀程式;驗證的包絡檔 = 階段七 / 八,500 次 → 驗證的 T_c)。"""
    tim, Sv, Tcv = P["tim"], P["S_val"], P["Tc_val"]
    out = {}

    def lab_ab(c, q):
        a = np.array(P["Rab_val"][c][MAIN_AB]["nerr_sr"], float).mean()
        sp = {B: speed(a, P["Eab_val"][c], tim[B], tim["net"][MAIN_AB][B], Sv, Tcv[B][0], "nerr_sr", [q])[str(q)] for B in ("64", "512")}
        return speed_label(sp["64"], sp["512"]), float(a)

    out["f1"] = lab_ab(COND_A, Q_MAIN)
    out["f2"] = lab_ab(F2_COND, F2_Q)
    am = float(s8.net_c(P["Rc_val"], MAIN_EXAM, MAIN_C, P["fc_val"][MAIN_EXAM]).mean())
    Q = K_MAIN * P["crlb_val"]
    sp = {B: speed(am, P["Ec_val"][MAIN_EXAM], tim[B], tim["net"][MAIN_C][B], Sv, Tcv[B][0], "nerr_c", [Q])[str(Q)] for B in ("64", "512")}
    out["f3"] = (speed_label(sp["64"], sp["512"]), am)
    return out


def same_direction(P, AB, C):
    """f4「同一方向」(協定 v0.4 寫定):(i) f1–f3 標籤;(ii) h5;(iii) h6;(iv) L-ideal 的天花板與漏判。逐項記錄相同 / 不同(描述)。"""
    out = {}
    lv = labels_val(P)
    le = {"f1": AB[COND_A]["nets"][MAIN_AB], "f2": AB[F2_COND]["nets"][MAIN_AB]}
    ex = {"f1": speed_label(le["f1"]["64"]["speed"][str(Q_MAIN)], le["f1"]["512"]["speed"][str(Q_MAIN)]),
          "f2": speed_label(le["f2"]["64"]["speed"][str(F2_Q)], le["f2"]["512"]["speed"][str(F2_Q)]),
          "f3": speed_label(C[MAIN_EXAM]["nets"][MAIN_C]["64"]["speed_k"][str(K_MAIN)], C[MAIN_EXAM]["nets"][MAIN_C]["512"]["speed_k"][str(K_MAIN)])}
    base_lab = lambda s: s.split("(")[0]                                                  # noqa: E731  (去掉「下界」的註記再比較類別)
    out["labels"] = {k: {"exam": ex[k], "val": lv[k][0], "same": base_lab(ex[k]) == base_lab(lv[k][0])} for k in ("f1", "f2", "f3")}
    print("    同一方向(i)f1–f3 的標籤:" + ";".join(f"{k} 考試「{v['exam']}」/ 驗證「{v['val']}」→ {'相同' if v['same'] else '不同'}"
                                          for k, v in out["labels"].items()))
    h5 = {}
    for e in ("L-paper", "L-ideal", "L-combo"):
        re_ = float(np.mean(C[e]["nets"][NEW_L]["nerr_c"]) / max(np.mean(C[e]["nets"][NEW_C]["nerr_c"]), 1e-30))
        lv_ = s8.net_c(P["Rc_val"], e, NEW_L, P["fc_val"][e]).mean() / max(s8.net_c(P["Rc_val"], e, NEW_C, P["fc_val"][e]).mean(), 1e-30)
        h5[e] = {"exam": re_, "val": float(lv_), "same": bool((re_ < 1) == (lv_ < 1))}
    out["h5"] = h5
    print("    同一方向(ii)h5 P6B4e8-L / P6B4e8-C:" + ";".join(f"{e} 考試 {v['exam']:.3f} / 驗證 {v['val']:.3f} → {'相同' if v['same'] else '不同'}"
                                                    for e, v in h5.items()))
    h6 = {}
    for c in AB_EVAL:
        ae, be = np.array(P["Rab"][c][NEW_L]["nerr_sr"]), np.array(P["Rab"][c][BASE]["nerr_sr"])
        av, bv = np.array(P["Rab_val"][c][NEW_L]["nerr_sr"]), np.array(P["Rab_val"][c][BASE]["nerr_sr"])
        rex, rva = s5b.ratio(ae, be), s5b.ratio(av, bv)
        h6[c] = {"exam": rex[1], "val": rva[1], "same_side": bool((rex[1] > 1) == (rva[1] > 1)),
                 "regress_exam": rex[0] == "較差", "regress_val": rva[0] == "較差"}
    out["h6"] = h6
    print("    同一方向(iii)h6 P6B4e8-L / P6B4e8(13 個條件):" + ";".join(f"{c} {v['exam']:.2f}/{v['val']:.2f}{'' if v['same_side'] else '✗'}"
                                                               for c, v in h6.items()))
    print(f"       在 1 的同一側:{sum(v['same_side'] for v in h6.values())} / {len(h6)};退步(考試):{[c for c, v in h6.items() if v['regress_exam']]};"
          f"退步(驗證):{[c for c, v in h6.items() if v['regress_val']]}")
    tim = P["tim"]
    ce = C["L-ideal"]["iter_best_tc512"][2] < float(np.mean(C["L-ideal"]["nets"][NEW_L]["nerr_c"]))
    bv = envelope(P["Ec_val"]["L-ideal"], P["Tc_val"]["512"][0], tim["512"], P["S_val"], "nerr_c")
    cv = float(bv[2].mean()) < float(s8.net_c(P["Rc_val"], "L-ideal", NEW_L, P["fc_val"]["L-ideal"]).mean())
    me = C["L-ideal"]["nets"][NEW_L]["vac"]["all"]["sv_r"] < 1
    mv = s8.vac_stats(s8.pooled_conf(P["Rc_val"], "L-ideal", NEW_L))["sv_r"] < 1
    out["scope"] = {"ceiling": {"exam": bool(ce), "val": bool(cv), "same": bool(ce == cv)}, "missed": {"exam": bool(me), "val": bool(mv), "same": bool(me == mv)}}
    print(f"    同一方向(iv)L-ideal:迭代法包絡(T_c 以內)最佳 < P6B4e8-L —— 考試 {ce} / 驗證 {cv} → {'相同' if ce == cv else '不同'};"
          f"P6B4e8-L 的 SV 召回率 < 1 —— 考試 {me} / 驗證 {mv} → {'相同' if me == mv else '不同'}")
    return out


def f5_report(P, AB, C):
    """f5:SSIM 與主指標的排名一致性(描述;同階段七 b7):每個條件 / 考卷內,網路(3 seeds 平均)+ 迭代法的關鍵設定,兩兩比較。"""
    out = {}
    for part, keys, R_, IT, mk in (("ab", AB_EVAL, AB, P["IT_ab"], "nerr_sr"), ("c", EXAMS, C, P["IT_c"], "nerr_c")):
        agree = {"amp": [0, 0], "ph": [0, 0]}
        for k in keys:
            items = [(float(np.mean(R_[k]["nets"][nm][mk])), *R_[k]["nets"][nm]["ssim"]) for nm in NETS]
            items += [(float(np.mean(x[mk])), float(np.mean(x["ssim_amp"])), float(np.mean(x["ssim_ph"]))) for x in IT.get(k, [])]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if items[i][0] == items[j][0]:
                        continue
                    for kk, col in (("amp", 1), ("ph", 2)):
                        agree[kk][1] += 1
                        agree[kk][0] += int((items[i][0] < items[j][0]) == (items[i][col] > items[j][col]))
        out[part] = agree
    print(f"  (f5)SSIM 與主指標的排名一致(描述):A / B 振幅 {out['ab']['amp'][0]}/{out['ab']['amp'][1]}、相位 {out['ab']['ph'][0]}/{out['ab']['ph'][1]};"
          f"C 振幅 {out['c']['amp'][0]}/{out['c']['amp'][1]}、相位 {out['c']['ph'][0]}/{out['c']['ph'][1]}")
    return out


# ============================================================================
# 圖(協定 §八;事先寫定,不挑圖;描述)
# ============================================================================
def figures(V, P, keep, out_dir, where):
    plt = s5._plt()
    if plt is None:
        check("matplotlib 可用(畫圖需要)", False)
        return
    from matplotlib.patches import Circle
    out_dir.mkdir(parents=True, exist_ok=True)
    tim, S, Tc = P["tim"], P["S"], P["Tc"]

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    # 1. 時間 vs 誤差(包絡含延長段;T_c 以虛線標出;網路為點)
    for part, keys, E_, key, ylab, Rk in (("ab", AB_ENV, P["Eab"], "nerr_sr", "nerr_sr on U", "ab"), ("c", EXAMS, P["Ec"], "nerr_c", "nerr_c on U", "c")):
        for B in ("64", "512"):
            nc = len(keys)
            cols = 4 if nc > 5 else nc
            rows = int(math.ceil(nc / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
            for k, c in enumerate(keys):
                ax = axs[k // cols, k % cols]
                ts, vs, best = [], [], float("inf")
                for t, v, _, _ in env_points(E_[c], S, tim[B], key):
                    best = min(best, v)
                    if t > 0:
                        ts.append(t)
                        vs.append(best)
                ax.plot(ts, vs, color="#0b0b0b", lw=1.3, drawstyle="steps-post", label="iterative envelope")
                ax.axvline(Tc[B][0], color="#8a5cd1", lw=0.8, ls="--", label="T_c")
                for nm, col in ((MAIN_AB if part == "ab" else MAIN_C, "#1baf7a"), (BASE if part == "c" else NEW_L, "#2a78d6")):
                    y = float(np.mean(V[part][c]["nets"][nm][key]))
                    ax.plot([tim["net"][nm][B]], [max(y, 1e-7)], "o", color=col, ms=6, label=tg(nm))
                if part == "ab":
                    for q in QS:
                        ax.axhline(q, color="#d6452a" if q == Q_MAIN else "#e8a598", lw=0.7, ls=":")
                else:
                    ax.axhline(K_MAIN * V["crlb"][c]["crlb_c"], color="#d6452a", lw=0.8, ls=":", label=f"{K_MAIN} x CRLB")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(c, fontsize=9)
                ax.set_xlabel("time per whole image (ms)")
                ax.grid(True, color="#e4e3df", which="both", lw=0.5)
            for k in range(nc, rows * cols):
                axs[k // cols, k % cols].axis("off")
            axs[0, 0].set_ylabel(ylab)
            axs[0, 0].legend(fontsize=6.5, frameon=False)
            fig.suptitle(f"{where}: time vs error, batch {B} (envelope includes the extension; only conclusions before T_c)", fontsize=10)
            fig.tight_layout()
            save(fig, f"f_time_{part}_b{B}.png")
    # 2. 各條件的誤差:網路 vs 同時間包絡
    for part, keys, key, nm in (("ab", AB_ENV, "nerr_sr", MAIN_AB), ("c", EXAMS, "nerr_c", MAIN_C)):
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(keys) + 2), 4))
        x = np.arange(len(keys))
        ax.plot(x, [np.mean(V[part][c]["nets"][nm][key]) for c in keys], "o", color="#1baf7a", label=tg(nm))
        for B, mk, col in (("64", "s", "#9a9994"), ("512", "^", "#0b0b0b")):
            ax.plot(x, [np.mean(V[part][c]["nets"][nm][B]["opp"][2]) for c in keys], mk, color=col, mfc="none", label=f"envelope at net time, b{B}")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(key)
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{where}: network vs iterative envelope at the same time", fontsize=9)
        fig.tight_layout()
        save(fig, f"f_cond_{part}.png")
    # 3. 可重現性(考試 / 驗證,90% 區間;[0.8, 1.25] 為等價範圍)
    items = [(c, V["f4"]["main"][c][MAIN_AB]) for c in AB_EVAL] + [(f"{MAIN_EXAM} (L)", V["f4"]["main"][MAIN_EXAM][MAIN_C])]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (k, r) in enumerate(items):
        ax.plot([i, i], r["ci90"], color="#2a78d6", lw=2)
        ax.plot([i], [r["ratio"]], "o", color="#0b0b0b", ms=4)
    ax.axhspan(REP_LO, REP_HI, color="#1baf7a", alpha=0.12)
    ax.axhline(1.0, color="#9a9994", lw=0.8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([k for k, _ in items], rotation=35, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("exam / validation (mean error)")
    ax.set_title(f"{where}: reproducibility (P6B4e8 on A/B, P6B4e8-L on L-paper); bar = 90% bootstrap interval", fontsize=9)
    ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    fig.tight_layout()
    save(fig, "f_repro.png")
    # 3b. FRC(L-paper;各網路;描述)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    refl = V["c"][MAIN_EXAM]["refl"]
    cols_ = {"P6B4": "#9a9994", "P6B4r": "#e8a33d", BASE: "#2a78d6", NEW_C: "#8a5cd1", NEW_L: "#1baf7a"}
    for k_, mk in enumerate(("W", "Mo")):
        ax = axs[k_]
        for nm in NETS:
            cur = np.array(V["c"][MAIN_EXAM]["nets"][nm]["frc"][mk])
            ax.plot(np.arange(len(cur)) / s8.FRC_N, cur, color=cols_[nm], lw=1.2, label=tg(nm))
        for x in refl:
            ax.axvline(x["ring"] / s8.FRC_N, color="#bbbbbb", lw=0.6)
            ax.text(x["ring"] / s8.FRC_N, 1.02, x["hk"], fontsize=6, ha="center")
        ax.set_ylim(-0.2, 1.08)
        ax.set_xlabel("spatial frequency (cycles / px)")
        ax.set_title(f"{MAIN_EXAM}, {'WS2' if mk == 'W' else 'MoS2'}: pooled FRC vs truth", fontsize=8)
        ax.grid(True, color="#e4e3df", lw=0.5)
    axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "f_frc.png")
    # 4. 重建圖:主判網路 seed 0 + 同時間的迭代法包絡;樣本 = 第 10 / 50 / 90 百分位、最差、6 個只依真值抽出的隨機樣本
    rngf = np.random.default_rng(FIG_SEED)
    for part, k in [("ab", c) for c in FIG_AB] + [("c", MAIN_EXAM)]:
        if k not in keep:
            continue
        kp = keep[k]
        U = kp["U"]
        if part == "ab":
            err = np.where(P["valid_ab"][k], P["Rab"][k][MAIN_AB]["sr"][0], np.nan)
            idx = np.where(P["valid_ab"][k])[0]
        else:
            cc = P["Rc"][k][MAIN_C]["c"][0]
            err = np.where(np.isfinite(cc), cc, P["fc"][k])
            idx = np.arange(len(err))
        order = idx[np.argsort(err[idx], kind="stable")]
        picks = [("p10", order[int(0.1 * (len(order) - 1))]), ("p50", order[int(0.5 * (len(order) - 1))]),
                 ("p90", order[int(0.9 * (len(order) - 1))]), ("worst", order[-1])]
        picks += [(f"rand{j}", int(v)) for j, v in enumerate(rngf.choice(idx, size=min(N_RAND_FIG, len(idx)), replace=False))]
        cols = [("truth", None)] + [(lab, lab) for lab in kp["img"]]
        y0, y1, x0, x1 = b7._bbox(U, a7.MARGIN)
        fig, axs = plt.subplots(len(picks), len(cols), figsize=(2.6 * len(cols), 2.5 * len(picks)), squeeze=False, constrained_layout=True)
        hp = None
        for r_, (pn, i) in enumerate(picks):
            o = kp["O"][i]
            for k_, (lab, key) in enumerate(cols):
                ax = axs[r_, k_]
                if key is None:
                    z, v = o, None
                else:
                    z, v = a7.align_show(kp["img"][key][i], o, U)
                hp = ax.imshow(np.angle(z[y0:y1, x0:x1].numpy()), cmap="viridis", vmin=0, vmax=float(P["phmax"] if part == "ab" else 0.9))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title((lab if r_ == 0 else "") + ("" if v is None else f"\nnerr_sr {v:.4f}"), fontsize=6)
                if part == "c":
                    vac = kp["vac"]
                    sel = vac.fid == i
                    pr_i = kp["pred"].get(key) if key is not None else None
                    pr_i = pr_i[sel] if pr_i is not None else None
                    for j, (lab_t, pos) in enumerate(zip(vac.lab[sel], vac.pos[sel])):
                        if int(lab_t) == 0:
                            if pr_i is not None and int(pr_i[j]) != 0:
                                ax.plot([float(pos[1]) - x0], [float(pos[0]) - y0], "x", color="#ff8c00", ms=4, mew=1.0)
                            continue
                        colr = "white" if pr_i is None else ("#1baf7a" if int(pr_i[j]) != 0 else "#d6452a")
                        ax.add_patch(Circle((float(pos[1]) - x0, float(pos[0]) - y0), 2.2 if int(lab_t) == 1 else 3.0, fill=False, color=colr,
                                            lw=0.8, ls="-" if int(lab_t) == 1 else "--"))
            axs[r_, 0].set_ylabel(f"#{i} ({pn})", fontsize=7)
        fig.colorbar(hp, ax=list(axs[:, -1]), fraction=0.03, label="phase (rad)")
        fig.suptitle(f"{where}: {k}, seed 0; rows = p10 / p50 / p90 / worst of {tg(MAIN_AB if part == 'ab' else MAIN_C)} + random (truth only)"
                     + ("; circles = truth SV (solid) / DV (dashed): green caught, red missed; orange x = false alarm" if part == "c" else ""),
                     fontsize=8)
        save(fig, f"f_recon_{k}.png")


# ============================================================================
# 內建檢查(協定 §七)
# ============================================================================
REPRO_CONFS = ["AP|one", "ePIE-C|rand", "AP-C:both|const", "ePIE-C:prb|zero", "AP-C:posA|khio20", "K-HIO"]   # 每類設定各 1 個
REPRO_IT = 20


def load_s8():
    J8 = json.load(open(S8_JSON))
    ok = J8.get("script_md5") == S8_MD5 and J8.get("complete") is True and not J8.get("smoke")
    return J8, ok


def load_c8():
    J = json.load(open(C8_JSON))
    ok = J.get("meta", {}).get("md5") == C8_MD5 and J.get("complete") is True and not J.get("meta", {}).get("smoke")
    return J, ok


def val_env_paths():
    p = {c: (a7.cond_json(c) if c in b7.OLD_CONDS else b7.env_json(c)) for c in AB_ENV}
    p.update({e: s8.env_json(e) for e in EXAMS})
    return p


def load_val_env(fc8):
    """驗證資料的包絡檔(階段七 7a / 7b、階段八):版本與停止點核對;C 以階段八的失敗值補上 nerr_c。"""
    E, bad = {}, []
    want = {c: c7.A7_MD5 if c in b7.OLD_CONDS else c7.B7_MD5 for c in AB_ENV}
    want.update({e: S8_MD5 for e in EXAMS})
    for k, p in val_env_paths().items():
        if not p.exists():
            bad.append(f"{p.name} 不存在")
            continue
        J = json.load(open(p))
        m = J["meta"]
        if m.get("script_md5") != want[k] or not J.get("checks_ok") or m.get("iters") != list(a7.ITERS):
            bad.append(f"{p.name} 版本、檢查或停止點不符")
            continue
        E[k] = J["env"]
        if k in EXAMS:
            for conf in E[k].values():
                for rec in conf.values():
                    s8.finalize_rec(rec, fc8[k])
    return E, bad


@torch.no_grad()
def check_repro(dev, n):
    """協定 §七 1(v0.4):驗證資料 seed 0 的全部場,網路與迭代法(每類設定各 1 個、20 次)= 階段七 / 八的結果檔(相對差 < 1e-4)。
    另:CRLB 的前 2 個場 = 8c 的結果檔。"""
    J8, ok8 = load_s8()
    J8c, okc = load_c8()
    fc8 = J8["verdict"]["fail_c"]
    Ev, bad = load_val_env(fc8)
    check("驗證資料的結果檔:scan8.json / scan8c.json 完整且版本正確;階段七 / 八的 17 個包絡檔版本、檢查、停止點正確", ok8 and okc and not bad,
          ";".join(bad))
    if not (ok8 and okc and not bad):
        return
    worst = {"net_ab": 0.0, "it_ab": 0.0, "net_c": 0.0, "it_c": 0.0}
    for c in AB_EVAL:
        d = setup_ab(0, dev, c, n, VAL_AB)
        O, U = d["O"], d["geo"].U
        for nm in NETS:
            kind, model, norm = s8.get_net(nm, 0, d["cfg"], d["geo"].pr, dev, roots_real())
            v = a7.metrics7(s8.run_net(kind, model, norm, d), O, U)["nerr_sr"]
            worst["net_ab"] = max(worst["net_ab"], reldiff(v, J8["orig"][c][nm]["nerr_sr"][0]))
            del model
        if c in AB_ENV:
            for conf in REPRO_CONFS:
                v = a7.metrics7(b7.iter_est(d, conf, REPRO_IT, dev), O, U)["nerr_sr"]
                worst["it_ab"] = max(worst["it_ab"], reldiff(v, Ev[c][conf][str(REPRO_IT)]["nerr_sr"][0]))
        del d
    for e in EXAMS:
        d = setup_c(0, dev, e, n, VAL_C)
        for nm in NETS:
            kind, model, norm = s8.get_net(nm, 0, d["cfg"], d["geo"].pr, dev, roots_real())
            cv = s8.lat_metrics(s8.run_net(kind, model, norm, d), d)["c"].numpy()
            v = float(np.where(np.isfinite(cv), cv, fc8[e]).mean())
            worst["net_c"] = max(worst["net_c"], reldiff(v, J8["verdict"]["exam"][e]["nets"][nm]["nerr_c"][0]))
            del model
        for conf in REPRO_CONFS:
            cv = s8.lat_metrics(s8.iter_est_L(d, conf, REPRO_IT, dev), d)["c"].numpy()
            v = float(np.where(np.isfinite(cv), cv, fc8[e]).mean())
            worst["it_c"] = max(worst["it_c"], reldiff(v, Ev[e][conf][str(REPRO_IT)]["nerr_c"][0]))
        del d
    d = setup_c(0, dev, "L-ideal", 2, VAL_C)
    ph = c8.Phys.of(d)
    rows = J8c["parts"]["A"]["T1"]["ideal"]
    wc = max(reldiff(c8.crlb_field(ph, d["O"][i], extra=False)["T"], rows[i]["T"]) for i in range(2))
    check(f"重現(協定 §七 1):驗證資料 seed 0 的全部 {n} 個場 —— 5 個網路(13 個條件、5 份考卷)與迭代法 {len(REPRO_CONFS)} 個設定 × {REPRO_IT} 次"
          f" = 階段七 / 八的結果檔(相對差 < {REPRO_TOL:.0e});CRLB 的前 2 個場 = 8c(< {REPRO_TOL:.0e})",
          max(list(worst.values()) + [wc]) < REPRO_TOL,
          "、".join(f"{k} {v:.1e}" for k, v in worst.items()) + f"、CRLB {wc:.1e}")


@torch.no_grad()
def check_paths(dev):
    """協定 §七 2:以驗證(與 smoke)資料的基底跑本程式的路徑 = scan_7a.setup_cond / scan_8.setup_exam(量測、位置、離焦、物體完全相同)。"""
    n = 8 if not QUICK else 4
    worst = 0.0
    bad = []
    for s in (SEEDS[0], SEEDS[-1]):
        for c in AB_EVAL:
            a, b = setup_ab(s, dev, c, n, VAL_AB), a7.setup_cond(s, dev, c, n)
            dd = [float((a["fields"] - b["fields"]).abs().max()), max(float((x - y).abs().max()) for x, y in zip(a["counts"], b["counts"])),
                  float((a["z"] - b["z"]).abs().max()), float((a["sign"] - b["sign"]).abs().max())]
            worst = max([worst] + dd)
            if a["init"] != b["init"]:
                bad.append(f"{c} s{s} init")
        for e in EXAMS:
            for base, smk in ((VAL_C, False), (dict(s8.SEED_SMOKE_SET), True)):
                a, b = setup_c(s, dev, e, n, base), s8.setup_exam(s, dev, e, n, smoke=smk)
                dd = [float((a["O"] - b["O"]).abs().max()), max(float((x - y).abs().max()) for x, y in zip(a["counts"], b["counts"])),
                      float((a["z"] - b["z"]).abs().max()), float((a["sign"] - b["sign"]).abs().max()), float((a["dose"] - b["dose"]).abs().max())]
                worst = max([worst] + dd)
                if a["init"] != b["init"] or not torch.equal(a["vac"].lab, b["vac"].lab) or a["obj_seeds"] != b["obj_seeds"]:
                    bad.append(f"{e} s{s}{' smoke' if smk else ''}")
    check("替換 seed 基底後的路徑(協定 §七 2):本程式的 A / B(13 個條件)、C(5 份考卷;驗證與 smoke 的基底)= scan_7a.setup_cond / scan_8.setup_exam"
          "(物體、量測、位置、離焦、劑量、起點 seed、真值的空缺標籤;逐位元)", worst == 0.0 and not bad, f"最大差 {worst:.1e};不同 {bad or '無'}")


def exam_seed_ranges(n):
    """期末考實際用到的 seed 範圍(含派生;協定 §二)。"""
    S_, out = list(SEEDS), {}
    smax = 1000 * max(S_)
    out["A/B 物體"] = (EXAM_AB["field"], EXAM_AB["field"] + 2000)                       # make_procedural:seed + 1000 × 種類(3 種)、+1(平移)
    out["A/B 雜訊 + 起點"] = (EXAM_AB["noise"], EXAM_AB["noise"] + smax + 1 + 100 + a6.NPOS ** 2)   # 起點 = 雜訊 + 1;K-HIO 起點 = 起點 + 100 + j
    out["A/B 位置"] = (EXAM_AB["pos"], EXAM_AB["pos"] + smax)
    out["A/B 離焦"] = (EXAM_AB["df"], EXAM_AB["df"] + smax)
    out["C 物體"] = (EXAM_C["obj"], EXAM_C["obj"] + n - 1)
    out["C 新缺陷"] = (EXAM_C["newdef"], EXAM_C["newdef"] + n - 1)
    out["C 雜訊"] = (EXAM_C["noise"], EXAM_C["noise"] + smax)
    out["C 位置"] = (EXAM_C["pos"], EXAM_C["pos"] + smax)
    out["C 離焦"] = (EXAM_C["df"], EXAM_C["df"] + smax)
    out["C 起點"] = (EXAM_C["init"], EXAM_C["init"] + smax + 100 + a6.NPOS ** 2)
    return out


@torch.no_grad()
def check_seeds(dev, n):
    """協定 §七 3:考試資料的 seed(含派生)與所有既有腳本的常數、訓練池、階段八的區間不重疊、彼此不重疊;保護會拒絕(只測拒絕,不產生考試資料)。"""
    found = s8.scan_seed_constants({Path(__file__).name})
    clash = sorted(f"{v}({','.join(sorted(found[v]))})" for v in found
                   if in_exam(v) and (v not in RESERVED or set(found[v]) - RESERVED[v]))
    rng_ = exam_seed_ranges(n)
    iv = sorted(rng_.values())
    disjoint = all(iv[i][1] < iv[i + 1][0] for i in range(len(iv) - 1)) and all(in_exam(lo) and in_exam(hi) for lo, hi in iv)
    others = [(50_000_000, 50_000_000 + 10_000_000 * max(SEEDS) + 10_000 * 1000), (80_000_000, 84_999_999)] + list(s8.MY_RANGES) \
        + list(c8.MY_RANGES_8C) + [(95_000_000, 96_999_999)]
    far = all(hi < EXAM_LO or lo > EXAM_HI for lo, hi in others)
    rej = []
    cfg = s5.load_cfg(0)
    for fn in (lambda: setup_ab(0, dev, "ideal", 1, EXAM_AB), lambda: setup_c(0, dev, "L-paper", 1, EXAM_C),
               lambda: s5.make_fields(cfg, 1, seed=EXAM_AB["field"]), lambda: s8.make_lattice([EXAM_C["obj"]], 16, dev, dx=0.358),
               lambda: s8.draw_params(EXAM_C["newdef"], 1), lambda: guard([EXAM_HI])):
        try:
            fn()
            rej.append(False)
        except ValueError:
            rej.append(True)
    check("seed(協定 §七 3):考試資料的 seed(含派生)彼此不重疊、都在保留區間內;與既有腳本的常數(grep)、訓練池 / 驗證 / 階段八 / 8c / 期中考的區間不重疊;"
          "保護會拒絕(A / B、C 的考試資料、物體產生器、晶格產生器;只測拒絕)",
          not clash and disjoint and far and all(rej),
          f"既有腳本的大整數 {len(found)} 個,落在 {EXAM_LO:,}–{EXAM_HI:,} 的(保留以外){clash or '無'};"
          + "、".join(f"{k} {lo:,}–{hi:,}" for k, (lo, hi) in rng_.items()) + f";拒絕 {rej}")


@torch.no_grad()
def check_extension(dev, plan):
    """協定 §七 4′:同一設定跑 n_s 次時,前 500 次(基本停止點)的結果 = 只跑 500 次的結果(迷你問題;相對差 < 1e-4)。"""
    n = 4
    S, lo = plan["S"], plan["base_max"]
    worst = {}
    d = setup_ab(0, dev, "combo", n, VAL_AB)
    dl = setup_c(0, dev, "L-combo", n, VAL_C)
    for dd, tag in ((d, "A/B combo"), (dl, "C L-combo")):
        O0 = a6.make_start("rand", dd, None, dev)
        for meth in a7.METHODS7:
            full = [i for i in S[f"{meth}|rand"] if i > 0]
            short = [i for i in full if i <= lo]
            a = a7.run_method7(meth, dd, full, O0)
            b = a7.run_method7(meth, dd, short, O0)
            w = max(float((a[i] - b[i]).abs().norm() / b[i].abs().norm().clamp_min(1e-30)) for i in short)
            worst[f"{tag} {meth}"] = w if np.isfinite(w) else float("inf")
        kf = sorted(set(S["K-HIO"]) | set(a7.KHIO_N.values()))
        ks = [k for k in kf if k <= lo]
        wa, wb = a7.khio_windows7(dd, dev, kf), a7.khio_windows7(dd, dev, ks)
        worst[f"{tag} K-HIO"] = max(float((wa[k] - wb[k]).abs().norm() / wb[k].abs().norm().clamp_min(1e-30)) for k in ks)
        del wa, wb
    mx = max(worst.values())
    check(f"延長的一致性(協定 §七 4′):85 個設定的 15 種演算法(rand 起點 + K-HIO)跑到 n_s 時,前 {lo} 次的停止點 = 只跑 {lo} 次(A / B 與 C 各 {n} 個場;"
          f"相對差 < {REPRO_TOL:.0e})", mx < REPRO_TOL, f"最大 {mx:.1e}(" + max(worst, key=worst.get) + ")")


def check_crlb():
    """協定 §七 4:CRLB 的程式(同 8c 的檢查 (1)):解析 Fisher = autograd、零空間、Cholesky = 特徵值分解、Fisher 恆等式(迷你問題)。"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ph, O = c8.mini_problem(dev)
    inc = ph.ill >= c8.TAU
    Fp, pid, Np = c8.fisher(ph, O, inc)
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
    ej = float((Fp - Fa).norm() / Fa.norm())
    G, gfull = c8.gauge_vecs(O, pid, Np, ph.U)
    nul = float((Fp @ (gfull / gfull.norm())).norm() / Fp.norm())
    ev, Vv = torch.linalg.eigh(Fp)
    nz = int((ev < 1e-9 * ev.max()).sum())
    ch = c8.fisher_chol(ph, O)
    r = c8.crlb_from(ph, ch)
    keep = ev >= 1e-9 * ev.max()
    Cm = (Vv[:, keep] / ev[keep]) @ Vv[:, keep].T
    Uidx = pid.view(ph.F, ph.F)[ph.U]
    cols = torch.cat([Uidx, Uidx + Np])
    Cuu = Cm[cols][:, cols]
    g1 = G[cols, 0] / G[cols, 0].norm()
    et = abs(r["T"] / float(Cuu.trace() - g1 @ Cuu @ g1) - 1)
    lam = ph.lam(O[None])[0]
    nr = torch.stack([torch.poisson(lam, generator=torch.Generator(device=dev).manual_seed(c8.SEED_MINI + 1 + k)) for k in range(c8.MINI_MC)])
    q, nexp = c8.score_q(ph, ch, nr, 1.0)
    qm = float(q.mean() / nexp)
    check("CRLB 的程式(協定 §七 4;同 8c):解析 Fisher = autograd(< 1e-6);整體相位是零空間(< 1e-10)且只有 1 個近零特徵值;"
          "Cholesky = 特徵值分解(< 1e-6);Fisher 恆等式 E[sᵀ I_F⁻¹ s] = 可辨識的參數數(< 3%)",
          ej < 1e-6 and nul < 1e-10 and nz == 1 and et < 1e-6 and abs(qm - 1) < 0.03,
          f"雅可比 {ej:.1e};零空間 {nul:.1e};近零特徵值 {nz} 個;T 相對差 {et:.1e};恆等式 {qm:.4f}")


def model_paths():
    out = {}
    for s in SEEDS:
        out[f"P6B4_s{s}"] = s5c.run_dir("P6B4", PROBE, s, a6.N_TRAIN) / "final.pt"
        out[f"P6B4r_s{s}"] = b7.model_dir(s) / "final.pt"
        out[f"{BASE}_s{s}"] = d7.model_dir("e", s) / "final.pt"
        out[f"{NEW_L}_s{s}"] = s8.model_dir("L", s) / "final.pt"
        out[f"{NEW_C}_s{s}"] = s8.model_dir("C", s) / "final.pt"
    return out


def check_models(write):
    """協定 §七 6(凍結):15 個模型檔的 md5;與既有結果檔記下的 md5 交叉核對(P6B4e8 的起點 = P6B4、P6B4e8-L / -C 的起點 = P6B4e8);
    --check 寫出 final_models.json,--run 要求相同。"""
    paths = model_paths()
    M = {k: md5(p) for k, p in paths.items()}
    cross = []
    for s in SEEDS:
        r_e = json.load(open(d7.model_dir("e", s) / "result.json"))
        cross.append(r_e.get("init_md5") == M[f"P6B4_s{s}"])
        for kind in ("L", "C"):
            r = json.load(open(s8.model_dir(kind, s) / "result.json"))
            cross.append(r.get("init_md5") == M[f"{BASE}_s{s}"] and r.get("script_md5") == S8_MD5)
    p = models_json()
    if write:
        if p.exists() and json.load(open(p))["md5"] != M:
            raise SystemExit(f"❌ {p} 已存在但模型的 md5 不同:模型被改過?先告訴 Claude")
        if not p.exists():
            dump_json({"md5": M, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, p)
        same = True
    else:
        same = p.exists() and json.load(open(p))["md5"] == M
    check("凍結(協定 §七 6):15 個模型檔的 md5 = --check 記下的(final_models.json);P6B4e8 的起點 = P6B4、P6B4e8-L / -C 的起點 = P6B4e8(結果檔記下的 md5)",
          same and all(cross), f"交叉核對 {sum(cross)} / {len(cross)}")
    return M


def check_ns(write, T512):
    """協定 §七 5:每個設定的 n_s 由階段八的計時檔算出,印出並存檔(考試之前固定);--run 要求與存檔相同。"""
    plan = plan_of(False, T512)
    rec = {"T_H": plan["T_H"], "ns": plan["ns"], "raw": plan["raw"], "S": plan["S"]}
    p = ns_json()
    if write:
        if p.exists():
            old = json.load(open(p))
            if {k: old[k] for k in ("ns", "S")} != {"ns": rec["ns"], "S": rec["S"]}:
                raise SystemExit(f"❌ {p} 已存在但 n_s 不同:先告訴 Claude")
        else:
            dump_json({**rec, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "s8_md5": S8_MD5}, p)
        same = True
    else:
        same = p.exists() and {k: json.load(open(p))[k] for k in ("ns", "S")} == {"ns": rec["ns"], "S": rec["S"]}
    check(f"延長的次數(協定 §七 5):T_H = AP-C × {plan['hi']} = {plan['T_H']:.3f} ms(階段八計時檔 b512);n_s 存檔且與存檔相同(final_ns.json)",
          same, "、".join(f"{m} {v}({plan['raw'][m]:.1f})" for m, v in plan["ns"].items()))
    return plan


def check_files(kind):
    """依賴的程式版本與需要的檔案。"""
    ok = s8.check_files("final")                                                       # 階段七 / 八的程式、模型、結果檔(scan_8 的檢查)
    m8, mc = md5(Path(s8.__file__).resolve()), md5(Path(c8.__file__).resolve())
    here = Path(__file__).resolve().parent
    dbad = [f for f, v in DEPEND.items() if not (here / f).exists() or md5(here / f) != v]
    check(f"scan_8.py、scan_8c.py = 階段八 / 8c 正式結果所用的版本;依賴的 {len(DEPEND)} 個程式 = install_final.sh 的版本",
          m8 == S8_MD5 and mc == C8_MD5 and not dbad, f"md5 {m8[:8]}… / {mc[:8]}…" + (f";不同:{dbad}" if dbad else ""))
    need = list(model_paths().values()) + [s8.model_dir(k, s) / "result.json" for k in ("L", "C") for s in SEEDS]
    need += [S8_JSON, C8_JSON, s8.DOSE_JSON] + list(val_env_paths().values())
    missing = [str(p) for p in need if not p.exists()]
    check(f"期末考另外需要的 {len(need)} 個檔案都存在(模型、階段八 / 8c 的結果、劑量、驗證的包絡檔)", not missing,
          "缺:" + ", ".join(missing) if missing else "")
    return ok and m8 == S8_MD5 and mc == C8_MD5 and not missing and not dbad


def input_files():
    """凍結的輸入:設定、模型(權重與正規化)、劑量、階段八 / 8c 的結果、驗證的包絡檔。"""
    files = [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS] + [S8_JSON, C8_JSON, s8.DOSE_JSON]
    files += list(val_env_paths().values())
    for p in model_paths().values():
        files += [p, p.parent / "result.json"]
    return files


def manifest():
    """smoke 通過時記下、--run 時比對:本程式、依賴的程式與全部輸入檔的 md5。"""
    here = Path(__file__).resolve().parent
    m = {"scan_final.py": me_md5()}
    m.update({f: md5(here / f) for f in DEPEND})
    m.update({str(p): md5(p) for p in input_files()})
    return m


def manifest_key(m):
    return hashlib.md5(json.dumps(m, sort_keys=True).encode()).hexdigest()


def timing512():
    J8, ok = load_s8()
    if not ok:
        raise SystemExit(f"❌ {S8_JSON} 不完整或版本不同:無法決定延長的次數")
    return J8["timing"]["512"]


def checks_full(dev):
    """--check:全部(協定 §七 1–6)。"""
    print("=" * 70)
    print("期末考:內建檢查(只用驗證資料;不產生考試資料)")
    print("=" * 70)
    torch.backends.cudnn.benchmark = True
    plan = check_ns(True, timing512())
    check_models(True)
    check_seeds(dev, s5b.N_FIELDS or s5.load_cfg(0).eval_n)
    check_paths(dev)
    check_extension(dev, plan)
    check_crlb()
    check_repro(dev, s5b.N_FIELDS or s5.load_cfg(0).eval_n)
    print(f"\n     裝置:{dev}({a7.gpu_name(dev)})")
    return plan


def checks_light(dev):
    """--run 前:n_s 與模型與 --check 時相同、seed 與保護、路徑。"""
    plan = check_ns(False, timing512())
    check_models(False)
    check_seeds(dev, s5b.N_FIELDS or s5.load_cfg(0).eval_n)
    check_paths(dev)
    return plan


# ============================================================================
# 物體的不重疊檢查(協定 §二;--run 開始時,只產生物體、不量測)
# ============================================================================
@torch.no_grad()
def overlap_ab(n):
    """A / B:考試物體的非空白物體塊(同期中考的做法)vs 驗證、期中考、舊測試、§十五、訓練池。產生器的結果與要求的數量有關
    (src/procedural.py 的混合數量與排列)→ 訓練池以原本的數量重產(每塊 CHUNK 個場;每個 seed 的第一塊與最後一塊;使用者 10/12:取樣)。
    主要的保證是 seed 不重疊(檢查 3);這裡是第二道確認。"""
    cfg = s5.load_cfg(0)
    d_t = s5.tile_size(cfg)
    s0 = (cfg.canvas - d_t) // 2
    n_t = n * s5.TILES ** 2
    crop = (lambda t: t[:, :, s0:s0 + d_t, s0:s0 + d_t])
    he = s5c._tile_hashes(crop(p41.make_home(cfg, n_t, seed=EXAM_AB["field"])))
    others = {"驗證": VAL_AB["field"], "期中考": 95_000_000, "舊測試": cfg.test_seed + s5.FIELD_SEED_OFFSET, "§十五": a6.s5e.CONFIRM_SEED}
    inter = {k: len(he & s5c._tile_hashes(crop(p41.make_home(cfg, n_t, seed=v)))) for k, v in others.items()}
    last = max(0, math.ceil(a6.N_TRAIN / s5c.CHUNK) - 1)
    nt = 0
    hit = 0
    for s in s5c.SEEDS:
        for k in sorted({0, last}):
            m = min(s5c.CHUNK, a6.N_TRAIN - k * s5c.CHUNK)                               # 同 scan_5c.make_train_fields 的分塊數量
            ht = s5c._tile_hashes(crop(p41.make_home(cfg, m * s5.TILES ** 2, seed=s5c.train_chunk_seed(s, k))))
            hit += len(he & ht)
            nt += len(ht)
    inter[f"訓練池(每個 seed 的第 0 與第 {last} 塊,原本的數量)"] = hit
    check("物體的不重疊(A / B):考試物體的非空白物體塊與驗證場、期中考、舊測試、§十五、訓練池(取樣,原本的數量)都沒有相同",
          all(v == 0 for v in inter.values()), f"考試的物體塊 {len(he)} 個;訓練池比對 {nt} 個;相同:" + "、".join(f"{k} {v}" for k, v in inter.items()))
    return all(v == 0 for v in inter.values())


@torch.no_grad()
def overlap_c(dev, n):
    """C:晶格場(相位,四捨五入到 1e-4 rad 再雜湊,不受浮點加總順序影響)。考試場(整張 F,含新缺陷)vs 驗證、smoke(同大小);
    考試場在訓練的方框大小(80 × 80)vs **全部**訓練池(原本的設定:材料隨機、方框大小;使用者 10/12)。"""
    import probe_4_2b as pb
    cfg = s5.load_cfg(0)
    pr = pb.build_probes(cfg, dev)[PROBE]
    geo = a6.Geom(cfg, pr, dev)
    dx = s8.pixel_size(pr, geo.W)
    Uc = geo.U.cpu()
    S_box = s5c.geometry(cfg)[1]

    def hashes(seeds, size, alt=True, nd=None):
        out = set()
        for i0 in range(0, len(seeds), 4096):
            sd = seeds[i0:i0 + 4096]
            mats = [s8.MATS[(i0 + i) % 2] for i in range(len(sd))] if alt else None
            f, _ = s8.make_lattice(sd, size, dev, materials=mats, newdef_seeds=None if nd is None else nd[i0:i0 + 4096],
                                   U=Uc if nd is not None else None, dx=dx)
            out |= {hashlib.md5(np.round(x.cpu().numpy(), 4).tobytes()).hexdigest() for x in f[:, 1]}
            del f
        return out

    ex = [EXAM_C["obj"] + i for i in range(n)]
    he = hashes(ex, geo.F) | hashes(ex, geo.F, nd=[EXAM_C["newdef"] + i for i in range(n)])
    vs = [VAL_C["obj"] + i for i in range(n)]
    inter = {"驗證": len(he & hashes(vs, geo.F)),
             "驗證(新缺陷)": len(he & hashes(vs, geo.F, nd=[VAL_C["newdef"] + i for i in range(n)])),
             "smoke": len(he & hashes([s8.SEED_SMOKE_SET["obj"] + i for i in range(min(n, 64))], geo.F))}
    hb = hashes(ex, S_box)
    ntr = s8.LAT_POOL_N
    inter[f"訓練池(全部 {ntr} 個,方框 {S_box})"] = len(hb & hashes([s8.SEED_TRAIN_OBJ + i for i in range(ntr)], S_box, alt=False))
    check("物體的不重疊(C):考試的晶格場(含新缺陷)與晶格的驗證場、smoke 都沒有相同;考試場在方框大小與全部訓練池(材料隨機、原本的設定)都沒有相同",
          all(v == 0 for v in inter.values()), f"考試的場 {len(he)} 個;相同:" + "、".join(f"{k} {v}" for k, v in inter.items()))
    return all(v == 0 for v in inter.values())


# ============================================================================
# 彙整流程(--run final;smoke 用驗證資料代替)
# ============================================================================
def cached(ctx, name, fn):
    """彙整的分段暫存(中斷可續跑;程式版本或設定不同 → 拒絕續跑)。"""
    p = cache_dir(ctx["root"]) / f"{name}.pt"
    key = {"md5": me_md5(), "smoke": ctx["smoke"], "n": ctx["n"], "n_val": ctx["n_val"], "manifest": ctx["manifest"]}
    if p.exists():
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if obj.get("key") != key:
            raise SystemExit(f"❌ {p} 由不同版本或設定產生:不續跑。先告訴 Claude")
        print(f"  (沿用暫存 {p.name})", flush=True)
        return obj["val"]
    val = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    torch.save({"key": key, "val": val}, tmp)
    os.replace(tmp, p)
    return val


def timing_all(dev):
    tim = a7.timing(dev)
    tim["net"] = s8.time_nets8(dev, roots_real())
    return tim


def run_final(dev, ctx):
    me = me_md5()
    smoke, root, n, nv, plan = ctx["smoke"], ctx["root"], ctx["n"], ctx["n_val"], ctx["plan"]
    S = plan["S"]
    Bs = ["64", "512"]
    lg = (lambda x: print(x, flush=True))
    where = "final exam (smoke: validation data)" if smoke else "final exam (held-out data)"
    # 1. 考試的包絡檔
    E, bad = {}, []
    for part, name in TASKS:
        p = env_json(name, root)
        if not p.exists():
            bad.append(f"{p.name} 不存在")
            continue
        J = json.load(open(p))
        m = J["meta"]
        want_base = ctx["base_ab"] if part == "ab" else ctx["base_c"]
        if m["script_md5"] != me or m["S"] != S or m["n"] != n or not J["checks_ok"] or bool(m["smoke"]) != smoke or m["base"] != want_base:
            bad.append(f"{p.name} 的版本、停止點、場數、資料或檢查不符")
            continue
        E[name] = J["env"]
    check(f"{len(TASKS)} 個包絡檔都在、同一版程式、停止點與場數一致、檢查全過", not bad, ";".join(bad))
    J8, ok8 = load_s8()
    J8c, okc = load_c8()
    fc8 = J8["verdict"]["fail_c"]
    Ev, badv = load_val_env(fc8)
    check("驗證的結果檔(scan8.json、scan8c.json、階段七 / 八的 17 個包絡檔)完整、版本正確", ok8 and okc and not badv, ";".join(badv))
    if bad or badv or not (ok8 and okc):
        return None
    dose = s8.load_dose()
    t0 = time.time()
    # 2. 計時、T_c
    lg("\n  計時(ms / 整張)")
    tim = cached(ctx, "timing", lambda: timing_all(dev))
    Tc = {B: t_c(S, tim[B]) for B in Bs}
    S_val = stops_val(a7.ITERS)
    Tcv = {B: t_c(S_val, tim[B]) for B in Bs}
    lg(f"  T_c:b64 {Tc['64'][0]:.2f} ms、b512 {Tc['512'][0]:.2f} ms(經過 {time.time() - t0:.0f} 秒)")
    # 3. 網路(考試、驗證)
    rr = roots_real()

    def nets_ab(base, nn, want_keep):
        kp = {} if want_keep else None
        R, v = eval_nets_ab(dev, base, nn, rr, kp, lg)
        return R, v, kp

    def nets_c(base, nn, want_keep):
        kp = {} if want_keep else None
        R, info = eval_nets_c(dev, base, nn, rr, kp, lg)
        return R, info, kp

    lg(f"\n  網路(考試資料):A / B 13 個條件、C 5 份考卷 × 3 seeds × 5 個網路({n} 個場)")
    Rab, valid_ab, keep_ab = cached(ctx, "nets_ab", lambda: nets_ab(ctx["base_ab"], n, True))
    Rc, info_c, keep_c = cached(ctx, "nets_c", lambda: nets_c(ctx["base_c"], n, True))
    lg(f"\n  網路(驗證資料,可重現性的分母;{nv} 個場)")
    Rab_v, valid_ab_v, _ = cached(ctx, "nets_ab_val", lambda: nets_ab(VAL_AB, nv, False))
    Rc_v, info_c_v, _ = cached(ctx, "nets_c_val", lambda: nets_c(VAL_C, nv, False))
    if not smoke:
        dm = max(reldiff(Rab_v[c][nm]["nerr_sr"][i], J8["orig"][c][nm]["nerr_sr"][i]) for c in AB_EVAL for nm in NETS for i in range(len(SEEDS)))
        dc = max(reldiff(float(np.where(np.isfinite(x), x, fc8[e]).mean()), J8["verdict"]["exam"][e]["nets"][nm]["nerr_c"][i])
                 for e in EXAMS for nm in NETS for i, x in enumerate(Rc_v[e][nm]["c"]))
        check(f"重現:驗證資料上 5 個網路的 nerr_sr(13 個條件)、nerr_c(5 份考卷)= 階段八的結果檔(3 seeds;相對差 < {REPRO_TOL:.0e})",
              max(dm, dc) < REPRO_TOL, f"最大差 {dm:.1e} / {dc:.1e}")
    # 4. 失敗值、CRLB
    fc = {e: s8.fail_value(E[e], Rc, e) for e in EXAMS}
    for e in EXAMS:
        for conf in E[e].values():
            for rec in conf.values():
                s8.finalize_rec(rec, fc[e])
    lg(f"\n  CRLB(考試資料;{ctx['n_crlb']} 個物體 × 2 組)")
    def crlb_checked():                                                              # 核對通過才回傳(→ 才會存進暫存)
        A_ = crlb_part(dev, ctx["base_c"], ctx["n_crlb"], lg)
        if not validate_crlb(A_, ctx["n_crlb"]):
            raise SystemExit("❌ [CRLB] 核對未通過:不存暫存、停止。把輸出貼給 Claude")
        return A_

    A = cached(ctx, "crlb", crlb_checked)
    if not validate_crlb(A, ctx["n_crlb"]):                                           # 讀回暫存時也重新核對
        return None
    crl = crlb_table(A, dose)
    # 5. 各部分的數字 → 迭代法的關鍵設定(重算)
    Eab = {c: E[c] for c in AB_ENV}
    Ec = {e: E[e] for e in EXAMS}
    AB = rows_ab(Rab, valid_ab, Eab, tim, S, Tc)
    C = rows_c(Rc, info_c, Ec, tim, S, Tc, fc, crl)
    V0 = {"ab": AB, "c": C}
    lg("\n  迭代法的關鍵設定(重算,3 seeds):SSIM、核對包絡檔、重建圖")
    IT_ab, w_ab, img_ab = cached(ctx, "iter_ab", lambda: iter_key_ab(dev, Eab, V0, n, ctx["base_ab"], FIG_AB, lg))
    IT_c, w_c, img_c = cached(ctx, "iter_c", lambda: iter_key_c(dev, Ec, V0, fc, n, ctx["base_c"], [MAIN_EXAM], lg))
    print(f"  描述性核對(不阻擋):重算的迭代法 vs 包絡檔,最大相對差 A / B {w_ab:.1e}、C {w_c:.1e}")
    keep = dict(keep_ab)
    keep.update(keep_c)
    for k, im in list(img_ab.items()) + list(img_c.items()):
        keep[k]["img"].update(im)
    P = {"tim": tim, "S": S, "S_val": S_val, "Tc": Tc, "Tc_val": Tcv, "plan": plan, "Rab": Rab, "valid_ab": valid_ab, "Rab_val": Rab_v,
         "valid_ab_val": valid_ab_v, "Rc": Rc, "info_c": info_c, "Rc_val": Rc_v, "Eab": Eab, "Ec": Ec,
         "Eab_val": {c: Ev[c] for c in AB_ENV}, "Ec_val": {e: Ev[e] for e in EXAMS}, "fc": fc, "fc_val": fc8, "crlb": crl,
         "crlb_val": J8c["report"]["crlb"][MAIN_EXAM]["crlb_c"], "IT_ab": IT_ab, "IT_c": IT_c, "phmax": s5.load_cfg(0).phase_max}
    V = report(P, AB, C, smoke)
    final = final_json(root)
    part = Path(str(final) + ".partial")                                             # 進行中 / 檢查未過:只寫暫存檔;全部通過才產生最終檔
    res = {"verdict": V, "timing": tim, "crlb_rows": A, "iter_key": {"ab": IT_ab, "c": IT_c, "recompute_maxrel": [w_ab, w_c]},
           "stats": {e: info_c[e]["stats"] for e in EXAMS}, "n": n, "n_val": nv, "checks_ok": all_ok(), "smoke": smoke, "script_md5": me,
           "s8_md5": S8_MD5, "c8_md5": C8_MD5, "elapsed_s": time.time() - t0, "complete": False}
    dump_json(res, part)
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        figures(V, P, keep, ctx["fig_dir"], where)
    except Exception as ex:                                               # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"畫圖完成(判讀不受影響,已存於 {final.name})", False, f"{type(ex).__name__}: {ex}")
    res["checks_ok"] = all_ok()
    res["complete"] = bool(all_ok())
    if res["complete"]:
        dump_json(res, final)
        part.unlink(missing_ok=True)
        print(f"  結果:{final}(經過 {time.time() - t0:.0f} 秒)")
    else:
        dump_json(res, part)
        print(f"  ❌ 有檢查未通過:結果只存在 {part}(沒有產生最終檔;經過 {time.time() - t0:.0f} 秒)")
    return res


# ============================================================================
# smoke(協定 §九:驗證資料代替考試資料、縮小版;印出時間預估;寫下 PASSED)
# ============================================================================
def smoke(dev, me):
    print("#" * 70)
    print(f"迷你全流程(--smoke):內建檢查、{len(TASKS)} 個包絡 job({a7.SMOKE_FIELDS} 個場、停止點 ≤ {NS_X * max(a7.SMOKE_ITERS)})、彙整;"
          f"資料 = 驗證資料;輸出到 {SMOKE_DIR}")
    print("#" * 70)
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)                                          # 只刪這個 smoke 專用資料夾
    SMOKE_DIR.mkdir(parents=True)
    t0 = time.time()
    man0 = manifest()                                                                 # smoke 開始時的快照:結束時必須相同才寫 PASSED
    plan_real = checks_full(dev)
    if not all_ok():
        print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    tchk = time.time() - t0
    plan = plan_of(True, timing512())
    n = a7.SMOKE_FIELDS
    te = {}
    for k, (part, name) in enumerate(TASKS):
        t1 = time.time()
        run_task(dev, k, n, VAL_AB if part == "ab" else VAL_C, plan, SMOKE_DIR, smoke=True)
        te[name] = time.time() - t1
    ctx = {"smoke": True, "root": SMOKE_DIR, "n": n, "n_val": n, "plan": plan, "base_ab": VAL_AB, "base_c": VAL_C,
           "fig_dir": SMOKE_DIR / "figs", "n_crlb": 2 if QUICK else min(n, 8), "manifest": manifest_key(man0)}
    tf = time.time()
    res = run_final(dev, ctx)
    tfin = time.time() - tf
    nf = s5b.N_FIELDS or s5.load_cfg(0).eval_n
    it_r = sum(st[-1] for st in plan_real["S"].values())
    it_s = sum(st[-1] for st in plan["S"].values())
    k_it = it_r / max(it_s, 1)
    est = {nm: v * (nf / n) * k_it for nm, v in te.items()}
    ab_h = sum(est[c] for c in AB_ENV) / 3600
    c_h = sum(est[e] for e in EXAMS) / 3600
    print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒(內建檢查 {tchk:.0f};包絡 " + "、".join(f"{k} {v:.0f}" for k, v in te.items())
          + f" 秒;彙整 {tfin:.0f} 秒)")
    print(f"  粗估正式的包絡 job(線性外推:場數 × {nf / n:.0f}、迭代次數 × {k_it:.0f};偏保守,實際以 log 為準):"
          f"A / B 每個約 {ab_h / len(AB_ENV):.1f} 小時(合計 {ab_h:.1f})、C 每份約 {c_h / len(EXAMS):.1f} 小時(合計 {c_h:.1f});"
          f"單一 job 最長約 {max(est.values()) / 3600:.1f} 小時(run_final_env.sh 的時限 5 小時;超過會中斷,重送同一個編號會續跑)")
    if all_ok() and res is not None:
        man1 = manifest()
        if man1 != man0:
            changed = sorted(k for k in set(man0) | set(man1) if man0.get(k) != man1.get(k))
            print(f"\n❌ smoke 進行中,依賴的程式或輸入檔改變了:{changed[:5]} → 不寫 PASSED,把輸出貼給 Claude")
            sys.exit(1)
        json.dump({"script_md5": me, "manifest": man0, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
        print("\n✅ 迷你全流程全部通過 → 可以送出 run_final_env.sh,全部成功後再送 run_final_agg.sh")
        sys.exit(0)
    print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
    sys.exit(1)


def need_passed(me):
    if not PASSED.exists():
        raise SystemExit(f"❌ 找不到 {PASSED}:先在 dev 節點跑 python scan_final.py --smoke,全部通過才可送件")
    J = json.load(open(PASSED))
    if J["script_md5"] != me:
        raise SystemExit(f"❌ smoke 通過時的 scan_final.py(md5 {J['script_md5']})與現在的({me})不同:用現在的版本重跑 --smoke")
    now = manifest()
    diff = sorted(k for k in set(now) | set(J.get("manifest", {})) if now.get(k) != J.get("manifest", {}).get(k))
    if diff:
        raise SystemExit(f"❌ smoke 通過之後,依賴的程式或輸入檔改變了:{diff[:5]}{' …' if len(diff) > 5 else ''} → 先告訴 Claude")
    print(f"✅ smoke 已通過(同一版本;依賴與輸入 {len(now)} 個檔案相同;{J['time']})")


# ============================================================================
# 主程式
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", type=str, help="0..16(包絡 job)或 final(彙整)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = me_md5()
    print(f"scan_final.py md5 {me};GPU {a7.gpu_name(dev)}")
    if not (a.check or a.smoke or a.run is not None):
        ap.print_help()
        return
    if not check_files("run" if a.run is not None else "check"):
        print("\n❌ 缺檔案或版本不同,停下來")
        sys.exit(1)
    if a.check:
        checks_full(dev)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.smoke:
        smoke(dev, me)
    # ---- 正式(期末考資料只用一次)----
    need_passed(me)
    is_final = a.run == "final"
    if not is_final and not (a.run.isdigit() and 0 <= int(a.run) < len(TASKS)):
        raise SystemExit(f"❌ --run 只接受 0..{len(TASKS) - 1} 或 final")
    n = s5b.N_FIELDS or s5.load_cfg(0).eval_n
    out = final_json()
    if out.exists():                                                                  # 最終檔只在全部通過時產生 → 存在即代表考試已完成
        raise SystemExit(f"❌ {out} 已存在:期末考已經完成。為避免覆蓋或再次使用考試資料,先告訴 Claude")
    if not is_final:
        k = int(a.run)
        p = env_json(task_name(k))
        if p.exists():
            raise SystemExit(f"❌ {p} 已存在:這個 job 已經跑完。為避免覆蓋,先告訴 Claude")
    plan = checks_light(dev)
    if not all_ok():
        print("\n❌ 內建檢查未通過 → 沒有執行。把輸出貼給 Claude")
        sys.exit(1)
    enable_exam()
    if is_final:
        ok = overlap_ab(n) and overlap_c(dev, n)
    else:
        ok = overlap_ab(n) if TASKS[k][0] == "ab" else overlap_c(dev, n)
    if not ok:
        print("\n❌ 考試物體與既有資料重疊 → 停止。把輸出貼給 Claude")
        sys.exit(1)
    if is_final:
        ctx = {"smoke": False, "root": RUN_ROOT, "n": n, "n_val": n, "plan": plan, "base_ab": EXAM_AB, "base_c": EXAM_C,
               "fig_dir": FIG_DIR, "n_crlb": n, "manifest": manifest_key(manifest())}
        res = run_final(dev, ctx)
        print("\n" + ("✅ 內建檢查全部通過" if all_ok() and res else "❌ 有項目未通過(判讀先不要採信)"))
        print("把完整輸出與 figs_final/ 的圖傳給 Claude")
        sys.exit(0 if all_ok() and res else 1)
    part = TASKS[k][0]
    run_task(dev, k, n, EXAM_AB if part == "ab" else EXAM_C, plan, RUN_ROOT, smoke=False)
    print("\n" + (f"✅ {task_name(k)} 完成、檢查全過" if all_ok() else f"❌ {task_name(k)} 有檢查未通過"))
    sys.exit(0 if all_ok() else 1)


if __name__ == "__main__":
    main()
