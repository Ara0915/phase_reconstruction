#!/usr/bin/env python
"""期中考(階段五協定 §二十;優化方案 §九)。不重新訓練,只考一次。

陣容(期中考之前寫定):主要 P6B4(B + 4 級);事先宣告的對照 P6B(B + 3 級)、B(舊快線)。
期中考場:512 個場,物體 seed 95,000,000;雜訊 96,000,000 + 1000 × seed;迭代法的隨機起點 同上 + 1。
對手:在期中考場上重跑整套迭代法(同 5-2b / §十五),重算包絡;同一 job 重新計時。
另附視覺化(§20.7):seed 0 的網路 + 同時間的迭代法,事先寫定的樣本(第 10 / 50 / 90 百分位、最差、6 個隨機)。

用法(需在計算節點執行):
    python scan_5i.py --check   # 只跑內建檢查(只用驗證場,不碰期中考場)
    python scan_5i.py --smoke   # 送件前的迷你全流程:用「驗證場」代替期中考場跑完整流程(dev 節點);通過才可正式執行
    python scan_5i.py           # 正式的期中考(run_scan5i_mid.sh);結果檔已存在時拒絕重跑
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
import scan_5h as h                                                  # noqa: E402  (載入 scan_5g 並讓 P6B4 可建構)

g = h.g
s5, s5b, s5c, s5d, s5e = h.s5, h.s5b, h.s5c, h.s5d, h.s5e
RUN_ROOT = s5.RUN_ROOT
QUICK = h.QUICK
PROBE = h.PROBE
SEEDS = h.SEEDS
N_TRAIN = h.N_TRAIN

PRIMARY = "P6B4"
LINEUP = ["B", "P6B", "P6B4"]                 # 顯示順序:舊快線 → 3 級 → 4 級(主要)
MID_FIELD_SEED = 95_000_000
MID_NOISE_BASE = 96_000_000
FINAL_FIELD_SEED = 97_000_000                 # 期末考(理想條件)保留;只用在不重疊檢查
REPRO_TOL = 1e-5
PS_TOL = 1e-6                                 # 相對於 max(1, 值);float32 的加總順序差異遠小於此
VIZ_TOL = 1e-4
REPRO_CONFIGS = ["ePIE-C|rand", "AP-C|zero"]  # + K-HIO(同 §十五 的檢查)
PCTS = [10, 50, 90]
RAND_SEED = 2026
N_RANDOM = 6
MARGIN = 6
AMP_MIN = 0.05
SUF = "_quick" if QUICK else ""
VAL_ENV = s5e.VAL_ENV
P5_RAW = RUN_ROOT / f"scan5h_p5_raw{SUF}.json"
MID_ENV = RUN_ROOT / f"scan5i_mid_envelope{SUF}.json"
MID_RAW = RUN_ROOT / f"scan5i_mid_raw{SUF}.json"
MID_JSON = RUN_ROOT / f"scan5i_mid{SUF}.json"
MD5_JSON = RUN_ROOT / f"scan5i_mid_md5{SUF}.json"
SMOKE_DIR = RUN_ROOT / f"scan5i_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
FIG_DIR = Path("figs_scan5i")
COL = {"iter_b64": "#9a9994", "iter_b512": "#0b0b0b", "B": "#2a78d6", "P6B": "#0b6e4f", "P6B4": "#d6452a"}

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and h.all_ok()


def absdiff(a, b):
    """|a − b|;非有限值(NaN / inf)一律當成 inf,讓比較不會因 NaN 而誤判為通過。"""
    d = abs(float(a) - float(b))
    return d if np.isfinite(d) else float("inf")


def ps_close(mean_ps, fm):
    """逐樣本平均 vs field_metrics:差 / max(1, |值|) < PS_TOL(誤差 ≤ 1 時就是絕對差 < 1e-6)。"""
    d = absdiff(mean_ps, fm)
    return d / max(1.0, abs(float(fm))) if np.isfinite(d) else float("inf")


def mid_seeds(cfg, s):
    return {"field": MID_FIELD_SEED, "noise": MID_NOISE_BASE + 1000 * s, "init": MID_NOISE_BASE + 1000 * s + 1}


def md5(path):
    m = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


def model_dir(x, s):
    return s5c.run_dir(x, PROBE, s, N_TRAIN)


# ============================================================================
# 逐樣本誤差、整體相位對齊、物體類型(與 scan_5g_viz.py 相同的做法)
# ============================================================================
def per_sample(est, O, R0):
    """逐樣本 nerr_ph(與 scan_5.field_metrics 同一公式);空樣本 = NaN。"""
    m = R0.to(est.real.dtype)
    a, t = est * m, O * m
    E = (t.abs() ** 2).sum((1, 2))
    cross = (a * t.conj()).sum((1, 2)).abs()
    v = ((a.abs() ** 2).sum((1, 2)) + E - 2 * cross) / E.clamp_min(1e-12)
    return torch.where(E > 1e-12, v, torch.full_like(v, float("nan")))


def align_phase(est, O, R0):
    """乘上最佳的整體相位(R0 上),= 評分時的對齊方式。"""
    m = R0.to(est.real.dtype)
    c = ((O * m) * (est * m).conj()).sum((1, 2))
    return est * torch.exp(1j * torch.angle(c))[:, None, None]


def kind_labels(cfg, n_obj, seed):
    """每個物體的類型編號(同 src.procedural.make_procedural 的排列:各類型依比例串接,再以 seed 洗牌)。"""
    kinds = list(cfg.proc_kinds)
    w = list(cfg.proc_weights)[:len(kinds)]
    per = [int(n_obj * x / sum(w)) for x in w]
    per[-1] += n_obj - sum(per)
    lab = torch.cat([torch.full((p,), i, dtype=torch.long) for i, p in enumerate(per) if p > 0])
    perm = torch.randperm(n_obj, generator=torch.Generator().manual_seed(seed))
    return lab[perm], kinds


def tile_kind_map(cfg, n, seed, F):
    """每個場的逐像素類型圖 [n, F, F](同 scan_5.make_fields 的拼貼與循環平移)。"""
    d, T = s5.tile_size(cfg), s5.TILES
    lab, kinds = kind_labels(cfg, n * T * T, seed)
    lab = lab.reshape(n, T, T)
    km = lab[:, :, None, :, None].expand(n, T, d, T, d).reshape(n, T * d, T * d)
    sh = torch.randint(0, d, (n, 2), generator=torch.Generator().manual_seed(seed + 1))
    km = torch.stack([torch.roll(km[i], (int(sh[i, 0]), int(sh[i, 1])), dims=(-2, -1)) for i in range(n)])
    return km, kinds


def kinds_check(cfg, fseed):
    """類型推算的核對:照 make_procedural 的做法逐類型產生、洗牌後,須與 make_home 逐位元相同。"""
    import probe_4_1 as p41
    from src.procedural import GENERATORS
    n_chk = 60
    lab_c, kinds = kind_labels(cfg, n_chk, fseed)
    w = list(cfg.proc_weights)[:len(kinds)]
    per = [int(n_chk * x / sum(w)) for x in w]
    per[-1] += n_chk - sum(per)
    parts = [GENERATORS[k](q, cfg, seed=fseed + 1000 * j, target_support=cfg.match_support_to,
                           contrast_gamma=cfg.proc_contrast_gamma) for j, (k, q) in enumerate(zip(kinds, per)) if q > 0]
    manual = torch.cat(parts)[torch.randperm(n_chk, generator=torch.Generator().manual_seed(fseed))]
    return bool(torch.equal(manual, p41.make_home(cfg, n_chk, seed=fseed)))


# ============================================================================
# 迭代法:重算包絡選到的設定(同 scan_5d.measure_envelope 的做法;K-HIO 只算一次,多個設定共用)
# ============================================================================
@torch.no_grad()
def iterative_outputs(confs, d, dev):
    import probe_4_2b as pb
    cfg, pr, cp, bs, counts, st, n, F = (d[k] for k in ("cfg", "pr", "cp", "bs", "counts", "st", "n", "F"))
    ones = torch.ones(d["W"], d["W"], device=dev)
    kit = sorted(set(s5b.K_ITERS) | set(s5b.KHIO_N.values()))
    init = s5.k_init(counts[s5b.CENTER], cp, d["init"], dev)
    k_out = pb.hio_probe(init, counts[s5b.CENTER], bs, cp, kit, "K", pr, ones)
    outs = []
    for conf, it in confs:
        if conf == "K-HIO":
            outs.append(s5.k_to_field(k_out[it], pr, cfg, st[s5b.CENTER], F))
            continue
        meth, kind = conf.split("|")
        O0 = s5b.make_init(kind, n, F, cfg, cp, pr, bs, counts, st, d["init"], dev, k_out=k_out)
        outs.append(O0 if it == 0 else s5b.run_method(meth, counts, bs, cp, pr, st, [it], O0, d["init"])[it])
    del k_out
    return outs


# ============================================================================
# 網路(同 scan_5h.evaluate 的載入與推論);可同時算視覺化需要的迭代法
# ============================================================================
@torch.no_grad()
def run_nets(seeds_fn, dev, iter_confs=None):
    """回傳 (res, ps, keep, worst_ps)。
    res[x] = 每個 seed 的 {net: field_metrics, stages, alphas, params, train_nerr_ph, train_sec}
    ps[key] = 每個 seed 的逐樣本 nerr_ph(numpy;key = 網路名,或 iter_b64 / iter_b512 當 iter_confs 給定時)
    keep = seed 0 的估計(視覺化用)與共同物件"""
    res = {x: [] for x in LINEUP}
    ps = {x: [] for x in LINEUP}
    if iter_confs:
        ps.update({k: [] for k in iter_confs})
    keep = {}
    worst_ps = 0.0
    for s in SEEDS:
        d = s5d.setup_seed(s, seeds_fn, dev)
        cfg, cp, bs, O, R0, F, counts = d["cfg"], d["cp"], d["bs"], d["O"], d["R0"], d["F"], d["counts"]
        for x in LINEUP:
            md = model_dir(x, s)
            meta = json.load(open(md / "result.json"))
            net = h.build_model(x, cfg, d["pr"]).to(dev)
            net.load_state_dict(torch.load(md / "final.pt", map_location=dev))
            net.eval()
            rec = {"params": meta["params"], "train_nerr_ph": meta["train_nerr_ph"], "train_sec": meta.get("train_sec")}
            if x == "B":
                pred = g.predict(net, counts, bs, meta["norm"], x, cp)
            else:
                O0, outs = g.predict(net, counts, bs, meta["norm"], x, cp, all_stages=True)
                pred = outs[-1]
                st_fields = [s5c.to_field(O0, cfg, F)] + [s5c.to_field(torch.polar(o[:, 0], o[:, 1]), cfg, F) for o in outs]
                rec["stages"] = [s5.field_metrics(f, O, R0)["nerr_ph"] for f in st_fields]
                rec["alphas"] = net.alphas()
                if s == SEEDS[0] and x == PRIMARY:
                    keep["stages"] = [f.cpu() for f in st_fields]
                del st_fields
            est = s5c.to_field(torch.polar(pred[:, 0], pred[:, 1]), cfg, F)
            rec["net"] = s5.field_metrics(est, O, R0)
            e = per_sample(est, O, R0).cpu().numpy()
            worst_ps = max(worst_ps, ps_close(np.nanmean(e), rec["net"]["nerr_ph"]))
            res[x].append(rec)
            ps[x].append(e)
            if s == SEEDS[0]:
                keep[x] = est.cpu()
            del net, pred, est
        if iter_confs:
            outs = iterative_outputs(list(iter_confs.values()), d, dev)
            for k, est in zip(iter_confs, outs):
                ps[k].append(per_sample(est, O, R0).cpu().numpy())
                if s == SEEDS[0]:
                    keep[k] = est.cpu()
            del outs
        if s == SEEDS[0]:
            keep["d"] = {k: d[k] for k in ("cfg", "F", "n")}
            keep["O"], keep["R0"] = O.cpu(), R0.cpu()
        print(f"  [seed {s}] 完成", flush=True)
    return res, ps, keep, worst_ps


def vals(res, x):
    return np.array([r["net"]["nerr_ph"] for r in res[x]], float)


# ============================================================================
# 內建檢查(§20.4;只用驗證場,不碰期中考場)
# ============================================================================
@torch.no_grad()
def checks(dev):
    import probe_4_1 as p41
    torch.backends.cudnn.benchmark = True                                   # 同驗證場評估時的設定
    print("=" * 70)
    print("期中考:內建檢查(只用驗證場)")
    print("=" * 70)
    cfg = s5.load_cfg(0)

    # (1) 資料不重疊
    d_t = s5.tile_size(cfg)
    s0 = (cfg.canvas - d_t) // 2
    n_t = (s5b.N_FIELDS or cfg.eval_n) * s5.TILES ** 2
    crop = (lambda t: t[:, :, s0:s0 + d_t, s0:s0 + d_t])
    hm = s5c._tile_hashes(crop(p41.make_home(cfg, n_t, seed=MID_FIELD_SEED)))
    others = {"舊測試": cfg.test_seed + s5.FIELD_SEED_OFFSET, "§十五 測試": s5e.CONFIRM_SEED, "驗證": s5e.VAL_FIELD_SEED}
    inter = {k: len(hm & s5c._tile_hashes(crop(p41.make_home(cfg, n_t, seed=v)))) for k, v in others.items()}
    tr = torch.cat([crop(p41.make_home(cfg, 64 * s5.TILES ** 2, seed=s5c.train_chunk_seed(s, k)))
                    for s in s5c.SEEDS for k in (0, max(0, N_TRAIN // s5c.CHUNK - 1))])
    inter["訓練(抽樣)"] = len(hm & s5c._tile_hashes(tr))
    tr_max = s5c.train_chunk_seed(max(s5c.SEEDS), N_TRAIN // s5c.CHUNK) + 3000
    used = list(others.values()) + [p41.CALIB_SEED, p41.CALIB_SEED + 1, s5e.VAL_NOISE_BASE]
    far = MID_FIELD_SEED - tr_max > 1_000_000 and all(abs(MID_FIELD_SEED - u) > 1_000_000 for u in used)
    noise = [mid_seeds(cfg, s)[k] for s in SEEDS for k in ("noise", "init")]
    noise_ok = all(MID_NOISE_BASE <= v < FINAL_FIELD_SEED and v - MID_FIELD_SEED >= 1_000_000 for v in noise)
    check("資料不重疊:期中考場的 seed 遠離訓練 / 舊測試 / §十五 / 驗證 / 校準;非空白物體塊皆無相同;雜訊 seed 不落在期末考的範圍",
          far and noise_ok and all(v == 0 for v in inter.values()),
          f"期中考 seed {MID_FIELD_SEED:,}(雜訊 {min(noise):,}–{max(noise):,});訓練 seed 上限約 {tr_max:,};相同的塊:"
          + "、".join(f"{k} {v}" for k, v in inter.items()))

    # (2) 迭代法程式一致(驗證場)
    ref = json.load(open(VAL_ENV))["envelope"][PROBE]
    rep = s5d.measure_envelope(s5e.val_seeds, dev, configs=REPRO_CONFIGS)[PROBE]
    worst, cnt = 0.0, 0
    runs_ok = [ref[k]["run"] == sd["run"] for k, sd in enumerate(rep)]
    for k, sd in enumerate(rep):
        for conf, rec in sd["rec"].items():
            if conf.startswith("init|"):
                continue
            for it, m in rec.items():
                o = ref[k]["rec"].get(conf, {}).get(it)
                if o is None:
                    continue
                worst = max(worst, absdiff(o["nerr_ph"], m["nerr_ph"]))
                cnt += 1
    want = len(SEEDS) * (len(s5b.K_ITERS) + len(REPRO_CONFIGS) * (len(s5b.ITERS) + 1))
    check(f"迭代法程式一致:驗證場上重跑 {' / '.join(REPRO_CONFIGS)} / K-HIO = scan5e_val_envelope.json(< {REPRO_TOL:.0e})",
          all(runs_ok) and cnt == want and worst < REPRO_TOL, f"比對 {cnt} / {want} 個點,最大差 {worst:.1e}")

    # (3) 網路程式一致(驗證場)+ (4) 逐樣本計算一致
    old = json.load(open(P5_RAW))["results"]
    res, _, _, worst_ps = run_nets(s5e.val_seeds, dev)
    diffs = {}
    for x in LINEUP:
        o = np.array([r["net"]["nerr_ph"] for r in old[x]], float)
        diffs[x] = max(absdiff(a_, b_) for a_, b_ in zip(vals(res, x), o)) if len(o) == len(SEEDS) else float("inf")
    check(f"網路程式一致:{'、'.join(LINEUP)} 在驗證場上的逐 seed nerr_ph = §19.11 的評估(scan5h_p5_raw.json;< {REPRO_TOL:.0e})",
          all(v < REPRO_TOL for v in diffs.values()), " / ".join(f"{k} {v:.1e}" for k, v in diffs.items()))
    check(f"逐樣本計算一致:逐樣本 nerr_ph 的平均 = field_metrics(差 / max(1, 值) < {PS_TOL:.0e})", worst_ps < PS_TOL,
          f"最大差 {worst_ps:.1e}")
    print(f"\n     裝置:{dev}")
    return {x: [r["net"]["nerr_ph"] for r in old[x]] for x in LINEUP}


# ============================================================================
# 判讀(§20.6,結果出來前寫定)
# ============================================================================
def report(env, res, tim, val, smoke):
    where = "驗證場(smoke 代替期中考場)" if smoke else "期中考場(seed 95,000,000)"
    v = {}
    Bs = ["64", "512"]
    if smoke:
        print("\n" + "!" * 100)
        print("迷你流程:以下用「驗證場」代替期中考場 → 數字應重現 §19.11(P6B4 0.0195、P6B 0.0375、B 0.2459),不是期中考的結果")
        print("!" * 100)
    print("\n" + "=" * 100)
    print("計時(ms / 張;本 job 重新量;網路含輸入前處理)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {tim[B][k]:.4f}" for k in
                                         ["ePIE-C", "AP-C", f"K-HIO:{PROBE}"] + [f"net:{x}" for x in LINEUP]))

    print("\n" + "=" * 100)
    print(f"{where}:網路(DEF2,R0 的 nerr_ph,3 seeds)與 vs 包絡")
    print("=" * 100)
    for x in LINEUP:
        a = vals(res, x)
        va = np.array(val[x], float)
        st = s5b.ratio(a, va)
        tr = np.mean([r["train_nerr_ph"] for r in res[x]])
        tag = "(主要)" if x == PRIMARY else "(對照)"
        print(f"  {x}{tag}:{a.mean():.4f}({' '.join(f'{y:.4f}' for y in a)})  驗證場 {va.mean():.4f}"
              f"  → (m4) 本場 / 驗證場 比值 {st[1]:.3f}(z {st[2]:+.1f})  訓練場 {tr:.4f}  參數 {res[x][0]['params']:,}")
        gv = {"nerr": list(a), "val": list(va), "vs_val": list(st)}
        if "stages" in res[x][0]:
            sm = np.array([r["stages"] for r in res[x]]).mean(0)
            gv["stages"] = sm.tolist()
            gv["alphas"] = [r["alphas"] for r in res[x]]
            print("     各級 nerr_ph(0 = 起點):" + " → ".join(f"{y:.4f}" for y in sm))
        for B in Bs:
            tn = tim[B][f"net:{x}"]
            e = s5b.envelope(env, PROBE, tn, tim[B])
            r = s5b.ratio(a, e[2])
            gv[B] = {"t_net": tn, "opp": [e[0], e[1], list(e[2]), e[3]], "ratio": list(r)}
            print(f"     batch {B}:{tn:.4f} ms vs 對手 {s5b.fmt_conf(e[0])}({e[1]} 次){e[2].mean():.4f}"
                  f" → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}")
        gv["useful"] = gv["64"]["ratio"][0] == "較準"
        gv["robust"] = gv["useful"] and gv["512"]["ratio"][0] == "較準"
        print(f"     ▶ {'穩健有用(batch 64 與 512 都較準)' if gv['robust'] else ('有用(只在 batch 64)' if gv['useful'] else '未達有用')}")
        v[x] = gv

    def pair(a_x, b_x):
        r = s5b.ratio(vals(res, a_x), vals(res, b_x))
        return r, tim["64"][f"net:{a_x}"] / tim["64"][f"net:{b_x}"], tim["512"][f"net:{a_x}"] / tim["512"][f"net:{b_x}"]

    print("\n" + "=" * 100)
    print("配對比較(比值 = 逐 seed 誤差比值的幾何平均,< 1 = 前者較準)")
    print("=" * 100)
    for a_x, b_x, tag in [("P6B4", "P6B", "(m2)"), ("P6B4", "B", "(m3)"), ("P6B", "B", "(m3)")]:
        r, t64, t512 = pair(a_x, b_x)
        print(f"  {tag} {a_x} vs {b_x}:誤差比值 {r[1]:.3f}(z {r[2]:+.1f});推論時間 × {t64:.2f}(b64)/ × {t512:.2f}(b512)")
        v[f"{a_x}_vs_{b_x}"] = {"ratio": list(r), "t64": t64, "t512": t512}

    m1 = v[PRIMARY]["robust"]
    r2 = v["P6B4_vs_P6B"]["ratio"]
    m2 = bool(r2[1] < 1 and r2[2] < -2)
    m4 = v[PRIMARY]["vs_val"][0] == "較差"
    v.update({"m1": m1, "m2": m2, "m4_optimistic": m4})
    print("\n" + "=" * 100)
    print("判讀(協定 §20.6,結果出來前寫定)")
    print("=" * 100)
    print(f"  (m1) P6B4 穩健有用(b64 與 b512 都較準):{'✅' if m1 else '❌'}")
    print(f"  (m2) P6B4 比 P6B 準(比值 < 1 且 z < −2):{'✅' if m2 else '❌'}(比值 {r2[1]:.3f},z {r2[2]:+.1f})")
    print(f"  (m3) 整個優化期:P6B4 vs B 比值 {v['P6B4_vs_B']['ratio'][1]:.3f};P6B vs B 比值 {v['P6B_vs_B']['ratio'][1]:.3f}(描述)")
    vv = v[PRIMARY]["vs_val"]
    print(f"  (m4) P6B4 本場 / 驗證場 比值 {vv[1]:.3f}(z {vv[2]:+.1f})"
          + (" → ⚠️ 達「較差」(≥ 1.25 且 z > 2):結論須註明「驗證場的結果偏樂觀」" if m4 else " → 未達「較差」"))
    print("  (m5) 對照組:" + ";".join(f"{x} {'穩健有用' if v[x]['robust'] else ('只在 b64 有用' if v[x]['useful'] else '未達有用')}"
                                   for x in LINEUP if x != PRIMARY))
    if m1 and m2:
        concl = "期中考確認:P6B4 為主力,優化成果(含第 4 級)在新資料上成立"
    elif m1:
        concl = "P6B4 在新資料上仍穩健有用;第 4 級的額外優勢未在新資料上確認(P6B4 仍為主力,不看期中考結果換網路)"
    else:
        concl = "優化的結論未在新資料上確認;照實回報,與使用者討論下一步(不在期中考場上調整任何東西)"
    if m4:
        concl += ";驗證場的結果偏樂觀"
    v["conclusion"] = concl
    print(f"\n  ▶ 結論(事先寫定的寫法):{concl}" + ("  [smoke:無意義]" if smoke else ""))
    return v


# ============================================================================
# 視覺化(§20.7;描述,不改變判讀)
# ============================================================================
def visualize(ps, keep, labels, fseed, out_dir, where):
    plt = s5._plt()
    if plt is None:
        check("matplotlib 可用(視覺化需要)", False)
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, F = keep["d"]["cfg"], keep["d"]["F"]
    O, R0 = keep["O"], keep["R0"]
    order = ["iter_b64", "iter_b512", "B", "P6B", PRIMARY]
    err0 = {k: ps[k][0] for k in order}                                     # seed 0
    valid = (((O.abs() ** 2) * R0).sum((1, 2)) > 1e-12).numpy()             # 只依真值(R0 內有物體);與結果無關
    rank = np.nan_to_num(err0[PRIMARY], nan=np.inf)                         # 萬一 P6B4 輸出 NaN:當成最差,不會被藏起來
    idx_sorted = np.where(valid)[0][np.argsort(rank[valid], kind="stable")]
    picks = {f"p{p}": int(idx_sorted[min(len(idx_sorted) - 1, int(round(p / 100 * (len(idx_sorted) - 1))))]) for p in PCTS}
    picks["worst"] = int(idx_sorted[-1])
    rng = np.random.default_rng(RAND_SEED)
    rand_ids = sorted(int(i) for i in rng.choice(np.where(valid)[0], size=min(N_RANDOM, int(valid.sum())), replace=False))

    # ---- 逐樣本統計(3 個 seed 各自算,再平均)----
    stats = {"where": where, "n_valid": int(valid.sum()), "picks": {}, "random": rand_ids, "per_method": {}}
    print(f"\n逐樣本 nerr_ph({where};{int(valid.sum())} 個非空樣本;3 個 seed 各自算再平均)")
    for k in order:
        rows = []
        for e in ps[k]:
            e = e[valid]
            rows.append([e.mean(), np.median(e), np.percentile(e, 10), np.percentile(e, 90), (e < 0.05).mean(),
                         (e < 0.1).mean(), e.max()])
        m = np.mean(rows, 0)
        stats["per_method"][k] = dict(zip(["mean", "median", "p10", "p90", "frac_lt_0.05", "frac_lt_0.1", "max"],
                                          map(float, m)))
        print(f"  {labels[k].splitlines()[0]:<46} 平均 {m[0]:.4f}  中位數 {m[1]:.4f}  第 10 / 90 百分位 {m[2]:.4f} / {m[3]:.4f}"
              f"  < 0.05 {100 * m[4]:.0f}%  < 0.1 {100 * m[5]:.0f}%  最差 {m[6]:.3f}")
    for ref in ["iter_b512", "iter_b64", "P6B"]:
        w = [float((a[valid] < b[valid]).mean()) for a, b in zip(ps[PRIMARY], ps[ref])]
        stats[f"p6b4_beats_{ref}_frac"] = w
        print(f"  P6B4 逐樣本比「{labels[ref].splitlines()[0]}」準的比例:"
              + " / ".join(f"{100 * x:.1f}%" for x in w) + f"(平均 {100 * np.mean(w):.1f}%)")
    for p, i in picks.items():
        stats["picks"][p] = {"index": i, **{k: float(err0[k][i]) for k in order}}

    # ---- 依 R0 內的主要物體類型分組 ----
    check("類型推算:逐類型產生再洗牌 = make_home(逐位元,60 個物體)", kinds_check(cfg, fseed))
    km, kinds = tile_kind_map(cfg, O.shape[0], fseed, F)
    E = (O.abs() ** 2) * R0
    share = torch.stack([(E * (km == k)).sum((1, 2)) for k in range(len(kinds))], 1)
    dom = share.argmax(1).numpy()
    stats["by_kind"] = {"n": {kn: int(((dom == j) & valid).sum()) for j, kn in enumerate(kinds)}}
    print(f"\n依 R0 內的主要物體類型分組(3 個 seed 各自算再平均;樣本數:"
          + "、".join(f"{kn} {stats['by_kind']['n'][kn]}" for kn in kinds) + ")")
    for k in order:
        row = {}
        for j, kn in enumerate(kinds):
            sel = valid & (dom == j)
            if not sel.any():
                row[kn] = {"mean": None, "median": None}
                continue
            row[kn] = {"mean": float(np.mean([e[sel].mean() for e in ps[k]])),
                       "median": float(np.mean([np.median(e[sel]) for e in ps[k]]))}
        stats["by_kind"][k] = row
        print(f"  {labels[k].splitlines()[0]:<46} " + "  ".join(
            f"{kn} 平均 {row[kn]['mean']:.4f} / 中位數 {row[kn]['median']:.4f}" if row[kn]["mean"] is not None else f"{kn} —"
            for kn in kinds))

    # ---- 圖 ----
    ys, xs = torch.where(R0)
    y0, y1 = max(0, int(ys.min()) - MARGIN), min(F, int(ys.max()) + 1 + MARGIN)
    x0, x1 = max(0, int(xs.min()) - MARGIN), min(F, int(xs.max()) + 1 + MARGIN)
    r0c = R0[y0:y1, x0:x1].numpy()
    pm = cfg.phase_max

    def show(ax, im, cm, lo, hi):
        cmap = plt.get_cmap(cm).copy()
        cmap.set_bad("#d9d9d9")
        hnd = ax.imshow(im, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.imshow(np.where(r0c, np.nan, 0.0), cmap="Greys", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        return hnd

    def cbar(fig, hnd, axrow, label):
        cb = fig.colorbar(hnd, ax=list(axrow), fraction=0.012, pad=0.01)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    def crop_imgs(field, i, truth):
        t = O[i, y0:y1, x0:x1].numpy()
        z = t if truth else align_phase(field[i:i + 1], O[i:i + 1], R0)[0, y0:y1, x0:x1].numpy()
        ph = np.where(np.abs(t) > AMP_MIN, np.angle(z), np.nan)
        return np.abs(z), ph, np.abs(z - t)

    def save(fig, name):
        if not fig.get_constrained_layout():
            fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    rows = ["amplitude", "phase (rad)", "|error|"]
    cols = [("truth", None)] + [(k, keep[k]) for k in order]
    for name, i in picks.items():
        fig, axs = plt.subplots(3, len(cols), figsize=(2.2 * len(cols) + 0.6, 7.2), constrained_layout=True)
        for c, (k, f) in enumerate(cols):
            amp, ph, e = crop_imgs(f, i, k == "truth")
            ha = show(axs[0, c], amp, "gray", 0, 1)
            hp = show(axs[1, c], ph, "viridis", 0, pm)
            if k == "truth":
                axs[2, c].axis("off")
                axs[0, c].set_title("truth", fontsize=8)
            else:
                he = show(axs[2, c], e, "magma", 0, 0.5)
                axs[0, c].set_title(f"{labels[k]}\nnerr {err0[k][i]:.3f}", fontsize=7.5)
        for r, nm in enumerate(rows):
            axs[r, 0].set_ylabel(nm, fontsize=9)
        cbar(fig, ha, axs[0], "amplitude")
        cbar(fig, hp, axs[1], "phase (rad)")
        cbar(fig, he, axs[2], "|error|")
        what = "worst sample" if name == "worst" else f"{name[1:]}th percentile"
        fig.suptitle(f"{where}: field #{i} (P6B4 error = {what}; seed-0 networks). Scored region = R0 (outside dimmed); "
                     f"grey phase = true amplitude < {AMP_MIN}; error colour scale 0-0.5", fontsize=8.5)
        save(fig, f"mid_{name}.png")

    for part in range(0, len(rand_ids), 3):
        ids = rand_ids[part:part + 3]
        fig, axs = plt.subplots(2 * len(ids), len(cols), figsize=(2.2 * len(cols) + 0.6, 2.35 * 2 * len(ids)),
                                squeeze=False, constrained_layout=True)
        for r, i in enumerate(ids):
            for c, (k, f) in enumerate(cols):
                amp, ph, _ = crop_imgs(f, i, k == "truth")
                ha = show(axs[2 * r, c], amp, "gray", 0, 1)
                hp = show(axs[2 * r + 1, c], ph, "viridis", 0, pm)
                ttl = "truth" if k == "truth" else f"{labels[k].splitlines()[0]}\nnerr {err0[k][i]:.3f}"
                axs[2 * r, c].set_title(ttl, fontsize=7.5)
            axs[2 * r, 0].set_ylabel(f"#{i}\namplitude", fontsize=8.5)
            axs[2 * r + 1, 0].set_ylabel(f"#{i}\nphase", fontsize=8.5)
            cbar(fig, ha, axs[2 * r], "amplitude")
            cbar(fig, hp, axs[2 * r + 1], "phase (rad)")
        fig.suptitle(f"{where}: random samples (numpy seed {RAND_SEED}, drawn from non-empty samples before looking at "
                     f"results); seed-0 networks; iterative at P6B4's time", fontsize=8.5)
        save(fig, f"mid_random_{part // 3 + 1}.png")

    # ---- 全景:整個掃描範圍(80×80;網路只輸出這個範圍),藍線 = R0(評分範圍)----
    b0, size, _ = s5c.geometry(cfg)
    r0full = R0[b0:b0 + size, b0:b0 + size].numpy().astype(float)

    def full_imgs(field, i, truth):
        t = O[i, b0:b0 + size, b0:b0 + size].numpy()
        z = t if truth else align_phase(field[i:i + 1], O[i:i + 1], R0)[0, b0:b0 + size, b0:b0 + size].numpy()
        return np.abs(z), np.where(np.abs(t) > AMP_MIN, np.angle(z), np.nan)

    for name, ids in [("p50", [picks["p50"]]), ("random", rand_ids[:3])]:
        fig, axs = plt.subplots(2 * len(ids), len(cols), figsize=(2.6 * len(cols) + 0.6, 2.75 * 2 * len(ids)),
                                squeeze=False, constrained_layout=True)
        for r, i in enumerate(ids):
            for c, (k, f) in enumerate(cols):
                amp, ph = full_imgs(f, i, k == "truth")
                for rr, (im, cm, hi) in enumerate([(amp, "gray", 1), (ph, "viridis", pm)]):
                    ax = axs[2 * r + rr, c]
                    cmap = plt.get_cmap(cm).copy()
                    cmap.set_bad("#d9d9d9")
                    hnd = ax.imshow(im, cmap=cmap, vmin=0, vmax=hi, interpolation="nearest")
                    ax.contour(r0full, levels=[0.5], colors="#2a78d6", linewidths=0.9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if rr == 0:
                        ha = hnd
                    else:
                        hp = hnd
                ttl = "truth" if k == "truth" else f"{labels[k].splitlines()[0]}\nnerr (R0) {err0[k][i]:.3f}"
                axs[2 * r, c].set_title(ttl, fontsize=7.5)
            axs[2 * r, 0].set_ylabel(f"#{i}\namplitude", fontsize=8.5)
            axs[2 * r + 1, 0].set_ylabel(f"#{i}\nphase", fontsize=8.5)
            cbar(fig, ha, axs[2 * r], "amplitude")
            cbar(fig, hp, axs[2 * r + 1], "phase (rad)")
        fig.suptitle(f"{where}: whole scan area ({size}x{size} px; networks only output this area). Blue line = R0 "
                     f"(the scored region); outside R0 is less illuminated and not scored. Same global-phase alignment "
                     f"as scoring; grey phase = true amplitude < {AMP_MIN}", fontsize=8.5)
        save(fig, f"mid_full_{name}.png")

    i = picks["p50"]
    st = keep["stages"]
    ste = [per_sample(f, O, R0).numpy() for f in st]
    stats["stages_seed0_mean"] = [float(np.nanmean(x)) for x in ste]
    scol = [(f"P6B4 start\nnerr {ste[0][i]:.3f}", st[0])] + [(f"after stage {k}\nnerr {ste[k][i]:.3f}", st[k])
                                                              for k in range(1, len(st))] + [("truth", None)]
    fig, axs = plt.subplots(3, len(scol), figsize=(2.2 * len(scol) + 0.6, 7.2), constrained_layout=True)
    for c, (ttl, f) in enumerate(scol):
        amp, ph, e = crop_imgs(f, i, f is None)
        ha = show(axs[0, c], amp, "gray", 0, 1)
        hp = show(axs[1, c], ph, "viridis", 0, pm)
        if f is None:
            axs[2, c].axis("off")
        else:
            he = show(axs[2, c], e, "magma", 0, 0.5)
        axs[0, c].set_title(ttl, fontsize=8)
    for r, nm in enumerate(rows):
        axs[r, 0].set_ylabel(nm, fontsize=9)
    cbar(fig, ha, axs[0], "amplitude")
    cbar(fig, hp, axs[1], "phase (rad)")
    cbar(fig, he, axs[2], "|error|")
    fig.suptitle(f"{where}: inside P6B4 - start -> each physics + CNN stage (field #{i}, median sample)", fontsize=9)
    save(fig, "mid_stages.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fin = {k: err0[k][valid][np.isfinite(err0[k][valid])] for k in order}
    lo = max(1e-8, min(float(fin[k].min()) for k in order if len(fin[k])) * 0.8)
    hi = max(float(fin[k].max()) for k in order if len(fin[k])) * 1.2
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    for k in order:
        ax.hist(fin[k], bins=bins, histtype="step", lw=1.6, color=COL[k], label=labels[k].replace("\n", " "))
    ax.set_xscale("log")
    ax.set_xlabel("per-sample nerr on R0 (global phase only)")
    ax.set_ylabel("samples")
    ax.set_title(f"{where}: per-sample error (seed-0 networks)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "mid_hist.png")
    json.dump(stats, open(out_dir / "mid_viz_stats.json", "w"), indent=2)
    return stats


def envelope_figure(env, res, tim, out_dir, where):
    plt = s5._plt()
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, B in zip(axs, ["64", "512"]):
        tt = sorted({s5b.cfg_time(c, it, tim[B], PROBE) for c in s5b.configs() for it in s5b.stops(c)} - {0.0})
        ax.plot(tt, [s5b.envelope(env, PROBE, T, tim[B])[2].mean() for T in tt], color="#0b0b0b", lw=2,
                drawstyle="steps-post", label="strongest iterative method at each time (envelope)")
        for x in LINEUP:
            a = vals(res, x)
            tn = tim[B][f"net:{x}"]
            e = s5b.envelope(env, PROBE, tn, tim[B])
            r = s5b.ratio(a, e[2])
            ax.plot([tn], [a.mean()], "o", ms=8, color=COL[x], zorder=4,
                    label=f"{x}: {a.mean():.4f} at {tn:.3f} ms (ratio {r[1]:.3f})")
            ax.plot([tn], [e[2].mean()], "x", ms=7, color=COL[x], zorder=4)
            ax.plot([tn, tn], [a.mean(), e[2].mean()], ":", color=COL[x], lw=1)
        ax.plot([], [], "x", color="#555", label="envelope at the network's own time")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{PROBE}, batch {B} ({where})", fontsize=10)
        ax.set_xlabel("time per sample (ms)")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    axs[0].set_ylabel("nerr on R0 (global phase only)")
    fig.tight_layout()
    p = out_dir / "mid_vs_envelope.png"
    fig.savefig(p, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


# ============================================================================
# 流程
# ============================================================================
def measure(dev, seeds_fn, env, val, out_dir, fig_dir, fseed, smoke):
    import probe_4_2b as pb
    where = "validation fields (smoke)" if smoke else "midterm fields (seed 95,000,000)"
    cfg0 = s5.load_cfg(0)
    probes0 = {k: v for k, v in pb.build_probes(cfg0, dev).items() if k in s5.PROBES}
    print("\n  重新計時:迭代法與網路", flush=True)
    tim = s5b.timing(cfg0, probes0, dev)
    tg = g.time_groups(LINEUP, dev)
    for B in tg:
        tim[B].update({f"net:{x}": t for x, t in tg[B].items()})
    iter_confs, labels = {}, {}
    for B in ["64", "512"]:
        conf, it, _, _ = s5b.envelope(env, PROBE, tim[B][f"net:{PRIMARY}"], tim[B])
        iter_confs[f"iter_b{B}"] = (conf, it)
        labels[f"iter_b{B}"] = f"iterative b{B}: {s5b.fmt_conf(conf)} x{it}\n(= P6B4's time at batch {B})"
    for x in LINEUP:
        labels[x] = f"{x} (b64 {tim['64'][f'net:{x}']:.3f} ms)"
    print("\n  網路(與視覺化用的迭代法)", flush=True)
    res, ps, keep, worst_ps = run_nets(seeds_fn, dev, iter_confs)
    check(f"逐樣本計算一致(本場):逐樣本 nerr_ph 的平均 = field_metrics(差 / max(1, 值) < {PS_TOL:.0e})", worst_ps < PS_TOL,
          f"最大差 {worst_ps:.1e}")
    worst_it = 0.0
    for k, (conf, it) in iter_confs.items():
        ref = s5b.vals(env, PROBE, conf, it)
        worst_it = max(worst_it, max(absdiff(np.nanmean(e), r) for e, r in zip(ps[k], ref)))
    check(f"視覺化的迭代法 = 包絡檔(逐 seed nerr_ph,< {VIZ_TOL:.0e})", worst_it < VIZ_TOL,
          " / ".join(f"{k}: {s5b.fmt_conf(c)} ×{i}" for k, (c, i) in iter_confs.items()) + f";最大差 {worst_it:.1e}")
    raw = out_dir / f"scan5i_mid_raw{SUF}.json"
    dump_json({"results": res, "timing": tim, "iter_confs": iter_confs, "val": val, "smoke": smoke, "quick": QUICK,
               "checks_ok": all_ok(), "per_sample": {k: [e.tolist() for e in v] for k, v in ps.items()}}, raw)
    print(f"  原始結果先存檔:{raw}")
    if not all_ok():
        print("\n⚠️ 上面有內建檢查未通過 —— 以下的判讀先不要採信,把完整輸出貼給 Claude")
    verdict = report(env, res, tim, val, smoke)
    final = out_dir / f"scan5i_mid{SUF}.json"
    out = {"results": res, "timing": tim, "verdict": verdict, "viz_stats": None, "smoke": smoke, "quick": QUICK,
           "checks_ok": all_ok()}
    dump_json(out, final)                                                  # 判讀先存(畫圖若失敗,判讀仍在)
    print("\n" + "=" * 100)
    print("視覺化與逐樣本統計(§20.7;描述,不改變判讀)")
    print("=" * 100)
    try:
        out["viz_stats"] = visualize(ps, keep, labels, fseed, fig_dir, where)
        envelope_figure(env, res, tim, fig_dir, where)
    except Exception as e:                                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"視覺化完成(判讀不受影響,已存於 {final.name})", False, f"{type(e).__name__}: {e}")
    out["checks_ok"] = all_ok()
    dump_json(out, final)
    return verdict


def dump_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def check_files():
    need = [Path(f) for f in ("scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py", "scan_5e.py", "scan_5g.py", "scan_5h.py",
                              "probe_4_2b.py", "probe_4_1.py", "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    need += [model_dir(x, s) / f for x in LINEUP for s in SEEDS for f in ("final.pt", "result.json")]
    need += [VAL_ENV, P5_RAW]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在(含陣容 {'、'.join(LINEUP)} 的 9 個模型)", not missing,
          "缺:" + ", ".join(missing) if missing else "")
    return not missing


def freeze_record():
    rec = {f"{x}_s{s}": md5(model_dir(x, s) / "final.pt") for x in LINEUP for s in SEEDS}
    print("\n凍結的模型(final.pt 的 md5):")
    for k, v in rec.items():
        print(f"  {k:<10} {v}")
    bad, n = [], 0
    for x in ("P6B", "P6B4"):
        for s in SEEDS:
            im = json.load(open(model_dir(x, s) / "result.json")).get("init_md5")
            if im is None:
                continue
            n += 1
            if im != rec[f"B_s{s}"]:
                bad.append(f"{x} s{s}")
    check(f"凍結一致:P6B、P6B4 訓練時記下的起點(init_md5)= 現在的 B 模型({n} 個有紀錄)", not bad,
          "不一致:" + "、".join(bad) if bad else "")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑內建檢查(只用驗證場)")
    ap.add_argument("--smoke", action="store_true", help="用驗證場代替期中考場跑完整流程")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = md5(Path(__file__).resolve())
    print(f"scan_5i.py md5 {me}")
    if not check_files():
        print("\n❌ 缺檔案,停下來")
        sys.exit(1)

    if a.check:
        checks(dev)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)

    if a.smoke:
        print("#" * 70)
        print(f"迷你全流程(--smoke):用驗證場代替期中考場;輸出到 {SMOKE_DIR},不碰期中考場")
        print("#" * 70)
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)                                      # 只刪這個 smoke 專用資料夾
        SMOKE_DIR.mkdir(parents=True)
        t0 = time.time()
        freeze_record()
        val = checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
        env = json.load(open(VAL_ENV))["envelope"]
        measure(dev, s5e.val_seeds, env, val, SMOKE_DIR, SMOKE_DIR / "figs", s5e.VAL_FIELD_SEED, smoke=True)
        print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒")
        if all_ok():
            json.dump({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
            print("\n✅ 迷你全流程全部通過 → 可以送出正式的期中考(sbatch run_scan5i_mid.sh)")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)

    # ---- 正式的期中考 ----
    for p in (MID_RAW, MID_JSON):
        if p.exists():
            raise SystemExit(f"❌ {p} 已存在:期中考已經跑過(只考一次)。重跑的數字會相同,但為避免覆蓋,先告訴 Claude")
    if not PASSED.exists():
        raise SystemExit(f"❌ 找不到 {PASSED}:先在 dev 節點跑 python scan_5i.py --smoke,全部通過才可送件")
    pm = json.load(open(PASSED))["script_md5"]
    if pm != me:
        raise SystemExit(f"❌ smoke 通過時的 scan_5i.py(md5 {pm})與現在的({me})不同:用現在的版本重跑 --smoke")
    print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")
    frozen = freeze_record()
    val = checks(dev)
    if not all_ok():
        print("\n❌ 內建檢查未通過 → 沒有碰期中考場。把輸出貼給 Claude")
        sys.exit(1)
    json.dump({"models_md5": frozen, "script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(MD5_JSON, "w"), indent=2)
    old_env = json.load(open(MID_ENV)) if MID_ENV.exists() else {}
    if (old_env.get("field_seed") == MID_FIELD_SEED and old_env.get("noise_base") == MID_NOISE_BASE
            and old_env.get("quick") == QUICK):
        print(f"\n  期中考場的包絡已存在(同一個 job 先前中斷時留下),沿用:{MID_ENV}")
        env = old_env["envelope"]
    else:
        print("\n期中考場:迭代法包絡(DEF2,全部設定)", flush=True)
        env = s5d.measure_envelope(mid_seeds, dev)
        tmp = MID_ENV.with_suffix(".tmp")
        json.dump({"envelope": env, "field_seed": MID_FIELD_SEED, "noise_base": MID_NOISE_BASE, "quick": QUICK}, open(tmp, "w"))
        os.replace(tmp, MID_ENV)
        print(f"  存檔:{MID_ENV}")
    measure(dev, mid_seeds, env, val, RUN_ROOT, FIG_DIR, MID_FIELD_SEED, smoke=False)
    print("\n" + ("✅ 內建檢查全部通過" if all_ok() else "❌ 有項目未通過(判讀先不要採信)"))
    print("把完整輸出與 figs_scan5i/ 的圖傳給 Claude")
    sys.exit(0 if all_ok() else 1)


if __name__ == "__main__":
    main()
