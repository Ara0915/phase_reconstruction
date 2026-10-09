#!/usr/bin/env python
"""階段五 新測試場確認(階段五協定 §十五;不重新訓練)。

問:DEF2 的 B、C 網路(100k 訓練)在「從未使用過的測試場」上,是否仍比最強的迭代法準?
   迭代法的包絡也在新測試場上重跑(4 演算法 × 6 起點 + K-HIO,同 5-2b),網路直接載入已訓練的模型。

內建檢查(全部通過才量測):
  1. 新測試場與舊測試場、訓練場的非空白物體塊無相同;seed 範圍不重疊
  2. 同一支程式在舊測試場上重跑:迭代法(ePIE-C+rand、AP-C+zero、K-HIO)= scan5b_raw.json;
     網路(B、C 100k)= scan5c_raw_n100000.json(逐 seed nerr_ph,< 1e-5)

用法(需在計算節點執行;需 scan_5.py、scan_5b.py、scan_5c.py 與其依賴、5-2b / 5-3 的結果檔、B / C 的模型):
    python scan_5d.py --check   # 只跑內建檢查
    python scan_5d.py           # 內建檢查 + 新測試場的量測與判讀(用 run_scan5d.sh 送 job)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5 as s5                                                  # noqa: E402
import scan_5b as s5b                                                # noqa: E402
import scan_5c as s5c                                                # noqa: E402  (匯入時已關閉 TF32,同 5-3)

RUN_ROOT = s5.RUN_ROOT
PROBE = "DEF2"
GROUPS = ["B", "C"]
N_TRAIN = s5c.N_TRAIN if s5c.QUICK else 100_000
NEW_FIELD_SEED = 90_000_000
NEW_NOISE_BASE = 91_000_000
REPRO_TOL = 1e-5
REPRO_CONFIGS = ["ePIE-C|rand", "AP-C|zero"]           # + K-HIO
FIG_DIR = Path("figs_scan5d")

QUICK = s5c.QUICK
OLD_5B = RUN_ROOT / ("scan5b_raw_quick.json" if QUICK else "scan5b_raw.json")
OLD_5C = RUN_ROOT / ("scan5c_raw_quick.json" if QUICK else f"scan5c_raw_n{N_TRAIN}.json")

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


# ---- 舊 / 新測試場的 seed(舊 = 5-2b / 5-3 的公式,逐位元相同)----
def old_seeds(cfg, s):
    si = list(s5.SCANS).index(s5b.SCAN)
    pi_ = s5.PROBES.index(PROBE)
    return {"field": cfg.test_seed + s5.FIELD_SEED_OFFSET,
            "noise": cfg.test_seed + s + 20_000 + 1000 * pi_ + 100 * si,
            "init": cfg.test_seed + s}


def new_seeds(cfg, s):
    return {"field": NEW_FIELD_SEED, "noise": NEW_NOISE_BASE + 1000 * s, "init": NEW_NOISE_BASE + 1000 * s + 1}


def setup_seed(s, seeds_fn, dev):
    """回傳這個 seed 的共同物件:cfg、探針、量測、真值、R0 …(量測方式同 5-2b / 5-3)。"""
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[PROBE]
    n = s5b.N_FIELDS or cfg.eval_n
    F, W = s5.field_size(cfg), cfg.canvas
    bs = beamstop_mask(cfg, device=dev)
    sd = seeds_fn(cfg, s)
    fields = s5.make_fields(cfg, n, seed=sd["field"], device=dev)
    O = torch.polar(fields[:, 0], fields[:, 1])
    st = s5.window_starts(cfg, s5b.SCAN)
    cp = s5.scan_cfg(cfg, PROBE, pr, s5.SCANS[s5b.SCAN][2])
    counts, _ = s5.measure_scan(fields, PROBE, pr, cp, bs, st, seed=sd["noise"])
    R0 = s5.footprint_mask(cfg, pr, s5.window_starts(cfg, "1x1"), dev)
    return dict(cfg=cfg, pr=pr, n=n, F=F, W=W, bs=bs, O=O, st=st, cp=cp, counts=counts, R0=R0, init=sd["init"])


# ============================================================================
# 迭代法的包絡(同 scan_5b.measure 的迴圈,只把 seed 參數化)
# ============================================================================
@torch.no_grad()
def measure_envelope(seeds_fn, dev, configs=None):
    """configs = None → 全部設定(同 5-2b);否則只跑列出的設定(另加 K-HIO)。回傳與 scan5b 相同結構的 {PROBE: [...]}。"""
    import probe_4_2b as pb
    res = {PROBE: []}
    for s in s5b.SEEDS:
        d = setup_seed(s, seeds_fn, dev)
        cfg, pr, cp, bs, counts, st, O, R0, n, F = (d[k] for k in ("cfg", "pr", "cp", "bs", "counts", "st", "O", "R0",
                                                                "n", "F"))
        ones = torch.ones(d["W"], d["W"], device=dev)
        rec = {}
        kit = sorted(set(s5b.K_ITERS) | set(s5b.KHIO_N.values()))
        init = s5.k_init(counts[s5b.CENTER], cp, d["init"], dev)
        k_out = pb.hio_probe(init, counts[s5b.CENTER], bs, cp, kit, "K", pr, ones)
        rec["K-HIO"] = {str(it): s5.field_metrics(s5.k_to_field(k_out[it], pr, cfg, st[s5b.CENTER], F), O, R0)
                        for it in s5b.K_ITERS}
        for kind in s5b.INITS:
            want = [m for m in s5b.METHODS if configs is None or f"{m}|{kind}" in configs]
            if not want:
                continue
            O0 = s5b.make_init(kind, n, F, cfg, cp, pr, bs, counts, st, d["init"], dev, k_out=k_out)
            rec[f"init|{kind}"] = s5.field_metrics(O0, O, R0)
            for meth in want:
                outs = s5b.run_method(meth, counts, bs, cp, pr, st, s5b.ITERS, O0, d["init"])
                rec[f"{meth}|{kind}"] = {str(it): s5.field_metrics(outs[it], O, R0) for it in s5b.ITERS}
                rec[f"{meth}|{kind}"]["0"] = rec[f"init|{kind}"]
                del outs
        del k_out
        res[PROBE].append({"run": f"{s5.BASE}_s{s}", "rec": rec})
        print(f"  [{s5.BASE}_s{s}] 迭代法完成", flush=True)
    return res


# ============================================================================
# 網路(載入已訓練的模型;同 scan_5c.evaluate 的做法)
# ============================================================================
@torch.no_grad()
def measure_nets(seeds_fn, dev, hybrid=True):
    res = {PROBE: {g: [] for g in GROUPS}}
    for s in s5b.SEEDS:
        d = setup_seed(s, seeds_fn, dev)
        cfg, pr, cp, bs, counts, st, O, R0, n, F = (d[k] for k in ("cfg", "pr", "cp", "bs", "counts", "st", "O", "R0",
                                                                "n", "F"))
        for g in GROUPS:
            net, norm, meta = s5c.load_net(g, PROBE, s, N_TRAIN, cfg, pr, dev)
            if net is None:
                raise SystemExit(f"❌ 缺少模型:{g} {PROBE} seed {s}(n{N_TRAIN})")
            pred = s5c.predict(net, counts, bs, norm, g, cp)
            est = s5c.to_field(torch.polar(pred[:, 0], pred[:, 1]), cfg, F)
            rec = {"net": s5.field_metrics(est, O, R0), "params": meta["params"],
                   "train_nerr_ph": meta["train_nerr_ph"], "train_nerr_pooled": meta.get("train_nerr_pooled"),
                   "history_last": meta["history"][-1]}
            if hybrid:                                                       # 網路當起點(同 5-3:R0 內網路,其餘 const)
                a0 = s5b.const_amp(counts[s5b.CENTER], bs, cp, pr)
                cst = torch.polar(a0[:, None, None].expand(n, F, F).contiguous(),
                                  torch.full((n, F, F), cfg.phase_max / 2, device=dev))
                O0 = torch.where(R0, est, cst)
                for meth in s5b.METHODS:
                    outs = s5b.run_method(meth, counts, bs, cp, pr, st, s5b.ITERS, O0, d["init"])
                    rec[meth] = {str(it): s5.field_metrics(outs[it], O, R0) for it in s5b.ITERS}
                    del outs
            res[PROBE][g].append(rec)
            del net, pred, est
        print(f"  [seed {s}] 網路完成", flush=True)
    return res


# ============================================================================
# 內建檢查
# ============================================================================
@torch.no_grad()
def checks(dev):
    import probe_4_1 as p41
    torch.backends.cudnn.benchmark = True                                   # 同 5-3 評估時的設定(網路程式一致)
    print("=" * 70)
    print("新測試場確認:內建檢查")
    print("=" * 70)
    cfg = s5.load_cfg(0)

    # (1) 資料不重疊
    d = s5.tile_size(cfg)
    s0 = (cfg.canvas - d) // 2
    n_t = (s5b.N_FIELDS or cfg.eval_n) * s5.TILES ** 2
    crop = (lambda t: t[:, :, s0:s0 + d, s0:s0 + d])
    new_t = crop(p41.make_home(cfg, n_t, seed=NEW_FIELD_SEED))
    old_t = crop(p41.make_home(cfg, n_t, seed=cfg.test_seed + s5.FIELD_SEED_OFFSET))
    tr_t = torch.cat([crop(p41.make_home(cfg, 64 * s5.TILES ** 2, seed=s5c.train_chunk_seed(s, k)))
                      for s in s5c.SEEDS for k in (0, max(0, N_TRAIN // s5c.CHUNK - 1))])
    hn = s5c._tile_hashes(new_t)
    inter_old, inter_tr = len(hn & s5c._tile_hashes(old_t)), len(hn & s5c._tile_hashes(tr_t))
    tr_max = s5c.train_chunk_seed(max(s5c.SEEDS), N_TRAIN // s5c.CHUNK) + 3000
    used = [cfg.test_seed + s5.FIELD_SEED_OFFSET, p41.CALIB_SEED, p41.CALIB_SEED + 1]
    far = NEW_FIELD_SEED - tr_max > 1_000_000 and all(NEW_FIELD_SEED - u > 1_000_000 for u in used)
    check("資料不重疊:新測試場的 seed 遠離訓練 / 舊測試 / 校準;非空白物體塊與舊測試、訓練(抽樣)皆無相同",
          far and inter_old == 0 and inter_tr == 0,
          f"新 seed {NEW_FIELD_SEED:,};訓練 seed 上限約 {tr_max:,};相同的塊:舊測試 {inter_old}、訓練 {inter_tr}")

    # (2) 迭代法程式一致(舊測試場)
    old5b = json.load(open(OLD_5B))["results"][PROBE]
    rep = measure_envelope(old_seeds, dev, configs=REPRO_CONFIGS)[PROBE]
    worst, cnt = 0.0, 0
    runs_ok = [old5b[k]["run"] == sd["run"] for k, sd in enumerate(rep)]
    for k, sd in enumerate(rep):
        for conf, rec in sd["rec"].items():
            if conf.startswith("init|"):
                continue
            for it, m in rec.items():
                o = old5b[k]["rec"].get(conf, {}).get(it)
                if o is None:
                    continue
                worst = max(worst, abs(o["nerr_ph"] - m["nerr_ph"]))
                cnt += 1
    want = len(s5b.SEEDS) * (len(s5b.K_ITERS) + len(REPRO_CONFIGS) * (len(s5b.ITERS) + 1))
    check(f"迭代法程式一致:舊測試場上重跑 {' / '.join(REPRO_CONFIGS)} / K-HIO = 5-2b(< {REPRO_TOL:.0e})",
          all(runs_ok) and cnt == want and worst < REPRO_TOL, f"比對 {cnt} / {want} 個點,最大差 {worst:.1e}")

    # (3) 網路程式一致(舊測試場)
    old5c = json.load(open(OLD_5C))["results"][PROBE]
    repn = measure_nets(old_seeds, dev, hybrid=False)[PROBE]
    worst, cnt = 0.0, 0
    for g in GROUPS:
        for k, r in enumerate(repn[g]):
            o = old5c[g][k]
            if o is None:
                continue
            worst = max(worst, abs(o["net"]["nerr_ph"] - r["net"]["nerr_ph"]))
            cnt += 1
    check(f"網路程式一致:B、C(n{N_TRAIN})在舊測試場上 = 5-3 的評估(< {REPRO_TOL:.0e})",
          cnt == len(GROUPS) * len(s5b.SEEDS) and worst < REPRO_TOL, f"比對 {cnt} 個網路,最大差 {worst:.1e}")
    print(f"\n     裝置:{dev}")
    return old5c


# ============================================================================
# 判讀(協定 §15.5,結果出來前寫定)
# ============================================================================
def report(env, nets, tim, old5c):
    v = {}
    Bs = ["64", "512"]
    print("\n" + "=" * 100)
    print("計時(ms / 張;本 job 重新量)")
    print("=" * 100)
    keep = ["ePIE-C", "ePIE", "AP-C", "AP", f"K-HIO:{PROBE}"] + [f"net:{g}:{PROBE}" for g in GROUPS]
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {tim[B][k]:.4f}" for k in keep))

    print("\n" + "=" * 100)
    print(f"新測試場:迭代法的包絡({PROBE};R0 的 nerr_ph,3 seeds 平均;k = 預算 / ePIE 3x3 掃一次)")
    print("=" * 100)
    for B in Bs:
        te = tim[B]["ePIE-C"]
        row = []
        for k in s5b.BUDGET_K:
            e = s5b.envelope(env, PROBE, k * te, tim[B])
            row.append(f"k={k}:{e[2].mean():.3f}({s5b.fmt_conf(e[0])} {e[1]})")
        print(f"  batch {B}:" + "  ".join(row))

    print("\n" + "=" * 100)
    print(f"新測試場:網路({PROBE},n{N_TRAIN};3 seeds)")
    print("=" * 100)
    res = nets
    for g in GROUPS:
        a = s5c.nvals(res, PROBE, g)
        old = np.array([r["net"]["nerr_ph"] for r in old5c[g]], float)
        st = s5b.ratio(a, old)
        print(f"  {g}:新 {a.mean():.4f}({' '.join(f'{x:.4f}' for x in a)})  舊 {old.mean():.4f}"
              f"  → (c4) 新 / 舊 比值 {st[1]:.3f}(z {st[2]:+.1f})"
              f"  完整對齊 {s5c.nvals(res, PROBE, g, 'nerr_al').mean():.4f}  選翻轉 {s5c.nvals(res, PROBE, g, 'twin_frac').mean():.2f}")
        gv = {"new": list(a), "old": list(old), "stability": list(st)}
        for B in Bs:
            tn = tim[B][f"net:{g}:{PROBE}"]
            e = s5b.envelope(env, PROBE, tn, tim[B])
            r = s5b.ratio(a, e[2])
            te = s5c.env_time_to(env, PROBE, a.mean(), tim[B])
            hy = {}
            for q in s5c.Q_TARGETS:
                th, hm, hi = s5c.hyb_time_to(res, PROBE, g, q, tim[B], tn)
                tq = s5c.env_time_to(env, PROBE, q, tim[B])
                hy[str(q)] = [th, tq, (th / tq) if (th is not None and tq) else None, hm, hi]
            gv[B] = {"t_net": tn, "opp": [e[0], e[1], list(e[2]), e[3]], "ratio": list(r), "same_quality_time": te,
                     "hybrid": hy}
            print(f"  (n4) {g} batch {B}:網路 {tn:.4f} ms、{a.mean():.4f} vs 對手 {s5b.fmt_conf(e[0])}({e[1]} 次)"
                  f" {e[2].mean():.4f} → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}")
            print("       (n5) 包絡要 " + (f"{te:.4f} ms 才達到網路的誤差 → 網路快 {te / tn:.1f} 倍" if te is not None
                                        else "在所有設定內都達不到網路的誤差"))
            print("       (n6) 網路起點 + 迭代(至少 1 次)vs 包絡,達到 q 的時間:" + ";".join(
                f"q={q}:" + (f"{h[0]:.4f} vs {h[1]:.4f} ms(× {h[2]:.2f};{h[3]} {h[4]} 次)" if h[2] is not None
                             else ("包絡在時間 0 就達到" if h[1] == 0 else
                                   (f"起點達到({h[3]} {h[4]} 次)、包絡達不到" if h[0] is not None and h[1] is None
                                    else ("皆達不到" if h[0] is None and h[1] is None else "起點達不到"))))
                for q, h in hy.items()))
        useful64 = gv["64"]["ratio"][0] == "較準"
        robust = useful64 and gv["512"]["ratio"][0] == "較準"
        start_ok = any(all((gv[B]["hybrid"][str(q)][2] is not None and gv[B]["hybrid"][str(q)][2] <= 0.8)
                           or (gv[B]["hybrid"][str(q)][0] is not None and gv[B]["hybrid"][str(q)][1] is None)
                           for B in Bs) for q in s5c.Q_TARGETS)
        gv.update({"useful64": useful64, "robust": robust, "start_useful": start_ok})
        v[g] = gv

    print("\n" + "=" * 100)
    print("判讀(協定 §15.5,結果出來前寫定)")
    print("=" * 100)
    c1 = v["B"]["robust"]
    c2 = v["C"]["useful64"]
    print(f"  (c1) B 組穩健有用(兩種 batch 都較準):{'✅ 通過 → 在新測試場上確認' if c1 else '❌ 未通過'}")
    print(f"  (c2) C 組有用(batch 64 較準):{'✅ 通過' if c2 else '❌ 未通過'}")
    for g in GROUPS:
        print(f"  (c3) {g} 組起點有用:{'✅ 通過' if v[g]['start_useful'] else '❌ 未通過'}")
    print("  (c4) 穩定性:見上方「新 / 舊 比值」(描述)")
    v["c1"], v["c2"] = c1, c2
    return v


def make_figure(env, nets, tim):
    plt = s5._plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    col = {"B": "#2a78d6", "C": "#eb6834"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, B in zip(axs, ["64", "512"]):
        tt = sorted({s5b.cfg_time(c, it, tim[B], PROBE) for c in s5b.configs() for it in s5b.stops(c)} - {0.0})
        ax.plot(tt, [s5b.envelope(env, PROBE, T, tim[B])[2].mean() for T in tt], color="#0b0b0b", lw=2,
                label="iterative envelope (new test set)")
        for g in GROUPS:
            a = s5c.nvals(nets, PROBE, g)
            tn = tim[B][f"net:{g}:{PROBE}"]
            ax.plot([tn], [a.mean()], "o", color=col[g], ms=7, label=f"net {g}")
            pts = sorted((tn + it * tim[B][m], s5c.hvals(nets, PROBE, g, m, it).mean())
                         for m in s5b.METHODS for it in s5b.ITERS)
            xs, ys, cur = [], [], float("inf")
            for t, y in pts:
                cur = min(cur, y)
                xs.append(t)
                ys.append(cur)
            ax.plot(xs, ys, color=col[g], lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{PROBE}, batch {B} (new test fields)", fontsize=10)
        ax.set_xlabel("time per sample (ms)")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    axs[0].set_ylabel("nerr on R0 (global phase only)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Confirmation on unseen test fields: nets (dots; dashed = net start + best iterative) vs envelope",
                 fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "scan5d_confirm.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


# ============================================================================
def check_files():
    need = [Path(f) for f in ("scan_5.py", "scan_5b.py", "scan_5c.py", "probe_4_2b.py", "probe_4_1.py",
                              "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in s5b.SEEDS]
    need += [OLD_5B, OLD_5C]
    need += [s5c.run_dir(g, PROBE, s, N_TRAIN) / f for g in GROUPS for s in s5b.SEEDS for f in ("final.pt", "result.json")]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在(含 B、C 的 6 個模型與其 result.json)", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑內建檢查")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    if not check_files():
        print("\n❌ 缺檔案,停下來")
        sys.exit(1)
    old5c = checks(dev)
    if a.check or not ok_all:
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過,先不要量測"))
        sys.exit(0 if ok_all else 1)
    print("\n計時")
    import probe_4_2b as pb
    cfg0 = s5.load_cfg(0)
    probes0 = {k: v for k, v in pb.build_probes(cfg0, dev).items() if k in s5.PROBES}
    tim = s5b.timing(cfg0, probes0, dev)
    tnet = s5c.time_nets(N_TRAIN, dev)
    for B in tnet:
        tim[B].update({f"net:{k}": val for k, val in tnet[B].items()})
    print("\n新測試場:迭代法")
    env = measure_envelope(new_seeds, dev)
    json.dump({"envelope": env, "timing": tim, "quick": QUICK},
              open(RUN_ROOT / ("scan5d_partial.json" if not QUICK else "scan5d_partial_quick.json"), "w"))
    print("\n新測試場:網路")
    nets = measure_nets(new_seeds, dev)
    raw = RUN_ROOT / ("scan5d_raw.json" if not QUICK else "scan5d_raw_quick.json")
    json.dump({"envelope": env, "nets": nets, "timing": tim, "quick": QUICK}, open(raw, "w"))
    print(f"  原始結果先存檔:{raw}")
    verdict = report(env, nets, tim, old5c)
    json.dump({"envelope": env, "nets": nets, "timing": tim, "verdict": verdict, "quick": QUICK},
              open(RUN_ROOT / ("scan5d.json" if not QUICK else "scan5d_quick.json"), "w"))
    make_figure(env, nets, tim)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))
    print("把完整輸出貼給 Claude")


if __name__ == "__main__":
    main()
