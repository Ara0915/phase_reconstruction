#!/usr/bin/env python
"""NAF 的 LayerNorm 實作優化:只計時、不訓練(階段五協定 §十七)。

實作 1:LayerNorm2d → 內建 F.layer_norm(對通道維度;數學上相同,權重直接沿用)
實作 2:實作 1 + channels_last(整個網路與輸入)
各網路(B、C、NAF)一律取自己最快的等價實作;輔助:torch.compile(網路與 AP-C 一次迭代,分開報告,不進主判)。

檢查:已訓練的 NAF-B 新舊實作輸出差 < 1e-5;以新實作重算驗證場 nerr_ph 與 P7 評估差 < 1e-5。
判讀:以新時間重算 NAF-B vs 驗證場包絡;列出所有 NAF 候選的新時間(NAF-B 小只回報)。

用法(需在計算節點執行;需 scan_5e.py 與其依賴、P7 的結果檔與 NAF-B 模型):
    python scan_5f.py     # 用 run_scan5f.sh 送 job
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5 as s5                                                  # noqa: E402
import scan_5b as s5b                                                # noqa: E402
import scan_5c as s5c                                                # noqa: E402
import scan_5d as s5d                                                # noqa: E402
import scan_5e as s5e                                                # noqa: E402  (匯入時註冊 NAF 架構)

RUN_ROOT = s5.RUN_ROOT
QUICK = s5e.QUICK
PROBE = s5e.PROBE
N_TRAIN = s5e.N_TRAIN
SUF = s5e.SUF
P7_RAW = RUN_ROOT / f"scan5e_p7_raw{SUF}.json"
EQ_TOL = 1e-5

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


# ============================================================================
# 等價的快速實作
# ============================================================================
class LayerNorm2dFast(nn.Module):
    """與 scan_5e.LayerNorm2d 數學上相同(通道維度、有偏變異數、eps 1e-6),改用內建的 F.layer_norm。"""

    def __init__(self, ln):
        super().__init__()
        self.weight, self.bias, self.eps = ln.weight, ln.bias, ln.eps

    def forward(self, x):
        y = Fnn.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2).contiguous()                      # 回到一般(NCHW)排列:實作 1 只換 LayerNorm


def fast_ln(model):
    """把模型裡所有 LayerNorm2d 換成 LayerNorm2dFast(共用同一組參數)。"""
    for name, m in list(model.named_modules()):
        for cname, child in list(m.named_children()):
            if isinstance(child, s5e.LayerNorm2d):
                setattr(m, cname, LayerNorm2dFast(child))
    return model


class ChannelsLast(nn.Module):
    """整個網路與輸入改用 channels_last(輸入轉換計入時間)。"""

    def __init__(self, net):
        super().__init__()
        self.net = net.to(memory_format=torch.channels_last)

    def forward(self, inp):
        return self.net({**inp, "x": inp["x"].contiguous(memory_format=torch.channels_last)})


def variants_of(group, cfg, pr, dev, state=None):
    """回傳 {實作名稱: 模型}。state:已訓練的權重(可選)。"""
    def base():
        m = s5c.build_model(group, cfg, pr)
        if state is not None:
            m.load_state_dict(state)
        return m.to(dev).eval()
    out = {"原寫法": base()}
    if group.startswith("NAF"):
        out["實作1"] = fast_ln(base())
        out["實作2"] = ChannelsLast(fast_ln(base()))
    elif group in ("A", "B", "D"):
        out["channels_last"] = ChannelsLast(base())
    return out


# ============================================================================
# 計時
# ============================================================================
def timing_inputs(dev):
    cfg, pr, cp, bs = s5c.setup(0, PROBE, dev)
    b0, size, _ = s5c.geometry(cfg)
    res = {}
    for Bl in ["64", "512"]:
        B = int(Bl) if not QUICK else {"64": 8, "512": 16}[Bl]
        fields = s5.make_fields(cfg, B, seed=cfg.test_seed + 99, device=dev)
        cnt = s5c.measure_box(fields[:, :, b0:b0 + size, b0:b0 + size].contiguous(), PROBE, pr, cp, bs, cfg, seed=0)
        res[Bl] = (B, cnt, s5c.calibrate_norm(cnt))
    return cfg, pr, cp, bs, res


@torch.no_grad()
def time_model(net, group, B, cnt, norm, bs, cp, dev):
    for _ in range(3):
        net(s5c.prep(cnt, bs, norm, group, cp))
    ts = []
    for _ in range(s5b.TIME_REPS):
        s5._sync(dev)
        t0 = time.perf_counter()
        net(s5c.prep(cnt, bs, norm, group, cp))
        s5._sync(dev)
        ts.append((time.perf_counter() - t0) / B * 1e3)
    return float(np.median(ts))


@torch.no_grad()
def time_all(groups, dev):
    """{batch: {組: {實作: ms}}};先把所有模型各跑一次暖機。"""
    torch.backends.cudnn.benchmark = True
    cfg, pr, cp, bs, inputs = timing_inputs(dev)
    res = {}
    for Bl, (B, cnt, norm) in inputs.items():
        models = {g: variants_of(g, cfg, pr, dev) for g in groups}
        for g, vs in models.items():
            for m in vs.values():
                m(s5c.prep(cnt, bs, norm, g, cp))
        for g, vs in models.items():
            res.setdefault(Bl, {})[g] = {k: time_model(m, g, B, cnt, norm, bs, cp, dev) for k, m in vs.items()}
        del models
    return res


def best(t):
    k = min(t, key=t.get)
    return k, t[k]


# ============================================================================
# 等價檢查
# ============================================================================
@torch.no_grad()
def equivalence(dev):
    """NAF-B 與 B 的每個替代實作:輸出差、驗證場 nerr 與 P7 評估的差(原寫法也重算一次當對照)。
    回傳 {(組, 實作): 是否等價} 與明細。"""
    torch.backends.cudnn.benchmark = True                                   # 同 P7 評估時的設定
    raw_all = json.load(open(P7_RAW))["results"]
    detail = {}
    for g in (s5e.NAF_B, "B"):
        raw = raw_all[g]
        for i, s in enumerate(s5b.SEEDS):
            d = s5d.setup_seed(s, s5e.val_seeds, dev)
            path = s5c.run_dir(g, PROBE, s, N_TRAIN)
            state = torch.load(path / "final.pt", map_location=dev)
            norm = json.load(open(path / "result.json"))["norm"]
            vs = variants_of(g, d["cfg"], d["pr"], dev, state=state)
            preds = {k: s5c.predict(m, d["counts"], d["bs"], norm, g, d["cp"]) for k, m in vs.items()}
            for k, pk in preds.items():
                est = s5c.to_field(torch.polar(pk[:, 0], pk[:, 1]), d["cfg"], d["F"])
                dn = abs(s5.field_metrics(est, d["O"], d["R0"])["nerr_ph"] - raw[i]["net"]["nerr_ph"])
                do = float((pk - preds["原寫法"]).abs().max())
                o = detail.setdefault((g, k), [0.0, 0.0])
                detail[(g, k)] = [max(o[0], do), max(o[1], dn)]
            del vs, preds
    ok = {key: (v[0] < EQ_TOL and v[1] < EQ_TOL) for key, v in detail.items()}
    ctrl = [v[1] for (g, k), v in detail.items() if k == "原寫法"]
    check(f"對照:原寫法重算的驗證場 nerr_ph 與 P7 評估相同(< {EQ_TOL:.0e})", max(ctrl) < EQ_TOL, f"最大差 {max(ctrl):.1e}")
    for (g, k), v in detail.items():
        if k == "原寫法":
            continue
        check(f"等價:{g} {k} —— 輸出最大差與 nerr 差皆 < {EQ_TOL:.0e}(驗證場 3 seeds、已訓練權重)",
              ok[(g, k)], f"輸出 {v[0]:.1e};nerr {v[1]:.1e}")
    return ok, {f"{g}|{k}": v for (g, k), v in detail.items()}


# ============================================================================
# 輔助:torch.compile(網路與 AP-C 一次迭代)
# ============================================================================
def ap_step(O, meas, P, Pc, bs, den, delta, starts, W, phmax):
    """與 scan_5b.ap 的一次迭代相同(constrain=True)。"""
    Oj = torch.stack([O[:, y0:y0 + W, x0:x0 + W] for y0, x0 in starts], 1)
    psi = P * Oj
    E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    new_mag = torch.where(bs > 0, meas, E.abs())
    psi2 = torch.fft.ifft2(torch.fft.ifftshift(torch.polar(new_mag, torch.angle(E)), dim=(-2, -1)), norm="ortho")
    g = Pc * psi2
    num = delta * O
    for j, (y0, x0) in enumerate(starts):
        num[:, y0:y0 + W, x0:x0 + W] += g[:, j]
    O = num / den
    return torch.polar(O.abs().clamp(max=1.0), torch.angle(O).clamp(0, phmax))


@torch.no_grad()
def compile_aux(dev):
    out = {}
    if not hasattr(torch, "compile") or dev.type != "cuda":
        print("  (輔助)torch.compile:此環境不適用,略過")
        return {"status": "skipped"}
    from src.hio import _measured_amp
    cfg, pr, cp, bs, inputs = timing_inputs(dev)
    # 網路
    for g in ["B", s5e.NAF_B]:
        for Bl, (B, cnt, norm) in inputs.items():
            try:
                vs = variants_of(g, cfg, pr, dev)
                k = "實作2" if g.startswith("NAF") else "channels_last"
                ref = vs[k]
                comp = torch.compile(vs[k])
                t0 = time_model(ref, g, B, cnt, norm, bs, cp, dev)
                t1 = time_model(comp, g, B, cnt, norm, bs, cp, dev)
                out.setdefault(Bl, {})[g] = [t0, t1, t0 / t1]
            except Exception as e:                                           # noqa: BLE001
                out.setdefault(Bl, {})[g] = f"失敗:{type(e).__name__}"
    # AP-C 一次迭代
    st = s5.window_starts(cfg, s5b.SCAN)
    W, F = cfg.canvas, s5.field_size(cfg)
    wsum = s5b.illum_sum(cfg, pr, st, dev)
    delta = s5b.AP_DELTA * float(wsum.max())
    for Bl, (B, _, _) in inputs.items():
        try:
            fields = s5.make_fields(cfg, B, seed=cfg.test_seed + 99, device=dev)
            counts, _ = s5.measure_scan(fields, PROBE, pr, cp, bs, st, seed=0)
            meas = torch.stack([_measured_amp(c, cp) for c in counts], 1)
            O0 = s5.epie_init(B, F, cfg.phase_max, 0, dev)
            args = (meas, pr["P"], pr["P"].conj(), bs, wsum + delta, delta, st, W, cfg.phase_max)
            ref10 = s5b.ap(counts, bs, cp, pr, st, [10], True, O0)[10]
            step = torch.compile(ap_step)
            O = O0.clone()
            for _ in range(10):
                O = step(O, *args)
            diff = float((O - ref10).abs().max())
            ts = {}
            for name, fn in [("原寫法", ap_step), ("compile", step)]:
                O = O0.clone()
                for _ in range(3):
                    O = fn(O, *args)
                s5._sync(dev)
                t0 = time.perf_counter()
                for _ in range(20):
                    O = fn(O, *args)
                s5._sync(dev)
                ts[name] = (time.perf_counter() - t0) / 20 / B * 1e3
            out.setdefault(Bl, {})["AP-C"] = [ts["原寫法"], ts["compile"], ts["原寫法"] / ts["compile"], diff]
        except Exception as e:                                               # noqa: BLE001
            out.setdefault(Bl, {})["AP-C"] = f"失敗:{type(e).__name__}"
    return out


# ============================================================================
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    need = [Path(f) for f in ("scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py", "scan_5e.py")]
    need += [P7_RAW, s5e.VAL_ENV, s5e.PREP_JSON]
    need += [s5c.run_dir(g, PROBE, s, N_TRAIN) / f for g in ("B", "C", s5e.NAF_B) for s in s5b.SEEDS
             for f in ("final.pt", "result.json")]
    miss = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not miss, "缺:" + ", ".join(miss) if miss else "")
    if miss:
        sys.exit(1)

    print("\n" + "=" * 100)
    print("1. 等價檢查(協定 §17.3-1)")
    print("=" * 100)
    ok_impl, eq_detail = equivalence(dev)

    def allowed(g, k):
        """原寫法永遠可用;替代實作:NAF 以 NAF-B 的等價檢查為準(同一套程式碼),B 以 B 的檢查為準;其他組只用原寫法。"""
        if k == "原寫法":
            return True
        ref = s5e.NAF_B if g.startswith("NAF") else g
        return ok_impl.get((ref, k), False)

    print("\n" + "=" * 100)
    print("2. 新的時間(ms / 張;同一 job;各網路取自己最快的等價實作)")
    print("=" * 100)
    groups = ["B", "C", s5e.NAF_B] + s5e.SMALL_CANDIDATES
    tim_n = time_all(groups, dev)
    params = json.load(open(s5e.PREP_JSON))["params"]
    bestt = {}
    for Bl in ["64", "512"]:
        for g in groups:
            t = {k: v for k, v in tim_n[Bl][g].items() if allowed(g, k)}
            bestt.setdefault(Bl, {})[g] = best(t)
    tB = {Bl: bestt[Bl]["B"][1] for Bl in bestt}
    print(f"  {'網路':<10}{'參數':>11}   " + "   ".join(f"batch {Bl}:各實作 → 最快(× B)" for Bl in ["64", "512"]))
    for g in groups:
        cells = []
        for Bl in ["64", "512"]:
            det = " / ".join(f"{k} {v:.4f}" for k, v in tim_n[Bl][g].items())
            k, v = bestt[Bl][g]
            tie = "(與原寫法差 < 3%,視為相同)" if k != "原寫法" and v > 0.97 * tim_n[Bl][g]["原寫法"] else ""
            cells.append(f"{det} → {k} {v:.4f}(× {v / tB[Bl]:.2f}){tie}")
        print(f"  {g:<10}{params.get(g, 0):>11,}   " + "   ".join(cells))

    print("\n" + "=" * 100)
    print("3. 以新時間重新比較 NAF-B 與包絡(誤差沿用 P7 評估;迭代法同 job 重新計時)")
    print("=" * 100)
    import probe_4_2b as pb
    cfg0 = s5.load_cfg(0)
    probes0 = {k: v for k, v in pb.build_probes(cfg0, dev).items() if k in s5.PROBES}
    tim = s5b.timing(cfg0, probes0, dev)
    env = json.load(open(s5e.VAL_ENV))["envelope"]
    raw = json.load(open(P7_RAW))["results"]
    verdict = {}
    for g in ["B", "C", s5e.NAF_B]:
        a = np.array([r["net"]["nerr_ph"] for r in raw[g]], float)
        row = {}
        for Bl in ["64", "512"]:
            k, tn = bestt[Bl][g]
            t_old = tim_n[Bl][g]["原寫法"]
            e = s5b.envelope(env, PROBE, tn, tim[Bl])
            r = s5b.ratio(a, e[2])
            row[Bl] = {"impl": k, "t_new": tn, "t_old": t_old, "opp": [e[0], e[1], list(e[2])], "ratio": list(r)}
            print(f"  {g:<10} batch {Bl}:{k} {tn:.4f} ms(原寫法 {t_old:.4f},× {t_old / tn:.2f} 加速)"
                  f" nerr {a.mean():.4f} vs 對手 {s5b.fmt_conf(e[0])}({e[1]} 次){e[2].mean():.4f}"
                  f" → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}")
        verdict[g] = row
    nb = verdict[s5e.NAF_B]
    print(f"\n  NAF-B:推論時間 × {nb['64']['t_new'] / tB['64']:.2f} B(batch 64)/ × {nb['512']['t_new'] / tB['512']:.2f} B(batch 512);"
          f"batch 512 vs 包絡:{nb['512']['ratio'][0]}")
    small_ok = [g for g in s5e.SMALL_CANDIDATES if bestt["64"][g][1] <= s5e.SMALL_TIME_FRAC * tB["64"]]
    print("  NAF-B 小(只回報,協定 §17.3-5):新時間下 batch 64 ≤ 0.8 × B 的候選 = "
          + (", ".join(f"{g}({params.get(g, 0):,} 參數)" for g in small_ok) if small_ok else "無"))

    result = {"equivalence": eq_detail, "timing_nets": tim_n, "best": bestt, "timing_iter": tim, "verdict": verdict,
              "small_ok": small_ok, "quick": QUICK}
    json.dump(result, open(RUN_ROOT / f"scan5f{SUF}.json", "w"), indent=1)
    print(f"\n  主要結果先存檔:{RUN_ROOT / f'scan5f{SUF}.json'}")

    print("\n" + "=" * 100)
    print("4. 輔助:torch.compile(不進主判)")
    print("=" * 100)
    try:
        aux = compile_aux(dev)
    except Exception as e:                                                   # noqa: BLE001
        aux = {"status": f"失敗:{type(e).__name__}"}
        print(f"  (輔助)torch.compile 整體失敗:{type(e).__name__}(不影響主要結果)")
    for Bl, row in (aux.items() if isinstance(aux, dict) and "status" not in aux else []):
        for k, v in row.items():
            if isinstance(v, str):
                print(f"  batch {Bl} {k}:{v}")
            elif k == "AP-C":
                print(f"  batch {Bl} AP-C 一次迭代(只計迭代本身,不含前處理):{v[0]:.4f} → {v[1]:.4f} ms(× {v[2]:.2f});"
                      f"與原程式 10 次後最大差 {v[3]:.1e}")
            else:
                print(f"  batch {Bl} {k}:{v[0]:.4f} → {v[1]:.4f} ms(× {v[2]:.2f})")

    result["compile"] = aux
    json.dump(result, open(RUN_ROOT / f"scan5f{SUF}.json", "w"), indent=1)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過(未通過等價檢查的實作不採用)"))
    print("把完整輸出貼給 Claude")


if __name__ == "__main__":
    main()
