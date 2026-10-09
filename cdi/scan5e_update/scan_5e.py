#!/usr/bin/env python
"""優化第一步:前置 + P7(NAFNet 式骨幹)(階段五協定 §十六;優化方案 §六、§七)。

前置:驗證場(seed 80,000,000)與其迭代法包絡(之後所有優化方案共用)、預先計時。
P7:保留 U 型結構,零件換成 NAFBlock(Chen et al., ECCV 2022);輸入、輸出、loss、訓練設定與資料皆與 B 相同。
  NAF-B    寬度 32、每層 2 個區塊(參數量最接近 B;事先決定)
  NAF-B 小 由預先計時決定:batch 64 推論時間 ≤ 0.8 × B 的候選中參數最多者(沒有就不做)

用法(需在計算節點執行;需 scan_5.py、scan_5b.py、scan_5c.py、scan_5d.py 與其依賴、B / C 的 100k 模型):
    python scan_5e.py --prep             # 單元測試 → 驗證場包絡 → 預先計時與選定(run_scan5e_prep.sh)
    python scan_5e.py --train --task 0-5 # 訓練(run_scan5e_train.sh 的 array;0–2 = NAF-B、3–5 = NAF-B 小)
    python scan_5e.py --eval             # 驗證場上評估 B、C、NAF(run_scan5e_eval.sh)
    python scan_5e.py --check            # 只跑單元測試
"""
import argparse
import json
import os
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
import scan_5c as s5c                                                # noqa: E402  (匯入時已關閉 TF32,同 5-3)
import scan_5d as s5d                                                # noqa: E402

RUN_ROOT = s5.RUN_ROOT
QUICK = s5c.QUICK
PROBE = "DEF2"
N_TRAIN = s5c.N_TRAIN if QUICK else 100_000
VAL_FIELD_SEED = 80_000_000
VAL_NOISE_BASE = 81_000_000
CONFIRM_SEED = 90_000_000                    # §十五 已用
FINAL_SEED = 95_000_000                      # 最終確認場(保留,只在不重疊檢查中用到)
NAF_B = "NAFw32b2"
NAF_B_PARAMS = 1_916_674
SMALL_CANDIDATES = [f"NAFw{w}b{b}" for w in (16, 24) for b in (1, 2, 3)] + ["NAFw32b1"]
SMALL_TIME_FRAC = 0.8
SUF = "_quick" if QUICK else ""
PREP_JSON = RUN_ROOT / f"scan5e_prep{SUF}.json"
VAL_ENV = RUN_ROOT / f"scan5e_val_envelope{SUF}.json"
FIG_DIR = Path("figs_scan5e")

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def val_seeds(cfg, s):
    return {"field": VAL_FIELD_SEED, "noise": VAL_NOISE_BASE + 1000 * s, "init": VAL_NOISE_BASE + 1000 * s + 1}


# ============================================================================
# NAFNet 式 U 型網路
# ============================================================================
class LayerNorm2d(nn.Module):
    """對通道維度做 LayerNorm(每個像素各自正規化),同 NAFNet。"""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        v = (x - m).pow(2).mean(1, keepdim=True)
        x = (x - m) / torch.sqrt(v + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv3 = nn.Conv2d(c, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, 2 * c, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))          # 初始 0:區塊一開始等於恆等映射
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.sg(self.dw(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFUNet(nn.Module):
    """U 型(下採樣 3 次:80 → 40 → 20 → 10),零件為 NAFBlock;輸出頭與約束同 B(振幅 sigmoid、相位 φmax·sigmoid)。"""

    def __init__(self, cfg, width, n_blocks, cin=19, n_down=3):
        super().__init__()
        self.phase_max = cfg.phase_max
        self.intro = nn.Conv2d(cin, width, 3, padding=1)
        self.enc, self.down, self.up, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = width
        for _ in range(n_down):
            self.enc.append(nn.Sequential(*[NAFBlock(c) for _ in range(n_blocks)]))
            self.down.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(n_blocks)])
        for _ in range(n_down):
            self.up.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.dec.append(nn.Sequential(*[NAFBlock(c) for _ in range(n_blocks)]))
        self.end = nn.Conv2d(width, 2, 3, padding=1)

    def forward(self, inp):
        x = self.intro(inp["x"])
        skips = []
        for e, d in zip(self.enc, self.down):
            x = e(x)
            skips.append(x)
            x = d(x)
        x = self.mid(x)
        for u, d, k in zip(self.up, self.dec, reversed(skips)):
            x = d(u(x) + k)
        z = self.end(x)
        return torch.cat([torch.sigmoid(z[:, 0:1]), self.phase_max * torch.sigmoid(z[:, 1:2])], 1)


def parse_naf(name):
    w, b = name[len("NAFw"):].split("b")
    return int(w), int(b)


_orig_build = s5c.build_model


def build_model(group, cfg, pr):
    if group.startswith("NAF"):
        w, b = parse_naf(group)
        return NAFUNet(cfg, w, b)
    return _orig_build(group, cfg, pr)


s5c.build_model = build_model                                        # 讓 scan_5c 的 train / load_net 也認得 NAF


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


# ============================================================================
# 計時(同 scan_5c.time_nets,組別可任意列出)
# ============================================================================
@torch.no_grad()
def time_groups(groups, dev):
    torch.backends.cudnn.benchmark = True
    cfg, pr, cp, bs = s5c.setup(0, PROBE, dev)
    b0, size, _ = s5c.geometry(cfg)
    res = {}
    for Bl in ["64", "512"]:
        B = int(Bl) if not QUICK else {"64": 8, "512": 16}[Bl]
        fields = s5.make_fields(cfg, B, seed=cfg.test_seed + 99, device=dev)
        cnt = s5c.measure_box(fields[:, :, b0:b0 + size, b0:b0 + size].contiguous(), PROBE, pr, cp, bs, cfg, seed=0)
        norm = s5c.calibrate_norm(cnt)
        for g in groups:                                                     # 先把每個網路都跑過一次(GPU 暖機,避免第一個被量得偏慢)
            build_model(g, cfg, pr).to(dev).eval()(s5c.prep(cnt, bs, norm, g, cp))
        for g in groups:
            net = build_model(g, cfg, pr).to(dev).eval()
            net(s5c.prep(cnt, bs, norm, g, cp))
            net(s5c.prep(cnt, bs, norm, g, cp))
            ts = []
            for _ in range(s5b.TIME_REPS):
                s5._sync(dev)
                t0 = time.perf_counter()
                net(s5c.prep(cnt, bs, norm, g, cp))
                s5._sync(dev)
                ts.append((time.perf_counter() - t0) / B * 1e3)
            res.setdefault(Bl, {})[g] = float(np.median(ts))
            del net
    return res


# ============================================================================
# 單元測試
# ============================================================================
def unit_tests(dev, groups=None, data_check=True):
    import probe_4_1 as p41
    groups = groups or [NAF_B]
    print("=" * 70)
    print(f"優化 P7 單元測試(組:{' '.join(groups)})")
    print("=" * 70)
    cfg = s5.load_cfg(0)
    torch.manual_seed(0)
    # (1) LayerNorm2d = 對通道維度的 F.layer_norm
    ln = LayerNorm2d(8)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.uniform_(-0.5, 0.5)
    x = torch.randn(3, 8, 5, 5)
    ref = Fnn.layer_norm(x.permute(0, 2, 3, 1), (8,), ln.weight, ln.bias, eps=1e-6).permute(0, 3, 1, 2)
    check("LayerNorm2d = 通道維度的 F.layer_norm", float((ln(x) - ref).abs().max()) < 1e-5)
    # (2) 事先寫定的 NAF-B 參數量
    nb = n_params(build_model(NAF_B, cfg, None))
    check(f"NAF-B({NAF_B})參數量 = 事先寫定的 {NAF_B_PARAMS:,}", nb == NAF_B_PARAMS, f"{nb:,}")

    import probe_4_2b as pb
    from src.physics import beamstop_mask
    pr = pb.build_probes(cfg, dev)[PROBE]
    cp = s5.scan_cfg(cfg, PROBE, pr)
    bs = beamstop_mask(cfg, device=dev)
    b0, size, _ = s5c.geometry(cfg)
    # (3) 前向:輸出大小與範圍
    fld = s5.make_fields(cfg, 8, seed=11, device=dev)[:, :, b0:b0 + size, b0:b0 + size].contiguous()
    cnt = s5c.measure_box(fld, PROBE, pr, cp, bs, cfg, seed=1)
    norm = s5c.calibrate_norm(cnt)
    ok_f = True
    for g in groups:
        with torch.no_grad():
            out = build_model(g, cfg, pr).to(dev).eval()(s5c.prep(cnt, bs, norm, g, cp))
        ok_f &= (tuple(out.shape) == (8, 2, s5c.BOX_SIZE, s5c.BOX_SIZE) and bool(out[:, 0].min() >= 0)
                 and bool(out[:, 0].max() <= 1) and bool(out[:, 1].min() >= 0)
                 and bool(out[:, 1].max() <= cfg.phase_max + 1e-6))
    check("前向:輸出 [N, 2, 80, 80]、振幅 ∈ [0, 1]、相位 ∈ [0, φmax]", ok_f)
    # (4) 可學性(同 5-3 的標準)
    small = s5.make_fields(cfg, s5c.LEARN_N, seed=12, device=dev)[:, :, b0:b0 + size, b0:b0 + size].contiguous()
    Os = torch.polar(small[:, 0], small[:, 1])
    R0b = s5c.r0_box(cfg, pr, dev)
    cnt = s5c.measure_box(small, PROBE, pr, cp, bs, cfg, seed=3)
    norm = s5c.calibrate_norm(cnt)
    learn = {}
    for g in groups:
        torch.manual_seed(0)
        net = build_model(g, cfg, pr).to(dev).train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        inp = s5c.prep(cnt, bs, norm, g, cp)
        first = None
        for _ in range(s5c.LEARN_STEPS):
            loss = s5c.nerr_loss(net(inp), Os, R0b)
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            learn[g] = (first, float(s5c.nerr_loss(net(inp), Os, R0b)))
    check(f"可學性:{s5c.LEARN_N} 個樣本訓練 {s5c.LEARN_STEPS} 步,loss 降到初始的 50% 以下" + (";QUICK 只列出" if QUICK else ""),
          QUICK or all(b < 0.5 * a for a, b in learn.values()),
          ";".join(f"{k} {a:.3f} → {b:.3f}" for k, (a, b) in learn.items()))
    # (5) 驗證場不重疊
    if data_check:
        d = s5.tile_size(cfg)
        s0 = (cfg.canvas - d) // 2
        n_t = (s5b.N_FIELDS or cfg.eval_n) * s5.TILES ** 2
        crop = (lambda t: t[:, :, s0:s0 + d, s0:s0 + d])
        hv = s5c._tile_hashes(crop(p41.make_home(cfg, n_t, seed=VAL_FIELD_SEED)))
        others = {"舊測試": cfg.test_seed + s5.FIELD_SEED_OFFSET, "§十五 測試": CONFIRM_SEED, "最終確認": FINAL_SEED}
        inter = {k: len(hv & s5c._tile_hashes(crop(p41.make_home(cfg, n_t, seed=v)))) for k, v in others.items()}
        tr = torch.cat([crop(p41.make_home(cfg, 64 * s5.TILES ** 2, seed=s5c.train_chunk_seed(s, k)))
                        for s in s5c.SEEDS for k in (0, max(0, N_TRAIN // s5c.CHUNK - 1))])
        inter["訓練(抽樣)"] = len(hv & s5c._tile_hashes(tr))
        tr_max = s5c.train_chunk_seed(max(s5c.SEEDS), N_TRAIN // s5c.CHUNK) + 3000
        far = VAL_FIELD_SEED - tr_max > 1_000_000 and all(abs(VAL_FIELD_SEED - v) > 1_000_000 for v in others.values())
        check("驗證場不重疊:seed 遠離訓練 / 各測試場;非空白物體塊皆無相同", far and all(v == 0 for v in inter.values()),
              "相同的塊:" + "、".join(f"{k} {v}" for k, v in inter.items()))
    print(f"\n     裝置:{dev}")


# ============================================================================
# prep:驗證場包絡 + 預先計時
# ============================================================================
def prep(dev):
    if VAL_ENV.exists():
        print(f"  驗證場包絡已存在,沿用:{VAL_ENV}")
    else:
        print("  驗證場:迭代法包絡(DEF2,全部設定)", flush=True)
        env = s5d.measure_envelope(val_seeds, dev)
        tmp = VAL_ENV.with_suffix(".tmp")
        json.dump({"envelope": env, "field_seed": VAL_FIELD_SEED, "quick": QUICK}, open(tmp, "w"))
        os.replace(tmp, VAL_ENV)                                              # 寫完才換名,避免留下不完整的檔
        print(f"  存檔:{VAL_ENV}")
    print("\n  預先計時(未訓練權重;ms / 張)", flush=True)
    cfg = s5.load_cfg(0)
    groups = ["B", "C", NAF_B] + SMALL_CANDIDATES
    tim = time_groups(groups, dev)
    pr_cpu = s5c.setup(0, PROBE, torch.device("cpu"))[1]
    params = {g: n_params(build_model(g, cfg, pr_cpu)) for g in groups}
    tB = tim["64"]["B"]
    for g in groups:
        print(f"    {g:<10} 參數 {params[g]:>10,}  batch 64 {tim['64'][g]:.4f}(× {tim['64'][g] / tB:.2f} B)"
              f"  batch 512 {tim['512'][g]:.4f}(× {tim['512'][g] / tim['512']['B']:.2f} B)")
    ok = [g for g in SMALL_CANDIDATES if tim["64"][g] <= SMALL_TIME_FRAC * tB]
    small = max(ok, key=lambda g: params[g]) if ok else None
    print(f"\n  NAF-B = {NAF_B}(事先決定)")
    print(f"  NAF-B 小 = {small if small else '無(沒有候選的 batch 64 時間 ≤ 0.8 × B)→ 不做'}"
          f"(規則:batch 64 時間 ≤ {SMALL_TIME_FRAC} × B 的候選中參數最多者)")
    tmp = PREP_JSON.with_suffix(".tmp")
    json.dump({"naf_b": NAF_B, "naf_small": small, "timing": tim, "params": params, "quick": QUICK},
              open(tmp, "w"), indent=2)
    os.replace(tmp, PREP_JSON)
    print(f"  存檔:{PREP_JSON}")


def variants():
    d = json.load(open(PREP_JSON))
    return [d["naf_b"], d["naf_small"]]


# ============================================================================
# 評估(驗證場)
# ============================================================================
@torch.no_grad()
def evaluate(dev):
    import probe_4_2b as pb
    env = json.load(open(VAL_ENV))["envelope"]
    groups = ["B", "C"] + [g for g in variants() if g]
    cfg0 = s5.load_cfg(0)
    probes0 = {k: v for k, v in pb.build_probes(cfg0, dev).items() if k in s5.PROBES}
    print("  重新計時:迭代法與網路", flush=True)
    tim = s5b.timing(cfg0, probes0, dev)
    tg = time_groups(groups, dev)
    for B in tg:
        tim[B].update({f"net:{g}": v for g, v in tg[B].items()})
    res = {g: [] for g in groups}
    missing = []
    for s in s5b.SEEDS:
        d = s5d.setup_seed(s, val_seeds, dev)
        for g in groups:
            net, norm, meta = s5c.load_net(g, PROBE, s, N_TRAIN, d["cfg"], d["pr"], dev)
            if net is None:
                missing.append(f"{g}:s{s}")
                res[g].append(None)
                continue
            pred = s5c.predict(net, d["counts"], d["bs"], norm, g, d["cp"])
            est = s5c.to_field(torch.polar(pred[:, 0], pred[:, 1]), d["cfg"], d["F"])
            res[g].append({"net": s5.field_metrics(est, d["O"], d["R0"]), "params": meta["params"],
                           "train_nerr_ph": meta["train_nerr_ph"], "train_sec": meta.get("train_sec")})
            del net, pred, est
        print(f"  [seed {s}] 完成", flush=True)
    return env, res, tim, missing


def vals(res, g, field="nerr_ph"):
    if any(r is None for r in res[g]):
        return None
    return np.array([r["net"][field] for r in res[g]], float)


def report(env, res, tim, missing):
    v = {"missing": missing}
    Bs = ["64", "512"]
    groups = list(res)
    print("\n" + "=" * 100)
    print("計時(ms / 張;本 job 重新量)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {tim[B][k]:.4f}" for k in
                                         ["ePIE-C", "AP-C", f"K-HIO:{PROBE}"] + [f"net:{g}" for g in groups]))
    if missing:
        print(f"  ⚠️ 缺少的網路:{', '.join(missing)}")
    print("\n" + "=" * 100)
    print(f"驗證場({PROBE},R0 的 nerr_ph,3 seeds)與 vs 包絡")
    print("=" * 100)
    b = vals(res, "B")
    for g in groups:
        a = vals(res, g)
        if a is None:
            print(f"  {g}:缺")
            continue
        tr = np.mean([r["train_nerr_ph"] for r in res[g]])
        secs = [r["train_sec"] for r in res[g] if r["train_sec"]]
        print(f"  {g}:{a.mean():.4f}({' '.join(f'{x:.4f}' for x in a)})  參數 {res[g][0]['params']:,}  訓練場 {tr:.4f}"
              + (f"  訓練時間 {np.mean(secs) / 60:.0f} 分" if secs else ""))
        gv = {"nerr": list(a)}
        for B in Bs:
            tn = tim[B][f"net:{g}"]
            e = s5b.envelope(env, PROBE, tn, tim[B])
            r = s5b.ratio(a, e[2])
            gv[B] = {"t_net": tn, "opp": [e[0], e[1], list(e[2])], "ratio": list(r)}
            print(f"     (n4) batch {B}:{tn:.4f} ms vs 對手 {s5b.fmt_conf(e[0])}({e[1]} 次){e[2].mean():.4f}"
                  f" → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}")
        v[g] = gv
    print("\n" + "=" * 100)
    print("與 B 的配對比較(協定 §16.3;優化方案 §5.2 / §7.2)")
    print("=" * 100)
    for g in groups:
        if g in ("B", "C") or vals(res, g) is None or b is None:
            continue
        a = vals(res, g)
        acc = s5b.ratio(a, b)
        t64, t512 = tim["64"][f"net:{g}"] / tim["64"]["net:B"], tim["512"][f"net:{g}"] / tim["512"]["net:B"]
        more_acc = acc[1] < 1 and acc[2] < -2 and t64 <= 1.1               # 時間以 batch 64 為主判(協定 §16.3)
        faster = t64 <= 0.95 and t512 <= 1.0 and acc[1] <= 1.03
        thr_acc = acc[1] <= 0.9 and acc[2] < -2 and t64 <= 1.1
        thr_fast = t64 <= 0.83 and t512 <= 1.0 and acc[1] <= 1.03
        lab = ("有改善" if (more_acc or faster) else "無改善 / 變差")
        thr = "達門檻" if (thr_acc or thr_fast) else ("未達門檻" if (more_acc or faster) else "")
        print(f"  {g}:誤差比值 {acc[1]:.3f}(z {acc[2]:+.1f});推論時間 × {t64:.2f}(batch 64)/ × {t512:.2f}(batch 512)")
        print(f"     更準:{'✅' if more_acc else '—'}{'(達 −10% 門檻)' if thr_acc else ''}  "
              f"更快:{'✅' if faster else '—'}{'(達 1.2 倍門檻)' if thr_fast else ''}  → {lab}{'、' + thr if thr else ''}")
        v[g]["vs_B"] = {"acc": list(acc), "t64": t64, "t512": t512, "more_acc": more_acc, "faster": faster,
                        "thr_acc": thr_acc, "thr_fast": thr_fast}
    print("\n  採用與否:回報後與使用者討論決定(優化方案 §5.2b);採用者成為 P6 的基礎")
    return v


def make_figure(env, res, tim):
    plt = s5._plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    col = {"B": "#2a78d6", "C": "#eb6834"}
    for ax, B in zip(axs, ["64", "512"]):
        tt = sorted({s5b.cfg_time(c, it, tim[B], PROBE) for c in s5b.configs() for it in s5b.stops(c)} - {0.0})
        ax.plot(tt, [s5b.envelope(env, PROBE, T, tim[B])[2].mean() for T in tt], color="#0b0b0b", lw=2,
                label="iterative envelope (validation)")
        for g in res:
            a = vals(res, g)
            if a is None:
                continue
            ax.plot([tim[B][f"net:{g}"]], [a.mean()], "o", ms=7, color=col.get(g, "#1baf7a" if g == NAF_B else "#8a5cd1"),
                    label=g)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{PROBE}, batch {B} (validation fields)", fontsize=10)
        ax.set_xlabel("time per sample (ms)")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    axs[0].set_ylabel("nerr on R0 (global phase only)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    p = FIG_DIR / "scan5e_p7.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


# ============================================================================
def check_files(need_models=True, need_prep=False, need_env=False):
    need = [Path(f) for f in ("scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py", "probe_4_2b.py", "probe_4_1.py",
                              "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in s5b.SEEDS]
    if need_models:
        need += [s5c.run_dir(g, PROBE, s, N_TRAIN) / f for g in ("B", "C") for s in s5b.SEEDS
                 for f in ("final.pt", "result.json")]
    if need_prep:
        need.append(PREP_JSON)
    if need_env:
        need.append(VAL_ENV)
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    if a.check:
        if not check_files(need_models=False):
            sys.exit(1)
        unit_tests(dev, [NAF_B] + SMALL_CANDIDATES[:1])
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過"))
        sys.exit(0 if ok_all else 1)
    if a.prep:
        if not check_files(need_models=True):
            sys.exit(1)
        unit_tests(dev, [NAF_B])
        if not ok_all:
            print("\n❌ 單元測試未通過,停下來")
            sys.exit(1)
        prep(dev)
        print("\n✅ prep 完成")
        return
    if a.train:
        if not check_files(need_models=False, need_prep=True):
            sys.exit(1)
        if a.task is None or not 0 <= a.task <= 5:
            raise SystemExit("--train 需要 --task 0–5")
        vs = variants()
        g, s = vs[a.task // 3], s5c.SEEDS[a.task % 3]
        if g is None:
            print(f"[task {a.task}] NAF-B 小 未選定(預先計時沒有符合的候選)→ 不訓練,正常結束")
            return
        print(f"[task {a.task}] {g}、{PROBE}、seed {s}、訓練場 {N_TRAIN}", flush=True)
        unit_tests(dev, [g], data_check=False)
        if not ok_all:
            print("\n❌ 單元測試未通過,不訓練")
            sys.exit(1)
        s5c.train(g, PROBE, s, N_TRAIN, dev)
        print("✅ 訓練完成")
        return
    if a.eval:
        if not check_files(need_models=True, need_prep=True, need_env=True):
            sys.exit(1)
        env, res, tim, missing = evaluate(dev)
        raw = RUN_ROOT / f"scan5e_p7_raw{SUF}.json"
        json.dump({"results": res, "timing": tim, "missing": missing, "quick": QUICK}, open(raw, "w"))
        print(f"  原始結果先存檔:{raw}")
        verdict = report(env, res, tim, missing)
        json.dump({"results": res, "timing": tim, "verdict": verdict, "quick": QUICK},
                  open(RUN_ROOT / f"scan5e_p7{SUF}.json", "w"))
        make_figure(env, res, tim)
        print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))
        print("把完整輸出貼給 Claude")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
