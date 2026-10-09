#!/usr/bin/env python
"""階段七 7b:容忍型訓練(階段七協定 §十四)。架構不變(P6B4),訓練資料加入隨機的真實誤差 → P6B4r。

訓練(每個樣本各自抽):位置誤差 σ ~ U(0, 1.0 px)、9 個視窗各 N(0, σ²)(截在 ± 3σ,再減去 9 個視窗的平均);
      離焦 ε ~ {−0.30, −0.29, …, +0.30}(61 個真探針的探針庫,同孔徑、同總能量);
      劑量:一半 = 1,另一半 log-uniform[0.01, 1](計數 = Poisson(劑量 × 期望)/ 劑量)。部分同調不放進訓練。
      其他全部同 P6B4(從 B 出發、20 epochs、batch 128、學習率 2e-4 / 2e-3、OneCycle、梯度裁切 1.0、同一批訓練場)。
評估(12 個條件):7a 的 8 個(迭代法包絡沿用 7a 的條件檔)+ cohL / cohH(部分同調,只測試)+ posX / prbX(超出訓練範圍)。
比較方式(老師的建議):達到品質門檻 Q(nerr_sr = 0.05 / 0.02 / 0.01;0.02 為主)時,網路比迭代法包絡快多少;
      副指標 SSIM(振幅、相位);延續 7a 的同時間比值。

用法(需在計算節點執行):
    python scan_7b.py --check          # 只跑內建檢查
    python scan_7b.py --smoke          # 迷你全流程(極小的訓練、4 個新條件、彙整;dev 節點);通過才可正式執行
    python scan_7b.py --task 0..6      # run_scan7b_jobs.sh 的 array:0–2 = 訓練 P6B4r seed 0–2;3–6 = 新條件 cohL、cohH、posX、prbX
    python scan_7b.py --final          # 彙整(run_scan7b_final.sh;等 7 個 job 都完成)
"""
import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_7a as a7                                                 # noqa: E402  (7a-1;不修改,只在本程式內擴充條件)

a6, h, g = a7.a6, a7.h, a7.g
s5, s5b, s5c = a7.s5, a7.s5b, a7.s5c
RUN_ROOT = a7.RUN_ROOT
QUICK = a7.QUICK
PROBE = a7.PROBE
SEEDS = a7.SEEDS
SUF = a7.SUF
A7_MD5 = "44c350438076dc75ce875206d5d1c97f"  # 7a-1 正式結果所用的 scan_7a.py
NEW = "P6B4r"
NETS = ["P6B4", NEW]
PIPES = a7.PIPES                             # G9(主)、G25(附帶測量)
PRIMARY = "G9"
MAIN = a7.MAIN                               # nerr_sr
OLD_CONDS = list(a7.CONDS)
NEW_CONDS = ["cohL", "cohH", "posX", "prbX"]
CONDS = OLD_CONDS + NEW_CONDS
HELDOUT = ["cohL", "cohH"]
OUTRANGE = {"posX": "posH", "prbX": "prbH"}  # 超出範圍的條件 → 訓練範圍內最重的同類條件
NEW_SPEC = {"cohL": dict(pos=0.0, df=0.0, dose=1.0, coh=0.5), "cohH": dict(pos=0.0, df=0.0, dose=1.0, coh=1.0),
            "posX": dict(pos=1.5, df=0.0, dose=1.0), "prbX": dict(pos=0.0, df=0.45, dose=1.0)}
NEW_LABEL = {"cohL": "partial coherence 0.5px", "cohH": "partial coherence 1.0px", "posX": "position sigma 1.5px",
             "prbX": "defocus ±45%"}
MAIN_COND = "combo"
# ---- 訓練的隨機誤差(§14.2)----
TR_POS_MAX = 1.0                             # σ ~ U(0, 1.0 px)
TR_DF_MAX, TR_DF_STEP = 0.30, 0.01           # ε ∈ {−0.30, …, +0.30}
DF_GRID = [round(-TR_DF_MAX + k * TR_DF_STEP, 2) for k in range(int(round(2 * TR_DF_MAX / TR_DF_STEP)) + 1)]   # 61 個
TR_DOSE_P1 = 0.5                             # 劑量 = 1 的比例
TR_DOSE_LOGMIN = -2.0                        # 其餘:10^U(−2, 0)
Z_CLIP = a7.Z_CLIP                           # ± 3σ
ERR_SEED_BASE = 1_500_000_000                # 每步誤差的亂數基底(遠離其他 seed)
EVAL_ERR_SEED = 1_900_000_000                # 訓練場上「有誤差」的誤差(描述用)
# ---- 部分同調(§14.3):3 × 3 Gauss–Hermite ----
SQ3 = math.sqrt(3.0)
COH_NODES = [(-SQ3, 1.0 / 6.0), (0.0, 2.0 / 3.0), (SQ3, 1.0 / 6.0)]
# ---- 判讀(§14.5)----
QS = [0.05, 0.02, 0.01]
Q_MAIN = 0.02
SPEED_MIN = 1.25
EST_ABS, EST_GAIN = 0.05, 2.0                # 估計型架構的評估條件
SSIM_WIN, SSIM_SIG = 11, 1.5
AMP_MIN = a7.AMP_MIN
REPRO_TOL = a7.REPRO_TOL
COH_COUNT_TOL = 0.03                         # 部分同調:總計數 / 理想(無雜訊)與 1 的差
# ---- 檔案 ----
TASKS = [("train", 0), ("train", 1), ("train", 2)] + [("env", c) for c in NEW_CONDS]
SMOKE_DIR = RUN_ROOT / f"scan7b_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
SMOKE_TRAIN_N = s5c.N_TRAIN if QUICK else 2000      # ≥ CAL_N 與 TRAIN_EVAL_N
SMOKE_EPOCHS = 1
FIG_DIR = Path("figs_scan7b")
RECON_CONDS = ["ideal", "combo", "posH", "prbH", "cohH", "posX"]
COL = {"P6B4": "#9a9994", NEW: "#1baf7a", "iter": "#0b0b0b"}

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and a7.all_ok()


md5 = a7.md5
dump_json = a7.dump_json
absdiff = a7.absdiff


def me_md5():
    return md5(Path(__file__).resolve())


def env_json(c, root=None):
    return (root or RUN_ROOT) / f"scan7b_env_{c}{SUF}.json"


def final_json(root=None):
    return (root or RUN_ROOT) / f"scan7b{SUF}.json"


def model_dir(s, mroot=None):
    return (mroot / f"{NEW}_s{s}") if mroot is not None else s5c.run_dir(NEW, PROBE, s, a6.N_TRAIN)


def clabel(c):
    return a7.CLABEL[c]


# ============================================================================
# 新條件:擴充 7a 的條件表與量測(舊條件的量測完全不變;內建檢查)
# ============================================================================
a7.SPEC.update(NEW_SPEC)
a7.CLABEL.update(NEW_LABEL)
_a7_is_ideal = a7.is_ideal
_a7_measure = a7.measure_cond


def is_ideal(spec):
    return _a7_is_ideal(spec) and spec.get("coh", 0.0) == 0


def coh_modes(sig):
    """3 × 3 Gauss–Hermite:位移 (a·σ, b·σ)、權重 w_a·w_b(Σw = 1、平均位移 0、各方向變異數 σ²)。"""
    return [(a * sig, b * sig, wa * wb) for a, wa in COH_NODES for b, wb in COH_NODES]


@torch.no_grad()
def measure_coh(fields, geo, cp, bs, seed, sig, ops, poisson=None):
    """部分同調(有限光源大小):期望強度 = Σ_m w_m |F{P(r − s_m)·O}|²(探針模式的非同調疊加),再抽 Poisson。
    雜訊的抽法同 s5.measure_scan(torch.manual_seed(seed) 後依位置順序)。"""
    from src.physics import forward_measure
    W, P, dev = geo.W, geo.pr["P"], fields.device
    modes = [(ops.shift(P, torch.tensor(dy, device=dev), torch.tensor(dx, device=dev)), w) for dy, dx, w in coh_modes(sig)]
    poisson = cp.add_poisson if poisson is None else poisson
    torch.manual_seed(seed)
    counts = []
    for (y0, x0) in geo.starts:
        win = s5.crop(fields, y0, x0, W)
        Ow = torch.polar(win[:, 0], win[:, 1])
        I = None
        for Pm, w in modes:
            psi = Pm * Ow
            v = w * forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), bs, cp, add_poisson=False)
            I = v if I is None else I + v
        if poisson:
            I = torch.poisson(I.clamp_min(0)) * bs
        counts.append(I)
    return counts


@torch.no_grad()
def measure_cond(fields, geo, cp, bs, seed, spec, z, sign, ops, force_generic=False):
    if spec.get("coh", 0.0) > 0:
        if spec["pos"] != 0 or spec["df"] != 0 or spec["dose"] != 1:
            raise ValueError("部分同調只與理想的位置 / 探針 / 劑量組合(本協定的設計)")
        return measure_coh(fields, geo, cp, bs, seed, spec["coh"], ops)
    return _a7_measure(fields, geo, cp, bs, seed, spec, z, sign, ops, force_generic)


a7.is_ideal = is_ideal
a7.measure_cond = measure_cond


# ============================================================================
# 訓練用的隨機誤差量測(9 個視窗、方框 80 × 80)
# ============================================================================
def probe_bank(pr, dev):
    """DF_GRID 上的真探針 [61, W, W](同 scan_7a.true_probe_pair 的作法:同孔徑、總能量 = 名目)。"""
    import probe_4_2b as pb
    W = pr["P"].shape[-1]
    ka = float(pr["meta"]["ka"])
    d0 = float(pr["meta"]["defocus"]) * math.pi
    E0 = float((pr["P64"].abs() ** 2).sum())
    out = []
    for eps in DF_GRID:
        p = pb.aperture_probe(W, ka, defocus=d0 * (1.0 + eps))
        p = p * math.sqrt(E0 / float((p.abs() ** 2).sum()))
        out.append(p.to(torch.complex64))
    return torch.stack(out).to(dev)


def draw_train_errors(B, gen):
    """每個樣本的誤差(CPU 亂數,固定順序抽):δ [B, 9, 2](px,已減去 9 個視窗的平均)、探針庫編號 [B]、劑量 [B]、σ [B]。"""
    sig = torch.rand(B, generator=gen) * TR_POS_MAX
    z = torch.randn(B, 9, 2, generator=gen).clamp(-Z_CLIP, Z_CLIP)
    idx = torch.randint(0, len(DF_GRID), (B,), generator=gen)
    hi = torch.rand(B, generator=gen) < TR_DOSE_P1
    u = torch.rand(B, generator=gen)
    dose = torch.where(hi, torch.ones(B), 10.0 ** (TR_DOSE_LOGMIN * u))
    d = sig[:, None, None] * z
    d = d - d.mean(1, keepdim=True)
    return d, idx, dose, sig


@torch.no_grad()
def measure_rand(obj, rel, Pf, ops, cp, bs, delta, idx, dose, gen=None):
    """obj [B, 2, S, S](振幅、相位);rel:9 個視窗在方框中的左上角;Pf:探針庫的頻譜 [K, W, W];
    第 j 個視窗的探針 = 探針庫[idx] 平移 +δ_j(頻率空間);計數 = Poisson(劑量 × 期望)/ 劑量(gen = None → 不加雜訊)。
    期望強度的尺度同 src.physics.forward_measure。回傳 9 個 [B, W, W]。"""
    W = cp.canvas
    dev = obj.device
    delta, idx, dose = delta.to(dev), idx.to(dev), dose.to(dev)
    Sf = Pf[idx][:, None] * ops.ramp(delta[..., 0], delta[..., 1])                 # [B, 9, W, W]
    Pj = torch.fft.ifft2(Sf)
    win = torch.stack([obj[:, :, y:y + W, x:x + W] for y, x in rel], 1)            # [B, 9, 2, W, W]
    psi = Pj * torch.polar(win[:, :, 0], win[:, :, 1])
    E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    I = E.abs() ** 2 * (cp.photons_per_pix * W * W / cp.ref_energy)
    dd = dose[:, None, None, None]
    if gen is not None:
        I = torch.poisson((I * dd).clamp_min(0), generator=gen) / dd
    I = I * bs
    return [I[:, j] for j in range(I.shape[1])]


class TrainMeas:
    """訓練量測的共用物件(探針庫、平移工具、方框幾何)。"""

    def __init__(self, cfg, pr, cp, bs, dev):
        self.cfg, self.pr, self.cp, self.bs, self.dev = cfg, pr, cp, bs, dev
        self.rel = s5c.geometry(cfg)[2]
        self.ops = a7.Ops(pr, cfg.canvas, dev)
        self.bank = probe_bank(pr, dev)
        self.Pf = torch.fft.fft2(self.bank)
        self.gen = torch.Generator(device=dev)

    def __call__(self, obj, err_seed, noise_seed):
        gc = torch.Generator().manual_seed(int(err_seed))
        d, idx, dose, sig = draw_train_errors(obj.shape[0], gc)
        self.gen.manual_seed(int(noise_seed))
        return measure_rand(obj, self.rel, self.Pf, self.ops, self.cp, self.bs, d, idx, dose, self.gen), (d, idx, dose, sig)


# ============================================================================
# 訓練 P6B4r(同 scan_5h.train 的 P6B4 分支;唯一差別:每步的量測帶隨機誤差)
# ============================================================================
def nerr_on(model, sub, counts, norm, cp, bs, R0):
    """= scan_5h.train_nerr 的算法,量測由參數給(逐樣本平均、合併)。"""
    pred = g.predict(model, counts, bs, norm, "P6B4", cp)
    O = torch.polar(sub[:, 0], sub[:, 1])
    Et = ((O.abs() ** 2) * R0).sum((1, 2))
    tr = [float(s5c.nerr_loss(pred[i:i + 1], O[i:i + 1], R0)) for i in range(len(sub)) if Et[i] > 1e-12]
    return float(np.mean(tr)), float(s5c.nerr_loss(pred, O, R0))


def train_r(seed, dev, n_fields=None, epochs=None, out=None):
    n_fields = a6.N_TRAIN if n_fields is None else n_fields
    epochs = epochs or g.EPOCHS
    cfg, pr, cp, bs = s5c.setup(seed, PROBE, dev)
    R0 = s5c.r0_box(cfg, pr, dev)
    out = Path(out) if out is not None else model_dir(seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    b_dir = s5c.run_dir("B", PROBE, seed, a6.N_TRAIN)
    norm = json.load(open(b_dir / "result.json"))["norm"]                 # 同 B、P6B4 的輸入正規化
    fields = s5c.make_train_fields(cfg, n_fields, seed).to(dev)          # 同 P6B4 的訓練場(同 seed)
    print(f"[init] {NEW} s{seed}:訓練場 {len(fields)} 個({time.time() - t0:.1f}s)", flush=True)
    if len(fields) >= s5c.CAL_N:
        with torch.no_grad():
            cal = s5c.calibrate_norm(s5c.measure_box(fields[:s5c.CAL_N], PROBE, pr, cp, bs, cfg, seed=seed + 7))
        e_n = max(abs(cal[k] - norm[k]) / max(abs(norm[k]), 1e-12) for k in norm)
        check("輸入正規化:以訓練場(理想量測)重新校準 ≈ B 的常數(相對差 < 1e-3;實際使用 B 的常數,同 P6B4)",
              e_n < h.NORM_TOL, f"相對差 {e_n:.1e}")
    torch.manual_seed(seed)
    model = h.build_model("P6B4", cfg, pr).to(dev)                       # 架構 = P6B4
    model.init.load_state_dict(torch.load(b_dir / "final.pt", map_location=dev))
    ref, _, _ = s5c.load_net("B", PROBE, seed, a6.N_TRAIN, cfg, pr, dev)
    model.eval()
    with torch.no_grad():
        cnt = s5c.measure_box(fields[:64], PROBE, pr, cp, bs, cfg, seed=seed + 98)
        e_i = float((model.init(g.prep(cnt, bs, norm, "P6B4", cp)) - ref(s5c.prep(cnt, bs, norm, "B", cp))).abs().max())
    check("載入檢查:模型內的初始網路 = B(< 1e-5)", e_i < 1e-5, f"最大差 {e_i:.1e}")
    del ref
    if not all_ok():
        raise SystemExit("❌ 載入檢查未通過,不訓練")
    tm = TrainMeas(cfg, pr, cp, bs, dev)
    p_init = list(model.init.parameters())
    ids = {id(p) for p in p_init}
    opt = torch.optim.Adam([{"params": p_init, "lr": g.LR_INIT},
                            {"params": [p for p in model.parameters() if id(p) not in ids], "lr": g.LR_NEW}])
    max_lr = [g.LR_INIT, g.LR_NEW]
    steps = len(fields) // g.BATCH
    total = epochs * steps + 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total)
    n_par = g.n_params(model)
    print(f"[init] params={n_par:,};{epochs} epochs × {steps} 步;隨機誤差:位置 σ ~ U(0, {TR_POS_MAX}) px、"
          f"離焦 ± {TR_DF_MAX:.0%}({len(DF_GRID)} 個探針)、劑量 {TR_DOSE_P1:.0%} = 1 / 其餘 10^U({TR_DOSE_LOGMIN:g}, 0)", flush=True)
    meta = {"group": NEW, "arch": "P6B4", "probe": PROBE, "seed": seed, "n_train": n_fields, "stages": model.K,
            "epochs": epochs, "batch": g.BATCH, "lr": max_lr, "clip": g.CLIP, "norm": norm, "params": n_par,
            "train_seed_base": s5c.train_chunk_seed(seed, 0), "quick": QUICK, "init_md5": g._md5(b_dir / "final.pt"),
            "rand": {"pos_max": TR_POS_MAX, "df_grid": DF_GRID, "dose_p1": TR_DOSE_P1, "dose_logmin": TR_DOSE_LOGMIN,
                     "z_clip": Z_CLIP, "err_seed_base": ERR_SEED_BASE, "noise_seed_base": s5c.NOISE_SEED_BASE},
            "script_md5": me_md5()}
    json.dump(meta, open(out / "config_used.json", "w"), indent=2)
    ck = out / "ckpt.pt"
    start, hist = 0, []
    if ck.exists():
        d = torch.load(ck, map_location=dev, weights_only=False)
        if d.get("script_md5") != me_md5():
            raise SystemExit(f"❌ {ck} 是由不同版本的 scan_7b.py 存的({str(d.get('script_md5'))[:8]}…):不續跑。先告訴 Claude")
        model.load_state_dict(d["model"])
        opt.load_state_dict(d["opt"])
        sched.load_state_dict(d["sched"])
        start, hist = d["epoch"] + 1, d["history"]
        print(f"[resume] 從 epoch {start} 續跑", flush=True)
    gstep = start * steps
    for ep in range(start, epochs):
        model.train()
        gen = torch.Generator().manual_seed(seed * 100_003 + ep)            # 同 B / P6B4 的 batch 順序公式
        perm = torch.randperm(len(fields), generator=gen)
        acc = {"loss": 0.0, "final": 0.0, "stages": [0.0] * model.K}
        n_ok, n_skip, n_clip, gns, te = 0, 0, 0, [], time.time()
        for i in range(steps):
            idx = perm[i * g.BATCH:(i + 1) * g.BATCH].to(dev)
            obj = fields[idx]
            with torch.no_grad():
                counts, _ = tm(obj, ERR_SEED_BASE + seed * 10_000_019 + gstep, s5c.NOISE_SEED_BASE + seed * 10_000_019 + gstep)
                inp = g.prep(counts, bs, norm, "P6B4", cp)
            gstep += 1
            loss, ls = g.ds_loss(model(inp, all_stages=True)[1], torch.polar(obj[:, 0], obj[:, 1]), R0)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip += 1
                if sched.last_epoch + 1 < total:
                    sched.step()
                continue
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), g.CLIP))
            if not np.isfinite(gn):
                n_skip += 1
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch + 1 < total:
                    sched.step()
                continue
            opt.step()
            if sched.last_epoch + 1 < total:
                sched.step()
            n_ok += 1
            n_clip += gn > g.CLIP
            gns.append(gn)
            acc["loss"] += float(loss.detach())
            acc["final"] += float(ls[-1].detach())
            for k in range(model.K):
                acc["stages"][k] += float(ls[k].detach())
        for k in ("loss", "final"):
            acc[k] /= max(n_ok, 1)
        acc["stages"] = [v / max(n_ok, 1) for v in acc["stages"]]
        acc.update({"epoch": ep, "skipped": n_skip, "clip_frac": n_clip / max(n_ok, 1),
                    "grad_norm_median": float(np.median(gns)) if gns else None, "alphas": model.alphas(),
                    "sec": time.time() - te, "lr": sched.get_last_lr()})
        hist.append(acc)
        print(f"epoch {ep:03d}  loss={acc['loss']:.5f}  各級=" + " ".join(f"{v:.4f}" for v in acc["stages"])
              + "  α=" + " ".join(f"{v:.3f}" for v in acc["alphas"])
              + f"  梯度範數中位數 {acc['grad_norm_median'] or 0:.2e}(裁切 {100 * acc['clip_frac']:.0f}%)"
              + f"  跳過 {n_skip}  {acc['sec']:.1f}s", flush=True)
        tmp = out / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "history": hist, "script_md5": meta["script_md5"]}, tmp)
        os.replace(tmp, ck)
    model.eval()
    with torch.no_grad():
        sub = fields[:s5c.TRAIN_EVAL_N]
        c_id = s5c.measure_box(sub, PROBE, pr, cp, bs, cfg, seed=seed + 99)       # 同 scan_5h.train_nerr
        tr, pooled = nerr_on(model, sub, c_id, norm, cp, bs, R0)
        c_er, _ = tm(sub, EVAL_ERR_SEED + seed, seed + 99)
        tr_e, _ = nerr_on(model, sub, c_er, norm, cp, bs, R0)
    meta.update({"history": hist, "train_nerr_ph": tr, "train_nerr_pooled": pooled, "train_nerr_ph_rand": tr_e,
                 "alphas": model.alphas(), "train_sec": time.time() - t0, "skipped_total": sum(x["skipped"] for x in hist)})
    tmp = out / "final.tmp"
    torch.save(model.state_dict(), tmp)
    json.dump(meta, open(out / "result.json", "w"), indent=2)
    os.replace(tmp, out / "final.pt")                                     # final.pt 最後才出現(= 完成的標記)
    print(f"[done] {NEW} s{seed}:訓練場 nerr_ph 理想 {tr:.4f} / 隨機誤差 {tr_e:.4f}  α {' '.join(f'{v:.3f}' for v in model.alphas())}"
          f"  {time.time() - t0:.0f}s → {out}", flush=True)
    return meta


def load_net(name, s, cfg, pr, dev, mroot=None):
    """回傳 (網路, 輸入正規化)。P6B4 = 期中考凍結的(同 7a);P6B4r = 本階段訓練的(同架構)。"""
    if name == "P6B4":
        net, norm, _ = a6.load_p6b4(s, cfg, pr, dev)
        return net, norm
    md = model_dir(s, mroot)
    meta = json.load(open(md / "result.json"))
    net = h.build_model("P6B4", cfg, pr).to(dev)
    net.load_state_dict(torch.load(md / "final.pt", map_location=dev))
    net.eval()
    return net, meta["norm"]


# ============================================================================
# SSIM(副指標;§14.4)
# ============================================================================
_GW = {}


def _gwin(dev):
    if str(dev) not in _GW:
        x = torch.arange(SSIM_WIN, dtype=torch.float32) - (SSIM_WIN - 1) / 2
        w = torch.exp(-x ** 2 / (2 * SSIM_SIG ** 2))
        _GW[str(dev)] = (w / w.sum()).to(dev)
    return _GW[str(dev)]


def _blur(x, w):
    """可分離的高斯模糊(valid):x [N, H, W] → [N, H − 10, W − 10]。"""
    k = w.numel()
    x = torch.nn.functional.conv2d(x[:, None], w.view(1, 1, 1, k))
    x = torch.nn.functional.conv2d(x, w.view(1, 1, k, 1))
    return x[:, 0]


def ssim_map(x, y, L):
    w = _gwin(x.device)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mx, my = _blur(x, w), _blur(y, w)
    sxx = _blur(x * x, w) - mx * mx
    syy = _blur(y * y, w) - my * my
    sxy = _blur(x * y, w) - mx * my
    return ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))


_AL = {}


def aligned(est, O, U):
    """與主指標相同的對齊(整體平移 t + 線性相位斜坡 q,皆取 align_err 選定的值;再對齊整體相位)。[N, F, F]。"""
    F = O.shape[-1]
    _, _, _, t, q = a7.align_err(est, O, U)
    _, _, ky, kx, _, _, yc = a7._sh_mats(F, O.device)
    ph = 2 * math.pi * (ky * t[:, 0, None, None] + kx * t[:, 1, None, None])
    es = torch.fft.ifft2(torch.fft.fft2(est) * torch.polar(torch.ones_like(ph), ph)) * a7._ramp_field(q, yc)
    m = U.to(O.real.dtype)
    c = ((O * m) * (es * m).conj()).sum((1, 2))
    return es * torch.exp(1j * torch.angle(c))[:, None, None]


def _bbox(U, pad):
    ys, xs = torch.where(U)
    F = U.shape[-1]
    y0, y1 = max(0, int(ys.min()) - pad), min(F, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(F, int(xs.max()) + 1 + pad)
    return y0, y1, x0, x1


@torch.no_grad()
def ssim_eval(est, O, U, phmax, align=True):
    """每個樣本的 (振幅 SSIM, 相位 SSIM),U 內的平均。對齊同主指標;相位 = 真值相位 + angle(估計 · conj(真值))
    (避免 2π 跳躍),只在真值振幅 > AMP_MIN 處比較,其餘兩邊都設 0;值域:振幅 1、相位 phmax。
    估計發散(非有限值)的樣本記為 0。回傳兩個 [N] 的 tensor。"""
    pad = SSIM_WIN // 2
    y0, y1, x0, x1 = _bbox(U, pad)
    Uc = U[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
    sa, sp = [], []
    for i in range(0, O.shape[0], a7.MCH):
        e, o = est[i:i + a7.MCH], O[i:i + a7.MCH]
        fin = torch.isfinite(e.real).all((1, 2)) & torch.isfinite(e.imag).all((1, 2))
        e = torch.where(fin[:, None, None], e, torch.zeros_like(e))
        if align:
            e = aligned(e, o, U)
        e, o = e[:, y0:y1, x0:x1], o[:, y0:y1, x0:x1]
        am = ssim_map(e.abs(), o.abs(), 1.0)
        keep = o.abs() > AMP_MIN
        po = torch.where(keep, torch.angle(o), torch.zeros_like(o.real))
        pe = torch.where(keep, torch.angle(o) + torch.angle(e * o.conj()), torch.zeros_like(o.real))
        pmap = ssim_map(pe, po, phmax)
        mU = Uc.to(am.dtype)
        va = (am * mU).sum((1, 2)) / mU.sum()
        vp = (pmap * mU).sum((1, 2)) / mU.sum()
        sa.append(torch.where(fin & torch.isfinite(va), va, torch.zeros_like(va)))
        sp.append(torch.where(fin & torch.isfinite(vp), vp, torch.zeros_like(vp)))
    return torch.cat(sa), torch.cat(sp)


def ssim_mean(est, O, U, phmax):
    mU = U.to(O.real.dtype)
    valid = ((O.abs() ** 2) * mU).sum((1, 2)) > 1e-12
    a, p = ssim_eval(est, O, U, phmax)
    return float(a[valid].mean()), float(p[valid].mean())


# ============================================================================
# 內建檢查
# ============================================================================
@torch.no_grad()
def checks(dev):
    torch.backends.cudnn.benchmark = True
    print("=" * 70)
    print("階段七 7b:內建檢查")
    print("=" * 70)
    from src.physics import forward_measure
    n = 8 if not QUICK else 4
    # (1) 舊條件的量測不變
    d0 = a6.setup(0, dev, n)
    e_old = {}
    for c in OLD_CONDS:
        dc = a7.setup_cond(0, dev, c, n)
        ref = _a7_measure(dc["fields"], dc["geo"], dc["cp"], dc["bs"], a6.seeds_of(0)["noise"], a7.SPEC[c], dc["z"], dc["sign"],
                          dc["ops"])
        e_old[c] = max(float((a - b).abs().max()) for a, b in zip(dc["counts"], ref))
    e_id = max(float((a - b).abs().max()) for a, b in zip(d0["counts"], a7.setup_cond(0, dev, "ideal", n)["counts"]))
    same_ideal = all(_a7_is_ideal(a7.SPEC[c]) == is_ideal(a7.SPEC[c]) for c in OLD_CONDS)
    check("舊條件:擴充後的量測 = scan_7a 原本的量測(8 個條件,逐位元);理想 = 階段六 measure49;『是否理想』的判斷不變",
          max(e_old.values()) == 0.0 and e_id == 0.0 and same_ideal, f"最大差 {max(e_old.values()):.1e}、理想 {e_id:.1e}")
    # (2) 部分同調
    dz = a7.setup_cond(0, dev, "ideal", n, force_generic=True)
    geo, cp, bs, ops, fields = dz["geo"], dz["cp"], dz["bs"], dz["ops"], dz["fields"]
    cp0 = copy.deepcopy(cp)
    cp0.add_poisson = False
    md = coh_modes(1.0)
    wsum = sum(w for _, _, w in md)
    my = sum(w * a for a, _, w in md)
    vy = sum(w * a * a for a, _, w in md)
    ideal0 = a7.measure_cond(fields, geo, cp0, bs, 0, dict(pos=0.0, df=0.0, dose=1.0), dz["z"], dz["sign"], ops, True)
    c00 = measure_coh(fields, geo, cp0, bs, 0, 0.0, ops)
    e0 = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(c00, ideal0))
    c10 = measure_coh(fields, geo, cp0, bs, 0, 1.0, ops)
    tot = float(sum(c.sum() for c in c10) / sum(c.sum() for c in ideal0))
    m = bs > 0

    def contrast(cs):
        v = torch.stack(cs, 1)[..., m]                                     # [n, 49, 像素]
        return float((v.std(-1) / v.mean(-1).clamp_min(1e-12)).mean())

    k0, k1 = contrast(ideal0), contrast(c10)
    dn = a7.setup_cond(0, dev, "cohH", n)
    differ = max(float((a - b).abs().max()) for a, b in zip(dn["counts"], d0["counts"]))
    check("部分同調:Σw = 1、平均位移 0、變異數 σ²(Gauss–Hermite);σ = 0 → = 理想(無雜訊,相對差 < 1e-5);"
          f"σ = 1 px 時總計數 / 理想 在 1 ± {COH_COUNT_TOL};繞射圖對比下降(變糊);cohH 的量測(有雜訊)與理想不同",
          abs(wsum - 1) < 1e-12 and abs(my) < 1e-12 and abs(vy - 1) < 1e-12 and e0 < 1e-5 and abs(tot - 1) < COH_COUNT_TOL
          and k1 < k0 and differ > 0,
          f"Σw {wsum:.6f}、平均 {my:.1e}、變異數 {vy:.6f};σ = 0 差 {e0:.1e};總計數比 {tot:.4f};對比 {k0:.4f} → {k1:.4f}")
    # (3) 訓練用的量測
    cfg, pr, cpb, bsb = s5c.setup(0, PROBE, dev)
    tm = TrainMeas(cfg, pr, cpb, bsb, dev)
    cpb0 = copy.deepcopy(cpb)
    cpb0.add_poisson = False
    fb = s5c.make_train_fields(cfg, n, 0).to(dev)
    i0 = DF_GRID.index(0.0)
    zero = torch.zeros(n, 9, 2)
    ones = torch.ones(n)
    a = measure_rand(fb, tm.rel, tm.Pf, tm.ops, cpb, bsb, zero, torch.full((n,), i0), ones, None)
    b = s5c.measure_box(fb, PROBE, pr, cpb0, bsb, cfg, seed=0)
    e_id = max(float((x - y).abs().max() / y.abs().max()) for x, y in zip(a, b))
    sh = zero.clone()
    sh[..., 0], sh[..., 1] = 2.0, -1.0
    a2 = measure_rand(fb, tm.rel, tm.Pf, tm.ops, cpb, bsb, sh, torch.full((n,), i0), ones, None)
    ref = []
    for (y0, x0) in tm.rel:
        win = fb[:, :, y0:y0 + cfg.canvas, x0:x0 + cfg.canvas]
        psi = torch.roll(pr["P"], (2, -1), dims=(-2, -1)) * torch.polar(win[:, 0], win[:, 1])
        ref.append(forward_measure(torch.stack([psi.abs(), torch.angle(psi)], 1), bsb, cpb0))
    e_sh = max(float((x - y).abs().max() / y.abs().max()) for x, y in zip(a2, ref))
    e_bank = 0.0
    for lev in (0.1, 0.3):
        pp, pm = a7.true_probe_pair(pr, lev, dev)
        e_bank = max(e_bank, float((tm.bank[DF_GRID.index(lev)] - pp).abs().max()),
                     float((tm.bank[DF_GRID.index(-lev)] - pm).abs().max()))
    e_b0 = float((tm.bank[i0] - pr["P"]).abs().max() / pr["P"].abs().max())
    de, di, dd_, _ = draw_train_errors(n, torch.Generator().manual_seed(77))
    a3 = measure_rand(fb, tm.rel, tm.Pf, tm.ops, cpb, bsb, de, di, dd_, None)            # 有誤差、無雜訊
    e_err = float(sum((x - y).abs().sum() for x, y in zip(a3, a)) / sum(y.abs().sum() for y in a))
    gq = torch.Generator(device=dev).manual_seed(5)
    nb = max(n, 16)
    fq = s5c.make_train_fields(cfg, nb, 0).to(dev)
    zq = torch.zeros(nb, 9, 2)
    a4 = measure_rand(fq, tm.rel, tm.Pf, tm.ops, cpb, bsb, zq, torch.full((nb,), i0), torch.full((nb,), 0.1), gq)
    a5 = measure_rand(fq, tm.rel, tm.Pf, tm.ops, cpb, bsb, zq, torch.full((nb,), i0), torch.ones(nb), None)
    raw = float(sum((x * 0.1).sum() for x in a4) / sum(y.sum() for y in a5))
    integer = all(bool(torch.allclose(x * 0.1, (x * 0.1).round(), atol=1e-3, rtol=1e-5)) for x in a4)
    check("訓練量測:無誤差、無雜訊 = scan_5c.measure_box(相對差 < 1e-5);δ = (2, −1) px = 探針 torch.roll((2, −1))(< 1e-4);"
          "探針庫 ±10% / ±30% = scan_7a.true_probe_pair(< 1e-6)、0% = 名目探針(< 1e-5);隨機抽的誤差確實改變量測"
          "(無雜訊時與理想的相對 L1 差 > 1e-2);有雜訊時原始計數為整數、原始總計數 / 期望 ≈ 劑量 0.1(± 2%)",
          e_id < 1e-5 and e_sh < 1e-4 and e_bank < 1e-6 and e_b0 < 1e-5 and e_err > 1e-2 and abs(raw / 0.1 - 1) < 0.02 and integer,
          f"理想 {e_id:.1e}、平移 {e_sh:.1e}、探針庫 {e_bank:.1e} / {e_b0:.1e}、有誤差 vs 理想 {e_err:.3f}、原始計數比 {raw:.4f}")
    gc = torch.Generator().manual_seed(123)
    N = 20000
    d, idx, dose, sig = draw_train_errors(N, gc)
    d2, idx2, dose2, _ = draw_train_errors(N, torch.Generator().manual_seed(123))
    rep = bool(torch.equal(d, d2) and torch.equal(idx, idx2) and torch.equal(dose, dose2))
    zm = float(d.mean(1).abs().max())
    f1 = float((dose == 1).float().mean())
    lo = dose[dose < 1]
    lmean = float(torch.log10(lo).mean())
    rms = float(d.pow(2).mean().sqrt())
    rms_exp = TR_POS_MAX * math.sqrt(1.0 / 3.0) * math.sqrt(8.0 / 9.0)        # E[σ²] = 1/3;減去 9 個的平均 → × 8/9
    cnt = torch.bincount(idx, minlength=len(DF_GRID)).float()
    unif = float(cnt.max() / cnt.min())
    check("訓練的誤差抽樣:同 seed 可重現;每個樣本的 9 個位移平均 = 0;位移 RMS ≈ √(1/3 · 8/9) px(± 3%);"
          "劑量 = 1 的比例 ≈ 0.5(± 0.02)、其餘 log10 平均 ≈ −1(± 0.03)、最小 ≥ 0.01;61 個探針都會抽到(最多 / 最少 < 1.5)",
          rep and zm < 1e-5 and abs(rms / rms_exp - 1) < 0.03 and abs(f1 - 0.5) < 0.02 and abs(lmean + 1) < 0.03
          and float(dose.min()) >= 0.01 - 1e-9 and unif < 1.5 and int((cnt > 0).sum()) == len(DF_GRID),
          f"可重現 {rep}、平均最大 {zm:.1e}、RMS {rms:.4f}(預期 {rms_exp:.4f})、劑量 1 的比例 {f1:.3f}、log10 平均 {lmean:.3f}、"
          f"最小 {float(dose.min()):.4f}、探針次數 最多 / 最少 {unif:.2f}")
    # (4) SSIM 與對齊
    U, O, phmax = geo.U, dz["O"], cp.phase_max
    sa, sp = ssim_eval(O, O, U, phmax, align=False)
    Fz = geo.F
    kf = torch.fft.fftfreq(Fz, device=dev)
    ky, kx = torch.meshgrid(kf, kf, indexing="ij")
    ph = -2 * math.pi * (ky * 0.3 + kx * (-1.2))
    shf = torch.fft.ifft2(torch.fft.fft2(O) * torch.polar(torch.ones_like(ph), ph)) * torch.exp(torch.tensor(0.9j, device=dev))
    _, _, _, _, _, _, yc = a7._sh_mats(Fz, dev)
    qq = torch.tensor([[0.0005, -0.0008]], device=dev).expand(O.shape[0], 2)
    est = shf * a7._ramp_field(qq, yc).conj()
    ba, bp = ssim_eval(est, O, U, phmax)
    na, np_ = ssim_eval(est, O, U, phmax, align=False)
    noisy = O + 0.1 * torch.randn_like(O.real)
    za, zp = ssim_eval(noisy, O, U, phmax)
    bad = O.clone()
    bad[0] = float("nan")
    fa, fp = ssim_eval(bad, O, U, phmax)
    v_sr = a7.metrics7(est, O, U)[MAIN]
    check("SSIM:與真值相同 → 1(< 1e-5);已知平移 + 整體相位 + 相位斜坡,對齊後 > 0.98 且高於不對齊;加雜訊 → 下降;"
          "發散的樣本 → 0;同一估計的 nerr_sr 很小(對齊一致)",
          float((1 - sa).abs().max()) < 1e-5 and float((1 - sp).abs().max()) < 1e-5 and float(ba.min()) > 0.98
          and float(bp.min()) > 0.98 and float(ba.mean()) > float(na.mean()) and float(bp.mean()) > float(np_.mean())
          and float(za.mean()) < 0.99 and float(zp.mean()) < 0.99 and float(fa[0]) == 0.0 and float(fp[0]) == 0.0 and v_sr < 1e-3,
          f"相同:{float(sa.min()):.6f} / {float(sp.min()):.6f};對齊後 {float(ba.min()):.4f} / {float(bp.min()):.4f}"
          f"(不對齊 {float(na.mean()):.3f} / {float(np_.mean()):.3f});加雜訊 {float(za.mean()):.3f} / {float(zp.mean()):.3f};"
          f"NaN 樣本 {float(fa[0]):.1f};nerr_sr {v_sr:.1e}")
    # (5) 新條件的設定
    sp_ok = (NEW_SPEC["posX"]["pos"] > TR_POS_MAX and NEW_SPEC["prbX"]["df"] > TR_DF_MAX and all(c in a7.SPEC for c in NEW_CONDS)
             and len(DF_GRID) == 61 and DF_GRID[0] == -0.3 and DF_GRID[-1] == 0.3)
    check("新條件的設定:posX、prbX 超出訓練範圍;cohL / cohH 只用於測試;探針庫 61 個(−0.30 … +0.30)", sp_ok,
          "、".join(f"{c} {NEW_SPEC[c]}" for c in NEW_CONDS))
    print(f"\n     裝置:{dev}({a7.gpu_name(dev)})")


# ============================================================================
# 新條件的迭代法包絡(用 scan_7a 的 measure_condition,不改)
# ============================================================================
def run_env(dev, cond, n, iters, root, smoke):
    me = me_md5()
    t0 = time.time()
    sp = a7.SPEC[cond]
    print(f"\n  新條件 {cond}({clabel(cond)}):{sp}", flush=True)
    path = env_json(cond, root)
    partial = Path(str(path) + ".partial")
    side = Path(str(path) + ".partial.md5")                               # 暫存檔由哪一版 scan_7b.py 產生
    if partial.exists() and (not side.exists() or side.read_text().strip() != me):
        raise SystemExit(f"❌ {partial} 由不同(或未知)版本的 scan_7b.py 產生:不續跑。先告訴 Claude")
    side.write_text(me)
    env, nets, stats = a7.measure_condition(dev, cond, n, iters, log=lambda s: print(s, flush=True), partial=partial)
    nn = stats[0]["n"]
    if sp["pos"] > 0:
        r = np.mean([s["pos_std_px"] for s in stats]) / sp["pos"]
        check(f"[{cond}] 位置誤差的標準差 = 設定 × (1 ± 5%)", abs(r - 1) <= 0.05, f"實際 / 設定 = {r:.3f}")
    if sp["df"] > 0:
        f = np.mean([s["df_plus_frac"] for s in stats])
        check(f"[{cond}] 離焦符號約各半(± 3 個標準誤)", abs(f - 0.5) <= 3 * 0.5 / math.sqrt(nn), f"+ 的比例 {f:.3f}")
    if sp.get("coh", 0) > 0:
        rr = np.mean([s["raw_count_ratio"] for s in stats])
        check(f"[{cond}] 平均總計數 / 理想 在 1 ± {COH_COUNT_TOL}(劑量不變)", abs(rr - 1) <= COH_COUNT_TOL, f"{rr:.4f}")
    check(f"[{cond}] 量測與理想不同", all(s.get("counts_differ", False) for s in stats))
    out = {"meta": {"cond": cond, "spec": sp, "n": nn, "iters": list(iters), "seeds": list(SEEDS), "script_md5": me,
                    "a7_md5": md5(Path(a7.__file__).resolve()), "quick": QUICK, "smoke": smoke, "gpu": a7.gpu_name(dev),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0},
           "env": env, "nets": nets, "stats": stats, "checks_ok": all_ok()}
    dump_json(out, path)
    for f in (partial, side):
        if f.exists():
            os.remove(f)                                                  # 本程式自己的暫存檔
    print(f"  存檔:{path}(經過 {time.time() - t0:.0f} 秒)")
    for p in PIPES:
        print(f"    P6B4 {p}:{MAIN} {np.mean(nets[p][MAIN]):.4f}")
    return out


# ============================================================================
# 彙整:網路評估、比較方式、判讀
# ============================================================================
def t_match_conf(env, Q, tB, iters):
    """包絡最早達到 Q(3 seeds 平均 ≤ Q)的 (時間, 設定, 次數);到不了 → (inf, None, None)。時間相同時取先找到的。"""
    best = (float("inf"), None, None)
    for c in a7.configs7():
        for it in a7.stops7(c, iters):
            t = a7.cfg_time7(c, it, tB)
            if float(a7.env_vals7(env, c, it).mean()) <= Q and t < best[0]:
                best = (t, c, it)
    return best


def t_match_lo(env, Q, tB, iters):
    """達到時間的下界:真正越過 Q 的時刻落在「第一個 ≤ Q 的停止點」與「前一個停止點」之間 →
    各設定取前一個停止點的時間(第一個停止點就 ≤ Q 時:起點本身 = 該點的時間;K-HIO = 0 次的時間),再取最小。
    只看 500 次內達到的設定(同包絡的定義);沒有設定達到 → inf。"""
    best = float("inf")
    for c in a7.configs7():
        prev = None
        for it in a7.stops7(c, iters):
            if float(a7.env_vals7(env, c, it).mean()) <= Q:
                best = min(best, a7.cfg_time7(c, prev if prev is not None else (it if it == 0 else 0), tB))
                break
            prev = it
    return best


@torch.no_grad()
def eval_nets(dev, C, mroot, log=print):
    """12 個條件 × 3 seeds × 2 個網路 × 2 個流程:三種誤差 + SSIM。另存 seed 0 的重建圖與繞射圖(畫圖用)。"""
    R = {c: {nm: {p: {k: [] for k in a7.MKEYS + ("ssim_amp", "ssim_ph")} for p in PIPES} for nm in NETS} for c in CONDS}
    keep = {"img": {}, "diffr": {}}
    t0 = time.time()
    for c in CONDS:
        n = C[c]["meta"]["n"]
        for s in SEEDS:
            d = a7.setup_cond(s, dev, c, n)
            geo, O, U = d["geo"], d["O"], d["geo"].U
            phmax = d["cfg"].phase_max
            for nm in NETS:
                net, norm = load_net(nm, s, d["cfg"], geo.pr, dev, mroot)
                gs = a6.GlobalStages(net, geo.starts, geo.F, geo.pr, d["bs"], dev)
                for p in PIPES:
                    est = a6.run_pipe(p, net, norm, d, gs)
                    m = a7.metrics7(est, O, U, keep_ps=(s == 0 and p == PRIMARY))
                    sa, sp = ssim_mean(est, O, U, phmax)
                    for k in a7.MKEYS:
                        R[c][nm][p][k].append(m[k])
                    R[c][nm][p]["ssim_amp"].append(sa)
                    R[c][nm][p]["ssim_ph"].append(sp)
                    if s == 0 and p == PRIMARY:
                        R[c][nm][p]["ps_sr0"] = m["ps_sr"]
                        if c in RECON_CONDS:
                            keep["img"].setdefault(c, {})[nm] = est.cpu()
                del net, gs
            if s == 0:
                if c in RECON_CONDS:
                    keep["img"][c]["O"] = O.cpu()
                if c in ("ideal", "cohL", "cohH"):
                    keep["diffr"][c] = d["counts"][a6.CEN49][0].cpu()
                if c == CONDS[0]:
                    keep["geo"], keep["cfg"] = geo, d["cfg"]
            del d
        log(f"  [網路評估] {c} 完成(經過 {time.time() - t0:.0f} 秒)")
    return R, keep


def speed_row(a_mean, env, tB, t_net, iters):
    """x:加速倍數(迭代法用「第一個達到 Q 的停止點」的時間;停止點有間隔 → 偏向網路);
    x_lo:下界(迭代法用前一個停止點的時間;偏向迭代法)。真正的值在 [x_lo, x] 之間。inf = 迭代法 500 次內達不到。"""
    out = {}
    for Q in QS:
        t_it, conf, it = t_match_conf(env, Q, tB, iters)
        t_lo = t_match_lo(env, Q, tB, iters)
        reach = bool(a_mean <= Q)
        out[str(Q)] = {"reach": reach, "t_iter": t_it, "t_iter_lo": t_lo, "conf": conf, "it": it,
                       "x": (t_it / t_net) if reach else None, "x_lo": (t_lo / t_net) if reach else None}
    return out


def fmt_x(r):
    if not r["reach"]:
        return "網路未達"
    if not np.isfinite(r["x"]):
        return "迭代法未達(網路達到)"
    return f"× {r['x_lo']:.2f}–{r['x']:.2f}"


def report(C, R, tim, iters, smoke):
    Bs = ["64", "512"]
    V = {"cond": {}}
    if smoke:
        print("\n" + "!" * 100)
        print("迷你流程:訓練極少、場數少、停止點 ≤ 20 → 數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    print("\n" + "=" * 100)
    print(f"計時(ms / 整張影像;本 job 重新量;GPU {tim['gpu']};P6B4r 與 P6B4 同架構 → 同時間)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:網路 G9 {tim[B]['net:G9']:.4f}、G25 {tim[B]['net:G25']:.4f};K-HIO49 {tim[B]['K-HIO49']:.4f}"
              f";AP-C 每次 {tim[B]['AP-C']:.4f}、ePIE-C 每次 {tim[B]['ePIE-C']:.4f}")
    print("\n" + "=" * 100)
    print(f"各條件(主指標 {MAIN};3 seeds;SSIM:振幅 / 相位;比值 = 網路 / 同時間包絡,≤ 0.8 且 z < −2 = 較準)")
    print(f"加速倍數 = 迭代法包絡最早達到 Q 的時間 / 網路的時間;Q = {' / '.join(str(q) for q in QS)}(主:{Q_MAIN})")
    print("  以區間 [下界–上界] 表示:上界用第一個達到 Q 的停止點(停止點有間隔 → 迭代法的時間偏高、偏向網路),")
    print("  下界用前一個停止點(偏向迭代法);主判(b1)的「穩健加速」要求下界 ≥ 1.25。")
    print("  另:包絡是事後在 85 個設定中挑最好的(對迭代法有利);網路是一個固定的模型。")
    print("=" * 100)
    for c in CONDS:
        V["cond"][c] = {}
        env = C[c]["env"]
        tag = "(只測試)" if c in HELDOUT else ("(超出訓練範圍)" if c in OUTRANGE else "")
        print(f"\n  ■ {c}({clabel(c)}){tag}")
        for nm in NETS:
            V["cond"][c][nm] = {}
            for p in PIPES:
                a = np.array(R[c][nm][p][MAIN], float)
                row = {"nerr": list(a), "nerr_ph": R[c][nm][p]["nerr_ph"], "ssim_amp": R[c][nm][p]["ssim_amp"],
                       "ssim_ph": R[c][nm][p]["ssim_ph"], "shift": float(np.mean(R[c][nm][p]["shift"])),
                       "n_fail": int(np.sum(R[c][nm][p]["n_fail"]))}
                line = (f"    {nm:<5} {p}:{a.mean():.4f}({' '.join(f'{x:.4f}' for x in a)});SSIM "
                        f"{np.mean(row['ssim_amp']):.3f} / {np.mean(row['ssim_ph']):.3f}"
                        + (f";發散 {row['n_fail']} 個" if row["n_fail"] else ""))
                for B in Bs:
                    tn = tim[B][f"net:{p}"]
                    e = a7.envelope7(env, tn, tim[B], iters)
                    r = s5b.ratio(a, e[2])
                    sr = speed_row(a.mean(), env, tim[B], tn, iters)
                    row[B] = {"t": tn, "opp": [e[0], e[1], list(e[2]), e[3]], "ratio": list(r), "speed": sr}
                    line += (f"\n       b{B}:{tn:.3f} ms;同時間包絡 {s5b.fmt_conf(e[0])} ×{e[1]} {np.mean(e[2]):.4f} → 比值 {r[1]:.3f}"
                             f"(z {r[2]:+.1f})→ {r[0]};加速 " + "、".join(f"Q {q}:{fmt_x(sr[str(q)])}" for q in QS))
                print(line)
                V["cond"][c][nm][p] = row
        bo = a7.best_overall(env, iters)
        V["cond"][c]["iter_best"] = [bo[0], bo[1], float(bo[2].mean())]
        tq = "、".join(f"Q {q}:" + (f"{t_match_conf(env, q, tim['512'], iters)[0]:.2f} ms" if np.isfinite(
            t_match_conf(env, q, tim['512'], iters)[0]) else "500 次內到不了") for q in QS)
        print(f"    迭代法:最佳 {s5b.fmt_conf(bo[0])} ×{bo[1]} {bo[2].mean():.2e};b512 達到門檻的時間 {tq}")
    # ---- 判讀 ----
    print("\n" + "=" * 100)
    print("判讀(階段七協定 §14.5,結果出來前寫定)")
    print("=" * 100)
    r1 = V["cond"][MAIN_COND][NEW][PRIMARY]
    s64, s512 = r1["64"]["speed"][str(Q_MAIN)], r1["512"]["speed"][str(Q_MAIN)]
    if not s64["reach"]:
        b1 = "達不到"
    elif all(x["x_lo"] >= SPEED_MIN for x in (s64, s512)):
        b1 = "穩健加速"
    elif any(x["x_lo"] >= SPEED_MIN for x in (s64, s512)):
        b1 = "達到、只在 " + "、".join(f"b{B}" for B, x in (("64", s64), ("512", s512)) if x["x_lo"] >= SPEED_MIN) + " 穩健加速"
    elif any(x["x"] >= SPEED_MIN for x in (s64, s512)):
        b1 = "達到、加速不確定(只有上界 ≥ 1.25,真正的越過時間落在停止點的間隔內)"
    else:
        b1 = "達到、無明顯加速"
    V["b1"] = b1
    print(f"  (b1 主判)combo 下 P6B4r 的 G9(誤差 {np.mean(r1['nerr']):.4f})在 Q = {Q_MAIN}:{b1}"
          f"(b64 {fmt_x(s64)}、b512 {fmt_x(s512)})" + ("  [smoke:無意義]" if smoke else ""))
    p0 = V["cond"][MAIN_COND]["P6B4"][PRIMARY]
    print(f"       對照:凍結的 P6B4 在 combo 誤差 {np.mean(p0['nerr']):.4f};b64 {fmt_x(p0['64']['speed'][str(Q_MAIN)])}、"
          f"b512 {fmt_x(p0['512']['speed'][str(Q_MAIN)])}")
    print("  (b2) 各條件、各門檻的加速倍數:見上表")
    print("  (b3)/(b4)/(b6) P6B4r vs P6B4(G9;比值 = P6B4r / P6B4,≥ 1.25 且 z > 2 = 退步;≤ 0.8 且 z < −2 = 改善)")
    cmp = {}
    for c in CONDS:
        cmp[c] = {}
        parts = []
        for p in PIPES:
            r = s5b.ratio(np.array(V["cond"][c][NEW][p]["nerr"]), np.array(V["cond"][c]["P6B4"][p]["nerr"]))
            lab = {"較差": "退步", "較準": "改善", "相近": "相近"}[r[0]]
            cmp[c][p] = {"ratio": r[1], "z": r[2], "label": lab}
            parts.append(f"{p} {r[1]:.3f}(z {r[2]:+.1f})→ {lab}")
        tag = "(只測試)" if c in HELDOUT else ("(超出範圍)" if c in OUTRANGE else "")
        print(f"    {c:<6}{tag} " + ";".join(parts))
    V["cmp"] = cmp
    reg_old = [c for c in OLD_CONDS if cmp[c][PRIMARY]["label"] == "退步"]
    reg_coh = [c for c in HELDOUT if cmp[c][PRIMARY]["label"] == "退步"]
    V["b3"] = {"regress": reg_old, "ok": not reg_old}
    V["b4"] = {"regress": reg_coh, "ok": not reg_coh}
    V["b6"] = cmp["ideal"][PRIMARY]
    print("  (b3 不退步規則,7a 的 8 個條件、G9)" + (f"❌ 退步:{', '.join(reg_old)} → 7b 不算完全成功" if reg_old else "✅ 沒有退步"))
    print("  (b4 補 A 傷 B,部分同調、G9)" + (f"❌ 退步:{', '.join(reg_coh)} → 訓練造成新的弱點" if reg_coh else "✅ 沒有退步"))
    print(f"  (b6 理想條件的代價)P6B4r / P6B4 = {cmp['ideal'][PRIMARY]['ratio']:.3f}({cmp['ideal'][PRIMARY]['label']};描述)")
    print("  (b5) 超出範圍(G9 誤差;對照訓練範圍內最重的同類條件)")
    b5 = {}
    for cx, cin in OUTRANGE.items():
        b5[cx] = {}
        parts = []
        for nm in NETS:
            ex, ei = np.mean(V["cond"][cx][nm][PRIMARY]["nerr"]), np.mean(V["cond"][cin][nm][PRIMARY]["nerr"])
            b5[cx][nm] = {"err": ex, "err_in": ei, "x": ex / max(ei, 1e-12)}
            parts.append(f"{nm} {ei:.4f}({cin})→ {ex:.4f}({cx}),× {ex / max(ei, 1e-12):.2f}")
        print(f"    {cx}:" + ";".join(parts))
    V["b5"] = b5
    est_a, est_b = np.mean(V["cond"]["posL"][NEW][PRIMARY]["nerr"]), np.mean(V["cond"]["posL"]["P6B4"][PRIMARY]["nerr"])
    gain = est_b / max(est_a, 1e-12)
    trig = bool(est_a > EST_ABS or gain < EST_GAIN)
    V["estimate_type"] = {"posL_new": est_a, "posL_old": est_b, "gain": gain, "trigger": trig}
    print(f"  ▶ 估計型架構的評估條件(事先寫定):posL 下 P6B4r 的 G9 {est_a:.4f}(門檻 {EST_ABS})、比 P6B4 改善 × {gain:.2f}"
          f"(門檻 {EST_GAIN})→ " + ("容忍型對位置誤差不足 → 提出估計型(需另寫協定、使用者確認)" if trig else "不需要估計型"))
    if any(V["cond"][c][NEW][PRIMARY][B]["ratio"][0] == "較準" for c in HELDOUT for B in Bs):
        print("  ⚠️ 部分同調的「網路 vs 迭代法」:我們的迭代法沒有混合態修正(實務會用)→ 對迭代法偏不利,只作描述")
    V["coh_caveat"] = True
    return V


@torch.no_grad()
def iter_est(d, conf, it, dev):
    """重算一個迭代法設定(同 scan_7a.recon 的作法;K-HIO 在第 k 次的結果與最多跑幾次無關)。"""
    geo = d["geo"]
    need = []
    if conf == "K-HIO":
        need = [it]
    elif conf.split("|")[1] in a7.KHIO_N:
        need = [a7.KHIO_N[conf.split("|")[1]]]
    kw = None
    if need:
        win = a7.khio_windows7(d, dev, sorted(set(need)))
        kw = {k: a6.merge_windows(wv, geo)[:2] for k, wv in win.items()}
        del win
    if conf == "K-HIO":
        Of, cover = kw[it]
        return torch.where(cover, Of, torch.zeros_like(Of))
    meth, kind = conf.split("|")
    O0 = a6.make_start(kind, d, kw, dev)
    return O0 if it == 0 else a7.run_method7(meth, d, [it], O0)[it]


@torch.no_grad()
def iter_ssim(dev, C, V, tim, iters, keep, log=print):
    """關鍵時間點的迭代法 SSIM(重算):各條件的「網路(G9)時間的包絡設定」(b64、b512)與「各門檻的達到設定」。
    另核對重算的 nerr_sr = 條件檔(描述,不阻擋)。"""
    out = {}
    worst = 0.0
    t0 = time.time()
    for c in CONDS:
        env = C[c]["env"]
        want = {}
        for B in ("64", "512"):
            e = V["cond"][c][NEW][PRIMARY][B]["opp"]
            want.setdefault((e[0], e[1]), []).append(f"T_net b{B}")
            for Q in QS:
                r = V["cond"][c][NEW][PRIMARY][B]["speed"][str(Q)]
                if r["conf"] is not None:
                    want.setdefault((r["conf"], r["it"]), []).append(f"Q {Q} b{B}")
        out[c] = []
        res = {k: {"sa": [], "sp": [], "sr": []} for k in want}
        n = C[c]["meta"]["n"]
        for s in SEEDS:
            d = a7.setup_cond(s, dev, c, n)
            O, U = d["O"], d["geo"].U
            for (conf, it) in want:
                est = iter_est(d, conf, it, dev)
                sa, sp = ssim_mean(est, O, U, d["cfg"].phase_max)
                v = float(a7.metrics7(est, O, U)[MAIN])
                res[(conf, it)]["sa"].append(sa)
                res[(conf, it)]["sp"].append(sp)
                res[(conf, it)]["sr"].append(v)
                worst = max(worst, absdiff(v, a7.env_vals7(env, conf, it)[SEEDS.index(s)]))
                if s == 0 and c in RECON_CONDS and "T_net b512" in want[(conf, it)]:
                    keep["img"][c]["iter"] = est.cpu()
                del est
            del d
        for (conf, it), tags in want.items():
            r = res[(conf, it)]
            out[c].append({"conf": conf, "it": it, "tags": tags, "ssim_amp": r["sa"], "ssim_ph": r["sp"], "nerr_sr": r["sr"]})
        log(f"  [迭代法 SSIM] {c}:{len(want)} 個設定(經過 {time.time() - t0:.0f} 秒)")
    print(f"  描述性核對(不阻擋):重算的迭代法 nerr_sr vs 條件檔,最大差 {worst:.1e}")
    return out, worst


def ssim_report(V, R, S):
    """(b7) SSIM 與 nerr 的排名是否一致:網路 2 個(G9)+ 網路時間的迭代法(b64、b512),兩兩比較的一致比例。"""
    print("\n  (b7) SSIM 與 nerr_sr 的排名一致性(每個條件:P6B4、P6B4r 的 G9 與關鍵時間點的迭代法;兩兩比較;格式 nerr_sr / SSIM 振幅 / 相位)")
    b7 = {}
    allc = {"amp": [0, 0], "ph": [0, 0]}
    for c in CONDS:
        items = []
        for nm in NETS:
            items.append((nm, np.mean(R[c][nm][PRIMARY][MAIN]), np.mean(R[c][nm][PRIMARY]["ssim_amp"]),
                          np.mean(R[c][nm][PRIMARY]["ssim_ph"])))
        for x in S[c]:                                                    # 關鍵時間點的迭代法(同一設定只算一次)
            items.append((f"iter[{','.join(x['tags'])}]", np.mean(x["nerr_sr"]), np.mean(x["ssim_amp"]), np.mean(x["ssim_ph"])))
        agree = {"amp": [0, 0], "ph": [0, 0]}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][1] == items[j][1]:
                    continue
                for k, col in (("amp", 2), ("ph", 3)):
                    agree[k][1] += 1
                    agree[k][0] += int((items[i][1] < items[j][1]) == (items[i][col] > items[j][col]))
        for k in agree:
            allc[k][0] += agree[k][0]
            allc[k][1] += agree[k][1]
        b7[c] = {"items": items, "agree": agree}
        print(f"    {c:<6} " + ";".join(f"{nm} {e:.4f} / {a:.3f} / {p:.3f}" for nm, e, a, p in items)
              + f" → 一致 振幅 {agree['amp'][0]}/{agree['amp'][1]}、相位 {agree['ph'][0]}/{agree['ph'][1]}")
    print(f"    合計:振幅 {allc['amp'][0]}/{allc['amp'][1]}、相位 {allc['ph'][0]}/{allc['ph'][1]}(描述)")
    b7["total"] = allc
    V["b7"] = b7


# ============================================================================
# 圖(描述,不改變判讀)
# ============================================================================
def figures(C, V, R, tim, iters, keep, mroot, out_dir, where, smoke):
    plt = s5._plt()
    if plt is None:
        check("matplotlib 可用(畫圖需要)", False)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    def blank(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    xs = np.arange(len(CONDS))
    lab = [clabel(c) for c in CONDS]

    def shade(ax):
        ax.axvspan(len(OLD_CONDS) - 0.5, len(OLD_CONDS) + 1.5, color="#f3e6c4", alpha=0.5, lw=0)
        ax.axvspan(len(OLD_CONDS) + 1.5, len(CONDS) - 0.5, color="#e6eef8", alpha=0.6, lw=0)

    # 1. 總覽
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, B in zip(axs[:2], ["64", "512"]):
        shade(ax)
        for k, nm in enumerate(NETS):
            ax.plot(xs + (k - 0.5) * 0.2, [np.mean(V["cond"][c][nm][PRIMARY]["nerr"]) for c in CONDS], "o", color=COL[nm], ms=7,
                    label=f"{nm} G9")
        ax.plot(xs, [np.mean(V["cond"][c][NEW][PRIMARY][B]["opp"][2]) for c in CONDS], "x", color=COL["iter"], ms=8,
                label="iterative envelope at the network's time")
        for q in QS:
            ax.axhline(q, color="#d6452a" if q == Q_MAIN else "#e8a598", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"batch {B}: {MAIN} (dashed: quality thresholds; yellow = held-out, blue = out of range)", fontsize=8)
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
        ax.legend(fontsize=7, frameon=False)
    ax = axs[2]
    shade(ax)
    for p, ls in (("G9", "-"), ("G25", ":")):
        ax.plot(xs, [V["cmp"][c][p]["ratio"] for c in CONDS], ls, marker="o", color=COL[NEW], ms=4, label=f"P6B4r / P6B4, {p}")
    for yv in (0.8, 1.25):
        ax.axhline(yv, color="#0b0b0b", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=7)
    ax.set_title("tolerant training vs frozen network (> 1.25 = regression, < 0.8 = improvement)", fontsize=8)
    ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{where}: P6B4 (frozen) vs P6B4r (trained with random errors)", fontsize=10)
    fig.tight_layout()
    save(fig, "b_summary.png")

    # 2. 加速倍數
    fig, axs = plt.subplots(1, len(QS), figsize=(6 * len(QS), 4.8), sharey=True)
    combos = [(nm, B) for nm in NETS for B in ("64", "512")]
    wbar = 0.2
    for ax, Q in zip(axs, QS):
        shade(ax)
        top = 1.0
        for k, (nm, B) in enumerate(combos):
            for i, c in enumerate(CONDS):
                r = V["cond"][c][nm][PRIMARY][B]["speed"][str(Q)]
                if r["reach"] and np.isfinite(r["x"]):
                    top = max(top, r["x"])
        for k, (nm, B) in enumerate(combos):
            xv, yv, yl, xn, xi = [], [], [], [], []
            for i, c in enumerate(CONDS):
                r = V["cond"][c][nm][PRIMARY][B]["speed"][str(Q)]
                xp = i + (k - 1.5) * wbar
                if not r["reach"]:
                    xn.append(xp)
                elif not np.isfinite(r["x"]):
                    xi.append(xp)
                else:
                    xv.append(xp)
                    yv.append(r["x"])
                    yl.append(r["x"] - r["x_lo"])
            col = COL[nm]
            ax.bar(xv, yv, width=wbar, color=col, alpha=1.0 if B == "512" else 0.55, label=f"{nm} b{B}")
            ax.errorbar(xv, yv, yerr=[yl, [0.0] * len(yl)], fmt="none", ecolor="#0b0b0b", elinewidth=0.8, capsize=1.5)
            ax.plot(xn, [0.12] * len(xn), "x", color=col, ms=6)
            ax.plot(xi, [top * 2] * len(xi), "^", color=col, ms=6)
        ax.axhline(1.0, color="#0b0b0b", lw=0.7)
        ax.axhline(SPEED_MIN, color="#0b0b0b", lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(0.08, top * 4)
        ax.set_xticks(xs)
        ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"speed-up at quality {MAIN} <= {Q}" + (" (main)" if Q == Q_MAIN else "")
                     + "\nbar = upper value, whisker = lower bound (stop-point spacing)"
                     + "\nx at bottom = network does not reach; triangle at top = iterative never reaches", fontsize=8)
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    axs[0].set_ylabel("iterative time to reach Q / network time")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{where}: time-to-quality speed-up of the network (G9) over the iterative envelope", fontsize=10)
    fig.tight_layout()
    save(fig, "b_speedup.png")

    # 3. 時間 vs 誤差
    for B in ["64", "512"]:
        fig, axs = plt.subplots(3, 4, figsize=(18, 12), sharey=True)
        for ax, c in zip(axs.ravel(), CONDS):
            ts, vs = a7.env_curve(C[c]["env"], tim[B], iters)
            ax.plot(ts, vs, "-", color=COL["iter"], lw=1.6, drawstyle="steps-post", label="iterative envelope")
            for nm in NETS:
                a = np.mean(V["cond"][c][nm][PRIMARY]["nerr"])
                ax.plot([tim[B]["net:G9"]], [a], "o", color=COL[nm], ms=7, zorder=4, label=f"{nm} G9 {a:.4f}")
            for q in QS:
                ax.axhline(q, color="#d6452a" if q == Q_MAIN else "#e8a598", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(clabel(c) + (" (held-out)" if c in HELDOUT else (" (out of range)" if c in OUTRANGE else "")), fontsize=9)
            ax.grid(True, color="#e4e3df", which="both", lw=0.5)
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        for ax in axs[-1]:
            ax.set_xlabel("time per whole image (ms)")
        for ax in axs[:, 0]:
            ax.set_ylabel(f"{MAIN} on U")
        fig.suptitle(f"{where}: time vs error, batch {B} (dashed: quality thresholds)", fontsize=10)
        fig.tight_layout()
        save(fig, f"b_vs_envelope_b{B}.png")

    # 4. 重建圖
    geo, cfg = keep["geo"], keep["cfg"]
    U = geo.U.cpu()
    pm = cfg.phase_max
    y0, y1, x0, x1 = _bbox(U, a7.MARGIN)
    Uc = U[y0:y1, x0:x1].numpy()
    ps = np.array(R[MAIN_COND][NEW][PRIMARY]["ps_sr0"], float)
    ok = np.where(np.isfinite(ps))[0]
    i = int(ok[np.argsort(ps[ok], kind="stable")][len(ok) // 2])
    cols = [("O", "truth"), ("P6B4", "P6B4 G9"), (NEW, "P6B4r G9"), ("iter", "iterative (net's time, b512)")]
    conds = [c for c in RECON_CONDS if c in keep["img"]]
    fig, axs = plt.subplots(len(conds), 7, figsize=(15, 2.05 * len(conds) + 0.8), squeeze=False, constrained_layout=True)
    for r, c in enumerate(conds):
        im = keep["img"][c]
        o = im["O"][i]
        t = o[y0:y1, x0:x1].numpy()
        for k, (key, lb) in enumerate(cols):
            if key not in im:
                axs[r, k].axis("off")
                continue
            if key == "O":
                z, v = o, None
            else:
                z, v = a7.align_show(im[key][i], o, U)
            zc = z[y0:y1, x0:x1].numpy()
            hp = axs[r, k].imshow(np.where(np.abs(t) > AMP_MIN, np.angle(zc), np.nan), cmap="viridis", vmin=0, vmax=pm)
            axs[r, k].imshow(np.where(Uc, np.nan, 0.0), cmap="Greys", vmin=0, vmax=1, alpha=0.55)
            blank(axs[r, k])
            axs[r, k].set_title((lb + "\n" if r == 0 else "") + ("" if v is None else f"{MAIN} {v:.4f}"), fontsize=6.5)
            if key != "O":
                he = axs[r, 3 + k].imshow(np.where(Uc, np.abs(zc - t), np.nan), cmap="magma", vmin=0, vmax=0.5)
                blank(axs[r, 3 + k])
                if r == 0:
                    axs[r, 3 + k].set_title(f"|error| {lb}", fontsize=6.5)
        axs[r, 0].set_ylabel(clabel(c), fontsize=7)
    fig.colorbar(hp, ax=list(axs[:, 3]), fraction=0.03, label="phase (rad)")
    fig.colorbar(he, ax=list(axs[:, 6]), fraction=0.03, label="|error|")
    fig.suptitle(f"{where}: sample #{i} (median of P6B4r G9 in 'all three'); seed 0; aligned like the main metric", fontsize=9)
    save(fig, "b_recon.png")

    # 5. 訓練曲線
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in SEEDS:
        for nm, md, ls in ((NEW, model_dir(s, mroot), "-"), ("P6B4", s5c.run_dir("P6B4", PROBE, s, a6.N_TRAIN), ":")):
            p = md / "result.json"
            if not p.exists():
                continue
            hist = json.load(open(p)).get("history", [])
            if hist:
                ax.plot([x["epoch"] + 1 for x in hist], [x["final"] for x in hist], ls, color=COL[nm], lw=1.2,
                        label=f"{nm}" if s == SEEDS[0] else None)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss of the last stage (nerr on R0)")
    ax.set_title("training curves (P6B4r: with random errors, so higher is expected)", fontsize=9)
    ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "b_train.png")

    # 6. 部分同調的繞射圖
    if all(c in keep["diffr"] for c in ("ideal", "cohL", "cohH")):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, c in zip(axs, ("ideal", "cohL", "cohH")):
            im_ = ax.imshow(np.log10(keep["diffr"][c].numpy() + 1), cmap="magma")
            ax.set_title(f"centre diffraction pattern, {clabel(c)}\nlog10(counts + 1)", fontsize=8)
            blank(ax)
            fig.colorbar(im_, ax=ax, fraction=0.045)
        fig.suptitle(f"{where}: partial spatial coherence blurs the diffraction pattern (seed 0, field 0)", fontsize=9)
        save(fig, "b_coherence.png")


# ============================================================================
# 流程
# ============================================================================
def load_inputs(old_root, new_root, mroot, iters):
    """讀 7a 的 8 個條件檔 + 本階段的 4 個新條件檔,檢查版本、檢查結果、設定一致;檢查 3 個 P6B4r 模型。"""
    me = me_md5()
    C, bad = {}, []
    for c in CONDS:
        p = a7.cond_json(c, old_root) if c in OLD_CONDS else env_json(c, new_root)
        if not p.exists():
            bad.append(f"{p.name} 不存在")
            continue
        C[c] = json.load(open(p))
        m = C[c]["meta"]
        want = A7_MD5 if c in OLD_CONDS else me
        if m["script_md5"] != want:
            bad.append(f"{p.name} 的程式 md5 {m['script_md5'][:8]}… ≠ 預期 {want[:8]}…")
        if c in NEW_CONDS and m.get("a7_md5") != A7_MD5:
            bad.append(f"{p.name} 用的 scan_7a.py 不是 7a-1 的版本")
        if not C[c]["checks_ok"]:
            bad.append(f"{p.name} 有檢查未通過")
    if not bad:
        n0 = C[CONDS[0]]["meta"]["n"]
        for c in CONDS:
            if C[c]["meta"]["iters"] != list(iters) or C[c]["meta"]["n"] != n0:
                bad.append(f"{c} 的停止點或場數與其他條件不同")
    check("12 個條件檔都在(7a 的 8 個 + 新的 4 個)、版本正確、檢查全過、設定一致", not bad, ";".join(bad))
    mb = []
    for s in SEEDS:
        md = model_dir(s, mroot)
        if not ((md / "final.pt").exists() and (md / "result.json").exists()):
            mb.append(f"{md.name} 缺 final.pt / result.json")
            continue
        r = json.load(open(md / "result.json"))
        nb = json.load(open(s5c.run_dir("B", PROBE, s, a6.N_TRAIN) / "result.json"))["norm"]
        if r["norm"] != nb:
            mb.append(f"{md.name} 的輸入正規化 ≠ B 的")
        if r.get("script_md5") != me:
            mb.append(f"{md.name} 由不同版本的 scan_7b.py 訓練({str(r.get('script_md5'))[:8]}…)")
    check("3 個 P6B4r 模型都在、輸入正規化 = B 的(= P6B4 的)、由同一版程式訓練", not mb, ";".join(mb))
    return (C if not bad and not mb else None)


def run_final(dev, old_root, new_root, mroot, fig_dir, iters, smoke):
    me = me_md5()
    where = "validation fields (smoke)" if smoke else "validation fields, 7x7 scan"
    C = load_inputs(old_root, new_root, mroot, iters)
    if C is None:
        return None
    t0 = time.time()
    print("\n  計時(ms / 整張影像)", flush=True)
    tim = a7.timing(dev)
    print(f"  (計時完成,經過 {time.time() - t0:.0f} 秒;GPU {tim['gpu']})", flush=True)
    # t_match_conf 與 scan_7a.t_match 一致
    e_t = 0.0
    for c in CONDS:
        for q in QS:
            for B in ("64", "512"):
                ta, tb = t_match_conf(C[c]["env"], q, tim[B], iters)[0], a7.t_match(C[c]["env"], q, tim[B], iters)
                e_t = max(e_t, 0.0 if (ta == tb) else abs(ta - tb))
    check("門檻達到時間:本程式 = scan_7a.t_match(12 個條件 × 3 個門檻 × 2 種 batch)", e_t < 1e-12, f"最大差 {e_t:.1e}")
    print("\n  網路評估(12 個條件 × 3 seeds × P6B4 / P6B4r × G9 / G25)", flush=True)
    R, keep = eval_nets(dev, C, mroot, log=lambda s: print(s, flush=True))
    dmax = max(absdiff(R[c]["P6B4"][p]["nerr_ph"][i], C[c]["nets"][p]["nerr_ph"][i])
               for c in CONDS for p in PIPES for i in range(len(SEEDS)))
    dsr = max(absdiff(R[c]["P6B4"][p][MAIN][i], C[c]["nets"][p][MAIN][i]) for c in CONDS for p in PIPES for i in range(len(SEEDS)))
    check(f"重現:凍結 P6B4 的 G9 / G25 nerr_ph = 條件檔(12 個條件、3 seeds;< {REPRO_TOL:.0e})→ 量測與 7a 相同", dmax < REPRO_TOL,
          f"最大差 {dmax:.1e}(nerr_sr 最大差 {dsr:.1e},描述)")
    V = report(C, R, tim, iters, smoke)
    final = final_json(new_root)
    res = {"verdict": V, "nets": R, "timing": tim, "checks_ok": all_ok(), "smoke": smoke, "script_md5": me, "a7_md5": A7_MD5,
           "complete": False, "gpus": {c: C[c]["meta"]["gpu"] for c in CONDS}}
    dump_json(res, final)                                                  # 先存(下面出錯也不會丟掉判讀)
    print("\n" + "=" * 100)
    print("迭代法的 SSIM(關鍵時間點重算;副指標)")
    print("=" * 100)
    try:
        S, worst = iter_ssim(dev, C, V, tim, iters, keep, log=lambda s: print(s, flush=True))
        res["iter_ssim"] = S
        res["iter_repro_maxdiff"] = worst
        ssim_report(V, R, S)
    except Exception as e:                                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"迭代法 SSIM 完成(判讀不受影響,已存於 {final.name})", False, f"{type(e).__name__}: {e}")
    res["verdict"] = V
    res["checks_ok"] = all_ok()
    dump_json(res, final)
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        figures(C, V, R, tim, iters, keep, mroot, fig_dir, where, smoke)
    except Exception as e:                                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"畫圖完成(判讀不受影響,已存於 {final.name})", False, f"{type(e).__name__}: {e}")
    res["checks_ok"] = all_ok()
    res["complete"] = True
    dump_json(res, final)
    print(f"  結果:{final}")
    return res


def check_files(kind):
    """kind:check / smoke / train / env / final。"""
    need = [Path(f) for f in ("scan_7a.py", "scan_6a.py", "scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py", "scan_5e.py",
                              "scan_5g.py", "scan_5h.py", "probe_4_2b.py", "probe_4_1.py", "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    need += [s5c.run_dir(gp, PROBE, s, a6.N_TRAIN) / f for gp in ("B", "P6B4") for s in SEEDS for f in ("final.pt", "result.json")]
    if kind == "smoke":
        need += [a7.cond_json(c, a7.SMOKE_DIR) for c in OLD_CONDS]
    if kind == "final":
        need += [a7.cond_json(c) for c in OLD_CONDS]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在(含 B、P6B4 的模型" + ("、7a 的條件檔" if kind in ("smoke", "final") else "") + ")",
          not missing, "缺:" + ", ".join(missing) if missing else "")
    a7m = md5(Path(a7.__file__).resolve())
    check("scan_7a.py = 7a-1 正式結果所用的版本", a7m == A7_MD5, f"md5 {a7m}")
    return not missing and a7m == A7_MD5


def need_passed(me):
    if not PASSED.exists():
        raise SystemExit(f"❌ 找不到 {PASSED}:先在 dev 節點跑 python scan_7b.py --smoke,全部通過才可送件")
    pm = json.load(open(PASSED))["script_md5"]
    if pm != me:
        raise SystemExit(f"❌ smoke 通過時的 scan_7b.py(md5 {pm})與現在的({me})不同:用現在的版本重跑 --smoke")
    print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", type=int, choices=range(len(TASKS)))
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = me_md5()
    print(f"scan_7b.py md5 {me};GPU {a7.gpu_name(dev)}")
    kind = "check" if a.check else ("smoke" if a.smoke else ("final" if a.final else
                                                              (TASKS[a.task][0] if a.task is not None else "check")))
    if not check_files(kind):
        print("\n❌ 缺檔案或版本不同,停下來")
        sys.exit(1)
    if a.check:
        checks(dev)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.smoke:
        n7, it7 = a7.SMOKE_FIELDS, a7.SMOKE_ITERS
        print("#" * 70)
        print(f"迷你全流程(--smoke):訓練 3 個 seed × {SMOKE_EPOCHS} epoch × {SMOKE_TRAIN_N} 個場;4 個新條件({n7} 個場、"
              f"停止點 {it7});彙整(7a 的條件檔用 {a7.SMOKE_DIR.name});輸出到 {SMOKE_DIR}")
        print("#" * 70)
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)                                      # 只刪這個 smoke 專用資料夾
        SMOKE_DIR.mkdir(parents=True)
        t0 = time.time()
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
        tt = time.time()
        for s in SEEDS:
            train_r(s, dev, SMOKE_TRAIN_N, SMOKE_EPOCHS, model_dir(s, SMOKE_DIR))
        per = (time.time() - tt) / len(SEEDS)
        for c in NEW_CONDS:
            run_env(dev, c, n7, it7, SMOKE_DIR, smoke=True)
        run_final(dev, a7.SMOKE_DIR, SMOKE_DIR, SMOKE_DIR, SMOKE_DIR / "figs", it7, smoke=True)
        if not QUICK:
            steps = (a6.N_TRAIN // g.BATCH) * g.EPOCHS
            est = per / max(1, SMOKE_TRAIN_N // g.BATCH) * steps / 3600
            print(f"  預估正式訓練每個 seed 約 {est:.2f} 小時(依本次每步的時間;含載入的固定開銷 → 偏高);job 時限 6 小時")
        print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒")
        if all_ok():
            json.dump({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
            print("\n✅ 迷你全流程全部通過 → 可以送出正式的 job(先 run_scan7b_jobs.sh 的 array,再 run_scan7b_final.sh)")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    if a.task is not None:
        what, arg = TASKS[a.task]
        need_passed(me)
        if what == "train":
            md = model_dir(arg)
            if (md / "final.pt").exists():
                raise SystemExit(f"❌ {md}/final.pt 已存在:這個 seed 已經訓練完。為避免覆蓋,先告訴 Claude")
            checks(dev)
            if not all_ok():
                print("\n❌ 內建檢查未通過 → 沒有訓練。把輸出貼給 Claude")
                sys.exit(1)
            train_r(arg, dev)
            print("\n" + (f"✅ P6B4r seed {arg} 訓練完成、檢查全過" if all_ok() else f"❌ P6B4r seed {arg} 有檢查未通過"))
            sys.exit(0 if all_ok() else 1)
        p = env_json(arg)
        if p.exists():
            raise SystemExit(f"❌ {p} 已存在:這個條件已經跑完。為避免覆蓋,先告訴 Claude")
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有量測。把輸出貼給 Claude")
            sys.exit(1)
        run_env(dev, arg, None, a7.ITERS, RUN_ROOT, smoke=False)
        print("\n" + (f"✅ 新條件 {arg} 完成、檢查全過" if all_ok() else f"❌ 新條件 {arg} 有檢查未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.final:
        out = final_json()
        if out.exists() and json.load(open(out)).get("complete"):
            raise SystemExit(f"❌ {out} 已存在且完整:正式結果已經有了。為避免覆蓋,先告訴 Claude")
        need_passed(me)
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有彙整。把輸出貼給 Claude")
            sys.exit(1)
        res = run_final(dev, RUN_ROOT, RUN_ROOT, None, FIG_DIR, a7.ITERS, smoke=False)
        print("\n" + ("✅ 內建檢查全部通過" if all_ok() and res else "❌ 有項目未通過(判讀先不要採信)"))
        print("把完整輸出與 figs_scan7b/ 的圖傳給 Claude")
        sys.exit(0 if all_ok() and res else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
