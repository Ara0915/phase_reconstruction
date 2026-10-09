#!/usr/bin/env python
"""階段五 5-3:重疊掃描的網路(階段五協定 §十二)。

問:1) 看 9 張重疊繞射圖的網路,是否比只看 1 張的準?(B vs A)
   2) 物理放進架構(C)/ 放進 loss(D)有沒有幫助?
   3) 同時間下,網路是否穩健地比最強的迭代法(5-2b 的包絡)準?

四組(每組 × DEF2 / DISK × 3 seeds = 24 個網路):
  A  中央 1 張 → U-Net                                 監督
  B  9 張 → U-Net                                      監督
  C  9 張 → 每張估繞射相位 → 固定 IFFT → 以已知探針與位置合併 → U-Net   監督
  D  9 張 → U-Net                                      監督 + 物理 loss(振幅,λ = 1)
輸出:掃描範圍 80×80(場座標 16–95)的物體;監督 loss = R0 上的 nerr_ph(= 評分指標)。

用法(需在計算節點執行;需 scan_5.py、scan_5b.py、probe_4_2b.py、probe_4_1.py、ambiguity_check.py、
      5-2b 的 scan5b_raw.json):
    python scan_5c.py --check                 # 全部單元測試
    python scan_5c.py --train --task 0..23    # 訓練一個網路(run_scan5c_train.sh 的 array)
    python scan_5c.py --eval                  # 評估全部網路(run_scan5c_eval.sh)
"""
import argparse
import hashlib
import json
import math
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
from src.config import Cfg                                          # noqa: E402

# 嚴格 fp32(協定 §12.2、§12.4):關掉 TF32,網路的訓練、評估與計時都不走較低精度的捷徑(對網路較保守)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

RUN_ROOT = s5.RUN_ROOT
OUT_JSON = RUN_ROOT / "scan5c.json"
FIG_DIR = Path("figs_scan5c")
GROUPS = ["A", "B", "C", "D"]
PROBES = s5.PROBES
SEEDS = s5.SEEDS
SCAN = "3x3s8"
CENTER = 4
N_TRAIN = 20_000
EPOCHS = 40
BATCH = 128
LR = 2e-3
FFT_K = 4
LAMBDA_PHYS = 1.0
CAL_N = 2000                                 # 正規化常數用的訓练場數(同 train.py 的 pool[:2000])
TRAIN_EVAL_N = 512                           # 訓練場上的 nerr(看過擬合)
CHUNK = 1000                                 # 產生訓練場的分塊大小
TRAIN_SEED_BASE = 50_000_000                 # 訓練場 seed = 基底 + 10^7 × 網路 seed + 10^4 × 分塊;與測試(2011)、校準(2026)遠離
Q_TARGETS = [0.1, 0.03, 0.01]                # (n6) 目標品質
LEARN_STEPS, LEARN_N = 200, 16               # 可學性測試
EVAL_CHUNK = 128
NOISE_SEED_BASE = 1_000_000_000               # 訓練時每步的 Poisson seed 基底(遠離測試量測的 seed)

QUICK = os.environ.get("CDI_QUICK") == "1"
if QUICK:
    N_TRAIN, EPOCHS, BATCH, CAL_N, TRAIN_EVAL_N, CHUNK = 64, 2, 16, 32, 16, 32
    LEARN_STEPS = 60
    Q_TARGETS = [0.5, 0.3]

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def task_of(t):
    """array 編號 → (組, 探針, seed):0–5 = A,6–11 = B,…;每組內 DEF2 s0–2、DISK s0–2。"""
    return GROUPS[t // 6], PROBES[(t // 3) % 2], SEEDS[t % 3]


def run_dir(group, probe, seed, n_train):
    return RUN_ROOT / f"scan5c_{group}_{probe}_n{n_train}_s{seed}"


# ============================================================================
# 幾何:掃描範圍方框(80×80)
# ============================================================================
def geometry(cfg):
    """方框在場中的左上角、方框大小、9 個視窗在方框中的左上角。"""
    st = s5.window_starts(cfg, SCAN)
    b0 = min(min(y for y, _ in st), min(x for _, x in st))
    size = max(max(y for y, _ in st), max(x for _, x in st)) + cfg.canvas - b0
    return b0, size, [(y - b0, x - b0) for y, x in st]


def r0_box(cfg, pr, dev):
    b0, size, _ = geometry(cfg)
    R0 = s5.footprint_mask(cfg, pr, s5.window_starts(cfg, "1x1"), dev)
    return R0[b0:b0 + size, b0:b0 + size]


def to_field(box, cfg, F):
    """方框(complex [N, B, B])放回場座標(方框外 = 0)。"""
    b0, size, _ = geometry(cfg)
    f = torch.zeros(box.shape[0], F, F, dtype=box.dtype, device=box.device)
    f[:, b0:b0 + size, b0:b0 + size] = box
    return f


# ============================================================================
# 訓練場
# ============================================================================
def train_chunk_seed(seed, k):
    return TRAIN_SEED_BASE + 10_000_000 * seed + 10_000 * k


def make_train_fields(cfg, n, seed):
    """n 個訓練場(與測試場同一個產生器 = scan_5.make_fields),只存方框 [n, 2, B, B](CPU)。"""
    b0, size, _ = geometry(cfg)
    out = []
    for k in range(math.ceil(n / CHUNK)):
        m = min(CHUNK, n - k * CHUNK)
        f = s5.make_fields(cfg, m, seed=train_chunk_seed(seed, k))
        out.append(f[:, :, b0:b0 + size, b0:b0 + size].contiguous())
    return torch.cat(out)


# ============================================================================
# 輸入(同階段四:log 強度、自相關、beamstop 遮罩)
# ============================================================================
def calibrate_norm(counts):
    """counts:9 個 [N, W, W] 的清單。全域常數(同 src.physics.calibrate_input_norm)。"""
    c = torch.stack(counts, 1)
    logI = torch.log1p(c)
    ac = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(c, dim=(-2, -1)), norm="ortho").real, dim=(-2, -1))
    return {"log_mean": float(logI.mean()), "log_std": float(logI.std().clamp_min(1e-6)),
            "ac_scale": float(ac.abs().amax(dim=(-2, -1)).mean().clamp_min(1e-12))}


def prep(counts, bs, norm, group, cfg_p):
    """量測 → 網路輸入。回傳 dict:x(補零到方框大小的通道)、x9(C 用,各張 3 通道)、meas(量測振幅)。"""
    from src.hio import _measured_amp
    use = [counts[CENTER]] if group == "A" else counts
    c = torch.stack(use, 1)                                                   # [N, J, W, W]
    logI = (torch.log1p(c) - norm["log_mean"]) / norm["log_std"]
    ac = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(c, dim=(-2, -1)), norm="ortho").real,
                            dim=(-2, -1)) / norm["ac_scale"]
    m = bs.expand(c.shape[0], 1, *bs.shape)
    out = {}
    if group == "C":
        J = c.shape[1]
        out["x9"] = torch.stack([logI, ac, m.expand(-1, J, -1, -1)], 2)       # [N, J, 3, W, W]
        out["meas"] = _measured_amp(c, cfg_p)                                 # [N, J, W, W]
    else:
        x = torch.cat([logI, ac, m], 1)                                       # [N, 2J+1, W, W]
        p = (BOX_SIZE - x.shape[-1]) // 2
        out["x"] = Fnn.pad(x, (p, p, p, p))
    return out


BOX_SIZE = 80                                # 由 geometry() 檢查(單元測試)


def in_channels(group):
    return 3 if group == "A" else 19


# ============================================================================
# 模型
# ============================================================================
def box_unet(cfg, cin):
    from src.model import UNet
    net = UNet(cfg, cin=cin)
    net.register_buffer("sup", torch.ones(BOX_SIZE, BOX_SIZE))              # 不再有方框 support
    return net


class PlainNet(nn.Module):
    """A / B / D:U-Net(同階段四,base 32、下採樣 3 次)。"""

    def __init__(self, cfg, group):
        super().__init__()
        self.unet = box_unet(cfg, in_channels(group))

    def forward(self, inp):
        return self.unet(inp["x"])


class PhysNet(nn.Module):
    """C:每張估 K 組繞射相位 → 固定 IFFT → 以已知探針、已知位置合併(= AP 的合併)→ U-Net。"""

    def __init__(self, cfg, pr, starts_box):
        super().__init__()
        k = FFT_K
        self.k = k
        self.det = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 3 * k, 1))                                          # 每個假設:cos、sin、beamstop 內振幅
        self.unet = box_unet(cfg, 2 * k + 1)
        self.starts = starts_box
        W = cfg.canvas
        wsum = torch.zeros(BOX_SIZE, BOX_SIZE)
        for y0, x0 in starts_box:
            wsum[y0:y0 + W, x0:x0 + W] += pr["Pa"].cpu() ** 2
        self.register_buffer("Pc", pr["P"].conj().cpu())
        self.register_buffer("wsum", wsum)
        self.delta = s5b.AP_DELTA * float(wsum.max())

    def merge(self, psi):
        """psi:complex [N, J, K, W, W] → 物體 [N, K, B, B] = Σ_j P* ψ_j / (Σ_j |P|² + δ)。"""
        N, J, K, W, _ = psi.shape
        num = torch.zeros(N, K, BOX_SIZE, BOX_SIZE, dtype=psi.dtype, device=psi.device)
        for j, (y0, x0) in enumerate(self.starts):
            num[..., y0:y0 + W, x0:x0 + W] = num[..., y0:y0 + W, x0:x0 + W] + self.Pc * psi[:, j]
        return num / (self.wsum + self.delta)

    def forward(self, inp):
        x9, meas = inp["x9"], inp["meas"]
        N, J, C, W, _ = x9.shape
        k = self.k
        h = self.det(x9.reshape(N * J, C, W, W)).reshape(N, J, 3 * k, W, W)
        a, b, c = h[:, :, :k], h[:, :, k:2 * k], Fnn.softplus(h[:, :, 2 * k:])
        bsm = x9[:, :, 2:3]                                                   # beamstop 遮罩(1 = 有量測)
        amp = torch.where(bsm > 0, meas[:, :, None], c)
        nrm = torch.sqrt(a * a + b * b + 1e-6)
        E = torch.complex(amp * a / nrm, amp * b / nrm)
        psi = torch.fft.ifft2(torch.fft.ifftshift(E, dim=(-2, -1)), norm="ortho")
        O = self.merge(psi)
        w = (self.wsum / self.wsum.max()).expand(N, 1, BOX_SIZE, BOX_SIZE)
        return self.unet(torch.cat([O.real, O.imag, w], 1))


def build_model(group, cfg, pr):
    if group == "C":
        return PhysNet(cfg, pr, geometry(cfg)[2])
    return PlainNet(cfg, group)


# ============================================================================
# loss
# ============================================================================
def nerr_loss(pred, target, m):
    """R0 上的 nerr_ph(每個樣本各自對齊整體相位;batch 內加總再正規化)。pred [N, 2, B, B],target complex [N, B, B]。"""
    mk = m.to(pred.dtype)
    a = torch.polar(pred[:, 0], pred[:, 1]) * mk
    t = target * mk
    cr = (a * t.conj()).sum((1, 2))
    cross = torch.sqrt(cr.real ** 2 + cr.imag ** 2 + 1e-12)
    Ea, Et = (a.abs() ** 2).sum((1, 2)), (t.abs() ** 2).sum((1, 2))
    return (Ea.sum() + Et.sum() - 2 * cross.sum()) / Et.sum().clamp_min(1e-12)


def phys_loss(pred, counts, bs, cfg_p, pr, starts_box):
    """振幅型物理 loss:Σ bs (|F{P·Ô_j}| − √(C_j/s))² / Σ bs C_j/s(9 個視窗;不需真值)。"""
    from src.hio import _measured_amp
    W = cfg_p.canvas
    O = torch.polar(pred[:, 0], pred[:, 1])
    num = pred.new_zeros(())
    den = pred.new_zeros(())
    for j, (y0, x0) in enumerate(starts_box):
        E = torch.fft.fftshift(torch.fft.fft2(pr["P"] * O[:, y0:y0 + W, x0:x0 + W], norm="ortho"), dim=(-2, -1))
        mag = torch.sqrt(E.real ** 2 + E.imag ** 2 + 1e-12)
        meas = _measured_amp(counts[j], cfg_p)
        num = num + (bs * (mag - meas) ** 2).sum()
        den = den + (bs * meas ** 2).sum()
    return num / den.clamp_min(1e-12)


# ============================================================================
# 共同準備
# ============================================================================
def setup(seed, probe, dev):
    import probe_4_2b as pb
    from src.physics import beamstop_mask
    cfg = s5.load_cfg(seed)
    pr = pb.build_probes(cfg, dev)[probe]
    cp = s5.scan_cfg(cfg, probe, pr, s5.SCANS[SCAN][2])
    bs = beamstop_mask(cfg, device=dev)
    return cfg, pr, cp, bs


def measure_box(fields_box, probe, pr, cp, bs, cfg, seed):
    return s5.measure_scan(fields_box, probe, pr, cp, bs, geometry(cfg)[2], seed=seed)[0]


# ============================================================================
# 訓練
# ============================================================================
def train(group, probe, seed, n_train, dev, epochs=None, fields=None, quiet=False):
    import random
    epochs = EPOCHS if epochs is None else epochs
    cfg, pr, cp, bs = setup(seed, probe, dev)
    starts_box = geometry(cfg)[2]
    R0 = r0_box(cfg, pr, dev)
    out = run_dir(group, probe, seed, n_train)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    if fields is None:
        fields = make_train_fields(cfg, n_train, seed)
    fields = fields.to(dev)
    print(f"[init] {group} {probe} s{seed}:訓練場 {len(fields)} 個({time.time() - t0:.1f}s)", flush=True)
    cal_counts = measure_box(fields[:CAL_N], probe, pr, cp, bs, cfg, seed=seed + 7)
    norm = calibrate_norm(cal_counts)
    del cal_counts
    model = build_model(group, cfg, pr).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_par:,} norm={norm}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    steps = len(fields) // BATCH
    total = epochs * steps + 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total)
    meta = {"group": group, "probe": probe, "seed": seed, "n_train": n_train, "epochs": epochs, "batch": BATCH,
            "lr": LR, "fft_k": FFT_K, "lambda_phys": LAMBDA_PHYS, "norm": norm, "params": n_par,
            "train_seed_base": train_chunk_seed(seed, 0), "quick": QUICK}
    json.dump(meta, open(out / "config_used.json", "w"), indent=2)

    ck = out / "ckpt.pt"
    start, hist = 0, []
    if ck.exists():
        d = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(d["model"])
        opt.load_state_dict(d["opt"])
        sched.load_state_dict(d["sched"])
        start, hist = d["epoch"] + 1, d["history"]
        print(f"[resume] 從 epoch {start} 續跑", flush=True)

    gstep = start * steps
    for ep in range(start, epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 100_003 + ep)
        perm = torch.randperm(len(fields), generator=g)
        acc = {"loss": 0.0, "sup": 0.0, "phys": 0.0}
        n_ok, n_skip, te = 0, 0, time.time()
        for i in range(steps):
            idx = perm[i * BATCH:(i + 1) * BATCH].to(dev)
            obj = fields[idx]
            with torch.no_grad():
                counts = measure_box(obj, probe, pr, cp, bs, cfg, seed=NOISE_SEED_BASE + seed * 10_000_019 + gstep)
                inp = prep(counts, bs, norm, group, cp)
            gstep += 1
            pred = model(inp)
            sup = nerr_loss(pred, torch.polar(obj[:, 0], obj[:, 1]), R0)
            loss = sup
            ph = torch.zeros(())
            if group == "D":
                ph = phys_loss(pred, counts, bs, cp, pr, starts_box)
                loss = sup + LAMBDA_PHYS * ph
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip += 1
                if sched.last_epoch + 1 < total:
                    sched.step()
                continue
            loss.backward()
            opt.step()
            if sched.last_epoch + 1 < total:
                sched.step()
            n_ok += 1
            acc["loss"] += float(loss)
            acc["sup"] += float(sup)
            acc["phys"] += float(ph)
        for k in acc:
            acc[k] /= max(n_ok, 1)
        acc.update({"epoch": ep, "skipped": n_skip, "sec": time.time() - te, "lr": sched.get_last_lr()[0]})
        hist.append(acc)
        if not quiet:
            print(f"epoch {ep:03d}  loss={acc['loss']:.5f}  sup={acc['sup']:.5f}"
                  + (f"  phys={acc['phys']:.5f}" if group == "D" else "")
                  + f"  lr={acc['lr']:.2e}  跳過 {n_skip}  {acc['sec']:.1f}s", flush=True)
        tmp = out / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "history": hist}, tmp)
        os.replace(tmp, ck)                                               # 寫完才換名:時限砍在寫入中途也不會壞檔

    # 訓練場上的 nerr_ph(固定雜訊 seed;看過擬合)
    model.eval()
    with torch.no_grad():
        sub = fields[:TRAIN_EVAL_N]
        counts = measure_box(sub, probe, pr, cp, bs, cfg, seed=seed + 99)
        pred = predict(model, counts, bs, norm, group, cp)
        O = torch.polar(sub[:, 0], sub[:, 1])
        Et = ((O.abs() ** 2) * R0).sum((1, 2))
        tr = [float(nerr_loss(pred[i:i + 1], O[i:i + 1], R0)) for i in range(len(sub)) if Et[i] > 1e-12]   # 空樣本不列入(同 field_metrics)
        pooled = float(nerr_loss(pred, O, R0))                             # 與訓練 loss 同一種估計(batch 加總後正規化)
    meta.update({"history": hist, "train_nerr_ph": float(np.mean(tr)), "train_nerr_pooled": pooled,
                 "train_sec": time.time() - t0})
    torch.save(model.state_dict(), out / "final.pt")
    json.dump(meta, open(out / "result.json", "w"), indent=2)
    print(f"[done] {group} {probe} s{seed}:訓練場 nerr_ph {np.mean(tr):.4f}  {time.time() - t0:.0f}s → {out}", flush=True)
    return model, norm, hist


@torch.no_grad()
def predict(model, counts, bs, norm, group, cp, chunk=EVAL_CHUNK):
    outs = []
    n = counts[0].shape[0]
    for i in range(0, n, chunk):
        outs.append(model(prep([c[i:i + chunk] for c in counts], bs, norm, group, cp)))
    return torch.cat(outs)


# ============================================================================
# 單元測試
# ============================================================================
def _tile_hashes(t):
    """非空白物體塊的雜湊。產生器偶爾輸出全零的物體(約 0.07%),全零塊在任何 seed 下都相同,不代表資料重疊。"""
    return {hashlib.md5(x.numpy().tobytes()).hexdigest() for x in t if bool(x[0].abs().sum() > 0)}


@torch.no_grad()
def unit_tests(dev, groups=None):
    import probe_4_1 as p41
    from src.physics import beamstop_mask
    groups = groups or GROUPS
    print("=" * 70)
    print(f"5-3 單元測試(組:{' '.join(groups)})")
    print("=" * 70)
    cfg = s5.load_cfg(0)
    b0, size, stb = geometry(cfg)
    F, W = s5.field_size(cfg), cfg.canvas

    # (1) 座標
    ok = (size == BOX_SIZE and b0 == 16 and stb[CENTER] == (8, 8)
          and sorted(stb) == sorted((y, x) for y in (0, 8, 16) for x in (0, 8, 16)))
    check("座標:方框 = 場的 16–95(80×80);9 個視窗在方框中的位置 {0, 8, 16}²;中央 = (8, 8)", ok,
          f"方框起點 {b0}、大小 {size}、中央 {stb[CENTER]}")

    # (2) 資料不重疊:seed 範圍 + 抽樣比對物體塊
    d = s5.tile_size(cfg)
    s0 = (cfg.canvas - d) // 2
    used_test = {cfg.test_seed + s5.FIELD_SEED_OFFSET, p41.CALIB_SEED, p41.CALIB_SEED + 1}
    lo = min(train_chunk_seed(s, 0) for s in SEEDS)
    far = all(lo - u > 100_000 for u in used_test)
    n_t = 64 if QUICK else 512
    test_tiles = p41.make_home(cfg, n_t * s5.TILES ** 2, seed=cfg.test_seed + s5.FIELD_SEED_OFFSET)[:, :, s0:s0 + d, s0:s0 + d]
    tr_tiles = torch.cat([p41.make_home(cfg, 64 * s5.TILES ** 2, seed=train_chunk_seed(s, 0))[:, :, s0:s0 + d, s0:s0 + d]
                          for s in SEEDS])
    inter = _tile_hashes(test_tiles) & _tile_hashes(tr_tiles)
    n_empty = (int((test_tiles[:, 0].abs().sum((1, 2)) == 0).sum()), int((tr_tiles[:, 0].abs().sum((1, 2)) == 0).sum()))
    check("資料不重疊:訓練場的 seed 遠離測試 / 校準;抽樣的非空白物體塊無相同", far and not inter,
          f"訓練 seed 最小 {lo:,};測試 / 校準 {sorted(used_test)};相同的塊 {len(inter)}"
          f"(全零的空白塊不比:測試 {n_empty[0]} / 訓練 {n_empty[1]} 個)")

    import probe_4_2b as pb
    allp = pb.build_probes(cfg, dev)
    bs = beamstop_mask(cfg, device=dev)
    fld = s5.make_fields(cfg, 8, seed=11, device=dev)
    box = fld[:, :, b0:b0 + size, b0:b0 + size].contiguous()
    e_meas, e_loss, e_phys, e_merge = {}, {}, {}, {}
    for name in PROBES:
        pr = allp[name]
        cp = s5.scan_cfg(cfg, name, pr)
        # (3) 量測一致:方框版 = 場版(同 seed 逐位元)
        a = s5.measure_scan(fld, name, pr, cp, bs, s5.window_starts(cfg, SCAN), seed=5)[0]
        b = measure_box(box, name, pr, cp, bs, cfg, seed=5)
        e_meas[name] = max(float((x - y).abs().max()) for x, y in zip(a, b))
        # (4) loss = 指標
        R0b = r0_box(cfg, pr, dev)
        R0f = s5.footprint_mask(cfg, pr, s5.window_starts(cfg, "1x1"), dev)
        O = torch.polar(box[:, 0], box[:, 1])
        noisy = torch.stack([(box[:, 0] * 0.8 + 0.1).clamp(0, 1), (box[:, 1] + 0.3).clamp(0, cfg.phase_max)], 1)
        worst = 0.0
        for i in range(8):
            l1 = float(nerr_loss(noisy[i:i + 1], O[i:i + 1], R0b))
            est = to_field(torch.polar(noisy[i:i + 1, 0], noisy[i:i + 1, 1]), cfg, F)
            l2 = s5.field_metrics(est, torch.polar(fld[i:i + 1, 0], fld[i:i + 1, 1]), R0f)["nerr_ph"]
            worst = max(worst, abs(l1 - l2))
        e_loss[name] = worst
        # (5) 物理 loss:真值(無雜訊)≈ 0;錯的物體 > 1e-2
        c0 = Cfg.from_dict(cp.to_dict())
        c0.add_poisson = False
        cnt = measure_box(box, name, pr, c0, bs, cfg, seed=0)
        e_phys[name] = (float(phys_loss(box, cnt, bs, c0, pr, stb)), float(phys_loss(noisy, cnt, bs, c0, pr, stb)))
        # (6) C 的合併:真實出射波 → 真值(照明夠強處)
        net = PhysNet(cfg, pr, stb).to(dev)
        psi = torch.stack([pr["P"] * O[:, y0:y0 + W, x0:x0 + W] for y0, x0 in stb], 1)[:, :, None]
        Om = net.merge(psi)[:, 0]
        strong = net.wsum > 0.2 * net.wsum.max()
        e_merge[name] = float(((Om - O).abs() * strong).max())
    R0f = s5.footprint_mask(cfg, allp[PROBES[0]], s5.window_starts(cfg, "1x1"), dev)
    inside = all(bool(s5.footprint_mask(cfg, allp[k], s5.window_starts(cfg, "1x1"), dev)[b0:b0 + size, b0:b0 + size].sum()
                      == s5.footprint_mask(cfg, allp[k], s5.window_starts(cfg, "1x1"), dev).sum()) for k in PROBES)
    check("R0 完全在方框內(兩個探針)", inside, f"R0 {int(R0f.sum())} px")
    check("量測一致:方框上的量測 = 場上的量測(逐位元)", all(v == 0 for v in e_meas.values()),
          " / ".join(f"{k} {v:.1e}" for k, v in e_meas.items()))
    check("loss = 指標:單一樣本的 nerr_ph loss = scan_5.field_metrics(< 1e-5)", all(v < 1e-5 for v in e_loss.values()),
          " / ".join(f"{k} {v:.1e}" for k, v in e_loss.items()))
    check("物理 loss:真值(無雜訊)< 1e-6、錯的物體 > 1e-2",
          all(a < 1e-6 and b > 1e-2 for a, b in e_phys.values()),
          " / ".join(f"{k} 真值 {a:.1e}、錯 {b:.1e}" for k, (a, b) in e_phys.items()))
    check("C 的合併:真實出射波 → 真值(Σ|P|² > 0.2 max 處,< 1e-2)", all(v < 1e-2 for v in e_merge.values()),
          " / ".join(f"{k} {v:.1e}" for k, v in e_merge.items()))

    # (7) 前向與 (8) 可學性
    pr = allp[PROBES[0]]
    cp = s5.scan_cfg(cfg, PROBES[0], pr)
    cnt = measure_box(box, PROBES[0], pr, cp, bs, cfg, seed=1)
    norm = calibrate_norm(cnt)
    shapes, learn = {}, {}
    for gname in groups:
        torch.manual_seed(0)
        net = build_model(gname, cfg, pr).to(dev).eval()
        inp = prep(cnt, bs, norm, gname, cp)
        out = net(inp)
        shp = tuple(out.shape) == (8, 2, BOX_SIZE, BOX_SIZE)
        rng = bool(out[:, 0].min() >= 0 and out[:, 0].max() <= 1 and out[:, 1].min() >= 0
                   and out[:, 1].max() <= cfg.phase_max + 1e-6)
        cin = inp["x"].shape[1] if "x" in inp else inp["x9"].shape[2] * inp["x9"].shape[1]
        shapes[gname] = (shp and rng, cin, sum(p.numel() for p in net.parameters()))
    want_cin = {"A": 3, "B": 19, "C": 27, "D": 19}                           # C:9 張 × 3 通道(各張分開處理)
    check("前向:輸入通道 A 3 / B、D 19 / C 9×3;輸出 [N, 2, 80, 80]、振幅 ∈ [0, 1]、相位 ∈ [0, φmax]",
          all(v[0] and v[1] == want_cin[k] for k, v in shapes.items()),
          ";".join(f"{k} 輸入 {v[1]} 通道、參數 {v[2]:,}" for k, v in shapes.items()))
    with torch.enable_grad():
        small = s5.make_fields(cfg, LEARN_N, seed=12, device=dev)[:, :, b0:b0 + size, b0:b0 + size].contiguous()
        Os = torch.polar(small[:, 0], small[:, 1])
        R0b = r0_box(cfg, pr, dev)
        cnt = measure_box(small, PROBES[0], pr, cp, bs, cfg, seed=3)
        norm = calibrate_norm(cnt)
        for gname in groups:
            torch.manual_seed(0)
            net = build_model(gname, cfg, pr).to(dev).train()
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            inp = prep(cnt, bs, norm, gname, cp)
            first = None
            for _ in range(LEARN_STEPS):
                pred = net(inp)
                loss = nerr_loss(pred, Os, R0b)
                if gname == "D":
                    loss = loss + LAMBDA_PHYS * phys_loss(pred, cnt, bs, cp, pr, stb)
                first = loss.item() if first is None else first
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pred = net(inp)
                last = float(nerr_loss(pred, Os, R0b) + (LAMBDA_PHYS * phys_loss(pred, cnt, bs, cp, pr, stb)
                                                         if gname == "D" else 0.0))
            learn[gname] = (first, last)
    check(f"可學性:{LEARN_N} 個樣本訓練 {LEARN_STEPS} 步,loss 降到初始的 50% 以下" + (";QUICK 只列出" if QUICK else ""),
          QUICK or all(b < 0.5 * a for a, b in learn.values()),
          ";".join(f"{k} {a:.3f} → {b:.3f}" for k, (a, b) in learn.items()))
    print(f"\n     裝置:{dev}")


# ============================================================================
# 評估
# ============================================================================
def load_net(group, probe, seed, n_train, cfg, pr, dev):
    d = run_dir(group, probe, seed, n_train)
    if not (d / "final.pt").exists():
        return None, None, None
    meta = json.load(open(d / "result.json"))
    net = build_model(group, cfg, pr).to(dev)
    net.load_state_dict(torch.load(d / "final.pt", map_location=dev))
    return net.eval(), meta["norm"], meta


@torch.no_grad()
def time_nets(n_train, dev):
    """網路一次推論的時間(ms / 張;含輸入前處理),batch 64 與 512。時間與權重無關,用 seed 0 的結構。"""
    torch.backends.cudnn.benchmark = True
    res = {}
    for probe in PROBES:
        cfg, pr, cp, bs = setup(0, probe, dev)
        for Bl in ["64", "512"]:
            B = int(Bl) if not QUICK else {"64": 8, "512": 16}[Bl]          # QUICK(CPU)只縮小 batch 以免記憶體不足
            fields = s5.make_fields(cfg, B, seed=cfg.test_seed + 99, device=dev)
            b0, size, _ = geometry(cfg)
            cnt = measure_box(fields[:, :, b0:b0 + size, b0:b0 + size].contiguous(), probe, pr, cp, bs, cfg, seed=0)
            norm = calibrate_norm(cnt)
            for g in GROUPS:
                net = build_model(g, cfg, pr).to(dev).eval()
                net(prep(cnt, bs, norm, g, cp))                               # 暖機
                net(prep(cnt, bs, norm, g, cp))
                ts = []
                for _ in range(s5b.TIME_REPS):
                    s5._sync(dev)
                    t0 = time.perf_counter()
                    net(prep(cnt, bs, norm, g, cp))
                    s5._sync(dev)
                    ts.append((time.perf_counter() - t0) / B * 1e3)
                res.setdefault(Bl, {})[f"{g}:{probe}"] = float(np.median(ts))
    return res


@torch.no_grad()
def evaluate(n_train, dev):
    import probe_4_2b as pb
    raw5b = RUN_ROOT / ("scan5b_raw.json" if not QUICK else "scan5b_raw_quick.json")
    res5b = json.load(open(raw5b))["results"]
    print(f"  5-2b 的逐 seed 結果:{raw5b}")
    cfg0 = s5.load_cfg(0)
    probes0 = {k: v for k, v in pb.build_probes(cfg0, dev).items() if k in PROBES}
    print("  重新計時:迭代法(同 5-2b 的程式)", flush=True)
    tim = s5b.timing(cfg0, probes0, dev)
    print("  重新計時:網路", flush=True)
    tnet = time_nets(n_train, dev)
    for B in tnet:
        tim[B].update({f"net:{k}": v for k, v in tnet[B].items()})

    res = {p: {g: [] for g in GROUPS} for p in PROBES}
    missing = []
    si = list(s5.SCANS).index(SCAN)
    for s in SEEDS:
        cfg = s5.load_cfg(s)
        probes = {k: v for k, v in pb.build_probes(cfg, dev).items() if k in PROBES}
        n = s5b.N_FIELDS or cfg.eval_n
        F = s5.field_size(cfg)
        from src.physics import beamstop_mask
        bs = beamstop_mask(cfg, device=dev)
        fields = s5.make_fields(cfg, n, seed=cfg.test_seed + s5.FIELD_SEED_OFFSET, device=dev)
        O = torch.polar(fields[:, 0], fields[:, 1])
        st = s5.window_starts(cfg, SCAN)
        for probe in PROBES:
            pi_ = s5.PROBES.index(probe)
            pr = probes[probe]
            cp = s5.scan_cfg(cfg, probe, pr, s5.SCANS[SCAN][2])
            counts, _ = s5.measure_scan(fields, probe, pr, cp, bs, st,
                                        seed=cfg.test_seed + s + 20_000 + 1000 * pi_ + 100 * si)   # = 5-2 / 5-2b
            R0 = s5.footprint_mask(cfg, pr, s5.window_starts(cfg, "1x1"), dev)
            for g in GROUPS:
                net, norm, meta = load_net(g, probe, s, n_train, cfg, pr, dev)
                if net is None:
                    missing.append(f"{g}:{probe}:s{s}")
                    res[probe][g].append(None)
                    continue
                pred = predict(net, counts, bs, norm, g, cp)
                est = to_field(torch.polar(pred[:, 0], pred[:, 1]), cfg, F)
                rec = {"net": s5.field_metrics(est, O, R0), "train_nerr_ph": meta["train_nerr_ph"],
                       "train_nerr_pooled": meta["train_nerr_pooled"],
                       "history_last": meta["history"][-1], "params": meta["params"]}
                a0 = s5b.const_amp(counts[CENTER], bs, cp, pr)                # (n6) 網路當起點:R0 內用網路輸出,
                cst = torch.polar(a0[:, None, None].expand(n, F, F).contiguous(),   # 其餘用 const(同 khio 起點的做法)
                                  torch.full((n, F, F), cfg.phase_max / 2, device=dev))
                O0 = torch.where(R0, est, cst)
                for meth in s5b.METHODS:
                    outs = s5b.run_method(meth, counts, bs, cp, pr, st, s5b.ITERS, O0, cfg.test_seed + s)
                    rec[meth] = {str(it): s5.field_metrics(outs[it], O, R0) for it in s5b.ITERS}
                    del outs
                res[probe][g].append(rec)
                del net, pred, est
            print(f"  [seed {s}] {probe} 完成", flush=True)
    return res, tim, res5b, missing


# ---- 判讀 ----
def nvals(res, p, g, field="nerr_ph"):
    rs = res[p][g]
    if any(r is None for r in rs):
        return None
    return np.array([r["net"][field] for r in rs], float)


def hvals(res, p, g, meth, it):
    if it == 0:
        return nvals(res, p, g)
    return np.array([r[meth][str(it)]["nerr_ph"] for r in res[p][g]], float)


def env_time_to(res5b, p, q, tim_B):
    """包絡達到平均 nerr_ph ≤ q 的最短時間(所有 5-2b 設定);達不到回傳 None。"""
    best = None
    for conf in s5b.configs():
        for it in s5b.stops(conf):
            if s5b.vals(res5b, p, conf, it).mean() <= q:
                t = s5b.cfg_time(conf, it, tim_B, p)
                best = t if best is None else min(best, t)
    return best


def hyb_time_to(res, p, g, q, tim_B, tn, min_it=1):
    """網路起點 + 迭代法達到平均 nerr_ph ≤ q 的最短時間;預設至少跑 1 次(第 0 次 = 網路本身屬於 (n4)、(n5))。
    回傳 (時間, 方法, 次數) 或 (None, None, None)。"""
    best = (None, None, None)
    for meth in s5b.METHODS:
        for it in [it for it in [0] + s5b.ITERS if it >= min_it]:
            v = hvals(res, p, g, meth, it)
            if v is not None and v.mean() <= q:
                t = tn + it * tim_B[meth]
                if best[0] is None or t < best[0]:
                    best = (t, meth, it)
    return best


def report(res, tim, res5b, missing):
    v = {"missing": missing}
    Bs = ["64", "512"]
    print("\n" + "=" * 100)
    print("計時(ms / 張;本 job 重新量)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {t:.4f}" for k, t in tim[B].items() if not k.startswith("UNet")))
    if missing:
        print(f"  ⚠️ 缺少的網路:{', '.join(missing)}(相關比較略過)")

    for p in PROBES:
        print("\n" + "=" * 100)
        print(f"探針 {p}:R0 的 nerr_ph(3 seeds 平均;括號 = 逐 seed)/ 完整對齊 / 選翻轉 / 訓練場 nerr_ph")
        print("=" * 100)
        for g in GROUPS:
            a = nvals(res, p, g)
            if a is None:
                print(f"  {g}:缺")
                continue
            al, tw = nvals(res, p, g, "nerr_al"), nvals(res, p, g, "twin_frac")
            tr = np.array([r["train_nerr_ph"] for r in res[p][g]])
            last = np.array([r["history_last"]["sup"] for r in res[p][g]])
            pooled = np.array([r["train_nerr_pooled"] for r in res[p][g]])
            print(f"  {g}:{a.mean():.4f}({' '.join(f'{x:.4f}' for x in a)})/ {al.mean():.4f} / {tw.mean():.2f}"
                  f" / 訓練場 {tr.mean():.4f}(最後一個 epoch 的訓練 loss {last.mean():.4f})  參數 {res[p][g][0]['params']:,}")
            if pooled.mean() > 2 * last.mean():
                print(f"     ⚠️ 訓練場的 nerr(eval 模式,與訓練 loss 同一種估計 {pooled.mean():.4f})比訓練 loss 大 2 倍以上:"
                      "先檢查 BatchNorm 統計或訓練是否正常,再判讀")
        pv = {}
        # (n1)–(n3)
        for tag, g1, g2 in [("n1", "B", "A"), ("n2", "C", "B"), ("n3", "D", "B")]:
            a, b = nvals(res, p, g1), nvals(res, p, g2)
            if a is None or b is None:
                print(f"  ({tag}) {g1} vs {g2}:缺")
                continue
            r = s5b.ratio(a, b)
            pv[tag] = list(r)
            lab = {"n1": "網路能否利用重疊", "n2": "物理結構", "n3": "物理 loss"}[tag]
            print(f"  ({tag}) {lab}:{g1} vs {g2} 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {g1} {r[0]}")
        # (n4)、(n5)、(n6)
        for g in GROUPS:
            a = nvals(res, p, g)
            if a is None:
                continue
            gv = {}
            for B in Bs:
                tn = tim[B][f"net:{g}:{p}"]
                e = s5b.envelope(res5b, p, tn, tim[B])
                r = s5b.ratio(a, e[2])
                te = env_time_to(res5b, p, a.mean(), tim[B])
                sp = (te / tn) if te is not None else None
                hy = {}
                for q in Q_TARGETS:
                    th, hm, hi = hyb_time_to(res, p, g, q, tim[B], tn)
                    tq = env_time_to(res5b, p, q, tim[B])
                    hy[str(q)] = [th, tq, (th / tq) if (th is not None and tq) else None, hm, hi]
                gv[B] = {"t_net": tn, "opp": [e[0], e[1], list(e[2]), e[3]], "ratio": list(r),
                         "same_quality_time": te, "speedup": sp, "hybrid": hy}
                print(f"  (n4) {g} batch {B}:網路 {tn:.4f} ms、{a.mean():.4f} vs 對手 {s5b.fmt_conf(e[0])}({e[1]} 次)"
                      f" {e[2].mean():.4f} → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}")
                print("       (n5) 包絡要 " + (f"{te:.4f} ms 才達到網路的誤差 → 網路快 {sp:.1f} 倍" if te is not None
                                              else "在所有設定內都達不到網路的誤差"))
                def _lab(h):
                    if h[1] == 0:
                        return "包絡在時間 0 就達到(不適用)"
                    if h[2] is not None:
                        return f"{h[0]:.4f} vs {h[1]:.4f} ms(× {h[2]:.2f};{h[3]} {h[4]} 次)"
                    if h[0] is not None and h[1] is None:
                        return f"起點達到({h[3]} {h[4]} 次)、包絡達不到"
                    return "皆達不到" if h[0] is None and h[1] is None else "起點達不到"
                print("       (n6) 網路起點 + 迭代(至少 1 次)vs 包絡,達到 q 的時間:"
                      + ";".join(f"q={q}:" + _lab(h) for q, h in hy.items()))
            useful = gv["64"]["ratio"][0] == "較準"
            robust = useful and gv["512"]["ratio"][0] == "較準"
            start_ok = any(all((gv[B]["hybrid"][str(q)][2] is not None and gv[B]["hybrid"][str(q)][2] <= 0.8)
                               or (gv[B]["hybrid"][str(q)][0] is not None and gv[B]["hybrid"][str(q)][1] is None)
                               for B in Bs) for q in Q_TARGETS)
            gv.update({"useful": useful, "robust": robust, "start_useful": start_ok})
            print(f"  ▶ {g}:{'穩健有用' if robust else ('有用(只在 batch 64)' if useful else '未達有用')};"
                  f"起點{'有用' if start_ok else '未達有用'}(兩種 batch 下某個 q 縮短 ≥ 20%)")
            pv[g] = gv
        v[p] = pv
    # (n7) 探針
    print("\n" + "=" * 100)
    print("(n7) 探針:同一組的 DEF2 vs DISK(比值 < 1 = DEF2 較準)")
    for g in GROUPS:
        a, b = nvals(res, "DEF2", g), nvals(res, "DISK", g)
        if a is not None and b is not None:
            r = s5b.ratio(a, b)
            v.setdefault("n7", {})[g] = list(r)
            print(f"  {g}:{r[1]:.3f}(z {r[2]:+.1f})→ DEF2 {r[0]}")
    # 停損(§12.8)
    for p in PROBES:
        pv = v.get(p, {})
        if any(nvals(res, p, g) is None for g in GROUPS):
            print(f"  ⚠️ {p}:有組別缺少網路 → 停損(§12.8)先不判斷,補齊後再評估")
            continue
        if not any(pv.get(g, {}).get("useful") for g in GROUPS if g in pv):
            cand = [(nvals(res, p, g).mean(), g) for g in ["B", "C", "D"] if nvals(res, p, g) is not None]
            if cand:
                best = min(cand)[1]
                print(f"  ⚠️ {p}:四組都未達「有用」→ 依 §12.8,對 {best} 組(B/C/D 中誤差最低)做一次資料量 × 5 的重訓")
                v.setdefault("stoploss", {})[p] = best
    print("判讀準則見階段五協定 §12.5、§12.8(結果出來前已寫定)")
    return v


def make_figures(res, tim, res5b):
    plt = s5._plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    col = {"A": "#9a9994", "B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a"}
    for B in ["64", "512"]:
        fig, axs = plt.subplots(1, len(PROBES), figsize=(6.0 * len(PROBES), 4.4), sharey=True)
        for ax, p in zip(np.atleast_1d(axs), PROBES):
            tt = sorted({s5b.cfg_time(c, it, tim[B], p) for c in s5b.configs() for it in s5b.stops(c)} - {0.0})
            ax.plot(tt, [s5b.envelope(res5b, p, T, tim[B])[2].mean() for T in tt], color="#0b0b0b", lw=2,
                    label="iterative envelope (5-2b)")
            for g in GROUPS:
                a = nvals(res, p, g)
                if a is None:
                    continue
                tn = tim[B][f"net:{g}:{p}"]
                ax.plot([tn], [a.mean()], "o", color=col[g], ms=7, label=f"net {g}")
                # 網路起點 + 最佳迭代法的下緣
                pts = sorted((tn + it * tim[B][m], hvals(res, p, g, m, it).mean())
                             for m in s5b.METHODS for it in s5b.ITERS)
                xs, ys, cur = [], [], float("inf")
                for t, y in pts:
                    cur = min(cur, y)
                    xs.append(t)
                    ys.append(cur)
                ax.plot(xs, ys, color=col[g], lw=1, ls="--", alpha=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{p} (3x3s8, batch {B})", fontsize=10)
            ax.set_xlabel("time per sample (ms)")
            ax.grid(True, color="#e4e3df", which="both", lw=0.5)
        np.atleast_1d(axs)[0].set_ylabel("nerr on R0 (global phase only)")
        np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7)
        fig.suptitle("Stage 5-3: networks (dots; dashed = net start + best iterative) vs iterative envelope", fontsize=10)
        fig.tight_layout()
        path = FIG_DIR / f"scan5c_vs_envelope_b{B}.png"
        fig.savefig(path, dpi=130, facecolor="white")
        plt.close(fig)
        print(f"  圖:{path}")


# ============================================================================
def check_files(need_5b=False):
    need = [Path("scan_5.py"), Path("scan_5b.py"), Path("probe_4_2b.py"), Path("probe_4_1.py"),
            Path("ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    if need_5b:
        need.append(RUN_ROOT / ("scan5b_raw.json" if not QUICK else "scan5b_raw_quick.json"))
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="全部單元測試")
    ap.add_argument("--train", action="store_true", help="訓練一個網路(配合 --task)")
    ap.add_argument("--task", type=int, default=None, help="0–23:組 × 探針 × seed(見 task_of)")
    ap.add_argument("--eval", action="store_true", help="評估全部網路")
    ap.add_argument("--n-train", type=int, default=None, help="訓練場數(預設 20000;停損重訓用 100000)")
    a = ap.parse_args()
    n_train = a.n_train or N_TRAIN
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    if not check_files(need_5b=a.eval):
        print("\n❌ 缺檔案,停下來")
        sys.exit(1)
    if a.check:
        unit_tests(dev)
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過"))
        sys.exit(0 if ok_all else 1)
    if a.train:
        if a.task is None:
            raise SystemExit("--train 需要 --task")
        g, p, s = task_of(a.task)
        print(f"[task {a.task}] 組 {g}、探針 {p}、seed {s}、訓練場 {n_train}", flush=True)
        unit_tests(dev, groups=[g])
        if not ok_all:
            print("\n❌ 單元測試未通過,不訓練")
            sys.exit(1)
        train(g, p, s, n_train, dev)
        print("✅ 訓練完成")
        return
    if a.eval:
        res, tim, res5b, missing = evaluate(n_train, dev)
        raw = RUN_ROOT / (f"scan5c_raw_n{n_train}.json" if not QUICK else "scan5c_raw_quick.json")
        json.dump({"results": res, "timing": tim, "missing": missing, "quick": QUICK}, open(raw, "w"))
        print(f"  原始結果先存檔:{raw}")
        verdict = report(res, tim, res5b, missing)
        json.dump({"results": res, "timing": tim, "verdict": verdict, "n_train": n_train, "quick": QUICK},
                  open(RUN_ROOT / (f"scan5c_n{n_train}.json" if not QUICK else "scan5c_quick.json"), "w"))
        make_figures(res, tim, res5b)
        print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))
        print("把完整輸出貼給 Claude")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
