#!/usr/bin/env python
"""訓練進入點。

用法:
    python train.py --config configs/bs03.yaml --seed 0
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.config import Cfg
from src.data import PoolSampler, build_pool, count_unique
from src.losses import build_loss
from src.model import build_model
from src.physics import beamstop_mask, build_input, calibrate_flux, \
    calibrate_input_norm, forward_measure


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 實測 seed 波動僅 ±0.11 dB,通常不需要完全決定性;
    # benchmark=True 明顯較快,在付費叢集上划算。
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="/work/elviss0915/runs")
    ap.add_argument("--tag", default=None, help="覆寫輸出資料夾名稱")
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    name = args.tag or Path(args.config).stem
    out_dir = Path(args.out_root) / f"{name}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} config={args.config} seed={args.seed}", flush=True)

    # ---- 資料 ----
    t_pool = time.time()
    pool = build_pool(cfg, cfg.subset_n, seed=args.seed)
    sampler = PoolSampler(cfg, pool, device)
    # 記錄**實際**相異數:程序生成的隨機參數偶爾碰撞,
    # 名目 n_unique 與實際值可能略有出入(實測 20000 -> 19980)。
    n_uni = count_unique(pool)
    print(f"[init] sources={list(cfg.train_sources)} pool={sampler.n} "
          f"unique={n_uni}(名目 {cfg.n_unique}) "
          f"({time.time()-t_pool:.1f}s)", flush=True)

    # ---- 校正:一次算好,寫進 config_used.json ----
    # eval.py 必須讀回同一組值,否則 R-factor 與輸入表示法都與訓練時不可比。
    cal = pool[:2000].to(device)
    if cfg.ref_energy is None:
        cfg.ref_energy = calibrate_flux(cal.cpu())
    print(f"[init] ref_energy={cfg.ref_energy:.6f}", flush=True)

    bs_cal = beamstop_mask(cfg, device=device)
    if cfg.input_norm == "global" and cfg.log_mean is None:
        calibrate_input_norm(cal, bs_cal, cfg)
        print(f"[init] input_norm=global log_mean={cfg.log_mean:.4f} "
              f"log_std={cfg.log_std:.4f} ac_scale={cfg.ac_scale:.4g}", flush=True)

    json.dump({**cfg.to_dict(), "seed": args.seed, "config_path": args.config,
               "n_unique_actual": n_uni},
              open(out_dir / "config_used.json", "w"), indent=2)

    bs_mask = beamstop_mask(cfg, device=device)
    blocked = 1 - float(bs_mask.mean())
    print(f"[init] beamstop_r={cfg.beamstop_r} blocked={blocked*100:.2f}%"
          f"  randomize: photons={cfg.randomize_photons} "
          f"beamstop={cfg.randomize_beamstop}", flush=True)

    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_par:,} oracle_phase={cfg.oracle_phase}", flush=True)

    loss_fn = build_loss(cfg, device)
    # 分段反傳(協定 §8.5):只用於 arch = unroll;0 = 不分段(原行為)
    seg = int(getattr(cfg, "unroll_trunc", 0) or 0)
    if seg and getattr(cfg, "arch", "unet") != "unroll":
        raise ValueError("unroll_trunc 只適用於 arch = unroll")
    if seg:
        print(f"[init] 分段反傳:每 {seg} 輪切斷梯度,{cfg.unroll_t // seg} 段 loss 取平均", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    steps = sampler.n // cfg.batch_size
    # total_steps 留一點餘裕:OneCycleLR 在 __init__ 會先 step 一次,
    # 續跑時還原 state_dict 會有 off-by-one,剛好踩到上限就會拋
    # "Tried to step N+1 times"。多留幾步對 LR 曲線的影響可以忽略。
    total_steps = cfg.epochs * steps + 8
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total_steps)

    # ---- 續跑 ----
    ckpt_path = out_dir / "ckpt.pt"
    start_epoch = 0
    history = []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        # OneCycle 的學習率是跟著步數走完整週期的,
        # 不還原 scheduler 的話 LR 會從頭開始爬,訓練曲線就毀了。
        sched.load_state_dict(ck["sched"])
        # 還原亂數狀態,讓續跑跟未中斷的執行完全等價
        # (Poisson 取樣與 batch 順序都吃 RNG)
        torch.set_rng_state(ck["rng_cpu"])
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        np.random.set_state(ck["rng_np"])
        random.setstate(ck["rng_py"])
        start_epoch = ck["epoch"] + 1
        history = ck.get("history", [])
        print(f"[resume] 從 epoch {start_epoch} 續跑", flush=True)

    # ---- 訓練 ----
    t0 = time.time()
    for ep in range(start_epoch, cfg.epochs):
        model.train()
        acc = dict(total=0.0, amp=0.0, amp_in=0.0, amp_out=0.0,
                   phase=0.0, dc=0.0, freq=0.0, amb_twin=0.0, amb_shift=0.0)
        n_ok, n_skip, gnorms = 0, 0, []
        seg_acc = [0.0] * (cfg.unroll_t // seg) if seg else None
        te = time.time()
        for idx in sampler.epoch_indices(cfg.batch_size):
            obj = sampler.batch(idx)
            # 每個 batch 重抽物理參數。固定值會被網路當常數背下來;
            # 評估時一律使用 cfg 的固定值,以確保跨設定可比較。
            ph = cfg.photons_per_pix
            bm = bs_mask
            if cfg.randomize_photons is not None:
                lo, hi = cfg.randomize_photons
                u = float(torch.rand(1))
                ph = float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
            if cfg.randomize_beamstop is not None:
                lo, hi = cfg.randomize_beamstop
                r = float(lo + torch.rand(1) * (hi - lo))
                bm = beamstop_mask(cfg, radius=r, device=device)
            with torch.no_grad():
                counts = forward_measure(obj, bm, cfg, photons=ph)
                x = build_input(obj, counts, bm, cfg)
            opt.zero_grad(set_to_none=True)
            if seg:
                # 分段反傳(協定 §8.5):訓練目標 = 各段 loss 的平均;
                # 記錄的 loss 與各項(amp、phase…)取最後一段 = 實際輸出,與其他組可比
                outs = model.forward_segments(x)
                res = [loss_fn(o, obj, counts, bm, photons=ph) for o in outs]
                obj_ds = sum(r[0] for r in res) / len(res)
                loss, parts = res[-1]
                seg_l = [float(r[0].detach()) for r in res]
                obj_ds.backward()
            else:
                loss, parts = loss_fn(model(x), obj, counts, bm, photons=ph)
                loss.backward()
            if cfg.grad_clip is not None:
                # 裁切前的總梯度範數;非有限(loss 或梯度)則跳過本 step,學習率排程照常前進
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not (torch.isfinite(loss) and torch.isfinite(gn)):
                    opt.zero_grad(set_to_none=True)
                    n_skip += 1
                    if sched.last_epoch + 1 < total_steps:
                        sched.step()
                    continue
                gnorms.append(float(gn))
            opt.step()
            if sched.last_epoch + 1 < total_steps:
                sched.step()
            n_ok += 1
            if seg:
                for i, v in enumerate(seg_l):
                    seg_acc[i] += v
            acc["total"] += loss.item()
            for k in ("amp", "amp_in", "amp_out", "phase", "dc", "freq",
                      "amb_twin", "amb_shift"):
                acc[k] += parts[k]
        for k in acc:
            acc[k] /= max(n_ok, 1) if cfg.grad_clip is not None else steps
        if seg:
            acc["seg_loss"] = [v / max(n_ok, 1) for v in seg_acc]      # 各段(第 K、2K、…、T 輪)的平均 loss
            acc["ds_obj"] = float(np.mean(acc["seg_loss"]))            # 實際的訓練目標
        if cfg.grad_clip is not None:
            acc["skipped"] = n_skip
            acc["grad_norm_med"] = float(np.median(gnorms)) if gnorms else float("nan")
            acc["grad_norm_max"] = float(np.max(gnorms)) if gnorms else float("nan")
            acc["clip_frac"] = float(np.mean([g > cfg.grad_clip for g in gnorms])) if gnorms else float("nan")
        acc["epoch"] = ep
        acc["sec"] = time.time() - te
        acc["lr"] = sched.get_last_lr()[0]
        history.append(acc)
        print(f"epoch {ep:03d}  total={acc['total']:.6f}  amp={acc['amp']:.6f}  "
              f"phase={acc['phase']:.6f}  dc={acc['dc']:.6f}  "
              f"lr={acc['lr']:.2e}  {acc['sec']:.1f}s"
              + (f"  amb:翻轉 {acc['amb_twin']:.2f} 平移 {acc['amb_shift']:.2f}"
                 if cfg.amb_invariant else "")
              + (f"  梯度:中位 {acc['grad_norm_med']:.3g} 最大 {acc['grad_norm_max']:.3g} "
                 f"裁切 {100 * acc['clip_frac']:.0f}% 跳過 {acc['skipped']}"
                 if cfg.grad_clip is not None else "")
              + ("  各段 " + " ".join(f"{v:.3f}" for v in acc["seg_loss"]) if seg else ""),
              flush=True)

        # 每個 epoch 都存 —— 工作被時限砍掉是常態,不是意外
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "history": history,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "rng_np": np.random.get_state(),
                    "rng_py": random.getstate()}, ckpt_path)

    torch.save(model.state_dict(), out_dir / "final.pt")
    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    print(f"[done] {time.time()-t0:.1f}s  -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
