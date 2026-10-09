#!/usr/bin/env python
"""階段四 4-2a:展開 HIO(探針版)的輪數與修正網路 —— 訓練前以計時決定(協定 §3.5)。

規則(結果出來前寫定;只計時,不看任何重建品質):
  在 batch 64、同一 GPU 上計時;總時間不超過「P-HIO(探針 support)100 次」的前提下,
  選輪數 T 最多的組合(T ≤ 50);同 T 時選修正網路較大者(unet > c16 > c8),再選每輪修正者。
  若沒有組合能達到 T ≥ 10,改以 P-HIO 200 次為預算,並在結果中註明。

候選:
  修正網路  unet(原版小 U-Net,基底 12)/ c16(4 層卷積 16 通道)/ c8(3 層卷積 8 通道)
  修正頻率  每輪 / 每 2 輪(中間只跑物理步驟)

輸出:
  configs/probe_unroll.yaml          = probe_unet.yaml + arch / unroll_* 欄位(若已存在且不同則停)
  /work/elviss0915/runs/probe42_select.json

用法(需在計算節點執行,且須在 GPU 上 —— 計時才有意義):
    python select_unroll.py
"""
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
OUT_JSON = RUN_ROOT / "probe42_select.json"
BASE_CFG = Path("configs/probe_unet.yaml")
OUT_CFG = Path("configs/probe_unroll.yaml")
BATCH = 64
REPS = 5
T_MAX = 50
T_MIN = 10
KINDS = ["unet", "c16", "c8"]                 # 由大到小
EVERY = [1, 2]
T_GRID = [5, 10, 20, 30, 40, 50]
QUICK = os.environ.get("CDI_QUICK") == "1"    # 只供本地測程式流程
if QUICK:
    BATCH, REPS, T_GRID = 8, 1, [4, 8]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def time_call(fn, dev, reps=REPS):
    """單次呼叫的中位數秒數;單次太短時在內圈重複,使每次量測 ≥ 20 ms。"""
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
def main():
    from src.hio import hio, random_init
    from src.model import build_model
    from src.physics import (apply_probe, beamstop_mask, build_input, calibrate_flux,
                             calibrate_input_norm, forward_measure)
    from src.procedural import make_procedural

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:計時必須在計算節點的 GPU 上進行")
    cfg = Cfg.load(BASE_CFG)
    assert cfg.probe == "disk", "probe_unet.yaml 應為探針設定"

    # 以一小批 psi 校正(只為讓模型能建立與計時;數值不影響計時)
    raw = make_procedural(512, cfg, seed=4242, kinds=tuple(cfg.proc_kinds),
                          weights=tuple(cfg.proc_weights),
                          target_support=cfg.match_support_to,
                          contrast_gamma=cfg.proc_contrast_gamma)
    psi = apply_probe(raw, cfg).to(dev)
    cfg.ref_energy = calibrate_flux(psi.cpu())
    bs = beamstop_mask(cfg, device=dev)
    calibrate_input_norm(psi, bs, cfg)
    torch.manual_seed(0)
    counts = forward_measure(psi[:BATCH], bs, cfg)
    x = build_input(psi[:BATCH], counts, bs, cfg)
    init = random_init(counts, cfg, seed=0, device=dev)

    def phio(n):
        return 1e3 * time_call(lambda: hio(init, counts, bs, cfg, n_iter=n, beta=cfg.hio_beta),
                               dev) / BATCH

    def net_ms(kind, every, T):
        d = {**cfg.to_dict(), "arch": "unroll", "unroll_t": T,
             "unroll_refine": kind, "unroll_every": every}
        m = build_model(Cfg.from_dict(d)).to(dev).eval()
        return 1e3 * time_call(lambda: m(x), dev) / BATCH

    budgets = {100: phio(100), 200: phio(200)}
    print(f"裝置 {dev};batch {BATCH}")
    print(f"預算:P-HIO 100 次 {budgets[100]:.4f} ms/sample;200 次 {budgets[200]:.4f} ms/sample")

    table = {}
    for kind in KINDS:
        for every in EVERY:
            ts = {T: net_ms(kind, every, T) for T in T_GRID if T % every == 0}
            table[f"{kind}/{every}"] = ts
            print(f"  {kind:>4} 每 {every} 輪修正:" +
                  "  ".join(f"T{T} {v:.4f}" for T, v in ts.items()))

    def best_under(budget):
        cands = []
        for kind in KINDS:
            for every in EVERY:
                ts = table[f"{kind}/{every}"]
                Ts = np.array(sorted(ts), float)
                vs = np.array([ts[int(T)] for T in Ts])
                b, a = np.polyfit(Ts, vs, 1)             # 時間 ≈ a + b·T
                t_est = int(min(T_MAX, math.floor((budget - a) / b))) if b > 0 else T_MAX
                t_est -= t_est % every
                # 直接量測驗證,不通過就往下減
                while t_est >= every:
                    if net_ms(kind, every, t_est) <= budget:
                        break
                    t_est -= every
                if t_est >= every:
                    cands.append((t_est, KINDS[::-1].index(kind), -every, kind, every))
        return max(cands) if cands else None

    used = 100
    pick = best_under(budgets[100])
    if pick is None or pick[0] < T_MIN:
        used = 200
        pick = best_under(budgets[200])
    if pick is None:
        raise SystemExit("❌ 沒有任何組合能在預算內,停下來查")
    T, _, _, kind, every = pick
    ms = net_ms(kind, every, T)
    note = "" if used == 100 else "(依規則放寬為 P-HIO 200 次;需在結果中註明)"
    print(f"\n選定:unroll_refine = {kind}、unroll_every = {every}、unroll_t = {T}"
          f"(預算 P-HIO {used} 次 = {budgets[used]:.4f} ms;本組 {ms:.4f} ms){note}")

    # 產生 configs/probe_unroll.yaml(= probe_unet.yaml + 四欄)
    text = open(BASE_CFG, encoding="utf-8").read()
    head = ("# 階段四 4-2a:物理內嵌 U-Net(展開 HIO,探針 support)—— 由 select_unroll.py 依協定 §3.5 產生\n"
            "#\n# 與 probe_unet.yaml **只差 arch 與 unroll_* 欄位**(check_probe42.py 以程式驗證)。\n"
            f"# 計時(batch {BATCH}):本組 {ms:.4f} ms/sample;預算 P-HIO {used} 次 {budgets[used]:.4f} ms/sample\n")
    body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    new = (head + body.rstrip() + f"\narch: unroll\nunroll_t: {T}\nunroll_refine: {kind}\n"
           f"unroll_every: {every}\n")
    if OUT_CFG.exists():
        old = open(OUT_CFG, encoding="utf-8").read()
        strip = lambda s: [l for l in s.splitlines() if not l.startswith("#")]   # noqa: E731
        if strip(old) != strip(new):
            raise SystemExit(f"❌ {OUT_CFG} 已存在且設定不同 —— 不覆蓋。把本輸出貼給 Claude。")
        print(f"  {OUT_CFG} 已存在且設定相同,不重寫")
    else:
        open(OUT_CFG, "w", encoding="utf-8").write(new)
        print(f"  已寫入 {OUT_CFG}")

    json.dump({"batch": BATCH, "budget_iters": used, "budgets_ms": budgets,
               "table_ms": {k: {str(t): v for t, v in d.items()} for k, d in table.items()},
               "pick": {"unroll_t": T, "unroll_refine": kind, "unroll_every": every, "ms": ms},
               "device": str(dev), "quick": QUICK},
              open(OUT_JSON if not QUICK else RUN_ROOT / "probe42_select_quick.json", "w"),
              indent=2)
    print(f"  已寫入 {OUT_JSON if not QUICK else RUN_ROOT / 'probe42_select_quick.json'}")


if __name__ == "__main__":
    main()
