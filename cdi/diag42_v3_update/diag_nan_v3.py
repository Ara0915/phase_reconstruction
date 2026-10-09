#!/usr/bin/env python
"""展開版 v3 關卡診斷:訓練中約 15% 的步出現 nan / inf(協定 §8.11)。

只做前向與反向(不更新權重、不改任何既有檔案)。用 seed 0 訓練完的 probe_v3_unroll_s0/final.pt,
以與 train.py 相同的方式產生訓練批次(build_pool → forward_measure → build_input → forward_segments → 各段 loss 平均),
逐批檢查:
  1. 前向是否有限:逐步重跑展開過程(起點、每輪的傅立葉約束 / HIO 更新 / 修正、每段讀出),找出第一個非有限的位置
  2. 各段 loss 是否有限
  3. 反向梯度是否有限;若不有限,以 autograd 異常偵測找出產生 nan 的運算與它在前向中的程式位置
  4. 找出壞批次中是哪幾個樣本造成的
  5. 比較 check_probe42_v3 的 fp32 結果:同一批改成 float64 是否仍壞(區分「數值精度」與「真正的奇異點」)

用法(計算節點,GPU):
    python diag_nan_v3.py            # 預設 40 批(batch 128)
輸出:/work/elviss0915/runs/diag_nan_v3.json
"""
import json
import os
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN_ROOT = Path(os.environ.get("CDI_RUN_ROOT", "/work/elviss0915/runs"))
NAME = "probe_v3_unroll_s0"
QUICK = os.environ.get("CDI_QUICK") == "1"
N_BATCH = 4 if QUICK else 40
BS = 8 if QUICK else 128
N_ANOMALY = 3                      # 最多對幾個壞批次跑異常偵測(慢)


def finite(t):
    return bool(torch.isfinite(t).all()) if not torch.is_complex(t) else \
        bool(torch.isfinite(t.real).all() and torch.isfinite(t.imag).all())


@torch.no_grad()
def locate_forward(m, x):
    """逐步重跑 forward_segments,回傳第一個非有限的位置(字串),全部有限則回傳 None。"""
    bsm = x[:, -1]
    A = m.measured_amp(x)
    if not finite(A):
        return "量測振幅 A"
    h0 = m.det(x)
    if not finite(h0):
        return "偵測器端網路 det"
    g = m.field(x, h0[:, 0:1], h0[:, 1:2], torch.nn.functional.softplus(h0[:, 2:3]))[:, 0]
    if not finite(g):
        return "起點 field()"
    g = g + m._fix(m.refine[0](m._c2r(g)))
    if not finite(g):
        return "起點修正 refine[0]"
    k = 0
    for t in range(m.t):
        g_new = m.fourier_step(g, A, bsm)
        if not finite(g_new):
            return f"第 {t + 1} 輪 傅立葉約束"
        h = m.hio_step(g, g_new)
        if not finite(h):
            return f"第 {t + 1} 輪 HIO 更新"
        if (t + 1) % m.every == 0:
            k += 1
            feat = torch.cat([m._c2r(g_new), m._c2r(g), m._c2r(h)], dim=1)
            g = h + m._fix(m.refine[k](feat))
            if not finite(g):
                return f"第 {t + 1} 輪 修正 refine[{k}]"
        else:
            g = h
        if m.trunc and (t + 1) % m.trunc == 0:
            if not finite(m.readout(g)):
                return f"第 {t + 1} 輪 讀出"
    return None


def step(m, x, obj, counts, bs, loss_fn):
    """與 train.py 相同的一步(不更新權重)。回傳 (各段 loss 是否有限, 梯度是否有限, 總 loss)。"""
    m.zero_grad(set_to_none=True)
    outs = m.forward_segments(x)
    res = [loss_fn(o, obj, counts, bs)[0] for o in outs]
    seg_ok = [bool(torch.isfinite(r)) for r in res]
    tot = sum(res) / len(res)
    tot.backward()
    g_ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    return seg_ok, g_ok, float(tot.detach())


def anomaly(m, x, obj, counts, bs, loss_fn):
    """以異常偵測重跑同一步,回傳 (出錯的運算, 前向程式位置)。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                step(m, x, obj, counts, bs, loss_fn)
            return "(異常偵測下未重現)", ""
        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            fwd = ""
            for wi in w:
                s = str(wi.message)
                if "Traceback of forward call" in s or "forward call" in s:
                    lines = [l.strip() for l in s.splitlines() if "model.py" in l or "losses.py" in l
                             or "physics.py" in l]
                    code = [l.strip() for l in s.splitlines()]
                    # 取最後一個本專案檔案的位置與下一行程式碼
                    pos = [i for i, l in enumerate(code) if "/src/" in l]
                    if pos:
                        i = pos[-1]
                        fwd = code[i] + (" | " + code[i + 1] if i + 1 < len(code) else "")
                    elif lines:
                        fwd = lines[-1]
                    else:
                        files = [i for i, l in enumerate(code) if l.startswith("File ")]
                        if files:
                            i = files[-1]
                            fwd = code[i] + (" | " + code[i + 1] if i + 1 < len(code) else "")
            return msg, fwd


def main():
    from realign_eval import load
    from src.data import build_pool
    from src.losses import build_loss
    from src.physics import beamstop_mask, build_input, forward_measure

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    rd = RUN_ROOT / NAME
    cfg, m = load(rd, dev)
    assert cfg.unroll_readout == "natural" and cfg.unroll_trunc > 0
    m.train()
    loss_fn = build_loss(cfg, dev)
    bs = beamstop_mask(cfg, device=dev)
    pool = build_pool(cfg, N_BATCH * BS, seed=0).to(dev)
    print(f"[{rd.name}] {N_BATCH} 批 × {BS} 張,裝置 {dev}", flush=True)

    rows, ops, where_fwd, bad_idx = [], Counter(), Counter(), []
    for b in range(N_BATCH):
        obj = pool[b * BS:(b + 1) * BS]
        torch.manual_seed(10_000 + b)
        with torch.no_grad():
            counts = forward_measure(obj, bs, cfg)
            x = build_input(obj, counts, bs, cfg)
        loc = locate_forward(m, x)
        seg_ok, g_ok, tot = step(m, x, obj, counts, bs, loss_fn)
        bad = (loc is not None) or (not all(seg_ok)) or (not g_ok)
        row = {"batch": b, "forward_bad_at": loc, "seg_loss_finite": seg_ok, "grad_finite": g_ok,
               "loss": tot, "bad": bad}
        if bad:
            bad_idx.append(b)
            if loc:
                where_fwd[loc] += 1
            if len(bad_idx) <= N_ANOMALY:
                op, fwd = anomaly(m, x, obj, counts, bs, loss_fn)
                row["anomaly_op"], row["anomaly_forward"] = op, fwd
                ops[op] += 1
                # 哪些樣本造成的:逐張重跑
                culprits = []
                for i in range(BS):
                    so, go, _ = step(m, x[i:i + 1], obj[i:i + 1], counts[i:i + 1], bs, loss_fn)
                    if not (all(so) and go):
                        culprits.append(i)
                row["culprit_samples"] = culprits
                # float64 是否仍壞
                m64 = m.double()
                try:
                    so, go, _ = step(m64, x.double(), obj.double(), counts.double(), bs.double(), loss_fn)
                    row["float64_ok"] = bool(all(so) and go)
                except Exception as e:                    # noqa: BLE001
                    row["float64_ok"] = f"無法以 float64 執行:{type(e).__name__}"
                m.float()
        rows.append(row)
        print(f"  批 {b:02d}:" + ("❌" if bad else "✅")
              + (f" 前向在「{loc}」出現非有限" if loc else "")
              + ("" if all(seg_ok) else f";非有限的段 {[i + 1 for i, o in enumerate(seg_ok) if not o]}")
              + ("" if g_ok else ";梯度非有限")
              + (f"\n      異常偵測:{row['anomaly_op']}\n      前向位置:{row['anomaly_forward']}"
                 f"\n      造成的樣本 {len(row['culprit_samples'])} 張:{row['culprit_samples'][:10]}"
                 f"\n      float64 下:{'正常' if row['float64_ok'] is True else row['float64_ok'] if isinstance(row['float64_ok'], str) else '仍非有限'}"
                 if "anomaly_op" in row else ""), flush=True)

    n_bad = len(bad_idx)
    print("\n" + "=" * 80)
    print(f"壞批次 {n_bad}/{N_BATCH}({100 * n_bad / N_BATCH:.0f}%;訓練紀錄中每 epoch 約 20–25 / 156 步 = 13–16%)")
    print(f"前向出現非有限:{dict(where_fwd) if where_fwd else '無(前向全部有限)'}")
    print(f"異常偵測找到的運算:{dict(ops) if ops else '—'}")
    json.dump({"rows": rows, "n_bad": n_bad, "n_batch": N_BATCH, "quick": QUICK},
              open(RUN_ROOT / ("diag_nan_v3_quick.json" if QUICK else "diag_nan_v3.json"), "w"),
              indent=2, ensure_ascii=False)
    print("把完整輸出貼給 Claude")


if __name__ == "__main__":
    main()
