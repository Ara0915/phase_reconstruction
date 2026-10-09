#!/usr/bin/env python
"""展開版 v3 nan 診斷第三步(協定 §8.11):檢驗「空樣本」假說。

第二步發現兩件事:
  (1) build_pool 產生的物體會隨總數 n 改變 —— 第二步用的訓練池與第一步不同,所以沒有重現到第一步的壞批次;
  (2) 唯一找到的樣本是「空樣本」:物體完全落在探針圓盤外,出射波全為 0、光子數全為 0。
      單獨一張時,資料一致項的分母(整批光子數開根號的總和)= 0,只剩 1e-12 的下限 → 放大 1e12 倍,
      所以「單獨跑一張」的結果可能是人為的,不能直接當成原因。
另外:訓練池中空樣本約 0.15%,一批 128 張含至少一張空樣本的機率約 18%,與壞批次比例(13–16%)相近。

本腳本用與第一步**完全相同**的訓練池與量測(40 批 × 128),逐批記錄:
  - 是否壞(梯度非有限)、是否含空樣本
  - 壞批次:整批(不拆開)做邊界分析,定位來源運算
  - 壞且含空樣本的批次:拿掉空樣本後是否恢復正常
  - 空樣本在各段的預測振幅最大值、預測繞射強度的最小正值
判讀(結果出來前寫定):壞批次 ⊆ 含空樣本的批次,且拿掉空樣本即恢復 → 空樣本是原因。

用法(計算節點,GPU):
    python diag_nan_v3c.py
輸出:/work/elviss0915/runs/diag_nan_v3c.json
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_nan_v3b import fin, traced                                  # noqa: E402

RUN_ROOT = Path(os.environ.get("CDI_RUN_ROOT", "/work/elviss0915/runs"))
NAME = "probe_v3_unroll_s0"
QUICK = os.environ.get("CDI_QUICK") == "1"
N_BATCH = 4 if QUICK else 40          # 必須與 diag_nan_v3.py 相同(訓練池隨總數改變)
BS = 8 if QUICK else 128


def grads_ok(m):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())


def step(m, x, obj, counts, bs, loss_fn):
    m.zero_grad(set_to_none=True)
    outs = m.forward_segments(x)
    res = [loss_fn(o, obj, counts, bs)[0] for o in outs]
    (sum(res) / len(res)).backward()
    return all(bool(torch.isfinite(r)) for r in res), grads_ok(m)


def main():
    from realign_eval import load
    from src.data import build_pool
    from src.losses import build_loss
    from src.physics import beamstop_mask, build_input, forward_measure, probe_amplitude

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    cfg, m = load(RUN_ROOT / NAME, dev)
    m.train()
    loss_fn = build_loss(cfg, dev)
    bs = beamstop_mask(cfg, device=dev)
    disk = probe_amplitude(cfg, device=dev) > 0
    pool = build_pool(cfg, N_BATCH * BS, seed=0).to(dev)              # 與 diag_nan_v3.py 相同
    empty_all = ((pool[:, 0] * disk).flatten(1).sum(1) == 0)
    print(f"訓練池 {len(pool)} 張中空樣本 {int(empty_all.sum())} 張({100 * float(empty_all.float().mean()):.2f}%)",
          flush=True)

    rows = []
    for b in range(N_BATCH):
        obj = pool[b * BS:(b + 1) * BS]
        torch.manual_seed(10_000 + b)                                  # 與 diag_nan_v3.py 相同
        with torch.no_grad():
            counts = forward_measure(obj, bs, cfg)
            x = build_input(obj, counts, bs, cfg)
        empt = [int(i) for i in torch.nonzero(empty_all[b * BS:(b + 1) * BS]).flatten()]
        loss_ok, g_ok = step(m, x, obj, counts, bs, loss_fn)
        bad = not (loss_ok and g_ok)
        row = {"batch": b, "bad": bad, "empty_samples": empt}
        if bad:
            # 整批邊界分析
            m.zero_grad(set_to_none=True)
            outs, ts = traced(m, x)
            res = [loss_fn(o, obj, counts, bs)[0] for o in outs]
            (sum(res) / len(res)).backward()
            gf = [(n, fin(t.grad)) for n, t in ts]
            badt = [j for j, (_, f) in enumerate(gf) if f is False]
            if badt:
                j = max(badt)
                nxt = "該段的 loss(來源在 loss 的反傳)" if "段輸出" in gf[j][0] else \
                    next((gf[k][0] for k in range(j + 1, len(gf)) if gf[k][1] is not None), "loss")
                row["boundary"] = f"{gf[j][0]} → {nxt}"
                # nan 在批次中的哪幾張(依最下游 nan 張量的梯度)
                gt = ts[j][1].grad
                per = gt.reshape(gt.shape[0], -1)
                per_bad = ~(torch.isfinite(per.real if torch.is_complex(per) else per).all(1))
                row["nan_rows"] = [int(i) for i in torch.nonzero(per_bad).flatten()]
            else:
                row["boundary"] = "中間張量皆有限(nan 只在參數梯度)"
            # 拿掉空樣本後
            if empt:
                keep = torch.tensor([i for i in range(BS) if i not in empt], device=dev)
                lo, go = step(m, x[keep], obj[keep], counts[keep], bs, loss_fn)
                row["ok_without_empty"] = bool(lo and go)
        # 空樣本在各段的預測
        if empt:
            with torch.no_grad():
                outs = m.forward_segments(x[empt])
                row["empty_pred_amp_max_per_seg"] = [float(o[:, 0].max()) for o in outs]
                I = [torch.fft.fft2(torch.polar(o[:, 0], o[:, 1]), norm="ortho").abs() ** 2 for o in outs]
                row["empty_I_min_pos_per_seg"] = [float(v[v > 0].min()) if (v > 0).any() else 0.0 for v in I]
        rows.append(row)
        print(f"  批 {b:02d}:{'❌' if bad else '✅'}  空樣本 {empt if empt else '無'}"
              + (f"\n      邊界:{row.get('boundary')};nan 在第 {row.get('nan_rows')} 張" if bad else "")
              + (f"\n      拿掉空樣本後:{'恢復正常' if row['ok_without_empty'] else '仍然壞'}"
                 if "ok_without_empty" in row else "")
              + (f"\n      空樣本各段預測振幅最大 {[round(v, 4) for v in row['empty_pred_amp_max_per_seg']]}"
                 f"\n      空樣本各段預測強度最小正值 {['%.2g' % v for v in row['empty_I_min_pos_per_seg']]}"
                 if empt else ""), flush=True)

    bad_b = [r["batch"] for r in rows if r["bad"]]
    emp_b = [r["batch"] for r in rows if r["empty_samples"]]
    print("\n" + "=" * 80)
    print(f"壞批次 {bad_b}")
    print(f"含空樣本的批次 {emp_b}")
    print(f"壞批次中含空樣本:{len([b for b in bad_b if b in emp_b])}/{len(bad_b)};"
          f"含空樣本的批次中壞掉:{len([b for b in emp_b if b in bad_b])}/{len(emp_b)}")
    fixed = [r["ok_without_empty"] for r in rows if r["bad"] and "ok_without_empty" in r]
    print(f"壞且含空樣本的批次,拿掉空樣本後恢復正常:{sum(fixed)}/{len(fixed)}")
    verdict = bool(bad_b) and set(bad_b) <= set(emp_b) and all(fixed)
    print("判讀:" + ("空樣本是原因(壞批次都含空樣本,且拿掉即恢復)" if verdict
                    else "空樣本假說不成立或不完整,把完整輸出貼給 Claude"))
    json.dump({"rows": rows, "verdict_empty": verdict, "quick": QUICK},
              open(RUN_ROOT / ("diag_nan_v3c_quick.json" if QUICK else "diag_nan_v3c.json"), "w"),
              indent=2, ensure_ascii=False)
    print("把完整輸出貼給 Claude")


if __name__ == "__main__":
    main()
