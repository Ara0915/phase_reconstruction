#!/usr/bin/env python
"""展開版 v3 nan 診斷第二步(協定 §8.11):找出 nan 真正的來源運算。

第一步(diag_nan_v3.py)的異常偵測指向 loss 裡的 sqrt。但本地驗證:I_p 恰為 0 時 sqrt 的反傳會先算出 nan,
隨後被 clamp_min(0) 的反傳遮掉(最終梯度仍有限)—— 異常偵測遇到第一個 nan 就停,可能只是被遮掉的中間值。
本腳本改用「邊界法」:對造成 nan 的單一樣本,在展開過程的每個中間張量保留梯度;
nan 會從來源運算往上游擴散,所以「梯度有 nan 的最下游張量」與「梯度有限的下一個張量」之間,就是來源運算。

重現第一步的壞批次(同一個訓練池、同一個量測 seed),逐張找出造成的樣本並做邊界分析。

用法(計算節點,GPU):
    python diag_nan_v3b.py
輸出:/work/elviss0915/runs/diag_nan_v3b.json
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN_ROOT = Path(os.environ.get("CDI_RUN_ROOT", "/work/elviss0915/runs"))
NAME = "probe_v3_unroll_s0"
QUICK = os.environ.get("CDI_QUICK") == "1"
BAD_BATCHES = [0, 1] if QUICK else [9, 10, 17, 23, 28, 37]     # diag_nan_v3.py 找到的壞批次
BS = 8 if QUICK else 128
N_POOL = (max(BAD_BATCHES) + 1) * BS


def fin(t):
    if t is None:
        return None
    if torch.is_complex(t):
        return bool(torch.isfinite(t.real).all() and torch.isfinite(t.imag).all())
    return bool(torch.isfinite(t).all())


def traced(m, x):
    """與 forward_segments 相同的計算,但保留每個中間張量(依前向順序)。"""
    ts = []                                              # (名稱, 張量)

    def keep(name, t):
        if t.requires_grad:
            t.retain_grad()
        ts.append((name, t))
        return t

    bsm = x[:, -1]
    A = m.measured_amp(x)
    h0 = keep("det 輸出", m.det(x))
    g = keep("起點 field", m.field(x, h0[:, 0:1], h0[:, 1:2],
                                   torch.nn.functional.softplus(h0[:, 2:3]))[:, 0])
    g = keep("起點修正後 g0", g + m._fix(m.refine[0](m._c2r(g))))
    outs, k = [], 0
    for t in range(m.t):
        g_new = keep(f"第{t + 1}輪 g′(傅立葉約束後)", m.fourier_step(g, A, bsm))
        h = keep(f"第{t + 1}輪 h(HIO 更新後)", m.hio_step(g, g_new))
        if (t + 1) % m.every == 0:
            k += 1
            feat = torch.cat([m._c2r(g_new), m._c2r(g), m._c2r(h)], dim=1)
            g = keep(f"第{t + 1}輪 g(修正後)", h + m._fix(m.refine[k](feat)))
        else:
            g = h
        if (t + 1) % m.trunc == 0:
            o = keep(f"第{t + 1}輪 段輸出(readout)", m.readout(g))
            outs.append(o)
            g = g.detach()
    return outs, ts


def main():
    from realign_eval import load
    from src.data import build_pool
    from src.losses import build_loss
    from src.physics import beamstop_mask, build_input, forward_measure

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    cfg, m = load(RUN_ROOT / NAME, dev)
    m.train()
    loss_fn = build_loss(cfg, dev)
    bs = beamstop_mask(cfg, device=dev)
    pool = build_pool(cfg, N_POOL, seed=0).to(dev)
    report = []

    for b in BAD_BATCHES:
        obj = pool[b * BS:(b + 1) * BS]
        torch.manual_seed(10_000 + b)                    # 與 diag_nan_v3.py 相同
        with torch.no_grad():
            counts = forward_measure(obj, bs, cfg)
            x = build_input(obj, counts, bs, cfg)
        for i in range(BS):
            m.zero_grad(set_to_none=True)
            xi, oi, ci = x[i:i + 1], obj[i:i + 1], counts[i:i + 1]
            outs, ts = traced(m, xi)
            res = [loss_fn(o, oi, ci, bs)[0] for o in outs]
            (sum(res) / len(res)).backward()
            loss_ok = all(bool(torch.isfinite(r)) for r in res)
            if loss_ok and all(fin(p.grad) is not False for p in m.parameters()):
                continue
            # ---- 邊界:梯度為 nan 的最下游張量 ----
            gf = [(n, fin(t.grad)) for n, t in ts]
            bad = [j for j, (_, f) in enumerate(gf) if f is False]
            last_bad = max(bad) if bad else None
            if last_bad is None:
                nxt = "—"
            elif "段輸出" in gf[last_bad][0]:
                nxt = "該段的 loss(來源在 loss 的反傳)"
            else:
                nxt = next((gf[j][0] for j in range(last_bad + 1, len(gf)) if gf[j][1] is not None), "loss")
            # ---- 樣本特徵 ----
            with torch.no_grad():
                pred = outs[-1]
                sup = m.sup > 0
                psi_amp_in = float(oi[0, 0][sup].mean())
                pred_amp_max = float(pred[0, 0].max())
                n_zero_pred = int((pred[0, 0][sup] == 0).sum())
                E = torch.fft.fft2(torch.polar(pred[:, 0], pred[:, 1]), norm="ortho")
                I = E.abs() ** 2
                n_I_zero = int((I == 0).sum())
                I_min_pos = float(I[I > 0].min()) if (I > 0).any() else float("nan")
                seg_zero = [int((o[0, 0] == 0).all()) for o in outs]
            row = {"batch": b, "sample": i, "loss_finite": loss_ok,
                   "nan_boundary": {"最下游的 nan 梯度張量": gf[last_bad][0] if last_bad is not None else "無",
                                    "其下游(梯度有限)": nxt},
                   "first_param_nan": next((n for n, p in m.named_parameters()
                                            if p.grad is not None and not fin(p.grad)), None),
                   "psi 圓盤內平均振幅": psi_amp_in, "預測振幅最大值": pred_amp_max,
                   "圓盤內預測振幅為 0 的像素": n_zero_pred, "預測繞射強度恰為 0 的像素": n_I_zero,
                   "預測繞射強度最小正值": I_min_pos, "各段輸出是否全為 0": seg_zero}
            report.append(row)
            print(f"批 {b:02d} 樣本 {i:3d}:nan 來源在「{row['nan_boundary']['最下游的 nan 梯度張量']}」"
                  f"→「{nxt}」之間", flush=True)
            print(f"      psi 圓盤內平均振幅 {psi_amp_in:.4f};預測振幅最大 {pred_amp_max:.4f};"
                  f"圓盤內預測為 0 的像素 {n_zero_pred}/{int(sup.sum())};"
                  f"預測繞射強度為 0 的像素 {n_I_zero};最小正值 {I_min_pos:.3g};各段全為 0 {seg_zero}", flush=True)

    print("\n" + "=" * 80)
    if report:
        from collections import Counter
        c = Counter((r["nan_boundary"]["最下游的 nan 梯度張量"].split(" ", 1)[-1],
                     r["nan_boundary"]["其下游(梯度有限)"].split(" ", 1)[-1]) for r in report)
        print(f"造成 nan 的樣本 {len(report)} 張;來源位置統計:")
        for (a, b2), n in c.most_common():
            print(f"  {n} 次:「{a}」→「{b2}」之間")
    else:
        print("壞批次中沒有單張樣本能重現 nan(可能是批次層級的問題)")
    json.dump({"rows": report, "quick": QUICK}, open(RUN_ROOT / ("diag_nan_v3b_quick.json" if QUICK
              else "diag_nan_v3b.json"), "w"), indent=2, ensure_ascii=False)
    print("把完整輸出貼給 Claude")


if __name__ == "__main__":
    main()
