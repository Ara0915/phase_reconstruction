#!/usr/bin/env python
"""階段四 4-2a 展開版(分段反傳)診斷:訓練後為什麼連內建的 50 輪 HIO 都沒用上(協定 §8.7、§8.8)。

只做前向運算,不訓練、不改任何既有檔案。對每個 seed 的 probe_ds_unroll_s*,在與 check_probe42_ds.py
相同的 512 張測試物體、相同量測上,追蹤展開過程中第 0、10、20、30、40、50 輪的內部狀態 g,並用兩種方式讀出:

  自然讀出  振幅 = |g|、相位 = angle(g),截到約束範圍並乘上探針圓盤 —— 與 hio.py 的最後一步完全相同
  輸出頭    模型自己學到的輸出頭(model.readout)

五種情況:
  (1) 網路起點 + 修正開    = 訓練好的模型本身
  (2) 網路起點 + 修正關    同一個起點,只跑物理 HIO(修正網路的輸出不加上去)
  (3) 隨機起點 + 修正關    = P-HIO(健全性檢查:50 輪應接近 check 的 P-HIO 50 次 +0.59)
  (4) 隨機起點 + 修正開    訓練好的修正網路,放在好的起點上會怎樣
  (5) 未訓練模型的網路起點 + 修正關(未訓練時修正本來就是 0)—— 起點在訓練前就有問題嗎

判讀(結果出來前寫定,協定 §8.8;門檻 0.35 = §4.1 的 F):
  (1) 自然讀出 ≥ 0.35 而輸出頭 < 0.35         → 原因 ②:輸出頭沒學會讀出
  (1) 自然讀出 < 0.35、(2) ≥ 0.35              → 原因 ①:修正網路破壞了 HIO
  (1)(2) 皆 < 0.35、(3) ≥ 0.35                 → 原因 ④:網路給的起點讓 HIO 解不出來((5) 看訓練前是否已如此)
  (4) 明顯低於 (3)                             → 修正網路在好的起點上也有害(佐證 ①)
  以上可同時成立;原因 ③(梯度尖峰)由訓練紀錄判斷,本腳本不處理。

健全性檢查(不通過就停):模型輸出重現 check 的對齊後分數(差 ≤ 0.01);
  修正開時的第 50 輪經輸出頭 = model(x)(差 = 0);(3) 第 50 輪與 P-HIO 50 次差 ≤ 0.05。

用法(需在計算節點執行;需 realign_eval.py、ambiguity_check.py、check_probe42_ds.py 已安裝):
    python diag_unroll.py
輸出:/work/elviss0915/runs/diag_unroll.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN_ROOT = Path(os.environ.get("CDI_RUN_ROOT", "/work/elviss0915/runs"))
NAME = "probe_ds_unroll"
SEEDS = [0, 1, 2]
MARKS = [0, 10, 20, 30, 40, 50]
THR = 0.35
QUICK = os.environ.get("CDI_QUICK") == "1"     # 只供本地測程式流程
N_OBJ = 32 if QUICK else None
CHUNK = 128
REF_JSON = RUN_ROOT / ("probe42_ds_quick.json" if QUICK else "probe42_ds.json")

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}" + (f"\n     {detail}" if detail else ""), flush=True)
    ok_all &= bool(passed)


def natural(g, sup, pmax):
    """與 hio.py 最後一步相同的讀出。"""
    amp = g.abs().clamp(0, 1) * sup
    pha = torch.angle(g).clamp(0, pmax) * sup
    return torch.stack([amp, pha], dim=1)


@torch.no_grad()
def trajectory(m, g, A, bsm, refine):
    """照 UnrollNet.rounds 跑 T 輪;refine=False 時修正網路的輸出不加上去。
    回傳 {輪數: g} 與各修正輪的 |修正| / |h| 平均比例。"""
    outs, ratios, k = {0: g}, [], 0
    for t in range(m.t):
        g_new = m.fourier_step(g, A, bsm)
        h = m.hio_step(g, g_new)
        if (t + 1) % m.every == 0:
            k += 1
            if refine:
                feat = torch.cat([m._c2r(g_new), m._c2r(g), m._c2r(h)], dim=1)
                r = m._r2c(m.refine[k](feat))
                ratios.append(float(r.abs().mean() / h.abs().mean().clamp_min(1e-12)))
                g = h + r
            else:
                g = h
        else:
            g = h
        if (t + 1) in MARKS:
            outs[t + 1] = g
    return outs, ratios


def main():
    from check_probe42_ds import raw_home
    from realign_eval import load, score_all
    from src.config import Cfg
    from src.hio import hio, random_init
    from src.model import build_model
    from src.physics import apply_probe, beamstop_mask, build_input, forward_measure

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    ref = json.load(open(REF_JSON)) if REF_JSON.exists() else None
    res = {}

    for s in SEEDS:
        rd = RUN_ROOT / f"{NAME}_s{s}"
        cfg, model = load(rd, dev)
        assert cfg.arch == "unroll" and cfg.unroll_t == 50 and cfg.probe == "disk"
        n = N_OBJ or cfg.eval_n
        psi = apply_probe(raw_home(cfg, n), cfg).to(dev)
        bs = beamstop_mask(cfg, device=dev)
        torch.manual_seed(cfg.test_seed + s)                   # 與 check_probe42_ds.py 相同的量測
        counts = forward_measure(psi, bs, cfg)
        x = build_input(psi, counts, bs, cfg)
        sup, pmax = model.sup, cfg.phase_max

        torch.manual_seed(s)
        fresh = build_model(Cfg.from_dict(cfg.to_dict())).to(dev).eval()   # 未訓練(同架構、隨機初始)
        rinit = random_init(counts, cfg, seed=cfg.test_seed + s, device=dev)

        def aligned(pred):
            r, _ = score_all(model, psi, bs, cfg, pred)
            return r["aligned"]["frc_gain"]

        cases = {"1_net_on": (model, "net", True), "2_net_off": (model, "net", False),
                 "3_rand_off": (model, "rand", False), "4_rand_on": (model, "rand", True),
                 "5_fresh_off": (fresh, "net", False)}
        out = {}
        with torch.no_grad():
            y = torch.cat([model(x[i:i + CHUNK]) for i in range(0, n, CHUNK)])
            out["model_output"] = aligned(y)
            for key, (m, start, refine) in cases.items():
                nat = {T: [] for T in MARKS}
                head = {T: [] for T in MARKS}
                rat = []
                for i in range(0, n, CHUNK):
                    xs = x[i:i + CHUNK]
                    g0, A, bsm = m.start(xs)
                    if start == "rand":
                        ri = rinit[i:i + CHUNK]
                        g0 = torch.polar(ri[:, 0], ri[:, 1])
                    tr, r = trajectory(m, g0, A, bsm, refine)
                    rat += r
                    for T in MARKS:
                        nat[T].append(natural(tr[T], sup, pmax))
                        head[T].append(m.readout(tr[T]))
                    if key == "1_net_on" and i == 0:
                        dmax = float((m.readout(tr[50]) - y[:CHUNK]).abs().max())
                out[key] = {"natural": {T: aligned(torch.cat(nat[T])) for T in MARKS},
                            "head": {T: aligned(torch.cat(head[T])) for T in MARKS},
                            "refine_ratio": float(np.mean(rat)) if rat else 0.0}
            p50 = aligned(hio(rinit, counts, bs, cfg, n_iter=50, beta=cfg.hio_beta))
        out["phio50"] = p50
        res[s] = out

        print(f"\n=== seed {s} ===", flush=True)
        if ref is not None:
            rv = ref["results"]["nets"]["unroll"][s]["aligned"]["frc_gain"]
            check(f"s{s}:模型輸出重現 check_probe42_ds 的對齊後分數", abs(out["model_output"] - rv) <= 0.01,
                  f"{out['model_output']:+.4f} vs {rv:+.4f}")
        else:
            check(f"{REF_JSON.name} 存在", False)
        check(f"s{s}:修正開 + 輸出頭的第 50 輪 = model(x)", dmax == 0.0, f"最大差 {dmax}")
        d3 = out["3_rand_off"]["natural"][50] - p50
        check(f"s{s}:(3) 第 50 輪 ≈ hio.py 50 次(差 ≤ 0.05)", abs(d3) <= 0.05,
              f"{out['3_rand_off']['natural'][50]:+.4f} vs {p50:+.4f}(展開版取振幅多了 1e-12,HIO 對微小差異敏感,容許小差)")

    # ---- 報告(三個 seed 平均) ----
    print("\n" + "=" * 96)
    print("對齊後 FRC gain(3 seeds 平均);第 0 輪 = 起點")
    print("=" * 96)
    label = {"1_net_on": "(1) 網路起點 + 修正開(模型本身)", "2_net_off": "(2) 網路起點 + 修正關",
             "3_rand_off": "(3) 隨機起點 + 修正關(P-HIO)", "4_rand_on": "(4) 隨機起點 + 修正開",
             "5_fresh_off": "(5) 未訓練模型的起點 + 修正關"}
    avg = lambda f: float(np.mean([f(res[s]) for s in SEEDS]))            # noqa: E731
    print(f"  模型輸出 {avg(lambda r: r['model_output']):+.4f};hio.py P-HIO 50 次 {avg(lambda r: r['phio50']):+.4f}")
    print(f"  {'':<34}" + "".join(f"{'第' + str(T) + '輪':>9}" for T in MARKS) + "   |修正|/|h|")
    summ = {}
    for key in label:
        for rd_ in ["natural", "head"]:
            v = [avg(lambda r, T=T: r[key][rd_][T]) for T in MARKS]
            summ[(key, rd_)] = v
            ratio = avg(lambda r: r[key]["refine_ratio"])
            print(f"  {label[key] if rd_ == 'natural' else '':<30}{'自然' if rd_ == 'natural' else '輸出頭':>4}"
                  + "".join(f"{x:>+9.3f}" for x in v)
                  + (f"   {ratio:.3f}" if rd_ == "natural" and key in ("1_net_on", "4_rand_on") else ""))

    print("\n" + "=" * 96)
    print("判讀(協定 §8.8,結果出來前寫定;門檻 0.35)")
    print("=" * 96)
    n1, h1 = summ[("1_net_on", "natural")][-1], summ[("1_net_on", "head")][-1]
    n2, n3, n4, n5 = (summ[(k, "natural")][-1] for k in ["2_net_off", "3_rand_off", "4_rand_on", "5_fresh_off"])
    found = []
    if n1 >= THR and h1 < THR:
        found.append(f"② 輸出頭沒學會讀出:內部 g 自然讀出 {n1:+.3f} ≥ 0.35,輸出頭只有 {h1:+.3f}")
    if n1 < THR and n2 >= THR:
        found.append(f"① 修正網路破壞了 HIO:同一起點關掉修正 {n2:+.3f} ≥ 0.35,開著只有 {n1:+.3f}")
    if n1 < THR and n2 < THR and n3 >= THR:
        found.append(f"④ 網路給的起點讓 HIO 解不出來:網路起點關修正 {n2:+.3f},隨機起點 {n3:+.3f}"
                     f";訓練前的起點 {n5:+.3f}({'訓練前就已如此' if n5 < THR else '訓練後才變差'})")
    if n4 < n3 - 0.05:
        found.append(f"修正網路在好的起點上也有害:隨機起點修正開 {n4:+.3f} < 關 {n3:+.3f}(佐證 ①)")
    for f in found:
        print("  → " + f)
    if not found:
        print("  → 以上情況皆不成立,把完整輸出貼給 Claude")
    json.dump({"per_seed": {str(s): {k: v for k, v in r.items()} for s, r in res.items()},
               "found": found, "quick": QUICK},
              open(RUN_ROOT / ("diag_unroll_quick.json" if QUICK else "diag_unroll.json"), "w"), indent=2)
    print("\n" + ("✅ 健全性檢查全部通過" if ok_all else "❌ 健全性檢查有項目未通過:上面的判讀不可採信,貼給 Claude"))


if __name__ == "__main__":
    main()
