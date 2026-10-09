#!/usr/bin/env python
"""§1b.6 追加:近理想量測下的 HIO 精煉 —— 網路有沒有把量測裡的資訊用完?

背景(實驗設計 §八):
    拿掉 beamstop、劑量 x100 後,相位未知的網路 FRC gain 只多 +0.0067(z = 0.9),
    同條件下的 oracle 仍領先 +0.067。beamstop 與雜訊不是網路的主要限制。

本腳本要回答:
    量測品質提升時,「靠物理迭代的方法」會不會進步,而網路不會?

    若 HIO 系方法隨量測品質大幅進步、網路幾乎不動
    -> 資訊確實在量測裡,是網路沒有用上(候選 A)。

比較的 2 x 3 矩陣(皆為同一批 checkpoint,不重新訓練):

                      網路本身     網路 + HIO    純 HIO(隨機起點)
    現況 ideal_base      iter 0        掃描          掃描
    理想 ideal_bs0_ph1e5 iter 0        掃描          掃描

    oracle(ideal_bs0_ph1e5_oracle)的 metrics.json 作為天花板參考。

與 eval.py 的 HIO 區塊完全相同的流程(同一個 hio()、同一個 evaluate()),
差異只有三點:
    1. checkpoint 已存在,HIO 迭代數由本腳本指定(eval.py 只讀 config_used.json)
    2. 訓練分布用 eval_n(512)張,與 metrics.json 的 test 同一批樣本,
       故 iter 0 可直接對照 metrics.json 作為健全性檢查
    3. 網路起點與隨機起點共用同一份量測(counts),成對比較

結果另存 <run>/hio_ideal.json,不會動到 metrics.json。

用法(需在計算節點執行):
    python hio_ideal_conditions.py                 # 跑全部並印出彙整
    python hio_ideal_conditions.py --summary-only  # 只讀已存的 json 重印彙整
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from src.config import Cfg
from src.data import generalization_suites
from src.hio import hio, random_init
from src.metrics import evaluate
from src.model import build_model
from src.physics import beamstop_mask, build_input, forward_measure

RUN_ROOT = Path("/work/elviss0915/runs")
CONDITIONS = {
    "ideal_base":      "現況 bs=3  ph=1e3",
    "ideal_bs0_ph1e5": "理想 bs=0  ph=1e5",
}
ORACLE = "ideal_bs0_ph1e5_oracle"
SEEDS = [0, 1, 2]
ITERS = [0, 50, 200, 500, 1000, 2000, 5000]
INITS = ["network", "random"]
HOME = "procedural"
UNSEEN = ["mnist_test", "fashion_mnist", "random_shapes", "random_texture"]
SHORT = {"mnist_test": "mnist", "fashion_mnist": "fashion",
         "random_shapes": "shapes", "random_texture": "texture"}
# 既有結果(研究日誌 §11.3 / §11.4,訓練設定與 ideal_base 相同)作為重現參考。
# 本腳本用 512 張、不同的雜訊抽樣,故容忍放寬到 0.03。
REPRO = {("network", 200): 0.3116, ("random", 1000): 0.0599}
REPRO_TOL = 0.03
KEYS = ["frc_gain", "material_mae", "material_mae_trivial", "amp_psnr",
        "amp_psnr_trivial", "phase_rmse", "frc_res", "r_factor",
        "r_factor_noise_floor"]
SANITY_TOL = 0.01       # iter 0 與 metrics.json 的 test 應只差量測雜訊的一次抽樣


# ============================================================================
# 執行
# ============================================================================
@torch.no_grad()
def net_init(model, objs, counts, bs_mask, cfg, chunk=128):
    outs = []
    for i in range(0, len(objs), chunk):
        outs.append(model(build_input(objs[i:i + chunk], counts[i:i + chunk],
                                      bs_mask, cfg)))
    return torch.cat(outs, 0)


def run_one(run_dir, device):
    cfg = Cfg.from_dict(json.load(open(run_dir / "config_used.json")))
    assert cfg.ref_energy is not None, "config_used.json 缺 ref_energy"
    assert cfg.log_mean is not None, "config_used.json 缺 log_mean"
    assert not cfg.oracle_phase, "本腳本只用於相位未知的模型"
    seed = int(re.search(r"_s(\d+)$", run_dir.name).group(1))

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location=device))
    model.eval()
    bs_mask = beamstop_mask(cfg, device=device)

    print(f"\n[{run_dir.name}]  bs={cfg.beamstop_r:g}  ph={cfg.photons_per_pix:g}  "
          f"blocked={100 * (1 - bs_mask.mean().item()):.3f}%", flush=True)

    # 訓練分布用 eval_n,與 metrics.json 的 test 同一批;未見分布用 gen_n,與 eval.py 相同
    suites = {HOME: generalization_suites(cfg, cfg.eval_n)[HOME]}
    gen = generalization_suites(cfg, cfg.gen_n)
    suites.update({k: gen[k] for k in UNSEEN})

    # Poisson 抽樣可重現;每個 seed 用不同的抽樣,std 才有意義
    torch.manual_seed(cfg.test_seed + seed)

    out = {"run": run_dir.name, "seed": seed,
           "beamstop_r": cfg.beamstop_r, "photons_per_pix": cfg.photons_per_pix,
           "iters": ITERS, "hio_beta": cfg.hio_beta, "results": {}}

    for name, objs in suites.items():
        objs_d = objs.to(device)
        counts = forward_measure(objs_d, bs_mask, cfg)      # 兩種起點共用
        for init_src in INITS:
            if init_src == "network":
                init = net_init(model, objs_d, counts, bs_mask, cfg)
            else:
                init = random_init(counts, cfg, seed=cfg.test_seed + seed,
                                   device=device)
            per_it = {}
            for n_it in ITERS:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                ref = hio(init, counts, bs_mask, cfg, n_iter=n_it,
                          beta=cfg.hio_beta)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / len(objs_d) * 1e3
                r = evaluate(model, objs_d, bs_mask, cfg, pred_override=ref)
                row = {k: float(r[k]) for k in KEYS}
                row["hio_ms_per_sample"] = ms
                per_it[str(n_it)] = row
            out["results"].setdefault(name, {})[init_src] = per_it
            if name == HOME:
                g = [per_it[str(n)]["frc_gain"] for n in ITERS]
                print(f"  {init_src:<8} FRC gain  " +
                      "  ".join(f"{n}:{v:+.3f}" for n, v in zip(ITERS, g)),
                      flush=True)

    # ---- 健全性檢查:iter 0 的網路輸出應重現 metrics.json 的 test ----
    m = json.load(open(run_dir / "metrics.json"))["test"]["frc_gain"]
    it0 = out["results"][HOME]["network"]["0"]["frc_gain"]
    out["sanity"] = {"metrics_test_frc_gain": m, "iter0_frc_gain": it0}
    ok = abs(it0 - m) < SANITY_TOL
    print(f"  健全性:iter 0 = {it0:+.4f}  metrics.json test = {m:+.4f}  "
          f"差 {it0 - m:+.4f}  {'✅' if ok else '❌ 超過 ' + str(SANITY_TOL)}",
          flush=True)
    if not ok:
        raise SystemExit("iter 0 與 metrics.json 對不上 —— 評估流程與 eval.py 不一致,停下來查")

    json.dump(out, open(run_dir / "hio_ideal.json", "w"), indent=1)
    return out


# ============================================================================
# 彙整
# ============================================================================
def ms_(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def summarize():
    data = {}
    for c in CONDITIONS:
        runs = []
        for s in SEEDS:
            p = RUN_ROOT / f"{c}_s{s}" / "hio_ideal.json"
            if not p.exists():
                raise SystemExit(f"缺 {p},請先跑完整模式")
            runs.append(json.load(open(p)))
        data[c] = runs

    orc = [json.load(open(RUN_ROOT / f"{ORACLE}_s{s}" / "metrics.json"))["test"]
           for s in SEEDS]
    orc_frc = ms_([o["frc_gain"] for o in orc])
    orc_mat = ms_([o["material_mae"] for o in orc])

    def col(c, init, n_it, key, suite=HOME):
        return [r["results"][suite][init][str(n_it)][key] for r in data[c]]

    print("\n" + "=" * 78)
    print("一、訓練分布(procedural,eval_n 張)—— FRC gain,mean ± std(3 seeds)")
    print("=" * 78)
    hdr = f"{'迭代':>6}"
    for c in CONDITIONS:
        hdr += f"  {CONDITIONS[c][:2] + ' 網路+HIO':>16}{CONDITIONS[c][:2] + ' 純HIO':>16}"
    print(hdr)
    print("-" * 78)
    for n_it in ITERS:
        row = f"{n_it:>6}"
        for c in CONDITIONS:
            for init in INITS:
                m, s = ms_(col(c, init, n_it, "frc_gain"))
                row += f"  {m:+9.4f}±{s:.4f}"
        print(row)
    print(f"\n  oracle(理想條件,給繞射相位):FRC gain {orc_frc[0]:+.4f} ± {orc_frc[1]:.4f}"
          f"   材料 MAE {orc_mat[0]:.4f}")

    # ---- 重現檢查:現況條件應重現既有的 hio2_net / hio2_rand ----
    print("\n  重現檢查(現況條件 vs 研究日誌 §11.3 / §11.4):")
    for (init, n_it), ref in REPRO.items():
        if n_it not in ITERS:
            continue
        m, _ = ms_(col("ideal_base", init, n_it, "frc_gain"))
        lab = "網路+HIO" if init == "network" else "純 HIO"
        ok = abs(m - ref) < REPRO_TOL
        print(f"    {lab:<8}{n_it:>5} 次  {m:+.4f}  既有 {ref:+.4f}  差 {m - ref:+.4f}  "
              f"{'✅' if ok else '❌ 超過 ' + str(REPRO_TOL) + ',先停下來查'}")

    # ---- 最佳點:以真值選(樂觀)與以 R-factor 選(不需真值)----
    print("\n" + "=" * 78)
    print("二、各方法的最佳點")
    print("=" * 78)
    print("  以真值選 = 各迭代數中 FRC gain 平均最高者(樂觀,實驗上做不到)")
    print("  以 R 選  = 各迭代數中 R-factor 平均最低者(不需真值,實驗上可行)\n")
    print(f"  {'條件':<18}{'方法':<10}{'以真值選':>18}{'以 R 選':>22}{'材料 MAE(R選)':>16}")
    print("  " + "-" * 82)
    best = {}
    for c in CONDITIONS:
        for init in INITS:
            frc = {n: ms_(col(c, init, n, "frc_gain")) for n in ITERS}
            rf = {n: ms_(col(c, init, n, "r_factor"))[0] for n in ITERS}
            n_gt = max(ITERS, key=lambda n: frc[n][0])
            n_r = min(ITERS, key=lambda n: rf[n])
            mat = ms_(col(c, init, n_r, "material_mae"))
            best[(c, init)] = (n_gt, frc[n_gt], n_r, frc[n_r])
            lab = "網路+HIO" if init == "network" else "純 HIO"
            print(f"  {CONDITIONS[c]:<18}{lab:<10}"
                  f"{frc[n_gt][0]:+8.4f}({n_gt:>4} 次)"
                  f"{frc[n_r][0]:+12.4f}({n_r:>4} 次)"
                  f"{mat[0]:>12.4f}")

    # ---- 核心對照:量測品質提升帶來的增益 ----
    print("\n" + "=" * 78)
    print("三、核心對照:由現況 -> 理想,各方法進步多少(FRC gain,以 R 選的點)")
    print("=" * 78)
    b, i = "ideal_base", "ideal_bs0_ph1e5"
    net_b = ms_(col(b, "network", 0, "frc_gain"))
    net_i = ms_(col(i, "network", 0, "frc_gain"))
    rows = [("網路本身", net_b, net_i)]
    for init, lab in [("network", "網路+HIO"), ("random", "純 HIO")]:
        rows.append((lab, best[(b, init)][3], best[(i, init)][3]))
    print(f"  {'方法':<10}{'現況':>18}{'理想':>18}{'進步':>10}{'z':>7}")
    print("  " + "-" * 63)
    for lab, (mb, sb), (mi, si) in rows:
        se = np.sqrt(sb ** 2 / 3 + si ** 2 / 3)
        z = (mi - mb) / se if se > 0 else float("inf")
        print(f"  {lab:<10}{mb:+10.4f}±{sb:.4f}{mi:+10.4f}±{si:.4f}"
              f"{mi - mb:+10.4f}{z:>7.1f}")

    # ---- 未見分布贏過基準(網路+HIO,與 collect_hio.py 同定義)----
    print("\n" + "=" * 78)
    print("四、未見分布:模型 − 該分布平庸基準(PSNR,dB),網路+HIO")
    print("=" * 78)
    for c in CONDITIONS:
        print(f"\n  {CONDITIONS[c]}")
        print(f"  {'迭代':>6}" + "".join(f"{SHORT[s]:>10}" for s in UNSEEN)
              + f"{'贏過基準':>10}")
        for n_it in ITERS:
            row, win = f"  {n_it:>6}", 0
            for s in UNSEEN:
                v = (np.mean(col(c, "network", n_it, "amp_psnr", s))
                     - np.mean(col(c, "network", n_it, "amp_psnr_trivial", s)))
                row += f"{v:>10.2f}"
                win += v > 0
            print(row + f"{win:>7}/{len(UNSEEN)}")

    print("\n" + "=" * 78)
    print("判讀準則見 實驗設計_1b6 §九(結果出來前已寫定)")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for c in CONDITIONS:
            for s in SEEDS:
                run_one(RUN_ROOT / f"{c}_s{s}", device)
    summarize()


if __name__ == "__main__":
    main()
