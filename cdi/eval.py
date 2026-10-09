#!/usr/bin/env python
"""評估進入點。讀 final.pt,跑完整評估,輸出 metrics.json。

用法:
    python eval.py --run-dir /work/elviss0915/runs/bs03_s0
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.config import Cfg
from src.data import generalization_suites
from src.metrics import evaluate
from src.model import build_model
from src.hio import hio, random_init
from src.physics import (beamstop_mask, build_input, forward_measure,
                         input_channels)

HOME_MAP = {"mnist": "mnist_test", "fashion": "fashion_mnist",
            "shapes": "random_shapes", "texture": "random_texture",
            "procedural": "procedural"}


@torch.no_grad()
def inference_speed(model, cfg, device, batch_size=1, n_iter=100):
    model.eval()
    x = torch.randn(batch_size, input_channels(cfg), cfg.canvas,
                    cfg.canvas, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter / batch_size * 1e3   # ms/sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    args = ap.parse_args()

    run = Path(args.run_dir)
    # 讀回訓練當時實際生效的設定,包含 ref_energy ——
    # 重算會讓 R-factor 跟訓練時不可比。
    cfg = Cfg.from_dict(json.load(open(run / "config_used.json")))
    assert cfg.ref_energy is not None, "config_used.json 缺 ref_energy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / args.ckpt, map_location=device))
    bs_mask = beamstop_mask(cfg, device=device)
    print(f"[eval] {run.name}  ref_energy={cfg.ref_energy:.6f}", flush=True)

    out = {"run": run.name, "config": cfg.to_dict()}

    # ---- 泛化測試(所有分布都跑,含訓練分布本身) ----
    #
    # 重要:「主要指標」不能寫死用 MNIST。若模型是用 procedural 訓練的,
    # 拿 MNIST(它沒看過的分布)打分數,等同拿 Fashion-MNIST 考一個只學過
    # 數字的模型 —— 分數會很難看,但那不代表訓練失敗,是考錯科目。
    # (這是本專案曾實際犯過的錯:src_proc 用固定 MNIST 評估時 PSNR 只有
    #  21.5 dB、FRC 輸給平庸基準,一度誤判為訓練失敗;修正後見下方。)
    out["generalization"] = {}
    for name, objs in generalization_suites(cfg, cfg.gen_n).items():
        r = evaluate(model, objs, bs_mask, cfg)
        r.pop("_curves")
        out["generalization"][name] = r
        print(f"  [gen] {name:<16} PSNR {r['amp_psnr']:.2f}  "
              f"phase {r['phase_rmse']:.4f}  gain {r['frc_gain']:+.4f}", flush=True)
    g = out["generalization"]

    homes = [HOME_MAP[s] for s in cfg.train_sources if s in HOME_MAP]
    out["home_keys"] = homes

    # ---- 主要指標:訓練分布上的表現(用較大的 eval_n,較準) ----
    home_pool = generalization_suites(cfg, cfg.eval_n)
    parts = [home_pool[k] for k in homes if k in home_pool]
    out["test"] = evaluate(model, torch.cat(parts, 0) if len(parts) > 1
                           else parts[0], bs_mask, cfg)
    m = out["test"]
    print(f"\n  [home={homes}] PSNR {m['amp_psnr']:.2f} "
          f"(triv {m['amp_psnr_trivial']:.2f})  "
          f"phase {m['phase_rmse']:.4f} (triv {m['phase_rmse_trivial']:.4f}, "
          f"copy {m['phase_rmse_copy_amp']:.4f})", flush=True)
    print(f"  FRC AUC {m['frc_auc']:.4f} (triv {m['frc_auc_trivial']:.4f}, "
          f"gain {m['frc_gain']:+.4f})  res {m['frc_res']:.2f} px", flush=True)
    print(f"  R {m['r_factor']:.4f}  noise floor {m['r_factor_noise_floor']:.4f}",
          flush=True)
    # 材料反演:比相位 RMSE 更嚴格的檢驗。
    # 材料靠 φ/A 這個比值,除法會放大誤差 —— A 與 φ 各自看起來還行,
    # 合起來仍可能解不出材料。物理上這也正是相位成像的價值所在
    # (輕元素幾乎不吸收、在振幅上看不見,只能靠相位分辨)。
    mgain = (m['material_mae_trivial'] - m['material_mae']) / \
        max(m['material_mae_trivial'], 1e-12)
    print(f"  材料 MAE {m['material_mae']:.4f} "
          f"(平庸基準 {m['material_mae_trivial']:.4f},"
          f"贏 {100*mgain:+.1f}%)", flush=True)

    # ---- 附上「固定 MNIST」的分數,跨來源比較時可用同一把尺 ----
    out["mnist_fixed"] = g["mnist_test"]

    # ---- 多重實驗條件下的評估(probe robustness) ----
    #
    # 訓練時若隨機化了劑量與 beamstop,僅以單一固定條件評估並不公平:
    # 涵蓋廣泛條件的模型在單一條件上本就可能輸給專精者。
    # 先前「參數隨機化使表現下降」的結論即建立在此缺陷之上。
    # 此處掃描多組條件,量測「換條件仍能運作」的能力。
    if cfg.eval_photons or cfg.eval_beamstops:
        phs = list(cfg.eval_photons) or [cfg.photons_per_pix]
        bss = list(cfg.eval_beamstops) or [cfg.beamstop_r]
        out["robustness"] = {}
        home_objs = generalization_suites(cfg, cfg.eval_n)[homes[0]]
        for ph in phs:
            for br in bss:
                m_bs = beamstop_mask(cfg, radius=br, device=device)
                r = evaluate(model, home_objs, m_bs, cfg, photons=ph)
                r.pop("_curves", None)
                out["robustness"][f"ph{ph:g}_bs{br:g}"] = r
        vals = [v["frc_gain"] for v in out["robustness"].values()]
        sv = sorted(vals)
        print(f"\n  [robust] {len(vals)} 組條件  FRC gain "
              f"min {sv[0]:+.4f}  median {sv[len(sv)//2]:+.4f}  "
              f"max {sv[-1]:+.4f}", flush=True)

    # ---- HIO 後處理:掃描迭代數 ----
    #
    # 迭代數必須掃描而非固定 —— HIO 可能破壞網路的良好初始解,
    # 且迭代過多會開始擬合 Poisson 雜訊(見 config 的 hio_iters 說明)。
    # 同時記錄 R-factor:其不需 ground truth,
    # 可作為真實資料上選擇停止點的客觀依據。
    if cfg.hio_iters:
        out["hio"] = {}
        # 初始解來源：網路輸出 或 隨機起點（純 HIO，傳統方法基準）。
        # 兩者的迭代與評估流程完全相同，唯一差異為起點，故可直接比較
        # 「網路提供的初始解」在速度軸上值多少。
        src = "random" if cfg.hio_random_init else "network"
        out["hio_init"] = src
        for name, objs in generalization_suites(cfg, cfg.gen_n).items():
            objs_d = objs.to(device)
            counts = forward_measure(objs_d, bs_mask, cfg)
            if cfg.hio_random_init:
                init = random_init(counts, cfg, seed=cfg.test_seed, device=device)
            else:
                with torch.no_grad():
                    init = model(build_input(objs_d, counts, bs_mask, cfg))
            per_suite = {}
            for n_it in cfg.hio_iters:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                ref = hio(init, counts, bs_mask, cfg,
                          n_iter=n_it, beta=cfg.hio_beta)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                # 每張樣本的迭代耗時，對應計畫 benchmark 的
                # reconstruction speed 軸（純網路為 0.029 ms/sample）
                ms = (time.perf_counter() - t0) / len(objs_d) * 1e3
                r = evaluate(model, objs_d, bs_mask, cfg, pred_override=ref)
                r.pop("_curves", None)
                r["hio_ms_per_sample"] = ms
                per_suite[str(n_it)] = r
            out["hio"][name] = per_suite
            best = min(per_suite, key=lambda k: per_suite[k]["r_factor"])
            mark = "*" if name in homes else " "
            print(f"  [hio:{src[:4]}] {name:<16}{mark} R 最小於 {best:>4} 次  "
                  f"gain {per_suite[best]['frc_gain']:+.4f}  "
                  f"材料 {per_suite[best]['material_mae']:.4f}  "
                  f"{per_suite[best]['hio_ms_per_sample']:.2f} ms/sample",
                  flush=True)

    others = [k for k in g if k not in homes]
    if others:
        hp = max(g[k]["amp_psnr"] for k in homes)
        op = min(g[k]["amp_psnr"] for k in others)
        out["generalization_gap_db"] = hp - op
    else:
        out["generalization_gap_db"] = 0.0
    vals = [g[k]["amp_psnr"] for k in g]
    out["gen_spread_db"] = float(max(vals) - min(vals))
    print(f"\n  泛化落差 {out['generalization_gap_db']:.2f} dB "
          f"(home={homes} vs 最差的其他分布)", flush=True)
    print(f"  分布間離散 {out['gen_spread_db']:.2f} dB "
          f"(所有分布的 PSNR 最大差;越小越好)", flush=True)

    # ---- 推論速度(研究計畫 benchmark 的 reconstruction speed 軸) ----
    out["speed_ms_per_sample"] = {
        "batch1": inference_speed(model, cfg, device, 1),
        "batch64": inference_speed(model, cfg, device, 64),
    }
    print(f"  速度 {out['speed_ms_per_sample']['batch1']:.3f} ms/sample (b=1), "
          f"{out['speed_ms_per_sample']['batch64']:.4f} ms/sample (b=64)", flush=True)

    json.dump(out, open(run / "metrics.json", "w"), indent=2)
    print(f"[done] -> {run/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
