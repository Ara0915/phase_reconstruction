#!/usr/bin/env python
"""診斷材料反演在 procedural 上失效的原因。

背景:e40_proc_oracle(拿到完整繞射相位)的材料 MAE 0.3907,
      仍劣於平庸基準 0.304;而 e40_mnist_oracle 為 0.1366、優於基準 0.403。
      受控測試顯示 procedural 在各方面反而更有利
      (分母 A 中位數 0.407 vs 0.119、誤差放大倍率 0.61x vs 1.30x),
      故成因必在模型的實際輸出,而非物體或指標的性質。

用法(需在計算節點執行):
    python inspect_material.py --runs-root /work/elviss0915/runs \
        --runs e40_proc_oracle_s0 e40_mnist_oracle_s0
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

from src.config import Cfg
from src.data import generalization_suites
from src.metrics import recover_material
from src.model import build_model
from src.physics import beamstop_mask, build_input, forward_measure

HOME = {"mnist":"mnist_test","fashion":"fashion_mnist","shapes":"random_shapes",
        "texture":"random_texture","procedural":"procedural"}


@torch.no_grad()
def inspect(run):
    cfg = Cfg.from_dict(json.load(open(run/"config_used.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(run/"final.pt", map_location=dev)); model.eval()
    bs = beamstop_mask(cfg, device=dev)
    homes=[HOME[s] for s in cfg.train_sources if s in HOME]
    obj = generalization_suites(cfg, cfg.eval_n)[homes[0]].to(dev)
    counts = forward_measure(obj, bs, cfg)
    pred = model(build_input(obj, counts, bs, cfg))
    p, t = pred.cpu(), obj.cpu()
    sup = (t[:,0] > cfg.amp_floor).float()
    n = sup.sum().clamp_min(1)

    # 全域相位校正(與 material_error 一致)
    d=(p[:,1]-t[:,1])*sup; ns=sup.sum(dim=(-2,-1)).clamp_min(1)
    off=(d.sum(dim=(-2,-1))/ns)[:,None,None]
    pc=torch.stack([p[:,0],(p[:,1]-off*sup).clamp(0,cfg.phase_max)],1)

    mt, mp = recover_material(t,cfg), recover_material(pc,cfg)
    mbar=(mt*sup).sum()/n

    # 分項誤差來源
    only_amp = recover_material(torch.stack([pc[:,0], t[:,1]],1), cfg)
    only_pha = recover_material(torch.stack([t[:,0], pc[:,1]],1), cfg)

    def mae(x): return float(((x-mt).abs()*sup).sum()/n)
    return dict(
        run=run.name,
        amp_bias=float(((pc[:,0]-t[:,0])*sup).sum()/n),
        amp_mae=float(((pc[:,0]-t[:,0]).abs()*sup).sum()/n),
        pha_bias=float(((pc[:,1]-t[:,1])*sup).sum()/n),
        m_true_mean=float(mbar), m_pred_mean=float((mp*sup).sum()/n),
        mae_full=mae(mp), mae_amp_only=mae(only_amp), mae_pha_only=mae(only_pha),
        mae_triv=float(((mbar-mt).abs()*sup).sum()/n),
        sat0=float(((mp<1e-6)&(sup>0)).float().sum()/n),
        sat1=float(((mp>1-1e-6)&(sup>0)).float().sum()/n),
        sat0_t=float(((mt<1e-6)&(sup>0)).float().sum()/n),
        sat1_t=float(((mt>1-1e-6)&(sup>0)).float().sum()/n))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/work/elviss0915/runs")
    ap.add_argument("--runs", nargs="+", required=True)
    a=ap.parse_args()
    rs=[inspect(Path(a.runs_root)/r) for r in a.runs]

    print(f"\n{'run':<24}{'材料MAE':>9}{'僅振幅錯':>10}{'僅相位錯':>10}{'平庸基準':>10}")
    print("-"*66)
    for d in rs:
        print(f"{d['run']:<24}{d['mae_full']:>9.4f}{d['mae_amp_only']:>10.4f}"
              f"{d['mae_pha_only']:>10.4f}{d['mae_triv']:>10.4f}")
    print("\n『僅振幅錯』= 用模型振幅配真實相位;『僅相位錯』= 用真實振幅配模型相位。")
    print("哪一項接近完整誤差,主因就在該通道。\n")

    print(f"{'run':<24}{'振幅偏差':>10}{'振幅MAE':>10}{'相位偏差':>10}")
    print("-"*56)
    for d in rs:
        print(f"{d['run']:<24}{d['amp_bias']:>+10.4f}{d['amp_mae']:>10.4f}"
              f"{d['pha_bias']:>+10.4f}")
    print("\n偏差(有號)遠大於 0 代表系統性高估/低估,而非隨機誤差。\n")

    print(f"{'run':<24}{'真實m均值':>11}{'預測m均值':>11}"
          f"{'預測撞0%':>10}{'預測撞1%':>10}{'真實撞0%':>10}{'真實撞1%':>10}")
    print("-"*88)
    for d in rs:
        print(f"{d['run']:<24}{d['m_true_mean']:>11.4f}{d['m_pred_mean']:>11.4f}"
              f"{100*d['sat0']:>9.1f}%{100*d['sat1']:>9.1f}%"
              f"{100*d['sat0_t']:>9.1f}%{100*d['sat1_t']:>9.1f}%")
    print("\n撞邊界比例遠高於真實值,代表 φ/A 落在可解範圍外、被 clamp 吃掉。")


if __name__ == "__main__":
    main()
