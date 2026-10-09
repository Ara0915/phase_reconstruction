#!/usr/bin/env python
"""用修正後的 frc_resolution 重算所有已訓練模型的解析度。

背景:原本的 frc_resolution 直接回傳「FRC 曲線跌破 0.5 的那一格」,
      由於 FRC 只有約 15 個頻段,解析度只能落在 15 個離散值上
      (21.33 / 12.80 / 9.14 / 7.11 / 5.82 ...),相鄰兩格在 7-9 px
      區間相差 22-29%。實測 os64/os96/os128 三者皆為 7.1111±0.0000 ——
      並非表現相同,而是刻度太粗看不出差異。

修正:改為線性插值取連續值,並取「最後一次穿越門檻」以避免曲線抖動。

本腳本不重新訓練,只用既有的 final.pt 重算,並與 metrics.json 內
舊的解析度數值並列比較。

用法(需在計算節點執行):
    python recheck_resolution.py --runs-root /work/elviss0915/runs
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.config import Cfg
from src.data import generalization_suites
from src.metrics import frc_cropped, frc_auc, frc_resolution
from src.model import build_model
from src.physics import beamstop_mask, build_input, forward_measure

HOME_MAP = {"mnist": "mnist_test", "fashion": "fashion_mnist",
            "shapes": "random_shapes", "texture": "random_texture",
            "procedural": "procedural"}


@torch.no_grad()
def recompute(run):
    cfg = Cfg.from_dict(json.load(open(run / "config_used.json")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "final.pt", map_location=device))
    model.eval()
    bs_mask = beamstop_mask(cfg, device=device)

    homes = [HOME_MAP[s] for s in cfg.train_sources if s in HOME_MAP]
    suites = generalization_suites(cfg, cfg.eval_n)
    parts = [suites[k] for k in homes if k in suites]
    obj = (torch.cat(parts, 0) if len(parts) > 1 else parts[0]).to(device)

    counts = forward_measure(obj, bs_mask, cfg)
    pred = model(build_input(obj, counts, bs_mask, cfg))
    p, t = pred.cpu(), obj.cpu()

    triv_a = t[:, 0].mean(0, keepdim=True).expand_as(t[:, 0])
    sup = (t[:, 0] > cfg.amp_floor).float()
    const = (t[:, 1] * sup).sum() / sup.sum().clamp_min(1)
    triv = torch.stack([triv_a, const * sup], 1)

    f, v = frc_cropped(p, t, cfg)
    _, vt = frc_cropped(triv, t, cfg)
    return dict(res=frc_resolution(f, v), res_triv=frc_resolution(f, vt),
                auc=frc_auc(f, v), gain=frc_auc(f, v) - frc_auc(f, vt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/work/elviss0915/runs")
    args = ap.parse_args()

    groups = defaultdict(list)
    for d in sorted(Path(args.runs_root).glob("*_s*")):
        # 兩個檔案都要在:只有 final.pt 而無 metrics.json 的目錄
        # (例如中途重跑 train 但未跑 eval 的 smoke test)會讓後面讀檔失敗
        if not (d / "final.pt").exists() or not (d / "metrics.json").exists():
            continue
        groups[re.sub(r"_s\d+$", "", d.name)].append(d)

    print(f"\n{'setting':<18}{'舊解析度':>22}{'新解析度(插值)':>24}"
          f"{'平庸基準':>12}{'FRC gain':>18}")
    print("-" * 96)
    for name, dirs in sorted(groups.items()):
        old, new, trv, gains = [], [], [], []
        for d in dirs:
            mj = json.load(open(d / "metrics.json"))
            old.append(mj["test"]["frc_res"])
            r = recompute(d)
            new.append(r["res"]); trv.append(r["res_triv"]); gains.append(r["gain"])
        print(f"{name:<18}{np.mean(old):>14.4f}±{np.std(old):<7.4f}"
              f"{np.mean(new):>15.4f}±{np.std(new):<8.4f}"
              f"{np.mean(trv):>12.3f}"
              f"{np.mean(gains):>12.4f}±{np.std(gains):<6.4f}")

    print("\n舊值卡在 15 個離散格點上(21.33/12.80/9.14/7.11/5.82/...),")
    print("std=0.0000 多半代表刻度太粗而非表現一致。新值為插值後的連續量。")


if __name__ == "__main__":
    main()
