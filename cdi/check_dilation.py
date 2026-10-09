#!/usr/bin/env python
"""查證:dilation 實驗(ovs96 / ovs128 / ctl_d2 / ctl_d3)當時到底有沒有用上 dilation?

背景(實驗設計 §19.3):國網上的 src/model.py 不支援 dilation(設定會被靜默忽略),
其修改時間(9/2)早於這些 run(9/8)。但修改時間可能因上傳方式而不準,需要決定性的證據。

做法:同一份 final.pt,分別載入兩種模型結構再評分:
    無 dilation   = 國網當時的 model.py(卷積 padding=1,dilation 被忽略)
    有 dilation   = 支援 dilation 的版本(依 config 的 dilation 值)
兩者參數形狀完全相同、載入都不會報錯,但**權重只在訓練時的那種結構下才有意義**。
哪一種能重現該 run 的 metrics.json,當時就是用哪一種訓練與評估的。
另一種會因感受野錯位而大幅劣化。

用法(需在計算節點執行):
    python check_dilation.py
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.config import Cfg
from src.data import generalization_suites
from src.metrics import evaluate
from src.model import UNet
from src.physics import beamstop_mask, input_channels, support_mask

RUN_ROOT = Path("/work/elviss0915/runs")
RUNS = ["ovs64", "ovs96", "ovs128", "ctl_d2", "ctl_d3"]
SEEDS = [0, 1, 2]


# ---- 支援 dilation 的版本(與交給 Claude 的 model.py 相同的結構) ----
class DilDoubleConv(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        p = dilation
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=p, dilation=dilation),
            nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=p, dilation=dilation),
            nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x):
        return self.f(x)


class DilUNet(UNet):
    """結構與 UNet 相同,只是編碼器的卷積使用 cfg.dilation(解碼器維持 1,與原版一致)。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        cin, b, nd = input_channels(cfg), cfg.base_channels, cfg.n_down
        dil = cfg.dilation
        self.inc = DilDoubleConv(cin, b, dil)
        self.downs = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2), DilDoubleConv(b * 2 ** i, b * 2 ** (i + 1), dil))
            for i in range(nd)])


def score(model, cfg, run_dir, dev, seed):
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location=dev))
    model.to(dev).eval()
    homes = {"procedural": "procedural", "mnist": "mnist_test"}
    home = homes[list(cfg.train_sources)[0]]
    objs = generalization_suites(cfg, cfg.eval_n)[home].to(dev)
    torch.manual_seed(cfg.test_seed + seed)
    with torch.no_grad():
        return float(evaluate(model, objs, beamstop_mask(cfg, device=dev), cfg)["frc_gain"])


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'run':<12}{'dilation':>9}{'metrics.json':>14}{'無 dilation':>13}{'有 dilation':>13}   判定")
    print("-" * 76)
    for name in RUNS:
        for s in SEEDS:
            run_dir = RUN_ROOT / f"{name}_s{s}"
            if not (run_dir / "final.pt").exists():
                print(f"{run_dir.name:<12}  (找不到,略過)")
                continue
            cfg = Cfg.from_dict(json.load(open(run_dir / "config_used.json")))
            ref = json.load(open(run_dir / "metrics.json"))["test"]["frc_gain"]
            a = score(UNet(cfg), cfg, run_dir, dev, s)
            b = score(DilUNet(cfg), cfg, run_dir, dev, s)
            if cfg.dilation == 1:
                verdict = "(dilation=1,兩者相同,作對照)"
            elif abs(a - ref) < 0.01 and abs(b - ref) >= 0.03:
                verdict = "當時**沒有** dilation"
            elif abs(b - ref) < 0.01 and abs(a - ref) >= 0.03:
                verdict = "當時**有** dilation"
            else:
                verdict = "無法判定,貼給 Claude"
            print(f"{run_dir.name:<12}{cfg.dilation:>9}{ref:>+14.4f}{a:>+13.4f}{b:>+13.4f}   {verdict}")


if __name__ == "__main__":
    main()
