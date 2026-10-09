#!/usr/bin/env python
"""驗證振幅損失的正規化修改是否消除系統性低估。

背景:實測模型振幅存在系統性低估(MNIST -0.026、程序生成 -0.193)。
      該偏差為有號量,PSNR 與 L1 皆無法偵測,僅材料反演將其暴露。

      原假設為「空白區主導損失,促使模型整體壓低振幅」,
      但受控測試(整體縮放、含邊界溢出的模擬輸出)顯示
      舊版損失的最小值仍落在正確的 k=1.0,並未重現該誘因。
      故成因尚未確認,須以實際訓練比較新舊損失。

本腳本以相同資料、相同 seed、相同步數訓練兩個模型,
唯一差異為振幅損失的正規化方式,直接量測振幅偏差。

用法(需在計算節點執行):
    python verify_amp_bias.py --source procedural --steps 400
"""
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.config import Cfg
from src.data import build_pool
from src.model import build_model
from src.physics import beamstop_mask, build_input, calibrate_flux, forward_measure


def old_amp_loss(amp_p, amp_t, sup):
    """舊版:在整個輸出區域上平均(兩區權重由面積比例隱含決定)。"""
    return F.l1_loss(amp_p, amp_t)


def train_once(cfg, use_old, steps, device, n=128):
    torch.manual_seed(0)
    pool = build_pool(cfg, n, seed=0).to(device)
    cfg.ref_energy = calibrate_flux(pool.cpu())
    mask = beamstop_mask(cfg, device=device)
    counts = forward_measure(pool, mask, cfg, add_poisson=False)
    x = build_input(pool, counts, mask, cfg)

    model = build_model(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    sup = (pool[:, 0] > cfg.amp_floor).float()

    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        if use_old:
            # 舊版總損失:振幅為整區平均,相位項與新版相同。
            # 資料一致性項在兩版皆同,對本比較無影響,故略去以簡化。
            total = (old_amp_loss(pred[:, 0], pool[:, 0], sup)
                     + cfg.w_phase * ((pred[:, 1] - pool[:, 1]).abs() * sup).sum()
                     / sup.sum().clamp_min(1))
        else:
            sup_o = 1.0 - sup
            d = (pred[:, 0] - pool[:, 0]).abs()
            l_amp = ((d * sup).sum() / sup.sum().clamp_min(1)
                     + cfg.w_amp_out * (d * sup_o).sum() / sup_o.sum().clamp_min(1))
            total = (l_amp + cfg.w_phase * ((pred[:, 1] - pool[:, 1]).abs() * sup).sum()
                     / sup.sum().clamp_min(1))
        total.backward()
        opt.step()

    p = model(x).detach()
    bias = float(((p[:, 0] - pool[:, 0]) * sup).sum() / sup.sum())
    mae = float(((p[:, 0] - pool[:, 0]).abs() * sup).sum() / sup.sum())
    return bias, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="procedural")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--data-root", default="/work/elviss0915/data")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  source={a.source}  steps={a.steps}\n")
    print(f"{'振幅損失版本':<28}{'振幅偏差':>10}{'振幅MAE':>10}{'偏差佔MAE':>11}")
    print("-" * 62)
    for use_old, lab in [(True, "舊版(整區平均)"), (False, "新版(兩區各自正規化)")]:
        cfg = Cfg(train_sources=(a.source,), data_root=a.data_root)
        if a.source == "procedural":
            cfg.match_support_to = 0.0318
        b, m = train_once(cfg, use_old, a.steps, device)
        print(f"{lab:<28}{b:>+10.4f}{m:>10.4f}{abs(b)/max(m,1e-9):>10.1%}")
    print("\n偏差(有號)絕對值變小 = 修改有效消除系統性低估。")
    print("兩者相近 = 成因不在損失正規化,須另尋。")


if __name__ == "__main__":
    main()
