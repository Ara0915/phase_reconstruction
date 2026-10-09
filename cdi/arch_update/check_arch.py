#!/usr/bin/env python
"""架構比較(oracle):U-Net vs 內建反傅立葉轉換(fft)vs 加 self-attention(attn)。

問題(實驗設計 §十八):即使給了繞射相位,U-Net 在 polygon / bandpass 上仍學不會
反傅立葉轉換(0.11 / 0.04;同一量測下純 HIO 0.75 / 0.71)。換架構後能否學會?

三組皆為理想條件(beamstop 0、劑量 1e5)的 oracle,訓練設定逐欄相同,只差 arch:
    ideal_bs0_ph1e5_oracle   arch=unet(已有)
    arch_fft_oracle          arch=fft
    arch_attn_oracle         arch=attn

另列兩條參考線(皆不需訓練):
    直接反轉    量測振幅 × 真實相位 -> 反傅立葉轉換 -> 套 support 方框(= 不學任何東西的 oracle)
    純 HIO 5000 取自 oracle_proto.json(若存在)

用法(需在計算節點執行):
    python check_arch.py --configs-only   # 送件前:變因控制 + 架構單元測試
    python check_arch.py                  # 跑完後:收斂、各原型、速度、參數量
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Cfg                                        # noqa: E402
from src.model import build_model                                 # noqa: E402

RUN_ROOT = Path("/work/elviss0915/runs")
RUNS = {"unet": "ideal_bs0_ph1e5_oracle", "fft": "arch_fft_oracle", "attn": "arch_attn_oracle"}
SEEDS = [0, 1, 2]
SANITY_TOL = 0.01
ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


# ============================================================================
# 送件前
# ============================================================================
def configs_and_unit_test():
    print("=" * 70)
    print("變因控制")
    print("=" * 70)
    base = Cfg.load(f"configs/{RUNS['unet']}.yaml").to_dict()
    for a in ["fft", "attn"]:
        c = Cfg.load(f"configs/{RUNS[a]}.yaml").to_dict()
        diff = sorted(k for k in base if base[k] != c[k])
        check(f"{RUNS['unet']} vs {RUNS[a]} 只差 arch", diff == ["arch"] and c["arch"] == a,
              f"實際差異: {diff}")

    print("\n" + "=" * 70)
    print("架構單元測試")
    print("=" * 70)
    from src.physics import beamstop_mask, build_input, calibrate_flux, \
        calibrate_input_norm, forward_measure
    from src.procedural import make_procedural
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(name, radius=None, poisson=True):
        cfg = Cfg.load(f"configs/{name}.yaml")
        cfg.add_poisson = poisson
        obj = make_procedural(32, cfg, seed=5, kinds=tuple(cfg.proc_kinds),
                              weights=tuple(cfg.proc_weights),
                              target_support=cfg.match_support_to,
                              contrast_gamma=cfg.proc_contrast_gamma)
        cfg.ref_energy = calibrate_flux(obj)
        obj = obj.to(dev)
        bs = beamstop_mask(cfg, radius=radius, device=dev)
        calibrate_input_norm(obj, bs, cfg)
        return cfg, obj, bs, forward_measure(obj, bs, cfg)

    # fft:無雜訊、無遮罩、給真實相位 -> 固定轉換應完全還原物體
    cfg, obj, bs, counts = setup(RUNS["fft"], radius=-1, poisson=False)
    m = build_model(cfg).to(dev)
    x = build_input(obj, counts, bs, cfg)
    psi = m.field(x, x[:, 1:2], x[:, 2:3], torch.zeros_like(x[:, 1:2]))[:, 0]
    err = float((psi - torch.polar(obj[:, 0], obj[:, 1])).abs().max())
    check("fft:量測振幅 × 真實相位 -> 固定反轉換,完全還原物體", err < 1e-3, f"最大誤差 {err:.1e}")

    for a in ["unet", "fft", "attn"]:
        cfg, obj, bs, counts = setup(RUNS[a])
        m = build_model(cfg).to(dev)
        y = m(build_input(obj, counts, bs, cfg))
        y.sum().backward()
        good = (y.shape == obj.shape and bool(torch.isfinite(y).all())
                and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                        for p in m.parameters()))
        check(f"arch={a}:前向 / 反向正常", good,
              f"參數量 {sum(p.numel() for p in m.parameters()):,}")
    print(f"\n     裝置:{dev}")


# ============================================================================
# 跑完後
# ============================================================================
def convergence():
    print("\n" + "=" * 70)
    print("收斂(末 5 epoch 平均每步降幅佔 loss 比例;須相近才可比較)")
    print("=" * 70)
    for a, name in RUNS.items():
        fr, sec = [], []
        for s in SEEDS:
            h = json.load(open(RUN_ROOT / f"{name}_s{s}" / "history.json"))
            tot = [e["total"] for e in h]
            fr.append((tot[-6] - tot[-1]) / 5 / tot[-1])
            sec.append(np.mean([e["sec"] for e in h]))
        print(f"  {a:<5} {100 * np.mean(fr):.2f}%   單 epoch {np.mean(sec):.1f} s   最終 loss {tot[-1]:.4f}")


def direct_inverse(cfg, home, counts, dev):
    """不學任何東西的 oracle:量測振幅 × 真實繞射相位 -> 反傅立葉轉換 -> 套 support 方框。"""
    from src.physics import support_mask
    scale = cfg.photons_per_pix * cfg.canvas ** 2 / cfg.ref_energy
    A = torch.sqrt(counts.clamp_min(0) / scale)
    E_true = torch.fft.fftshift(torch.fft.fft2(torch.polar(home[:, 0], home[:, 1]),
                                               norm="ortho"), dim=(-2, -1))
    psi = torch.fft.ifft2(torch.fft.ifftshift(A * torch.exp(1j * torch.angle(E_true)),
                                              dim=(-2, -1)), norm="ortho")
    sup = support_mask(cfg, device=dev)
    return torch.stack([psi.abs().clamp(0, 1) * sup,
                        torch.angle(psi).clamp(0, cfg.phase_max) * sup], dim=1)


def evaluate_all():
    from realign_eval import load, net_out, proto_labels, score_all
    from src.data import generalization_suites
    from src.physics import beamstop_mask, forward_measure

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {a: [] for a in list(RUNS) + ["direct"]}
    meta = {a: {} for a in RUNS}
    kinds = None
    for a, name in RUNS.items():
        for s in SEEDS:
            run_dir = RUN_ROOT / f"{name}_s{s}"
            cfg, model = load(run_dir, dev)
            assert cfg.oracle_phase and getattr(cfg, "arch", "unet") == a
            bs = beamstop_mask(cfg, device=dev)
            home = generalization_suites(cfg, cfg.eval_n)["procedural"].to(dev)
            torch.manual_seed(cfg.test_seed + s)
            counts = forward_measure(home, bs, cfg)
            pred = net_out(model, home, counts, bs, cfg)
            labels, kinds = proto_labels(cfg.eval_n, cfg, home)

            def by_proto(p):
                rec = {"all": score_all(model, home, bs, cfg, p)[0]}
                for i, k in enumerate(kinds):
                    msk = (labels == i).to(dev)
                    rec[k] = score_all(model, home[msk], bs, cfg, p[msk])[0]
                return rec

            rec = by_proto(pred)
            mt = json.load(open(run_dir / "metrics.json"))
            if abs(rec["all"]["raw"]["frc_gain"] - mt["test"]["frc_gain"]) > SANITY_TOL:
                raise SystemExit(f"{run_dir.name}:未對齊 {rec['all']['raw']['frc_gain']:+.4f} "
                                 f"與 metrics.json {mt['test']['frc_gain']:+.4f} 對不上,停下來查")
            res[a].append(rec)
            meta[a].setdefault("speed", []).append(mt.get("speed_ms_per_sample", {}))
            meta[a]["params"] = sum(p.numel() for p in model.parameters())
            if a == "unet":
                res["direct"].append(by_proto(direct_inverse(cfg, home, counts, dev)))
            print(f"  [{run_dir.name}] 完成(未對齊 {rec['all']['raw']['frc_gain']:+.4f},"
                  f"metrics.json {mt['test']['frc_gain']:+.4f})", flush=True)
    return res, meta, kinds


def ms(v):
    a = np.array(v, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def report(res, meta, kinds):
    hio = None
    p = RUN_ROOT / "oracle_proto.json"
    if p.exists():
        d = json.load(open(p)).get("理想 bs=0 ph=1e5")
        if d:
            hio = {k: np.mean([r[k]["aligned"]["frc_gain"] for r in d["hio"]]) for k in ["all"] + kinds}

    cols = [("unet", "U-Net"), ("fft", "內建轉換"), ("attn", "attention"), ("direct", "直接反轉")]
    print("\n" + "=" * 96)
    print("oracle 各原型 FRC gain(對齊後,mean ± std,3 seeds)")
    print("=" * 96)
    print(f"  {'原型':<10}" + "".join(f"{c[1]:>18}" for c in cols) + f"{'純 HIO 5000':>14}")
    for k in ["all"] + kinds:
        row = f"  {('全部' if k == 'all' else k):<10}"
        for a, _ in cols:
            m, s = ms([r[k]["aligned"]["frc_gain"] for r in res[a]])
            row += f"{m:>+11.4f}±{s:.4f}"
        row += f"{hio[k]:>+14.4f}" if hio else f"{'—':>14}"
        print(row)

    print("\n  材料 MAE(對齊後;平庸基準約 0.30)")
    for k in ["all"] + kinds:
        row = f"  {('全部' if k == 'all' else k):<10}"
        for a, _ in cols:
            row += f"{np.mean([r[k]['aligned']['material_mae'] for r in res[a]]):>18.4f}"
        print(row)

    print("\n  與 U-Net 的差(對齊後 FRC gain,逐 seed 相減)")
    for a in ["fft", "attn"]:
        row = f"  {a:<6}"
        for k in kinds:
            d = [x[k]["aligned"]["frc_gain"] - u[k]["aligned"]["frc_gain"]
                 for x, u in zip(res[a], res["unet"])]
            m, s = ms(d)
            row += f"  {k} {m:+.4f}±{s:.4f}"
        print(row)

    print("\n  參數量與推論速度(metrics.json,ms/sample)")
    for a in RUNS:
        sp = meta[a]["speed"]
        b1 = np.mean([x.get("batch1", np.nan) for x in sp])
        b64 = np.mean([x.get("batch64", np.nan) for x in sp])
        print(f"  {a:<6} 參數 {meta[a]['params']:>10,}   batch 1: {b1:.3f}   batch 64: {b64:.4f}")
    print("\n判讀準則見 實驗設計_1b6 §十九(結果出來前已寫定)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-only", action="store_true")
    a = ap.parse_args()
    configs_and_unit_test()
    if a.configs_only:
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過"))
        sys.exit(0 if ok_all else 1)
    convergence()
    res, meta, kinds = evaluate_all()
    report(res, meta, kinds)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))


if __name__ == "__main__":
    main()
