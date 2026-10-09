#!/usr/bin/env python
"""第 2 步(相位未知):U-Net vs 內建轉換(單次)vs 展開 HIO —— 送件前檢查與結果比較。

三組皆為現況條件、歧異性不變 loss 開啟,訓練設定逐欄相同,只差 arch:
    amb_base          arch=unet(已有)
    arch_fft_amb      arch=fft     偵測器端估相位 -> 固定反轉換 -> 實空間 U-Net
    arch_unroll_amb   arch=unroll  展開 HIO 5 輪,每輪 = 固定物理 + 學習的修正

參考線(不需訓練):純 HIO 5000 次(取自 realign_eval.json,同條件、同批物體、對齊後)。

比較一律用對齊後的分數(歧異性不變 loss 允許輸出平移或翻轉版本)。

用法(需在計算節點執行;需 realign_eval.py、ambiguity_check.py 在同一資料夾):
    python check_arch2.py --configs-only   # 送件前:變因控制 + 架構單元測試
    python check_arch2.py                  # 跑完後
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
RUNS = {"unet": "amb_base", "fft": "arch_fft_amb", "unroll": "arch_unroll_amb"}
LABEL = {"unet": "U-Net", "fft": "內建轉換(單次)", "unroll": "展開 HIO"}
SEEDS = [0, 1, 2]
UNSEEN = ["mnist_test", "fashion_mnist", "random_shapes", "random_texture"]
SHORT = {"mnist_test": "mnist", "fashion_mnist": "fashion",
         "random_shapes": "shapes", "random_texture": "texture"}
SANITY_TOL = 0.01
FIG_DIR = Path("figs_arch2")
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
    from src.hio import hio
    from src.physics import beamstop_mask, build_input, calibrate_flux, \
        calibrate_input_norm, forward_measure, support_mask
    from src.procedural import make_procedural

    print("=" * 70)
    print("變因控制")
    print("=" * 70)
    base = Cfg.load(f"configs/{RUNS['unet']}.yaml").to_dict()
    for a in ["fft", "unroll"]:
        c = Cfg.load(f"configs/{RUNS[a]}.yaml").to_dict()
        diff = sorted(k for k in base if base[k] != c[k])
        check(f"{RUNS['unet']} vs {RUNS[a]} 只差 arch", diff == ["arch"] and c["arch"] == a,
              f"實際差異: {diff}")

    print("\n" + "=" * 70)
    print("架構單元測試")
    print("=" * 70)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(radius=None, poisson=True):
        cfg = Cfg.load(f"configs/{RUNS['unroll']}.yaml")
        cfg.add_poisson = poisson
        obj = make_procedural(32, cfg, seed=7, kinds=tuple(cfg.proc_kinds),
                              weights=tuple(cfg.proc_weights),
                              target_support=cfg.match_support_to,
                              contrast_gamma=cfg.proc_contrast_gamma)
        cfg.ref_energy = calibrate_flux(obj)
        obj = obj.to(dev)
        bs = beamstop_mask(cfg, radius=radius, device=dev)
        calibrate_input_norm(obj, bs, cfg)
        return cfg, obj, bs

    # (1) 未訓練的展開(修正網路輸出為 0)= hio.py 的同樣迭代數
    cfg, obj, bs = setup()
    counts = forward_measure(obj, bs, cfg)
    m = build_model(cfg).to(dev)
    x = build_input(obj, counts, bs, cfg)
    sup = support_mask(cfg, device=dev)
    g_ = torch.Generator().manual_seed(1)
    init = torch.stack([torch.rand(32, 64, 64, generator=g_).to(dev) * sup,
                        torch.rand(32, 64, 64, generator=g_).to(dev) * cfg.phase_max * sup], 1)
    with torch.no_grad():
        g = m.rounds(torch.polar(init[:, 0], init[:, 1]), m.measured_amp(x), x[:, -1])
        ref = hio(init, counts, bs, cfg, n_iter=cfg.unroll_t, beta=cfg.hio_beta)
    mine = torch.polar(g.abs().clamp(0, 1) * sup, torch.angle(g).clamp(0, cfg.phase_max) * sup)
    err = float((mine - torch.polar(ref[:, 0], ref[:, 1])).abs().max())
    check(f"未訓練的展開 {cfg.unroll_t} 輪 = hio.py 的 {cfg.unroll_t} 次迭代", err < 1e-3,
          f"最大差 {err:.1e}")

    # (2) 真值為固定點(無雜訊、無遮罩;相位縮到邊界內,見 §21.1 的數值註記)
    cfg, obj, bs = setup(radius=-1, poisson=False)
    obj = obj.clone()
    obj[:, 1] *= 0.95
    counts = forward_measure(obj, bs, cfg)
    m = build_model(cfg).to(dev)
    x = build_input(obj, counts, bs, cfg)
    t = torch.polar(obj[:, 0], obj[:, 1])
    with torch.no_grad():
        g = m.rounds(t, m.measured_amp(x), x[:, -1])
    err = float((g - t).abs().max())
    check("真值是展開的固定點(物理部分無誤)", err < 1e-3, f"最大差 {err:.1e}")

    # (3) 前向 / 反向
    cfg, obj, bs = setup()
    counts = forward_measure(obj, bs, cfg)
    for a in ["unet", "fft", "unroll"]:
        cfg.arch = a
        m = build_model(cfg).to(dev)
        y = m(build_input(obj, counts, bs, cfg))
        (y ** 2).sum().backward()
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
        fr, sec, fin, tw = [], [], [], []
        for s in SEEDS:
            h = json.load(open(RUN_ROOT / f"{name}_s{s}" / "history.json"))
            tot = [e["total"] for e in h]
            fr.append((tot[-6] - tot[-1]) / 5 / tot[-1])
            sec.append(np.mean([e["sec"] for e in h]))
            fin.append(tot[-1])
            tw.append(h[-1].get("amb_twin", 0.0))
        print(f"  {a:<7} {100 * np.mean(fr):.2f}%   最終 loss {np.mean(fin):.4f}   "
              f"單 epoch {np.mean(sec):.1f} s   末 epoch 選翻轉 {np.mean(tw):.2f}")


def evaluate_all():
    from ambiguity_check import centro_sym
    from realign_eval import load, net_out, proto_labels, score_all
    from src.data import generalization_suites
    from src.physics import beamstop_mask, forward_measure, support_mask

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res, meta, figs, kinds = {a: [] for a in RUNS}, {a: {} for a in RUNS}, {}, None
    for a, name in RUNS.items():
        for s in SEEDS:
            run_dir = RUN_ROOT / f"{name}_s{s}"
            cfg, model = load(run_dir, dev)
            assert getattr(cfg, "arch", "unet") == a and cfg.amb_invariant and not cfg.oracle_phase
            bs = beamstop_mask(cfg, device=dev)
            home = generalization_suites(cfg, cfg.eval_n)["procedural"].to(dev)
            gen = generalization_suites(cfg, cfg.gen_n)
            torch.manual_seed(cfg.test_seed + s)
            counts = forward_measure(home, bs, cfg)
            pred = net_out(model, home, counts, bs, cfg)
            labels, kinds = proto_labels(cfg.eval_n, cfg, home)
            whole, al = score_all(model, home, bs, cfg, pred)
            mt = json.load(open(run_dir / "metrics.json"))
            if abs(whole["raw"]["frc_gain"] - mt["test"]["frc_gain"]) > SANITY_TOL:
                raise SystemExit(f"{run_dir.name}:未對齊 {whole['raw']['frc_gain']:+.4f} "
                                 f"與 metrics.json {mt['test']['frc_gain']:+.4f} 對不上,停下來查")
            rec = {"all": whole,
                   "sym": centro_sym(al[:, 0], support_mask(cfg, device=dev))}
            for i, k in enumerate(kinds):
                msk = (labels == i).to(dev)
                rec[k] = score_all(model, home[msk], bs, cfg, pred[msk])[0]
            rec["unseen"] = {}
            for j, u in enumerate(UNSEEN):
                objs = gen[u].to(dev)
                torch.manual_seed(cfg.test_seed + s + 1000 * (j + 1))
                c = forward_measure(objs, bs, cfg)
                rec["unseen"][u] = score_all(model, objs, bs, cfg,
                                             net_out(model, objs, c, bs, cfg))[0]
            res[a].append(rec)
            meta[a].setdefault("speed", []).append(mt.get("speed_ms_per_sample", {}))
            meta[a]["params"] = sum(p.numel() for p in model.parameters())
            if s == SEEDS[0]:
                figs[a] = (home.cpu(), labels, kinds, al.cpu())
            print(f"  [{run_dir.name}] 完成(未對齊 {whole['raw']['frc_gain']:+.4f},"
                  f"metrics.json {mt['test']['frc_gain']:+.4f})", flush=True)
    return res, meta, figs, kinds


def ms(v):
    a = np.array(v, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def hio_ref(kinds):
    """純 HIO 5000(現況條件,對齊後):取自 realign_eval.json。"""
    p = RUN_ROOT / "realign_eval.json"
    if not p.exists():
        return None
    S = json.load(open(p))["proc"]["seeds"]
    out = {"all": np.mean([s["home"]["random"]["5000"]["aligned"]["frc_gain"] for s in S])}
    for k in kinds:
        out[k] = np.mean([s["proto"][k]["random_5000"]["aligned"]["frc_gain"] for s in S])
    un = {}
    for u in UNSEEN:
        un[u] = (np.mean([s["unseen"][u]["random"]["5000"]["aligned"]["amp_psnr"] for s in S])
                 - np.mean([s["unseen"][u]["random"]["5000"]["aligned"]["amp_psnr_trivial"] for s in S]))
    out["unseen"] = un
    return out


def report(res, meta, kinds):
    ref = hio_ref(kinds)
    print("\n" + "=" * 92)
    print("訓練分布 FRC gain(對齊後,mean ± std,3 seeds)")
    print("=" * 92)
    print(f"  {'原型':<10}" + "".join(f"{LABEL[a]:>20}" for a in RUNS) + f"{'純 HIO 5000':>14}")
    for k in ["all"] + kinds:
        row = f"  {('全部' if k == 'all' else k):<10}"
        for a in RUNS:
            m, s = ms([r[k]["aligned"]["frc_gain"] for r in res[a]])
            row += f"{m:>+13.4f}±{s:.4f}"
        row += f"{ref[k]:>+14.4f}" if ref else f"{'—':>14}"
        print(row)

    print("\n  材料 MAE(對齊後;平庸基準約 0.30;純 HIO 5000 約 0.106)")
    for k in ["all"] + kinds:
        row = f"  {('全部' if k == 'all' else k):<10}"
        for a in RUNS:
            row += f"{np.mean([r[k]['aligned']['material_mae'] for r in res[a]]):>20.4f}"
        print(row)

    print("\n  與 U-Net 的差(對齊後 FRC gain,逐 seed 相減)")
    for a in ["fft", "unroll"]:
        row = f"  {a:<7}"
        for k in ["all"] + kinds:
            d = [x[k]["aligned"]["frc_gain"] - u[k]["aligned"]["frc_gain"]
                 for x, u in zip(res[a], res["unet"])]
            m, s = ms(d)
            row += f"  {('全部' if k == 'all' else k)} {m:+.4f}±{s:.4f}"
        print(row)

    print(f"\n  中心對稱度(真值約 0.075):" + "   ".join(
        f"{a} {np.mean([r['sym'] for r in res[a]]):.3f}" for a in RUNS))

    print("\n" + "=" * 92)
    print("未見分布:相對增益 = 對齊後 PSNR − 該分布平庸基準(dB)")
    print("=" * 92)
    rows = [(LABEL[a], {u: np.mean([r["unseen"][u]["aligned"]["amp_psnr"] for r in res[a]])
                        - np.mean([r["unseen"][u]["aligned"]["amp_psnr_trivial"] for r in res[a]])
                        for u in UNSEEN}) for a in RUNS]
    if ref:
        rows.append(("純 HIO 5000", ref["unseen"]))
    for lab, v in rows:
        win = sum(x > 0 for x in v.values())
        print(f"  {lab:<16}" + "".join(f"{SHORT[u]} {v[u]:+6.2f}   " for u in UNSEEN) + f"贏 {win}/4")

    print("\n  參數量與推論速度(metrics.json,ms/sample)")
    for a in RUNS:
        sp = meta[a]["speed"]
        b1 = np.mean([x.get("batch1", np.nan) for x in sp])
        b64 = np.mean([x.get("batch64", np.nan) for x in sp])
        print(f"  {a:<7} 參數 {meta[a]['params']:>10,}   batch 1: {b1:.3f}   batch 64: {b64:.4f}")
    print("  (純 HIO 5000 次約 3.8 ms/sample)")
    print("\n判讀準則見 實驗設計_1b6 §二十一(結果出來前已寫定)")


def make_figure(figs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ 無 matplotlib,略過圖")
        return
    home, labels, kinds, _ = figs["unet"]
    rows = []
    for i, k in enumerate(kinds):
        rows += [(k, j) for j in (labels == i).nonzero()[:, 0][:2].tolist()]
    cols = [("GT", None)] + [(f"{a} (aligned)", a) for a in RUNS]
    FIG_DIR.mkdir(exist_ok=True)
    sl = slice(14, 50)
    for ch, cname in [(0, "amplitude"), (1, "phase")]:
        fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.2 * len(cols), 2.2 * len(rows)))
        for r, (k, j) in enumerate(rows):
            for c, (title, key) in enumerate(cols):
                img = home[j] if key is None else figs[key][3][j]
                x = img[ch].clone()
                if ch == 1:
                    x[img[0] < 0.05] = float("nan")
                a = ax[r, c]
                a.imshow(x[sl, sl].numpy(), cmap="gray" if ch == 0 else "viridis",
                         vmin=0, vmax=(1.0 if ch == 0 else float(np.pi / 2)))
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(title, fontsize=9)
                if c == 0:
                    a.set_ylabel(f"{k} #{j}", fontsize=9)
        fig.suptitle(f"Phase-unknown networks ({cname}), seed 0, current conditions", fontsize=10)
        fig.tight_layout()
        p = FIG_DIR / f"arch2_{cname}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  圖:{p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-only", action="store_true")
    a = ap.parse_args()
    configs_and_unit_test()
    if a.configs_only:
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過"))
        sys.exit(0 if ok_all else 1)
    convergence()
    res, meta, figs, kinds = evaluate_all()
    report(res, meta, kinds)
    make_figure(figs)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))


if __name__ == "__main__":
    main()
