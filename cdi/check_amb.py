#!/usr/bin/env python
"""平凡歧異性不變的訓練目標(amb_base)—— 送件前檢查與結果比較。

問題:網路的模糊(polygon / bandpass 上 ≈ 平庸基準)是否來自「猜不到位置與方向」?
做法:amb_base 與 ideal_base 只差 amb_invariant(loss 先把真值對齊到最接近的等價解)。
比較一律用「對齊後」的分數 —— amb_base 的網路本來就被允許輸出平移或翻轉版本。

用法(需在計算節點執行):
    python check_amb.py --configs-only   # 送件前:變因控制 + loss 單元測試
    python check_amb.py                  # 跑完後:收斂、主結果、各原型、未見分布、圖
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Cfg                                    # noqa: E402
from src.losses import _roll_batch, align_target, build_loss  # noqa: E402

RUN_ROOT = Path("/work/elviss0915/runs")
BASE, AMB = "ideal_base", "amb_base"
SEEDS = [0, 1, 2]
HIO_ITERS = [200, 1000]
UNSEEN = ["mnist_test", "fashion_mnist", "random_shapes", "random_texture"]
SHORT = {"mnist_test": "mnist", "fashion_mnist": "fashion",
         "random_shapes": "shapes", "random_texture": "texture"}
SANITY_TOL = 0.01
FIG_DIR = Path("figs_amb")

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
    a = Cfg.load(f"configs/{BASE}.yaml").to_dict()
    b = Cfg.load(f"configs/{AMB}.yaml").to_dict()
    diff = sorted(k for k in a if a[k] != b[k])
    check(f"{BASE} vs {AMB} 只差 amb_* 欄位",
          set(diff) <= {"amb_invariant", "amb_max_shift"} and "amb_invariant" in diff,
          f"實際差異: {diff}")
    check(f"{AMB} 的 amb_invariant = True,{BASE} 為 False",
          b["amb_invariant"] is True and a["amb_invariant"] is False)

    print("\n" + "=" * 70)
    print("loss 單元測試")
    print("=" * 70)
    from src.physics import beamstop_mask, calibrate_flux, forward_measure
    from src.procedural import make_procedural
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Cfg.load(f"configs/{AMB}.yaml")
    t = make_procedural(64, cfg, seed=3, kinds=tuple(cfg.proc_kinds),
                        weights=tuple(cfg.proc_weights),
                        target_support=cfg.match_support_to,
                        contrast_gamma=cfg.proc_contrast_gamma).to(dev)
    cfg.ref_energy = calibrate_flux(t.cpu())
    bs = beamstop_mask(cfg, device=dev)
    counts = forward_measure(t, bs, cfg)

    # 造一批已知的等價解:隨機平移(±amb_max_shift)+ 一半做共軛翻轉加 φ_max
    g = torch.Generator().manual_seed(0)
    k = cfg.amb_max_shift
    sy = torch.randint(-k, k + 1, (64,), generator=g).to(dev)
    sx = torch.randint(-k, k + 1, (64,), generator=g).to(dev)
    tw = (torch.rand(64, generator=g) < 0.5).to(dev)
    A, P = t[:, 0], t[:, 1]
    A2 = torch.where(tw[:, None, None], torch.flip(A, (-2, -1)), A)
    P2 = torch.where(tw[:, None, None], cfg.phase_max - torch.flip(P, (-2, -1)), P)
    eq = torch.stack([_roll_batch(A2, sy, sx), _roll_batch(P2, sy, sx)], 1)

    fa = torch.fft.fft2(torch.polar(A, P)).abs()
    fb = torch.fft.fft2(torch.polar(eq[:, 0], eq[:, 1])).abs()
    check("造出的版本確實與真值的繞射強度相同",
          float((fa - fb).abs().max() / fa.max()) < 1e-5)
    al, ftw, fmv = align_target(eq, t, cfg)
    err = float((torch.polar(al[:, 0], al[:, 1]) - torch.polar(eq[:, 0], eq[:, 1])).abs().max())
    check("align_target 完全還原已知的平移 + 翻轉", err < 1e-4,
          f"誤差 {err:.1e};選翻轉 {ftw:.2f}(實際 {float(tw.float().mean()):.2f})")
    _, parts = build_loss(cfg, dev)(eq, t, counts, bs)
    check("等價解的振幅與相位 loss 為 0", parts["amp"] < 1e-5 and parts["phase"] < 1e-5,
          f"amp {parts['amp']:.1e}  phase {parts['phase']:.1e}")
    c0 = Cfg.load(f"configs/{BASE}.yaml")
    c0.ref_energy = cfg.ref_energy
    _, p0 = build_loss(c0, dev)(eq, t, counts, bs)
    check(f"{BASE}(原 loss)對同一批等價解給出大 loss", p0["amp"] + p0["phase"] > 0.05,
          f"amp {p0['amp']:.3f}  phase {p0['phase']:.3f}")
    print(f"\n     裝置:{dev}")


# ============================================================================
# 跑完後
# ============================================================================
def convergence():
    print("\n" + "=" * 70)
    print("收斂(末 5 epoch 平均每步降幅佔 loss 比例;配對設定須對齊才可比較)")
    print("=" * 70)
    res = {}
    for name in [BASE, AMB]:
        fr, tw, mv = [], [], []
        for s in SEEDS:
            h = json.load(open(RUN_ROOT / f"{name}_s{s}" / "history.json"))
            tot = [e["total"] for e in h]
            fr.append((tot[-6] - tot[-1]) / 5 / tot[-1])
            tw.append(h[-1].get("amb_twin", 0.0))
            mv.append(h[-1].get("amb_shift", 0.0))
        res[name] = np.mean(fr)
        extra = (f"   末 epoch 選翻轉 {np.mean(tw):.2f}、有平移 {np.mean(mv):.2f}"
                 if name == AMB else "")
        print(f"  {name:<12} {100 * np.mean(fr):.2f}%{extra}")
    return res


def evaluate_all():
    from ambiguity_check import centro_sym
    from realign_eval import load, net_out, proto_labels, score_all
    from src.data import generalization_suites
    from src.hio import hio
    from src.physics import beamstop_mask, forward_measure, support_mask

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out, figs = {}, {}
    for name in [BASE, AMB]:
        out[name] = []
        for s in SEEDS:
            run_dir = RUN_ROOT / f"{name}_s{s}"
            cfg, model = load(run_dir, dev)
            bs = beamstop_mask(cfg, device=dev)
            home = generalization_suites(cfg, cfg.eval_n)["procedural"].to(dev)
            gen = generalization_suites(cfg, cfg.gen_n)
            torch.manual_seed(cfg.test_seed + s)
            counts = forward_measure(home, bs, cfg)
            pred = net_out(model, home, counts, bs, cfg)
            r0, al0 = score_all(model, home, bs, cfg, pred)
            m = json.load(open(run_dir / "metrics.json"))["test"]["frc_gain"]
            if abs(r0["raw"]["frc_gain"] - m) > SANITY_TOL:
                raise SystemExit(f"{run_dir.name}:未對齊 {r0['raw']['frc_gain']:+.4f} "
                                 f"與 metrics.json {m:+.4f} 對不上,停下來查")
            rec = {"net": r0, "hio": {},
                   "sym": centro_sym(al0[:, 0], support_mask(cfg, device=dev)),
                   "sym_gt": centro_sym(home[:, 0], support_mask(cfg, device=dev))}
            for n_it in HIO_ITERS:
                rec["hio"][n_it], _ = score_all(
                    model, home, bs, cfg, hio(pred, counts, bs, cfg, n_iter=n_it,
                                              beta=cfg.hio_beta))
            labels, kinds = proto_labels(cfg.eval_n, cfg, home)
            rec["proto"] = {}
            for i, k in enumerate(kinds):
                msk = (labels == i).to(dev)
                rec["proto"][k], _ = score_all(model, home[msk], bs, cfg, pred[msk])
            rec["unseen"] = {}
            for j, u in enumerate(UNSEEN):
                objs = gen[u].to(dev)
                torch.manual_seed(cfg.test_seed + s + 1000 * (j + 1))
                c = forward_measure(objs, bs, cfg)
                rec["unseen"][u], _ = score_all(model, objs, bs, cfg,
                                                net_out(model, objs, c, bs, cfg))
            out[name].append(rec)
            if s == SEEDS[0]:
                figs[name] = (home.cpu(), labels, kinds, al0.cpu())
            print(f"  [{run_dir.name}] 完成(未對齊 {r0['raw']['frc_gain']:+.4f},"
                  f"metrics.json {m:+.4f})", flush=True)
    return out, figs


def ms(v):
    a = np.array(v, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def diff_z(a, b):
    (ma, sa), (mb, sb) = ms(a), ms(b)
    se = np.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b))
    return mb - ma, ((mb - ma) / se if se > 0 else float("inf"))


def report(out):
    B, M = out[BASE], out[AMB]

    def col(R, *path):
        vals = []
        for r in R:
            x = r
            for p in path:
                x = x[p]
            vals.append(x)
        return vals

    print("\n" + "=" * 86)
    print("主結果:網路本身(訓練分布 512 張,對齊後,3 seeds)")
    print("=" * 86)
    print(f"  {'指標':<22}{BASE:>18}{AMB:>18}{'差':>10}{'z':>7}")
    rows = [("FRC gain(對齊後)", ("net", "aligned", "frc_gain")),
            ("FRC gain(未對齊)", ("net", "raw", "frc_gain")),
            ("材料 MAE(對齊後)", ("net", "aligned", "material_mae")),
            ("PSNR(對齊後)", ("net", "aligned", "amp_psnr")),
            ("錯配對齊 FRC", ("net", "mismatch", "frc_gain")),
            ("中心對稱度", ("sym",))]
    for lab, path in rows:
        a, b = col(B, *path), col(M, *path)
        d, z = diff_z(a, b)
        print(f"  {lab:<22}{ms(a)[0]:>+12.4f}±{ms(a)[1]:.4f}{ms(b)[0]:>+12.4f}±{ms(b)[1]:.4f}"
              f"{d:>+10.4f}{z:>7.1f}")
    print(f"  (真值的中心對稱度 {np.mean(col(B, 'sym_gt')):.3f};"
          f"選翻轉比例 {BASE} {np.mean(col(B, 'net', 'twin_frac')):.2f} / "
          f"{AMB} {np.mean(col(M, 'net', 'twin_frac')):.2f})")

    print("\n" + "=" * 86)
    print("各原型:網路本身,對齊後 FRC gain / 材料 MAE")
    print("=" * 86)
    for k in B[0]["proto"]:
        a = col(B, "proto", k, "aligned", "frc_gain")
        b = col(M, "proto", k, "aligned", "frc_gain")
        d, z = diff_z(a, b)
        ma = np.mean(col(B, "proto", k, "aligned", "material_mae"))
        mb = np.mean(col(M, "proto", k, "aligned", "material_mae"))
        print(f"  {k:<10}{ms(a)[0]:>+9.4f} / {ma:.3f}   ->  {ms(b)[0]:>+9.4f} / {mb:.3f}"
              f"   差 {d:>+.4f}(z {z:.1f})")

    print("\n" + "=" * 86)
    print("網路 + HIO(對齊後 FRC gain / 材料 MAE)")
    print("=" * 86)
    for n_it in HIO_ITERS:
        a = col(B, "hio", n_it, "aligned", "frc_gain")
        b = col(M, "hio", n_it, "aligned", "frc_gain")
        d, z = diff_z(a, b)
        print(f"  {n_it:>5} 次  {ms(a)[0]:+.4f} / {np.mean(col(B, 'hio', n_it, 'aligned', 'material_mae')):.3f}"
              f"   ->  {ms(b)[0]:+.4f} / {np.mean(col(M, 'hio', n_it, 'aligned', 'material_mae')):.3f}"
              f"   差 {d:+.4f}(z {z:.1f})")

    print("\n" + "=" * 86)
    print("未見分布:網路本身,相對增益(對齊後 PSNR − 平庸基準,dB)")
    print("=" * 86)
    for name, R in [(BASE, B), (AMB, M)]:
        row, win = f"  {name:<14}", 0
        for u in UNSEEN:
            v = (np.mean(col(R, "unseen", u, "aligned", "amp_psnr"))
                 - np.mean(col(R, "unseen", u, "aligned", "amp_psnr_trivial")))
            row += f"{SHORT[u]} {v:+6.2f}   "
            win += v > 0
        print(row + f"贏 {win}/4")
    print("\n判讀準則見 實驗設計_1b6 §十五(結果出來前已寫定)")


def make_figure(figs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ 無 matplotlib,略過圖")
        return
    home, labels, kinds, _ = figs[BASE]
    rows = []
    for i, k in enumerate(kinds):
        rows += [(k, j) for j in (labels == i).nonzero()[:, 0][:2].tolist()]
    cols = [("GT", None), (f"{BASE} (aligned)", BASE), (f"{AMB} (aligned)", AMB)]
    FIG_DIR.mkdir(exist_ok=True)
    sl = slice(14, 50)
    for ch, cname in [(0, "amplitude"), (1, "phase")]:
        fig, ax = plt.subplots(len(rows), 3, figsize=(6.6, 2.2 * len(rows)))
        for r, (k, j) in enumerate(rows):
            for c, (title, key) in enumerate(cols):
                img = home[j] if key is None else figs[key][3][j]
                x = img[ch].clone()
                if ch == 1:                                   # 振幅 ≈ 0 處的相位無定義,遮掉
                    x[img[0] < 0.05] = float("nan")
                a = ax[r, c]
                a.imshow(x[sl, sl].numpy(), cmap="gray" if ch == 0 else "viridis",
                         vmin=0, vmax=(1.0 if ch == 0 else float(np.pi / 2)))
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(title, fontsize=9)
                if c == 0:
                    a.set_ylabel(f"{k} #{j}", fontsize=9)
        fig.suptitle(f"Network output ({cname}), seed 0, current conditions", fontsize=10)
        fig.tight_layout()
        p = FIG_DIR / f"amb_{cname}.png"
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
    out, figs = evaluate_all()
    report(out)
    make_figure(figs)
    print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))


if __name__ == "__main__":
    main()
