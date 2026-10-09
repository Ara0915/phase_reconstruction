#!/usr/bin/env python
"""第二次報告用:新測試場(scan_5d,物體 seed 90,000,000)上 DEF2 B(10 萬個場訓練)的重建圖,
對照「同時間的最強迭代法」(batch 64 / 512,設定取自 scan5d.json 的判讀,不重新挑選)。只讀結果檔、不改任何東西。

輸出到 ~/cdi/figs_viz5d/:
  P9_recon_B_vs_iter.png         B 誤差的第 10 / 50 / 90 百分位樣本(seed 0 的網路):真值 / B / 迭代法 b64 / 迭代法 b512
  P9_recon_B_vs_iter_random.png  2 個隨機樣本(numpy seed 2027,只依樣本編號抽)
  (相位圖,振幅 < 0.05 處遮掉;下一列為 |誤差|;範圍 = R0 的外框 + 4 px,R0 外調淡;只對齊一個整體相位,同評分)

用法(計算節點,約 1–2 分鐘):  python viz_5d.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5d as d5                                                 # noqa: E402

s5, s5b, s5c = d5.s5, d5.s5b, d5.s5c
PROBE, GROUP, SEED = "DEF2", "B", 0
RES = d5.RUN_ROOT / ("scan5d.json" if not d5.QUICK else "scan5d_quick.json")
OUT = Path("figs_viz5d")
PCTS = [10, 50, 90]
RAND_SEED, N_RANDOM = 2027, 2
MARGIN, AMP_MIN, TOL = 4, 0.05, 1e-4


def per_sample(est, O, M):
    m = M.to(est.real.dtype)
    a, t = est * m, O * m
    E = (t.abs() ** 2).sum((1, 2))
    cross = (a * t.conj()).sum((1, 2)).abs()
    v = ((a.abs() ** 2).sum((1, 2)) + E - 2 * cross) / E.clamp_min(1e-12)
    return torch.where(E > 1e-12, v, torch.full_like(v, float("nan")))


@torch.no_grad()
def iterative(conf, it, d, dev):
    import probe_4_2b as pb
    cfg, pr, cp, bs, counts, st, n, F = (d[k] for k in ("cfg", "pr", "cp", "bs", "counts", "st", "n", "F"))
    ones = torch.ones(d["W"], d["W"], device=dev)
    need_k = conf == "K-HIO" or conf.split("|")[1] in s5b.KHIO_N
    k_out = None
    if need_k:
        kit = sorted(set(s5b.K_ITERS) | set(s5b.KHIO_N.values()))
        init = s5.k_init(counts[s5b.CENTER], cp, d["init"], dev)
        k_out = pb.hio_probe(init, counts[s5b.CENTER], bs, cp, kit, "K", pr, ones)
    if conf == "K-HIO":
        return s5.k_to_field(k_out[it], pr, cfg, st[s5b.CENTER], F)
    meth, kind = conf.split("|")
    O0 = s5b.make_init(kind, n, F, cfg, cp, pr, bs, counts, st, d["init"], dev, k_out=k_out)
    return O0 if it == 0 else s5b.run_method(meth, counts, bs, cp, pr, st, [it], O0, d["init"])[it]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not d5.QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    if not RES.exists():
        raise SystemExit(f"❌ 找不到 {RES}")
    R = json.load(open(RES))
    ver = R["verdict"][GROUP]
    opp = {B: (ver[B]["opp"][0], int(ver[B]["opp"][1]), float(np.mean(ver[B]["opp"][2]))) for B in ("64", "512")}
    tnet = {B: ver[B]["t_net"] for B in ("64", "512")}
    d = d5.setup_seed(SEED, d5.new_seeds, dev)
    cfg, pr, cp, bs, counts, O, R0, F = (d[k] for k in ("cfg", "pr", "cp", "bs", "counts", "O", "R0", "F"))
    net, norm, _ = s5c.load_net(GROUP, PROBE, SEED, d5.N_TRAIN, cfg, pr, dev)
    if net is None:
        raise SystemExit(f"❌ 缺少模型 {GROUP} {PROBE} seed {SEED}")
    with torch.no_grad():
        pred = s5c.predict(net, counts, bs, norm, GROUP, cp)
    est = {"B": s5c.to_field(torch.polar(pred[:, 0], pred[:, 1]), cfg, F)}
    for B, (conf, it, _) in opp.items():
        est[f"iter_b{B}"] = iterative(conf, it, d, dev)
    # 核對:與 scan5d.json 的數字一致(seed 0)
    ok = True
    vB = s5.field_metrics(est["B"], O, R0)["nerr_ph"]
    rB = R["nets"][PROBE][GROUP][SEED]["net"]["nerr_ph"]
    print(f"{'✅' if abs(vB - rB) < TOL else '❌'} B(seed 0)重算 {vB:.5f} vs 結果檔 {rB:.5f}")
    ok &= abs(vB - rB) < TOL
    for B, (conf, it, m3) in opp.items():
        v = s5.field_metrics(est[f"iter_b{B}"], O, R0)["nerr_ph"]
        r = R["envelope"][PROBE][SEED]["rec"][conf][str(it)]["nerr_ph"] if conf != "K-HIO" or str(it) in \
            R["envelope"][PROBE][SEED]["rec"][conf] else float("nan")
        good = abs(v - r) < TOL
        ok &= good
        print(f"{'✅' if good else '❌'} 迭代法 b{B}({conf.replace('|', '+')} × {it})重算 {v:.5f} vs 結果檔 {r:.5f}"
              f"(3 seeds 平均 {m3:.4f})")
    err = {k: per_sample(e, O, R0).cpu().numpy() for k, e in est.items()}
    print(f"  網路時間:b64 {tnet['64']:.4f} ms、b512 {tnet['512']:.4f} ms")
    print(f"  seed 0 的平均誤差:" + "、".join(f"{k} {np.nanmean(v):.4f}" for k, v in err.items()))
    for ref in ("iter_b64", "iter_b512"):
        print(f"  B 逐樣本比 {ref} 準的比例:{100 * np.nanmean(err['B'] < err[ref]):.1f}%")

    plt = s5._plt()
    OUT.mkdir(exist_ok=True)
    Oc, R0c = O.cpu(), R0.cpu()
    ys, xs = torch.where(R0c)
    y0, y1 = max(0, int(ys.min()) - MARGIN), min(F, int(ys.max()) + 1 + MARGIN)
    x0, x1 = max(0, int(xs.min()) - MARGIN), min(F, int(xs.max()) + 1 + MARGIN)
    Rc = R0c[y0:y1, x0:x1].numpy()
    pm = cfg.phase_max
    E = {k: e.cpu() for k, e in est.items()}
    cols = [("truth", None),
            ("B", f"network B\n({tnet['64']:.3f} / {tnet['512']:.3f} ms)"),
            ("iter_b64", f"strongest iterative at B's time, batch 64\n{opp['64'][0].replace('|', '+')} x{opp['64'][1]}"),
            ("iter_b512", f"strongest iterative at B's time, batch 512\n{opp['512'][0].replace('|', '+')} x{opp['512'][1]}")]

    def align(e, i):
        m = R0c.to(e.real.dtype)
        c = ((Oc[i] * m) * (e[i] * m).conj()).sum()
        return e[i] * torch.exp(1j * torch.angle(c))

    def draw(ids, name, what):
        fig, axs = plt.subplots(2 * len(ids), len(cols), figsize=(3.1 * len(cols) + 0.8, 3.0 * len(ids) * 2),
                                squeeze=False, constrained_layout=True)
        for r, i in enumerate(ids):
            t = Oc[i, y0:y1, x0:x1].numpy()
            for c, (k, lab) in enumerate(cols):
                z = t if k == "truth" else align(E[k], i)[y0:y1, x0:x1].numpy()
                ax = axs[2 * r, c]
                hp = ax.imshow(np.where(np.abs(t) > AMP_MIN, np.angle(z), np.nan), cmap="viridis", vmin=0, vmax=pm,
                               interpolation="nearest")
                ax.imshow(np.where(Rc, np.nan, 0.0), cmap="Greys", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ttl = "truth" if k == "truth" else f"nerr {err[k][i]:.3f}"
                ax.set_title(((lab + "\n") if (r == 0 and lab) else "") + ttl, fontsize=8)
                ax2 = axs[2 * r + 1, c]
                if k == "truth":
                    ax2.axis("off")
                else:
                    he = ax2.imshow(np.where(Rc, np.abs(z - t), np.nan), cmap="magma", vmin=0, vmax=0.5,
                                    interpolation="nearest")
                    ax2.set_xticks([])
                    ax2.set_yticks([])
            axs[2 * r, 0].set_ylabel(f"sample #{i}\nphase", fontsize=8)
            axs[2 * r + 1, 1].set_ylabel("|error|", fontsize=8)
        fig.colorbar(hp, ax=list(axs[0::2, -1]), fraction=0.04, label="phase (rad)")
        fig.colorbar(he, ax=list(axs[1::2, -1]), fraction=0.04, label="|error|")
        fig.suptitle(f"New test fields (DEF2, 3x3 scan, R0 = centre probe footprint; outside dimmed): {what}. "
                     "Seed-0 network; one global phase aligned (same as the score)", fontsize=9)
        p = OUT / name
        fig.savefig(p, dpi=160, facecolor="white")
        plt.close(fig)
        print(f"  圖:{p}")

    eB = err["B"]
    valid = np.where(np.isfinite(eB))[0]
    order = valid[np.argsort(eB[valid], kind="stable")]
    picks = [int(order[min(len(order) - 1, int(round(p / 100 * (len(order) - 1))))]) for p in PCTS]
    draw(picks, "P9_recon_B_vs_iter.png", "10th / 50th / 90th percentile of B's error")
    rng = np.random.default_rng(RAND_SEED)
    rnd = sorted(int(i) for i in rng.choice(O.shape[0], size=N_RANDOM, replace=False))
    draw(rnd, "P9_recon_B_vs_iter_random.png", f"random samples (numpy seed {RAND_SEED})")
    print("\n" + ("✅ 數字核對通過" if ok else "❌ 核對未通過(圖先不要用),把輸出貼給 Claude"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
