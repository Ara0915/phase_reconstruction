#!/usr/bin/env python
"""P6 的復原圖(描述用,不做判讀;協定 §18.8)。

在驗證場(seed 80,000,000)上,用 seed 0 的網路,畫出真值與各方法的復原結果:
  迭代法(同時間內最好的設定:P6B 在 batch 64 的時間、batch 512 的時間)、B、NAF-B、P6B;另畫 P6B 各級的變化。
樣本挑選規則(事先寫定,不挑圖):依 P6B 的逐樣本 nerr_ph 排序,取第 10、50、90 百分位的樣本(好 / 典型 / 差)。
顯示方式:與評分相同 —— 只對齊整體相位(R0 上);裁切 R0 周圍,R0 外調淡;相位只畫真值振幅 > 0.05 處(其餘為灰)。

用法(計算節點,約 1–3 分鐘;需 scan_5g.py 與 P6 的評估結果 scan5g_p6_raw.json):
    python scan_5g_viz.py
輸出:figs_scan5g/viz_overview.png(全景與物體類型)、viz_gallery.png(三種物體的樣子)、viz_p10 / p50 / p90.png、
      viz_stages.png、viz_hist.png 與 viz_stats.json;並印出依物體類型分組的誤差
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5g as g                                                  # noqa: E402  (也載入 scan_5 / 5b / 5c / 5d / 5e)

s5, s5b, s5c, s5d, s5e = g.s5, g.s5b, g.s5c, g.s5d, g.s5e
PROBE, NAF_B, QUICK = g.PROBE, g.NAF_B, g.QUICK
SEED = 0
PCTS = [10, 50, 90]
MARGIN = 6
AMP_MIN = 0.05
OUT = g.FIG_DIR


def per_sample(est, O, R0):
    """逐樣本 nerr_ph(與 scan_5.field_metrics 同一公式);空樣本 = NaN。"""
    m = R0.to(est.real.dtype)
    a, t = est * m, O * m
    E = (t.abs() ** 2).sum((1, 2))
    cross = (a * t.conj()).sum((1, 2)).abs()
    v = ((a.abs() ** 2).sum((1, 2)) + E - 2 * cross) / E.clamp_min(1e-12)
    return torch.where(E > 1e-12, v, torch.full_like(v, float("nan")))


def align_phase(est, O, R0):
    """乘上最佳的整體相位(R0 上),= 評分時的對齊方式。"""
    m = R0.to(est.real.dtype)
    c = ((O * m) * (est * m).conj()).sum((1, 2))
    return est * torch.exp(1j * torch.angle(c))[:, None, None]


def kind_labels(cfg, n_obj, seed):
    """每個物體的類型編號(同 src.procedural.make_procedural 的排列:各類型依比例串接,再以 seed 洗牌)。"""
    kinds = list(cfg.proc_kinds)
    w = list(cfg.proc_weights)[:len(kinds)]
    per = [int(n_obj * x / sum(w)) for x in w]
    per[-1] += n_obj - sum(per)
    lab = torch.cat([torch.full((p,), i, dtype=torch.long) for i, p in enumerate(per) if p > 0])
    perm = torch.randperm(n_obj, generator=torch.Generator().manual_seed(seed))
    return lab[perm], kinds


def tile_kind_map(cfg, n, seed, F):
    """每個場的逐像素類型圖 [n, F, F](同 scan_5.make_fields 的拼貼與循環平移)。"""
    d, T = s5.tile_size(cfg), s5.TILES
    lab, kinds = kind_labels(cfg, n * T * T, seed)
    lab = lab.reshape(n, T, T)
    km = lab[:, :, None, :, None].expand(n, T, d, T, d).reshape(n, T * d, T * d)
    sh = torch.randint(0, d, (n, 2), generator=torch.Generator().manual_seed(seed + 1))
    km = torch.stack([torch.roll(km[i], (int(sh[i, 0]), int(sh[i, 1])), dims=(-2, -1)) for i in range(n)])
    return km, sh, kinds


@torch.no_grad()
def iterative(conf, it, d, dev):
    """重算包絡選到的迭代法設定(同 scan_5d.measure_envelope 的做法)。"""
    import probe_4_2b as pb
    cfg, pr, cp, bs, counts, st, n, F = (d[k] for k in ("cfg", "pr", "cp", "bs", "counts", "st", "n", "F"))
    ones = torch.ones(d["W"], d["W"], device=dev)
    kit = sorted(set(s5b.K_ITERS) | set(s5b.KHIO_N.values()))
    init = s5.k_init(counts[s5b.CENTER], cp, d["init"], dev)
    k_out = pb.hio_probe(init, counts[s5b.CENTER], bs, cp, kit, "K", pr, ones)
    if conf == "K-HIO":
        return s5.k_to_field(k_out[it], pr, cfg, st[s5b.CENTER], F)
    meth, kind = conf.split("|")
    O0 = s5b.make_init(kind, n, F, cfg, cp, pr, bs, counts, st, d["init"], dev, k_out=k_out)
    if it == 0:
        return O0
    return s5b.run_method(meth, counts, bs, cp, pr, st, [it], O0, d["init"])[it]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    plt = s5._plt()
    if plt is None:
        raise SystemExit("❌ 沒有 matplotlib")
    raw = g.RUN_ROOT / f"scan5g_p6_raw{g.SUF}.json"
    tim = json.load(open(raw))["timing"]
    env = json.load(open(s5e.VAL_ENV))["envelope"]
    d = s5d.setup_seed(SEED, s5e.val_seeds, dev)
    cfg, O, R0, F = d["cfg"], d["O"], d["R0"], d["F"]

    est, labels = {}, {}
    with torch.no_grad():
        for grp in ["B", NAF_B, "P6B"]:
            net, norm, _ = s5c.load_net(grp, PROBE, SEED, g.N_TRAIN, cfg, d["pr"], dev)
            if grp == "P6B":
                O0, outs = g.predict(net, d["counts"], d["bs"], norm, grp, d["cp"], all_stages=True)
                stages = [s5c.to_field(O0, cfg, F)] + [s5c.to_field(torch.polar(o[:, 0], o[:, 1]), cfg, F) for o in outs]
                est[grp] = stages[-1]
            else:
                p = g.predict(net, d["counts"], d["bs"], norm, grp, d["cp"])
                est[grp] = s5c.to_field(torch.polar(p[:, 0], p[:, 1]), cfg, F)
            labels[grp] = f"{'NAF-B' if grp == NAF_B else grp} ({tim['64'][f'net:{grp}']:.3f} ms)"
            del net
    order = []
    checks = {}
    for B in ["64", "512"]:
        conf, it, v, t = s5b.envelope(env, PROBE, tim[B]["net:P6B"], tim[B])
        key = f"iter_b{B}"
        est[key] = iterative(conf, it, d, dev)
        labels[key] = f"iterative {s5b.fmt_conf(conf)} x{it}\n(= P6B time, batch {B})"
        mine = float(np.nanmean(per_sample(est[key], O, R0).cpu().numpy()))
        checks[key] = (conf, it, mine, float(v[SEED]))
        order.append(key)
    order += ["B", NAF_B, "P6B"]
    for key, (conf, it, mine, ref) in checks.items():
        ok = abs(mine - ref) < 1e-4
        print(f"{'✅' if ok else '❌'} 重算的迭代法 {conf} ×{it} 在 seed {SEED} 的 nerr_ph {mine:.4f} = 包絡檔 {ref:.4f}")

    err = {k: per_sample(est[k], O, R0).cpu().numpy() for k in order}
    stage_err = [per_sample(x, O, R0).cpu().numpy() for x in stages]
    valid = ~np.isnan(err["P6B"])
    idx_sorted = np.where(valid)[0][np.argsort(err["P6B"][valid])]
    picks = {p: int(idx_sorted[min(len(idx_sorted) - 1, int(round(p / 100 * (len(idx_sorted) - 1))))]) for p in PCTS}

    stats = {}
    print(f"\n逐樣本 nerr_ph(驗證場 seed {SEED} 的網路,{int(valid.sum())} 個非空樣本)")
    for k in order:
        e = err[k][valid]
        stats[k] = {"mean": float(e.mean()), "median": float(np.median(e)), "p10": float(np.percentile(e, 10)),
                    "p90": float(np.percentile(e, 90)), "frac_lt_0.05": float((e < 0.05).mean()),
                    "frac_lt_0.1": float((e < 0.1).mean())}
        print(f"  {labels[k].splitlines()[0]:<42} 平均 {e.mean():.4f}  中位數 {np.median(e):.4f}"
              f"  第 10 / 90 百分位 {np.percentile(e, 10):.4f} / {np.percentile(e, 90):.4f}"
              f"  < 0.05 的比例 {100 * (e < 0.05).mean():.0f}%  < 0.1 {100 * (e < 0.1).mean():.0f}%")
    win = float((err["P6B"][valid] < err["iter_b512"][valid]).mean())
    print(f"  P6B 比「batch 512 同時間的迭代法」準的樣本比例:{100 * win:.1f}%")
    stats["p6b_beats_iter_b512_frac"] = win
    stats["picks"] = {str(p): {"index": i, **{k: float(err[k][i]) for k in order}} for p, i in picks.items()}
    stats["stages_mean"] = [float(np.nanmean(x)) for x in stage_err]

    # 物體類型:逐像素類型圖;每個樣本以 R0 內真值能量最多的類型歸類
    import probe_4_1 as p41
    from src.procedural import GENERATORS
    fseed = s5e.val_seeds(cfg, SEED)["field"]
    n_all = O.shape[0]
    km, sh, kinds = tile_kind_map(cfg, n_all, fseed, F)
    km = km.to(dev)
    # 核對類型的推算:照 make_procedural 的做法逐類型產生,洗牌後須與 make_home 逐位元相同
    n_chk = 60
    lab_c, _ = kind_labels(cfg, n_chk, fseed)
    w = list(cfg.proc_weights)[:len(kinds)]
    per = [int(n_chk * x / sum(w)) for x in w]
    per[-1] += n_chk - sum(per)
    parts = [GENERATORS[k](q, cfg, seed=fseed + 1000 * j, target_support=cfg.match_support_to,
                           contrast_gamma=cfg.proc_contrast_gamma) for j, (k, q) in enumerate(zip(kinds, per)) if q > 0]
    manual = torch.cat(parts)[torch.randperm(n_chk, generator=torch.Generator().manual_seed(fseed))]
    ref_objs = p41.make_home(cfg, n_chk, seed=fseed)
    ok_k = bool(torch.equal(manual, ref_objs))
    print(f"{'✅' if ok_k else '❌'} 類型推算:逐類型產生再洗牌 = make_home(逐位元,{n_chk} 個物體)")
    E = (O.abs() ** 2) * R0
    share = torch.stack([(E * (km == k)).sum((1, 2)) for k in range(len(kinds))], 1)
    dom = share.argmax(1).cpu().numpy()
    stats["by_kind"] = {}
    print(f"\n依 R0 內的主要物體類型分組(各類型的樣本數:"
          + "、".join(f"{kinds[k]} {int(((dom == k) & valid).sum())}" for k in range(len(kinds))) + ")")
    for k in order:
        row = {}
        for j, kn in enumerate(kinds):
            e = err[k][valid & (dom == j)]
            row[kn] = {"mean": float(e.mean()) if len(e) else None, "median": float(np.median(e)) if len(e) else None}
        stats["by_kind"][k] = row
        print(f"  {labels[k].splitlines()[0]:<42} " + "  ".join(
            f"{kn} 平均 {row[kn]['mean']:.4f} / 中位數 {row[kn]['median']:.4f}" if row[kn]["mean"] is not None else f"{kn} —"
            for kn in kinds))

    # 裁切範圍:R0 的外框 ± MARGIN
    ys, xs = torch.where(R0)
    y0, y1 = max(0, int(ys.min()) - MARGIN), min(F, int(ys.max()) + 1 + MARGIN)
    x0, x1 = max(0, int(xs.min()) - MARGIN), min(F, int(xs.max()) + 1 + MARGIN)
    r0c = R0[y0:y1, x0:x1].cpu().numpy()
    pm = cfg.phase_max
    OUT.mkdir(exist_ok=True)

    def panels(ax_col, field, i, title, truth=False):
        t = O[i, y0:y1, x0:x1].cpu().numpy()
        if truth:
            z = t
        else:
            z = align_phase(field[i:i + 1], O[i:i + 1], R0)[0, y0:y1, x0:x1].cpu().numpy()
        amp, ph = np.abs(z), np.angle(z)
        ph = np.where(np.abs(t) > AMP_MIN, ph, np.nan)
        e = np.abs(z - t)
        imgs = [(amp, "gray", 0, 1), (ph, "viridis", 0.0, pm), (e, "magma", 0, 0.5)]
        for ax, (im, cm, lo, hi) in zip(ax_col, imgs):
            cmap = plt.get_cmap(cm).copy()
            cmap.set_bad("#d9d9d9")
            h = ax.imshow(im, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.imshow(np.where(r0c, np.nan, 0.0), cmap="Greys", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        ax_col[0].set_title(title, fontsize=8)
        if truth:
            ax_col[2].cla()
            ax_col[2].axis("off")
        return h

    rows = ["amplitude", "phase (rad)", "|error|"]
    for p, i in picks.items():
        cols = [("truth", None)] + [(k, est[k]) for k in order]
        fig, axs = plt.subplots(3, len(cols), figsize=(2.1 * len(cols), 6.6))
        for c, (k, f) in enumerate(cols):
            ttl = "truth" if k == "truth" else f"{labels[k]}\nnerr {err[k][i]:.3f}"
            panels(axs[:, c], f, i, ttl, truth=(k == "truth"))
        for r, name in enumerate(rows):
            axs[r, 0].set_ylabel(name, fontsize=9)
        fig.suptitle(f"Validation field #{i} (P6B error at the {p}th percentile; seed-{SEED} networks). "
                     f"Scored region = R0 (outside R0 is dimmed); grey phase = amplitude < {AMP_MIN}", fontsize=9)
        fig.tight_layout()
        path = OUT / f"viz_p{p}.png"
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  圖:{path}")

    i = picks[50]
    cols = [(f"P6B start\nnerr {stage_err[0][i]:.3f}", stages[0])] + \
           [(f"after stage {k}\nnerr {stage_err[k][i]:.3f}", stages[k]) for k in range(1, len(stages))]
    fig, axs = plt.subplots(3, len(cols) + 1, figsize=(2.1 * (len(cols) + 1), 6.6))
    for c, (ttl, f) in enumerate(cols):
        panels(axs[:, c], f, i, ttl)
    panels(axs[:, -1], None, i, "truth", truth=True)
    for r, name in enumerate(rows):
        axs[r, 0].set_ylabel(name, fontsize=9)
    fig.suptitle(f"P6B inside: start -> each physics + CNN stage (field #{i}, median sample)", fontsize=9)
    fig.tight_layout()
    path = OUT / "viz_stages.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  圖:{path}")

    # 全景:整個場(112×112)的真值、16 塊物體的類型、掃描範圍(80×80)與 R0;右邊是 P6B 在掃描範圍內的輸出
    i = picks[50]
    d_t = s5.tile_size(cfg)
    b0, size, _ = s5c.geometry(cfg)
    ab = {kn: kn[:4] for kn in kinds}
    Ot = O[i].cpu().numpy()
    z = align_phase(est["P6B"][i:i + 1], O[i:i + 1], R0)[0].cpu().numpy()
    box = np.full((F, F), np.nan)
    box[b0:b0 + size, b0:b0 + size] = np.abs(z)[b0:b0 + size, b0:b0 + size]
    r0n = R0.cpu().numpy().astype(float)
    fr = {kinds[k]: float(share[i, k] / share[i].sum().clamp_min(1e-12)) for k in range(len(kinds))}
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, im, cm, lo, hi, ttl in [
            (axs[0], np.abs(Ot), "gray", 0, 1, "truth amplitude (whole field, 4x4 objects)"),
            (axs[1], np.where(np.abs(Ot) > AMP_MIN, np.angle(Ot), np.nan), "viridis", 0, pm, "truth phase"),
            (axs[2], box, "gray", 0, 1, "P6B output (scan area only)")]:
        cmap = plt.get_cmap(cm).copy()
        cmap.set_bad("#d9d9d9")
        ax.imshow(im, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.add_patch(plt.Rectangle((b0 - 0.5, b0 - 0.5), size, size, fill=False, ec="#eb6834", lw=1.2))
        ax.contour(r0n, levels=[0.5], colors="#2a78d6", linewidths=0.8)
        for k in range(1, s5.TILES + 1):
            for v in [(k * d_t + int(sh[i, 0])) % F]:
                ax.axhline(v - 0.5, color="#f2c14e", lw=0.6, ls="--")
            for v in [(k * d_t + int(sh[i, 1])) % F]:
                ax.axvline(v - 0.5, color="#f2c14e", lw=0.6, ls="--")
        if ax is axs[0]:
            kmi = km[i].cpu().numpy()
            for ty in range(s5.TILES):
                for tx in range(s5.TILES):
                    cy = (ty * d_t + d_t // 2 + int(sh[i, 0])) % F
                    cx = (tx * d_t + d_t // 2 + int(sh[i, 1])) % F
                    ax.text(cx, cy, ab[kinds[int(kmi[cy, cx])]], color="#f2c14e", fontsize=8, ha="center", va="center")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Field #{i} (median sample). Orange = scan area (80x80); blue = R0 (scored); yellow dashed = object "
                 f"borders (fields are wrapped). R0 energy by type: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in fr.items()),
                 fontsize=9)
    fig.tight_layout()
    path = OUT / "viz_overview.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  圖:{path}")

    # 樣本圖鑑:三種類型各 4 個完整的物體(28×28;振幅 / 相位)
    fig, axs = plt.subplots(2 * len(kinds), 4, figsize=(8, 4.2 * len(kinds)))
    s0 = (cfg.canvas - d_t) // 2
    objs = ref_objs[:, :, s0:s0 + d_t, s0:s0 + d_t].numpy()
    for j, kn in enumerate(kinds):
        ids = [q for q in range(n_chk) if int(lab_c[q]) == j][:4]
        for c in range(4):
            for r, (im, cm, hi) in enumerate([(objs[ids[c], 0], "gray", 1), (objs[ids[c], 1], "viridis", pm)]):
                ax = axs[2 * j + r, c]
                if c < len(ids):
                    ax.imshow(im, cmap=cm, vmin=0, vmax=hi, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
            axs[2 * j, 0].set_ylabel(f"{kn}\namplitude", fontsize=8)
            axs[2 * j + 1, 0].set_ylabel(f"{kn}\nphase", fontsize=8)
    fig.suptitle("Object types used in training and test fields (whole 28x28 objects)", fontsize=9)
    fig.tight_layout()
    path = OUT / "viz_gallery.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  圖:{path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-3, 1, 60)
    col = {"iter_b64": "#9a9994", "iter_b512": "#0b0b0b", "B": "#2a78d6", NAF_B: "#1baf7a", "P6B": "#d6452a"}
    for k in order:
        ax.hist(err[k][valid], bins=bins, histtype="step", lw=1.5, color=col[k], label=labels[k].replace("\n", " "))
    ax.set_xscale("log")
    ax.set_xlabel("per-sample nerr on R0 (global phase only)")
    ax.set_ylabel("samples")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = OUT / "viz_hist.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  圖:{path}")
    json.dump(stats, open(OUT / "viz_stats.json", "w"), indent=2)
    print("\n✅ 完成。把上面的文字和 7 張圖傳給 Claude")


if __name__ == "__main__":
    main()
