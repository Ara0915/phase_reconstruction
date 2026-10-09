#!/usr/bin/env python
"""待辦 2:以「先對齊平凡歧異性再評分」重新評估既有結論。

背景(研究日誌 §十四):
    純 HIO 的輸出 97% 相對真值平移、49% 為共軛翻轉;未對齊時 FRC gain 0.063,
    對齊後 0.643。先前所有「以未對齊分數評 HIO」的結論都可能被同一個漏洞影響。

本腳本重新檢驗四項(皆不重新訓練):

  A. 未見分布 —— HIO 不依賴訓練分布,對齊後在未見分布上是否也贏過平庸基準?
                (§11.2 的「1/4」、§6.2 的「0/4」)
  B. MNIST    —— §11.6「HIO 自第一次迭代即破壞 MNIST 結果」是真的劣化,還是漂向等價解?
  C. R-factor —— §11.5「R-factor 僅在起點較差時與品質最佳點吻合」,
                改用對齊後的品質重判
  D. 各原型    —— §12c.4「HIO 對 lattice 特別有效」按原型分組量化
  E. 圖        —— 對齊前後的重建並列(取代 §12c 未對齊的視覺判讀)

對齊函式直接 import 自 ambiguity_check.py(同一份實作,已有自我檢查)。

用法(需在計算節點執行,需先跑過 hio_ideal_conditions.py):
    python realign_eval.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from ambiguity_check import align, self_test, to_c, to_obj
from src.config import Cfg
from src.data import generalization_suites
from src.hio import hio, random_init
from src.metrics import evaluate
from src.model import build_model
from src.physics import beamstop_mask, build_input, forward_measure
from src.procedural import GENERATORS

RUN_ROOT = Path("/work/elviss0915/runs")
OUT_JSON = RUN_ROOT / "realign_eval.json"
FIG_DIR = Path("figs_realign")
SEEDS = [0, 1, 2]

PROC_RUN = "ideal_base"                     # 程序生成、現況條件(= v2_proc 設定)
MNIST_RUNS = ["hio_mnist", "v2_mnist"]      # 依序嘗試,取第一個存在者
PROC_ITERS = [0, 50, 200, 500, 1000, 5000]
NET_MID = 200                               # 網路 + HIO 的代表點(§11.3 的 FRC 最佳點)
MNIST_ITERS = [0, 5, 20, 50, 200, 1000]
UNSEEN = ["mnist_test", "fashion_mnist", "random_shapes", "random_texture"]
SHORT = {"mnist_test": "mnist", "fashion_mnist": "fashion",
         "random_shapes": "shapes", "random_texture": "texture",
         "procedural": "proc"}
KEYS = ["frc_gain", "material_mae", "material_mae_trivial", "amp_psnr",
        "amp_psnr_trivial", "r_factor"]
REPRO_TOL = 0.005
SANITY_TOL = 0.01


# ============================================================================
# 共用
# ============================================================================
def load(run_dir, device):
    cfg = Cfg.from_dict(json.load(open(run_dir / "config_used.json")))
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location=device))
    model.eval()
    return cfg, model


@torch.no_grad()
def net_out(model, objs, counts, bs, cfg, chunk=128):
    return torch.cat([model(build_input(objs[i:i + chunk], counts[i:i + chunk], bs, cfg))
                      for i in range(0, len(objs), chunk)], 0)


def score(model, objs, bs, cfg, pred):
    r = evaluate(model, objs, bs, cfg, pred_override=pred)
    return {k: float(r[k]) for k in KEYS}


def score_all(model, objs, bs, cfg, pred):
    """回傳 未對齊 / 對齊 / 錯配對齊 三組分數,以及對齊後的輸出。"""
    raw = score(model, objs, bs, cfg, pred)
    al_c, tw, sh = align(to_c(pred), to_c(objs))
    al = to_obj(al_c)
    ali = score(model, objs, bs, cfg, al)
    mis_objs = torch.roll(objs, 1, dims=0)
    mis_c, _, _ = align(to_c(pred), to_c(mis_objs))
    mis = score(model, mis_objs, bs, cfg, to_obj(mis_c))
    return {"raw": raw, "aligned": ali, "mismatch": mis,
            "twin_frac": float(tw.float().mean())}, al


def sweep(model, objs, counts, bs, cfg, iters, seed, keep=None):
    """對兩種起點掃描迭代數。keep=(init, n_it) 時另回傳該點的輸出(畫圖用)。"""
    inits = {"network": net_out(model, objs, counts, bs, cfg),
             "random": random_init(counts, cfg, seed=cfg.test_seed + seed,
                                   device=objs.device)}
    res, kept = {}, {}
    for init_src, init in inits.items():
        res[init_src] = {}
        for n_it in iters:
            pred = hio(init, counts, bs, cfg, n_iter=n_it, beta=cfg.hio_beta)
            r, al = score_all(model, objs, bs, cfg, pred)
            res[init_src][str(n_it)] = r
            if keep and (init_src, n_it) in keep:
                kept[(init_src, n_it)] = (pred.cpu(), al.cpu())
    return res, kept


# ============================================================================
# D 用:重建程序生成樣本的原型標籤(與 make_procedural 完全相同的邏輯)
# ============================================================================
def proto_labels(n, cfg, objs):
    kinds = list(cfg.proc_kinds)
    w = list(cfg.proc_weights)[:len(kinds)]
    per = [int(n * x / sum(w)) for x in w]
    per[-1] += n - sum(per)
    lab = torch.cat([torch.full((p,), i) for i, p in enumerate(per) if p > 0])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.test_seed))
    labels = lab[perm]
    # 自我檢查:逐原型重新生成,必須與 make_procedural 的輸出逐位元相同
    offset = 0
    for i, (k, p) in enumerate(zip(kinds, per)):
        if p == 0:
            continue
        g = GENERATORS[k](p, cfg, seed=cfg.test_seed + 1000 * i,
                          target_support=cfg.match_support_to,
                          contrast_gamma=cfg.proc_contrast_gamma)
        idx = (labels == i).nonzero()[:, 0]
        src = perm[idx] - offset
        assert torch.equal(objs[idx].cpu(), g[src]), f"原型標籤重建失敗:{k}"
        offset += p
    return labels, kinds


# ============================================================================
# 各部分
# ============================================================================
def part_proc(device):
    """A + C + D + E 都用程序生成(現況條件)的 3 個 seed。"""
    out = {"seeds": []}
    figs = None
    for s in SEEDS:
        run_dir = RUN_ROOT / f"{PROC_RUN}_s{s}"
        cfg, model = load(run_dir, device)
        bs = beamstop_mask(cfg, device=device)
        home = generalization_suites(cfg, cfg.eval_n)["procedural"].to(device)
        gen = generalization_suites(cfg, cfg.gen_n)
        if s == SEEDS[0]:
            self_test(home, device)

        # 訓練分布:與 hio_ideal_conditions.py 相同的種子 -> 可逐點重現
        torch.manual_seed(cfg.test_seed + s)
        counts = forward_measure(home, bs, cfg)
        keep = {("network", 0), ("random", PROC_ITERS[-1]), ("network", NET_MID)}
        res_home, kept = sweep(model, home, counts, bs, cfg, PROC_ITERS, s, keep)
        ref = json.load(open(run_dir / "hio_ideal.json"))["results"]["procedural"]
        for init_src in ["network", "random"]:
            for n_it in PROC_ITERS:
                a = res_home[init_src][str(n_it)]["raw"]["frc_gain"]
                b = ref[init_src][str(n_it)]["frc_gain"]
                if abs(a - b) > REPRO_TOL:
                    raise SystemExit(f"{run_dir.name} {init_src} {n_it}:未對齊 {a:+.4f} "
                                     f"與 hio_ideal.json {b:+.4f} 對不上,停下來查")

        # D:按原型分組(同一份輸出,只是分組評分)
        labels, kinds = proto_labels(cfg.eval_n, cfg, home)
        per_proto = {}
        for (init_src, n_it), (pred, al) in kept.items():
            for i, k in enumerate(kinds):
                m = (labels == i).to(device)
                sub = home[m]
                per_proto.setdefault(k, {})[f"{init_src}_{n_it}"] = {
                    "raw": score(model, sub, bs, cfg, pred.to(device)[m]),
                    "aligned": score(model, sub, bs, cfg, al.to(device)[m]),
                    "n": int(m.sum())}

        # A:未見分布(各自固定種子,與評分呼叫的次數無關)
        res_unseen = {}
        for j, name in enumerate(UNSEEN):
            objs = gen[name].to(device)
            torch.manual_seed(cfg.test_seed + s + 1000 * (j + 1))
            c = forward_measure(objs, bs, cfg)
            res_unseen[name], _ = sweep(model, objs, c, bs, cfg, PROC_ITERS, s)

        out["seeds"].append({"run": run_dir.name, "home": res_home,
                             "unseen": res_unseen, "proto": per_proto})
        if s == SEEDS[0]:
            figs = (home.cpu(), labels, kinds, kept)
        print(f"  [{run_dir.name}] 完成", flush=True)
    return out, figs


def part_mnist(device):
    base = next((b for b in MNIST_RUNS
                 if all((RUN_ROOT / f"{b}_s{s}" / "final.pt").exists() for s in SEEDS)), None)
    if base is None:
        print("  ⚠️ 找不到 MNIST 訓練的 checkpoint,跳過 B")
        return None
    out = {"base": base, "seeds": []}
    for s in SEEDS:
        run_dir = RUN_ROOT / f"{base}_s{s}"
        cfg, model = load(run_dir, device)
        assert list(cfg.train_sources) == ["mnist"]
        bs = beamstop_mask(cfg, device=device)
        objs = generalization_suites(cfg, cfg.eval_n)["mnist_test"].to(device)
        torch.manual_seed(cfg.test_seed + s)
        counts = forward_measure(objs, bs, cfg)
        res, _ = sweep(model, objs, counts, bs, cfg, MNIST_ITERS, s)
        m = json.load(open(run_dir / "metrics.json"))["test"]["frc_gain"]
        it0 = res["network"]["0"]["raw"]["frc_gain"]
        if abs(it0 - m) > SANITY_TOL:
            raise SystemExit(f"{run_dir.name}:iter 0 {it0:+.4f} 與 metrics.json {m:+.4f} 對不上")
        out["seeds"].append({"run": run_dir.name, "res": res})
        print(f"  [{run_dir.name}] 完成(iter 0 = {it0:+.4f},metrics.json {m:+.4f})", flush=True)
    return out


# ============================================================================
# 彙整
# ============================================================================
def ms(v):
    a = np.array(v, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def get(seeds, path):
    vals = []
    for sd in seeds:
        x = sd
        for k in path:
            x = x[k]
        vals.append(x)
    return vals


def report(proc, mnist):
    S = proc["seeds"]

    # ---------------- A ----------------
    print("\n" + "=" * 88)
    print("A. 未見分布:相對增益 = 振幅 PSNR − 該分布平庸基準(dB),3 seeds 平均")
    print("   (§十 的主指標;正值 = 贏過基準)")
    print("=" * 88)
    for init_src, lab in [("network", "網路 + HIO"), ("random", "純 HIO")]:
        for kind, kl in [("raw", "未對齊"), ("aligned", "對齊後"), ("mismatch", "錯配對齊")]:
            print(f"\n  {lab}  [{kl}]")
            print(f"  {'迭代':>6}" + "".join(f"{SHORT[u]:>10}" for u in UNSEEN) + f"{'贏過基準':>10}")
            for n_it in PROC_ITERS:
                row, win = f"  {n_it:>6}", 0
                for u in UNSEEN:
                    p = np.mean(get(S, ["unseen", u, init_src, str(n_it), kind, "amp_psnr"]))
                    t = np.mean(get(S, ["unseen", u, init_src, str(n_it), kind, "amp_psnr_trivial"]))
                    row += f"{p - t:>+10.2f}"
                    win += (p - t) > 0
                print(row + f"{win:>7}/{len(UNSEEN)}")
    last = str(PROC_ITERS[-1])
    print(f"\n  同表的 FRC gain(對齊後 / 錯配對齊),純 HIO {last} 次:")
    for u in UNSEEN:
        a = ms(get(S, ["unseen", u, "random", last, "aligned", "frc_gain"]))
        b = np.mean(get(S, ["unseen", u, "random", last, "mismatch", "frc_gain"]))
        c = np.mean(get(S, ["unseen", u, "network", "0", "raw", "frc_gain"]))
        print(f"    {SHORT[u]:<8} 對齊後 {a[0]:+.4f}±{a[1]:.4f}   錯配 {b:+.4f}   (網路本身未對齊 {c:+.4f})")

    # ---------------- B ----------------
    print("\n" + "=" * 88)
    print("B. MNIST 訓練的模型:HIO 是否真的破壞結果(§11.6)")
    print("=" * 88)
    if mnist is None:
        print("  跳過(找不到 checkpoint)")
    else:
        M = mnist["seeds"]
        print(f"  checkpoint:{mnist['base']}_s*,訓練分布 mnist_test,512 張")
        print(f"  {'迭代':>6}{'網路+HIO 未對齊':>18}{'網路+HIO 對齊':>16}{'純HIO 未對齊':>15}"
              f"{'純HIO 對齊':>13}{'錯配':>9}")
        for n_it in MNIST_ITERS:
            v = [np.mean(get(M, ["res", i, str(n_it), k, "frc_gain"]))
                 for i, k in [("network", "raw"), ("network", "aligned"),
                              ("random", "raw"), ("random", "aligned"), ("random", "mismatch")]]
            print(f"  {n_it:>6}{v[0]:>+18.4f}{v[1]:>+16.4f}{v[2]:>+15.4f}{v[3]:>+13.4f}{v[4]:>+9.4f}")
        print("  材料 MAE(對齊後):網路本身 "
              f"{np.mean(get(M, ['res', 'network', '0', 'aligned', 'material_mae'])):.4f}   "
              f"純 HIO {MNIST_ITERS[-1]} 次 "
              f"{np.mean(get(M, ['res', 'random', str(MNIST_ITERS[-1]), 'aligned', 'material_mae'])):.4f}")

    # ---------------- C ----------------
    print("\n" + "=" * 88)
    print("C. R-factor 作為停止準則(程序生成,現況條件):R 最小點 vs 對齊後品質最佳點")
    print("=" * 88)
    print(f"  {'起點':<10}{'迭代':>6}{'R-factor':>11}{'FRC 未對齊':>13}{'FRC 對齊後':>13}")
    for init_src in ["network", "random"]:
        rs = {n: np.mean(get(S, ["home", init_src, str(n), "raw", "r_factor"])) for n in PROC_ITERS}
        fr = {n: np.mean(get(S, ["home", init_src, str(n), "raw", "frc_gain"])) for n in PROC_ITERS}
        fa = {n: np.mean(get(S, ["home", init_src, str(n), "aligned", "frc_gain"])) for n in PROC_ITERS}
        for n in PROC_ITERS:
            print(f"  {init_src:<10}{n:>6}{rs[n]:>11.4f}{fr[n]:>+13.4f}{fa[n]:>+13.4f}")
        nr = min(PROC_ITERS, key=lambda n: rs[n])
        nraw = max(PROC_ITERS, key=lambda n: fr[n])
        nal = max(PROC_ITERS, key=lambda n: fa[n])
        print(f"  -> R 最小於 {nr} 次;未對齊 FRC 最佳於 {nraw} 次;對齊後 FRC 最佳於 {nal} 次;"
              f"R 選點的對齊後 FRC 為最佳值的 {100 * fa[nr] / fa[nal]:.1f}%\n")

    # ---------------- D ----------------
    print("=" * 88)
    print("D. 按原型分組(程序生成,現況條件,對齊後 FRC gain / 材料 MAE)")
    print("=" * 88)
    kinds = list(S[0]["proto"].keys())
    last = PROC_ITERS[-1]
    cols = [("network_0", "網路本身"), (f"network_{NET_MID}", f"網路+HIO {NET_MID}"),
            (f"random_{last}", f"純 HIO {last}")]
    print(f"  {'原型':<10}{'n':>5}" + "".join(f"{c[1]:>22}" for c in cols) + f"{'HIO 增益':>14}")
    for k in kinds:
        n = S[0]["proto"][k]["network_0"]["n"]
        row = f"  {k:<10}{n:>5}"
        for key, _ in cols:
            f = ms(get(S, ["proto", k, key, "aligned", "frc_gain"]))
            mat = np.mean(get(S, ["proto", k, key, "aligned", "material_mae"]))
            row += f"{f[0]:>+10.4f}±{f[1]:.3f} / {mat:.3f}"
        gain = [a - b for a, b in zip(get(S, ["proto", k, f"random_{last}", "aligned", "frc_gain"]),
                                      get(S, ["proto", k, "network_0", "aligned", "frc_gain"]))]
        g = ms(gain)
        row += f"{g[0]:>+10.4f}±{g[1]:.3f}"
        print(row)
    print(f"  HIO 增益 = 純 HIO {last}(對齊)− 網路本身(對齊),逐 seed 相減")
    print("\n判讀準則見 實驗設計_1b6 §十三(結果出來前已寫定)")


def make_figure(figs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ 無 matplotlib,略過圖")
        return
    home, labels, kinds, kept = figs
    FIG_DIR.mkdir(exist_ok=True)
    last = PROC_ITERS[-1]
    cols = [("GT", None), ("Network", ("network", 0)),
            (f"Pure HIO {last} (raw)", ("random", last)),
            (f"Pure HIO {last} (aligned)", ("random", last))]
    rows = []
    for i, k in enumerate(kinds):
        idx = (labels == i).nonzero()[:, 0][:2].tolist()
        rows += [(k, j) for j in idx]
    sl = slice(14, 50)
    for ch, name in [(0, "amplitude"), (1, "phase")]:
        fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.2 * len(cols), 2.2 * len(rows)))
        for r, (k, j) in enumerate(rows):
            for c, (title, key) in enumerate(cols):
                if key is None:
                    img = home[j, ch]
                else:
                    pred, al = kept[key]
                    img = (al if "aligned" in title else pred)[j, ch]
                a = ax[r, c]
                a.imshow(img[sl, sl].numpy(), cmap="gray" if ch == 0 else "viridis",
                         vmin=0, vmax=(1.0 if ch == 0 else float(np.pi / 2)))
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(title, fontsize=9)
                if c == 0:
                    a.set_ylabel(f"{k} #{j}", fontsize=9)
        fig.suptitle(f"Realigned reconstructions ({name}), ideal_base_s0, current conditions",
                     fontsize=10)
        fig.tight_layout()
        p = FIG_DIR / f"realign_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  圖:{p}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("A / C / D:程序生成(現況條件)")
    proc, figs = part_proc(device)
    print("B:MNIST")
    mnist = part_mnist(device)
    json.dump({"proc": proc, "mnist": mnist}, open(OUT_JSON, "w"))
    report(proc, mnist)
    make_figure(figs)


if __name__ == "__main__":
    main()
