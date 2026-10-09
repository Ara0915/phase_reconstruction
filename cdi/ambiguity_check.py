#!/usr/bin/env python
"""平凡歧異性檢查 —— 寬鬆 support 是否留下了「一樣正確」的其他解?

背景(實驗設計 §10.5):
    HIO 使用的已知 support 是 32x32 方框(1024 px),程序生成物體只佔約 130 px。
    物體在方框內平移、或做共軛翻轉,繞射強度完全相同,support 約束排除不了。

本腳本檢查兩件事(不重新訓練,只用既有 checkpoint):

  ① 歧異性在實際上存不存在
     對 HIO 輸出「允許平移 + 共軛翻轉 + 全域相位」對齊真值後再評分。
     若對齊後分數大幅上升 -> HIO 其實解出了物體,只是位置/方向不同。
     以「隨機起點、0 次迭代」對齊後的上升量作為對齊本身造成的虛增底線。

  ② 網路的模糊是否來自對等價解取平均
     共軛翻轉 = 振幅繞中心轉 180 度。若網路對「原解 + 翻轉解」取平均,
     輸出的振幅會比真值更接近中心對稱。
     以 oracle(給了繞射相位,沒有歧異)作為對照。

與 hio_ideal_conditions.py 用同一組種子與抽樣順序,
故「未對齊」的數字應與其 hio_ideal.json 完全相同(內建檢查)。

用法(需在計算節點執行):
    python ambiguity_check.py
"""
import json
import re
from pathlib import Path

import numpy as np
import torch

from src.config import Cfg
from src.data import generalization_suites
from src.hio import hio, random_init
from src.metrics import evaluate
from src.model import build_model
from src.physics import beamstop_mask, build_input, forward_measure, support_mask

RUN_ROOT = Path("/work/elviss0915/runs")
CONDITIONS = {
    "ideal_base":      ("現況 bs=3 ph=1e3", "v2_proc_oracle"),
    "ideal_bs0_ph1e5": ("理想 bs=0 ph=1e5", "ideal_bs0_ph1e5_oracle"),
}
SEEDS = [0, 1, 2]
HOME = "procedural"
# (起點, 迭代數)。("random", 0) 是隨機起點本身 = 對齊虛增的底線
METHODS = [("network", 0), ("network", 200), ("network", 5000),
           ("random", 0), ("random", 200), ("random", 5000)]
LABEL = {("network", 0): "網路本身", ("network", 200): "網路+HIO 200",
         ("network", 5000): "網路+HIO 5000", ("random", 0): "隨機起點(底線)",
         ("random", 200): "純 HIO 200", ("random", 5000): "純 HIO 5000"}
REPRO_TOL = 0.005


# ============================================================================
# 平凡歧異性的三種變換
# ============================================================================
def twin(psi):
    """共軛翻轉:psi*(-r)。以畫布中心翻轉(i -> N-1-i),
    與嚴格的 i -> -i mod N 只差 1 px 平移,兩者繞射強度皆與原物體相同,
    而此版本讓置中的物體留在 support 方框內。"""
    return torch.conj(torch.flip(psi, dims=(-2, -1)))


def fft_mag(psi):
    return torch.fft.fft2(psi, norm="ortho").abs()


@torch.no_grad()
def align(pred_c, target_c):
    """在 {原解, 共軛翻轉} x {所有循環平移} x {全域相位} 中找最接近真值者。

    回傳 (對齊後的複數場, 是否選了翻轉, 平移量 [N,2])。
    """
    Ft = torch.fft.fft2(target_c)
    best_val, best_field, best_twin, best_shift = None, None, None, None
    for is_twin, c in [(False, pred_c), (True, twin(pred_c))]:
        # cc[s] = sum_r t(r) * conj(c(r - s)) -> 峰值處 t ≈ e^{iθ} c(r - s)
        cc = torch.fft.ifft2(Ft * torch.conj(torch.fft.fft2(c)))
        n, h, w = cc.shape
        flat = cc.reshape(n, -1)
        idx = flat.abs().argmax(dim=1)
        val = flat.abs().gather(1, idx[:, None])[:, 0]
        theta = torch.angle(flat.gather(1, idx[:, None])[:, 0])
        sy, sx = (idx // w), (idx % w)
        aligned = torch.stack([
            torch.roll(c[k], shifts=(int(sy[k]), int(sx[k])), dims=(-2, -1))
            for k in range(n)]) * torch.exp(1j * theta)[:, None, None]
        if best_val is None:
            best_val, best_field = val, aligned
            best_twin = torch.zeros(n, dtype=torch.bool, device=c.device)
            best_shift = torch.stack([sy, sx], 1)
        else:
            better = val > best_val
            best_val = torch.where(better, val, best_val)
            best_field = torch.where(better[:, None, None], aligned, best_field)
            best_twin = torch.where(better, torch.ones_like(best_twin), best_twin)
            best_shift = torch.where(better[:, None], torch.stack([sy, sx], 1),
                                     best_shift)
    # 平移量換成 [-N/2, N/2) 的有號值
    best_shift = (best_shift + h // 2) % h - h // 2
    return best_field, best_twin, best_shift


def to_obj(psi):
    return torch.stack([psi.abs(), torch.angle(psi)], dim=1)


def to_c(obj):
    return torch.polar(obj[:, 0], obj[:, 1])


# ============================================================================
# 內建自我檢查(不需 checkpoint)
# ============================================================================
def self_test(objs, device):
    t = to_c(objs[:32].to(device))
    # (a) 三種變換都不改變繞射強度
    for name, v in [("共軛翻轉", twin(t)),
                    ("平移 (5,-3)", torch.roll(t, (5, -3), (-2, -1))),
                    ("全域相位", t * np.exp(1j * 0.7))]:
        err = ((fft_mag(v) - fft_mag(t)).abs().max() / fft_mag(t).max()).item()
        assert err < 1e-5, f"{name} 改變了繞射強度(相對誤差 {err:.2e})"
    # (b) 對齊能把「平移 + 翻轉 + 相位」過的真值完全還原
    scrambled = torch.roll(twin(t), (4, -6), (-2, -1)) * np.exp(1j * 1.1)
    back, tw, sh = align(scrambled, t)
    err = ((back - t).abs().max()).item()
    assert err < 1e-4, f"對齊無法還原已知變換(誤差 {err:.2e})"
    assert tw.all(), "對齊沒有辨認出共軛翻轉"
    print("✅ 自我檢查:三種變換不改變繞射強度;對齊可完全還原已知的平移+翻轉+相位")


# ============================================================================
# ② 中心對稱度
# ============================================================================
def centro_sym(amp, sup):
    """每張樣本在 support 方框內,振幅與其 180 度旋轉的 Pearson 相關,取平均。"""
    a = amp * sup
    b = torch.flip(amp, dims=(-2, -1)) * sup
    m = sup > 0
    vals = []
    for k in range(a.shape[0]):
        x, y = a[k][m], b[k][m]
        x, y = x - x.mean(), y - y.mean()
        vals.append((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))
    return float(torch.stack(vals).mean())


# ============================================================================
# 主流程
# ============================================================================
def load(run_dir, device):
    cfg = Cfg.from_dict(json.load(open(run_dir / "config_used.json")))
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location=device))
    model.eval()
    return cfg, model


@torch.no_grad()
def forward_chunks(model, objs, counts, bs, cfg, chunk=128):
    return torch.cat([model(build_input(objs[i:i + chunk], counts[i:i + chunk], bs, cfg))
                      for i in range(0, len(objs), chunk)], 0)


def scalar(r):
    return {k: float(r[k]) for k in ["frc_gain", "material_mae", "amp_psnr"]}


def run_one(cond, seed, device, did_self_test):
    run_dir = RUN_ROOT / f"{cond}_s{seed}"
    cfg, model = load(run_dir, device)
    bs = beamstop_mask(cfg, device=device)
    objs = generalization_suites(cfg, cfg.eval_n)[HOME].to(device)
    if not did_self_test:
        self_test(objs, device)

    # 與 hio_ideal_conditions.py 相同:先設種子,第一個抽樣就是訓練分布的量測
    torch.manual_seed(cfg.test_seed + seed)
    counts = forward_measure(objs, bs, cfg)

    inits = {"network": forward_chunks(model, objs, counts, bs, cfg),
             "random": random_init(counts, cfg, seed=cfg.test_seed + seed,
                                   device=device)}
    ref = json.load(open(run_dir / "hio_ideal.json"))["results"][HOME]
    t_c = to_c(objs)
    objs_mis = torch.roll(objs, 1, dims=0)      # 每張輸出配到隔壁那張的真值
    out = {}
    for init_src, n_it in METHODS:
        pred = hio(inits[init_src], counts, bs, cfg, n_iter=n_it, beta=cfg.hio_beta)
        raw = scalar(evaluate(model, objs, bs, cfg, pred_override=pred))
        # 內建檢查:未對齊的數字應重現 hio_ideal.json
        prev = ref[init_src][str(n_it)]["frc_gain"]
        if abs(raw["frc_gain"] - prev) > REPRO_TOL:
            raise SystemExit(f"{run_dir.name} {init_src} {n_it}:未對齊 {raw['frc_gain']:+.4f} "
                             f"與 hio_ideal.json {prev:+.4f} 對不上,停下來查")
        al_c, tw, sh = align(to_c(pred), t_c)
        ali = scalar(evaluate(model, objs, bs, cfg, pred_override=to_obj(al_c)))
        # 錯配對照:把輸出對齊到「另一張」物體的真值再評分 = 對齊本身能製造的分數
        mis_c, _, _ = align(to_c(pred), to_c(objs_mis))
        mis = scalar(evaluate(model, objs_mis, bs, cfg, pred_override=to_obj(mis_c)))
        moved = (sh.abs().max(dim=1).values > 0)
        out[(init_src, n_it)] = dict(
            raw=raw, aligned=ali, mismatch=mis,
            twin_frac=float(tw.float().mean()),
            shift_frac=float(moved.float().mean()),
            shift_med=float(sh.abs().max(dim=1).values.float().median()))

    # ② 中心對稱度:真值、網路、oracle(同一批物體、同一份量測條件)
    sup = support_mask(cfg, device=device)
    sym = {"真值": centro_sym(objs[:, 0], sup),
           "網路": centro_sym(inits["network"][:, 0], sup)}
    orc_dir = RUN_ROOT / f"{CONDITIONS[cond][1]}_s{seed}"
    if (orc_dir / "final.pt").exists():
        ocfg, omodel = load(orc_dir, device)
        assert ocfg.oracle_phase
        assert (ocfg.beamstop_r, ocfg.photons_per_pix) == (cfg.beamstop_r, cfg.photons_per_pix)
        obs = beamstop_mask(ocfg, device=device)
        ocounts = forward_measure(objs, obs, ocfg)
        sym["oracle"] = centro_sym(forward_chunks(omodel, objs, ocounts, obs, ocfg)[:, 0], sup)
    print(f"  [{run_dir.name}] 完成", flush=True)
    return out, sym


def ms(v):
    a = np.array(v, float)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res, syms = {}, {}
    did = False
    for cond in CONDITIONS:
        res[cond], syms[cond] = [], []
        for s in SEEDS:
            o, y = run_one(cond, s, device, did)
            did = True
            res[cond].append(o)
            syms[cond].append(y)

    print("\n" + "=" * 86)
    print("① 允許平移 + 共軛翻轉 + 全域相位對齊後,訓練分布的 FRC gain(3 seeds)")
    print("=" * 86)
    for cond in CONDITIONS:
        print(f"\n  {CONDITIONS[cond][0]}")
        print(f"  {'方法':<16}{'未對齊':>10}{'對齊後':>16}{'錯配對齊':>10}"
              f"{'選翻轉':>9}{'有平移':>9}{'平移中位':>9}{'材料(對齊)':>12}")
        print("  " + "-" * 94)
        for key in METHODS:
            rs = res[cond]
            raw = ms([r[key]["raw"]["frc_gain"] for r in rs])[0]
            ali, ali_s = ms([r[key]["aligned"]["frc_gain"] for r in rs])
            mis = ms([r[key]["mismatch"]["frc_gain"] for r in rs])[0]
            tw = ms([r[key]["twin_frac"] for r in rs])[0]
            sf = ms([r[key]["shift_frac"] for r in rs])[0]
            sm = ms([r[key]["shift_med"] for r in rs])[0]
            mat = ms([r[key]["aligned"]["material_mae"] for r in rs])[0]
            print(f"  {LABEL[key]:<16}{raw:>+10.4f}{ali:>+10.4f}±{ali_s:.4f}{mis:>+10.4f}"
                  f"{100 * tw:>8.0f}%{100 * sf:>8.0f}%"
                  f"{sm:>8.1f}px{mat:>12.4f}")
    print("\n  錯配對齊 = 把輸出對齊到「另一張」物體的真值後的分數 = 對齊本身能製造的機率水準")
    print("  對齊後 明顯高於 錯配對齊,才代表輸出真的是該物體(只差位置/方向)")

    print("\n" + "=" * 86)
    print("② 振幅的中心對稱度(與 180 度旋轉的相關,support 方框內)")
    print("=" * 86)
    for cond in CONDITIONS:
        keys = [k for k in ["真值", "網路", "oracle"] if all(k in y for y in syms[cond])]
        cells = []
        for k in keys:
            m, s = ms([y[k] for y in syms[cond]])
            cells.append(f"{k} {m:.3f}±{s:.3f}")
        print(f"  {CONDITIONS[cond][0]:<18}" + "   ".join(cells))
        if "oracle" in keys:
            d = [y["網路"] - y["oracle"] for y in syms[cond]]
            m, s = ms(d)
            z = m / (s / np.sqrt(len(d))) if s > 0 else float("inf")
            print(f"  {'':<18}網路 − oracle = {m:+.3f}(z = {z:.1f})")

    print("\n判讀準則見 實驗設計_1b6 §十一(結果出來前已寫定)")


if __name__ == "__main__":
    main()
