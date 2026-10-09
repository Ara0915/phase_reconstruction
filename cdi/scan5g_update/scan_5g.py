#!/usr/bin/env python
"""優化第二步:P6 網路內建少數物理修正(階段五協定 §十八;優化方案 §六、§七)。

架構 = 已訓練的初始網路(warm start)→ K 級,每級:
  (1) 物理步:一次 AP 重疊投影(9 張的量測振幅、已知探針與位置;= AP 的一次迭代),可學的步長 α ∈ (0, 1)
  (2) 小 CNN 修正(殘差;4 層 3×3 卷積 3 → 32 → 32 → 32 → 2,最後一層初始為 0)
  (3) 投影回物體約束(振幅 ≤ 1、相位 ∈ [0, φmax];同 AP-C)
初始物體 = R0 內用初始網路的輸出、R0 外用常數(同 5-3「網路當起點」的做法;初始網路只在 R0 上訓練過)。
三組(各 3 seeds):
  P6B     B + 3 級(有物理步)          快線
  P6Bctl  B + 3 級(沒有物理步)        對照:分開「物理回饋」與「網路變大 + 多訓練」
  P6NAF   NAF-B + 2 級(有物理步)      準線

用法(需在計算節點執行;需 scan_5.py、scan_5b.py、scan_5c.py、scan_5d.py、scan_5e.py 與其依賴、
      B / NAF-B 的 100k 模型、驗證場包絡 scan5e_val_envelope.json):
    python scan_5g.py --check             # 單元測試(三組)
    python scan_5g.py --train --task 0-8  # 訓練(run_scan5g_train.sh 的 array;0–2 = P6B、3–5 = P6Bctl、6–8 = P6NAF)
    python scan_5g.py --eval              # 驗證場上評估(run_scan5g_eval.sh)
"""
import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5 as s5                                                  # noqa: E402
import scan_5b as s5b                                                # noqa: E402
import scan_5c as s5c                                                # noqa: E402  (匯入時已關閉 TF32,同 5-3)
import scan_5d as s5d                                                # noqa: E402
import scan_5e as s5e                                                # noqa: E402  (NAF 的定義)
from src.config import Cfg                                          # noqa: E402

RUN_ROOT = s5.RUN_ROOT
QUICK = s5c.QUICK
PROBE = "DEF2"
SEEDS = s5b.SEEDS
N_TRAIN = s5e.N_TRAIN                        # 100k(QUICK:64),與 B / NAF-B 相同
NAF_B = s5e.NAF_B
VARIANTS = {"P6B": ("B", 3, True), "P6Bctl": ("B", 3, False), "P6NAF": (NAF_B, 2, True)}   # 初始網路、級數、有無物理步
ORDER = ["P6B", "P6Bctl", "P6NAF"]
BASES = ["B", "C", NAF_B]                    # 評估時一起比較(同 job 重新計時)
EPOCHS = 20
BATCH = s5c.BATCH                            # 128
LR_INIT, LR_NEW = 2e-4, 2e-3                 # 初始網路 0.1 倍;新加的級(含步長)
CLIP = 1.0
HID = 32
EPS_E = 1e-8                                 # 物理步:E / √(|E|² + ε)(|E|² 的 0.1% 分位數約 4e-7)
EPS_P = 1e-6                                 # 投影:√(|O|² + ε²) 與 atan2 的梯度分母
PHYS_ONLY_ITERS = [1, 2, 3]                  # 「只有物理步、沒有 CNN、不訓練」的參考(網路起點 + 迭代法 1–3 次)
REPRO_TOL = 1e-5
SUF = "_quick" if QUICK else ""
P7_RAW = RUN_ROOT / f"scan5e_p7_raw{SUF}.json"
FIG_DIR = Path("figs_scan5g")
if QUICK:
    EPOCHS = 2

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def task_of(t):
    return ORDER[t // 3], SEEDS[t % 3]


# ============================================================================
# 模型
# ============================================================================
class _SafeAtan2(torch.autograd.Function):
    """前向 = torch.atan2;反向的分母加 ε²(|O| = 0 時梯度有限;振幅 → 0 時上游梯度也 → 0,乘積有界)。"""

    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, g):
        y, x = ctx.saved_tensors
        d = x * x + y * y + EPS_P ** 2
        return g * x / d, -g * y / d


class Refine(nn.Module):
    """小 CNN(殘差):[Re O, Im O, 照明權重] → 修正量 [ΔRe, ΔIm];最後一層初始為 0(一開始等於恆等)。"""

    def __init__(self, hid=HID):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, hid, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(True))
        self.out = nn.Conv2d(hid, 2, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.body(x))


class P6Net(nn.Module):
    def __init__(self, group, cfg, pr, bs, R0):
        super().__init__()
        base, K, phys = VARIANTS[group]
        self.group, self.K, self.phys = group, K, phys
        self.init = s5e.build_model(base, cfg, pr)                      # B → scan_5c.PlainNet;NAF → scan_5e.NAFUNet
        self.phase_max = cfg.phase_max
        self.starts = s5c.geometry(cfg)[2]
        W = cfg.canvas
        self.W = W
        Pa = pr["Pa"].detach().float().cpu()
        wsum = torch.zeros(s5c.BOX_SIZE, s5c.BOX_SIZE)
        for y0, x0 in self.starts:
            wsum[y0:y0 + W, x0:x0 + W] += Pa ** 2
        # 物理常數不存進權重檔(由 cfg、探針重建;與 scan_5b.ap 同一組量)
        self.register_buffer("P", pr["P"].detach().cpu().clone(), persistent=False)
        self.register_buffer("Pc", pr["P"].detach().cpu().conj().resolve_conj().clone(), persistent=False)
        self.register_buffer("wsum", wsum, persistent=False)
        self.register_buffer("wn", (wsum / wsum.max())[None, None].clone(), persistent=False)
        self.register_buffer("bs", bs.detach().cpu().clone(), persistent=False)
        self.register_buffer("R0", R0.detach().cpu().bool().clone(), persistent=False)
        self.register_buffer("pa2", (Pa ** 2).sum(), persistent=False)
        self.delta = s5b.AP_DELTA * float(wsum.max())
        self.a = nn.Parameter(torch.zeros(K)) if phys else None          # α = sigmoid(a),初始 0.5
        self.refine = nn.ModuleList([Refine() for _ in range(K)])
        self.eps_e, self.eps_p = EPS_E, EPS_P
        self.alpha_override = None                                      # 單元測試用(α = 1)

    def alphas(self):
        if not self.phys:
            return []
        return [float(v) for v in torch.sigmoid(self.a.detach())]

    def initial(self, inp):
        """初始物體(方框,complex [N, B, B]):R0 內 = 初始網路的輸出,R0 外 = 常數(振幅 a₀、相位 φmax / 2)。"""
        y = self.init(inp)
        On = torch.polar(y[:, 0], y[:, 1])
        m = inp["meas"][:, s5c.CENTER]
        a0 = ((m * m * self.bs).sum((-2, -1)) / self.pa2).sqrt().clamp(max=1.0)      # = scan_5b.const_amp
        c = torch.polar(a0[:, None, None].expand_as(On.real).contiguous(), torch.full_like(On.real, self.phase_max / 2))
        return torch.where(self.R0, On, c)

    def ap_step(self, O, meas):
        """一次 AP 重疊投影(不含物體約束;= scan_5b.ap 的一次迭代,限於方框)。O complex [N, B, B];meas [N, 9, W, W]。"""
        W = self.W
        Oj = torch.stack([O[:, y0:y0 + W, x0:x0 + W] for y0, x0 in self.starts], 1)
        E = torch.fft.fftshift(torch.fft.fft2(self.P * Oj, norm="ortho"), dim=(-2, -1))
        mag = torch.sqrt(E.real * E.real + E.imag * E.imag + self.eps_e)
        E2 = torch.where(self.bs > 0, E * (meas / mag), E)
        psi2 = torch.fft.ifft2(torch.fft.ifftshift(E2, dim=(-2, -1)), norm="ortho")
        g = self.Pc * psi2
        num = self.delta * O
        for j, (y0, x0) in enumerate(self.starts):
            num[:, y0:y0 + W, x0:x0 + W] = num[:, y0:y0 + W, x0:x0 + W] + g[:, j]
        return num / (self.wsum + self.delta)

    def project(self, O):
        re, im = O.real, O.imag
        amp = torch.sqrt(re * re + im * im + self.eps_p ** 2).clamp(max=1.0)
        ph = _SafeAtan2.apply(im, re).clamp(0.0, self.phase_max)
        return amp, ph

    def stage(self, k, O, meas):
        if self.phys:
            al = torch.sigmoid(self.a[k]) if self.alpha_override is None else self.alpha_override
            O = O + al * (self.ap_step(O, meas) - O)
        d = self.refine[k](torch.cat([O.real[:, None], O.imag[:, None], self.wn.expand(O.shape[0], -1, -1, -1)], 1))
        O = O + torch.complex(d[:, 0], d[:, 1])
        return self.project(O)

    def forward(self, inp, all_stages=False):
        O = self.initial(inp)
        O0, outs = O, []
        for k in range(self.K):
            amp, ph = self.stage(k, O, inp["meas"])
            O = torch.polar(amp, ph)
            outs.append(torch.stack([amp, ph], 1))
        return (O0, outs) if all_stages else outs[-1]


def build_model(group, cfg, pr):
    if group in VARIANTS:
        from src.physics import beamstop_mask
        dev = pr["P"].device
        return P6Net(group, cfg, pr, beamstop_mask(cfg, device=dev), s5c.r0_box(cfg, pr, dev))
    return s5e.build_model(group, cfg, pr)


s5c.build_model = build_model                                        # 讓 scan_5c.load_net 也認得 P6


def prep(counts, bs, norm, group, cp):
    """P6:初始網路的輸入(同 B)+ 9 張的量測振幅;其他組同 scan_5c.prep。"""
    if group in VARIANTS:
        from src.hio import _measured_amp
        out = s5c.prep(counts, bs, norm, "B", cp)
        out["meas"] = _measured_amp(torch.stack(counts, 1), cp)
        return out
    return s5c.prep(counts, bs, norm, group, cp)


@torch.no_grad()
def predict(model, counts, bs, norm, group, cp, chunk=s5c.EVAL_CHUNK, all_stages=False):
    n = counts[0].shape[0]
    if not all_stages:
        return torch.cat([model(prep([c[i:i + chunk] for c in counts], bs, norm, group, cp)) for i in range(0, n, chunk)])
    O0s, outs = [], []
    for i in range(0, n, chunk):
        O0, o = model(prep([c[i:i + chunk] for c in counts], bs, norm, group, cp), all_stages=True)
        O0s.append(O0)
        outs.append(o)
    return torch.cat(O0s), [torch.cat([o[k] for o in outs]) for k in range(len(outs[0]))]


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def ds_loss(outs, target, R0):
    """deep supervision:各級輸出在 R0 上的 nerr_ph 平均。"""
    ls = [s5c.nerr_loss(o, target, R0) for o in outs]
    return sum(ls) / len(ls), ls


# ============================================================================
# 計時(同 scan_5e.time_groups;P6 含量測振幅的前處理)
# ============================================================================
@torch.no_grad()
def time_groups(groups, dev):
    torch.backends.cudnn.benchmark = True
    cfg, pr, cp, bs = s5c.setup(0, PROBE, dev)
    b0, size, _ = s5c.geometry(cfg)
    res = {}
    for Bl in ["64", "512"]:
        B = int(Bl) if not QUICK else {"64": 8, "512": 16}[Bl]
        fields = s5.make_fields(cfg, B, seed=cfg.test_seed + 99, device=dev)
        cnt = s5c.measure_box(fields[:, :, b0:b0 + size, b0:b0 + size].contiguous(), PROBE, pr, cp, bs, cfg, seed=0)
        norm = s5c.calibrate_norm(cnt)
        for g in groups:                                                     # 先全部跑過一次(GPU 暖機)
            build_model(g, cfg, pr).to(dev).eval()(prep(cnt, bs, norm, g, cp))
        for g in groups:
            net = build_model(g, cfg, pr).to(dev).eval()
            net(prep(cnt, bs, norm, g, cp))
            net(prep(cnt, bs, norm, g, cp))
            ts = []
            for _ in range(s5b.TIME_REPS):
                s5._sync(dev)
                t0 = time.perf_counter()
                net(prep(cnt, bs, norm, g, cp))
                s5._sync(dev)
                ts.append((time.perf_counter() - t0) / B * 1e3)
            res.setdefault(Bl, {})[g] = float(np.median(ts))
            del net
    return res


# ============================================================================
# 單元測試(協定 §18.4)
# ============================================================================
def unit_tests(dev, groups=None):
    from src.hio import _measured_amp
    groups = groups or ORDER
    print("=" * 70)
    print(f"優化 P6 單元測試(組:{' '.join(groups)})")
    print("=" * 70)
    cfg, pr, cp, bs = s5c.setup(0, PROBE, dev)
    b0, size, stb = s5c.geometry(cfg)
    F = s5.field_size(cfg)
    st = s5.window_starts(cfg, s5b.SCAN)
    R0b = s5c.r0_box(cfg, pr, dev)
    fld = s5.make_fields(cfg, 8, seed=11, device=dev)
    box = fld[:, :, b0:b0 + size, b0:b0 + size].contiguous()
    Ot = torch.polar(box[:, 0], box[:, 1])
    cnt = s5c.measure_box(box, PROBE, pr, cp, bs, cfg, seed=5)
    meas = _measured_amp(torch.stack(cnt, 1), cp)
    norm = s5c.calibrate_norm(cnt)

    with torch.no_grad():
        # (1) 物理步 = AP 的一次迭代(α = 1、ε = 0)
        torch.manual_seed(0)
        net = build_model("P6B", cfg, pr).to(dev).eval()
        net.eps_e = 0.0
        Ob = torch.polar(box[:, 0] * 0.7 + 0.2, box[:, 1] * 0.8 + 0.1)
        mine = net.ap_step(Ob, meas)
        ref = s5b.ap(cnt, bs, cp, pr, st, [1], False, s5c.to_field(Ob, cfg, F))[1][:, b0:b0 + size, b0:b0 + size]
        e1 = float((mine - ref).abs().max())
        check("物理步 = scan_5b.ap 的一次迭代(不投影;α = 1、ε = 0;方框 vs 場,< 1e-5)", e1 < 1e-5, f"最大差 {e1:.1e}")

        # (1b) 整個前向:α = 1、小 CNN 為初始的 0、ε = 0 → 網路起點 + AP-C K 次
        net.alpha_override, net.eps_p = 1.0, 0.0
        O0, outs = net(prep(cnt, bs, norm, "P6B", cp), all_stages=True)
        refk = s5b.ap(cnt, bs, cp, pr, st, list(range(1, net.K + 1)), True, s5c.to_field(O0, cfg, F))
        e1b = max(float((torch.polar(outs[k][:, 0], outs[k][:, 1]) - refk[k + 1][:, b0:b0 + size, b0:b0 + size]).abs().max())
                  for k in range(net.K))
        check(f"整個前向(α = 1、小 CNN = 0、ε = 0)= 同一起點 + scan_5b.ap 投影版 {net.K} 次(< 1e-5)", e1b < 1e-5,
              f"最大差 {e1b:.1e}")

        # (1c) 常數振幅 = scan_5b.const_amp
        m = meas[:, s5c.CENTER]
        a0 = ((m * m * net.bs).sum((-2, -1)) / net.pa2).sqrt().clamp(max=1.0)
        e1c = float((a0 - s5b.const_amp(cnt[s5c.CENTER], bs, cp, pr)).abs().max())
        out_r0 = bool((O0[:, ~R0b] - torch.polar(a0[:, None].expand(-1, int((~R0b).sum())),
                                                 torch.full((8, int((~R0b).sum())), cfg.phase_max / 2, device=dev))
                       ).abs().max() < 1e-6)
        check("初始物體:R0 外 = 常數(a₀ = scan_5b.const_amp,< 1e-6;相位 φmax / 2)", e1c < 1e-6 and out_r0,
              f"a₀ 最大差 {e1c:.1e}")

        # (2) 固定點:真值(無雜訊)經物理步不變
        c0 = Cfg.from_dict(cp.to_dict())
        c0.add_poisson = False
        cnt0 = s5c.measure_box(box, PROBE, pr, c0, bs, cfg, seed=0)
        meas0 = _measured_amp(torch.stack(cnt0, 1), c0)
        strong = net.wsum > 0.2 * net.wsum.max()
        fx = {}
        for eps in (0.0, EPS_E):
            net.eps_e = eps
            d = (net.ap_step(Ot, meas0) - Ot).abs()
            fx[eps] = (float(d.max()), float((d * strong).max()), float((d * R0b).max()))
        check(f"固定點:真值(無雜訊)經物理步不變(ε = 0:整個方框 < 1e-4;ε = {EPS_E:g}:照明 > 0.2 max 處 < 1e-3)",
              fx[0.0][0] < 1e-4 and fx[EPS_E][1] < 1e-3,
              f"ε = 0:最大差 {fx[0.0][0]:.1e};ε = {EPS_E:g}:整個方框 {fx[EPS_E][0]:.1e}、照明強處 {fx[EPS_E][1]:.1e}、"
              f"R0 {fx[EPS_E][2]:.1e}")

        # (3) 恆等起點:對照組在初始化時,R0 上的輸出 = 初始網路的輸出
        e3 = {}
        for g in [g for g in groups if not VARIANTS[g][2]]:
            torch.manual_seed(0)
            nc = build_model(g, cfg, pr).to(dev).eval()
            inp = prep(cnt, bs, norm, g, cp)
            y = nc.init(inp)
            o = nc(inp)
            e3[g] = float(((torch.polar(o[:, 0], o[:, 1]) - torch.polar(y[:, 0], y[:, 1])).abs() * R0b).max())
        if e3:
            check("恆等起點:對照組初始化時 R0 上的輸出 = 初始網路的輸出(< 1e-5)", all(v < 1e-5 for v in e3.values()),
                  " / ".join(f"{k} {v:.1e}" for k, v in e3.items()))

        # (5) 前向:大小、範圍、參數量
        info, ok_f = [], True
        for g in groups:
            torch.manual_seed(0)
            nf = build_model(g, cfg, pr).to(dev).eval()
            out = nf(prep(cnt, bs, norm, g, cp))
            ok_f &= (tuple(out.shape) == (8, 2, s5c.BOX_SIZE, s5c.BOX_SIZE) and bool(torch.isfinite(out).all())
                     and bool(out[:, 0].min() >= 0) and bool(out[:, 0].max() <= 1) and bool(out[:, 1].min() >= 0)
                     and bool(out[:, 1].max() <= cfg.phase_max + 1e-6))
            info.append(f"{g} 參數 {n_params(nf):,}(初始網路 {n_params(nf.init):,} + 每級 {n_params(nf.refine[0]):,} × {nf.K}"
                        + (f" + 步長 {nf.K}" if nf.phys else "") + ")")
        check("前向:輸出 [N, 2, 80, 80]、有限、振幅 ∈ [0, 1]、相位 ∈ [0, φmax]", ok_f, ";".join(info))

        # (6) 存檔 → 載入 → 輸出相同(物理常數不存進權重檔、由 cfg 重建)
        e6 = {}
        with tempfile.TemporaryDirectory() as td:
            for g in groups:
                torch.manual_seed(1)
                n1 = build_model(g, cfg, pr).to(dev).eval()
                for p in n1.parameters():                                     # 讓每個參數都非預設值
                    p.add_(0.01 * torch.randn_like(p))
                torch.save(n1.state_dict(), Path(td) / "m.pt")
                n2 = build_model(g, cfg, pr).to(dev).eval()
                n2.load_state_dict(torch.load(Path(td) / "m.pt", map_location=dev))
                inp = prep(cnt, bs, norm, g, cp)
                e6[g] = float((n1(inp) - n2(inp)).abs().max())
        check("存檔 → 載入 → 輸出相同", all(v == 0 for v in e6.values()), " / ".join(f"{k} {v:.1e}" for k, v in e6.items()))

    # (4) 梯度安全:含全零振幅區域 / 全零量測的樣本,前向與反向皆有限
    ok4, gmax = True, {}
    for g in groups:
        torch.manual_seed(0)
        ng = build_model(g, cfg, pr).to(dev).train()
        for p in ng.parameters():
            if p.dim() == 4 and not p.abs().sum() > 0:                        # 小 CNN 最後一層(初始 0)也給值,讓梯度流過整條路
                p.data.normal_(0, 1e-2)
        cz = [c.clone() for c in cnt]
        for c in cz:
            c[0] = 0                                                          # 樣本 0:全零量測(a₀ = 0、E_meas = 0)
        inp = prep(cz, bs, norm, g, cp)
        O = ng.initial(inp)
        O = torch.where(torch.arange(s5c.BOX_SIZE, device=dev)[None, :, None] < 40, torch.zeros_like(O), O)  # 上半部全零
        O[1] = 0                                                              # 樣本 1:整個物體為 0
        O = O.detach().requires_grad_(True)
        outs, Oc = [], O
        for k in range(ng.K):
            amp, ph = ng.stage(k, Oc, inp["meas"])
            Oc = torch.polar(amp, ph)
            outs.append(torch.stack([amp, ph], 1))
        loss, _ = ds_loss(outs, Ot, R0b)
        ng.zero_grad()
        loss.backward()
        fin = bool(torch.isfinite(loss)) and bool(torch.isfinite(O.grad).all()) and all(
            bool(torch.isfinite(p.grad).all()) for p in ng.parameters() if p.grad is not None)
        # 從輸入開始的整個前向(樣本 0 全零量測)
        ng.zero_grad()
        loss2, _ = ds_loss(ng(inp, all_stages=True)[1], Ot, R0b)
        loss2.backward()
        fin &= bool(torch.isfinite(loss2)) and all(bool(torch.isfinite(p.grad).all())
                                                   for p in ng.parameters() if p.grad is not None)
        ok4 &= fin
        gmax[g] = float(torch.nn.utils.clip_grad_norm_(ng.parameters(), float("inf")))
    check("梯度安全:物體含全零區域、全零樣本與全零量測時,前向與反向皆有限", ok4,
          " / ".join(f"{k} 梯度範數 {v:.2e}" for k, v in gmax.items()))

    # (7) 可學性:16 個樣本訓練 200 步(從未訓練的初始網路開始),deep supervision loss 降到初始的 50% 以下
    small = s5.make_fields(cfg, s5c.LEARN_N, seed=12, device=dev)[:, :, b0:b0 + size, b0:b0 + size].contiguous()
    Os = torch.polar(small[:, 0], small[:, 1])
    cs = s5c.measure_box(small, PROBE, pr, cp, bs, cfg, seed=3)
    norm_s = s5c.calibrate_norm(cs)
    learn = {}
    for g in groups:
        torch.manual_seed(0)
        nl = build_model(g, cfg, pr).to(dev).train()
        opt = torch.optim.Adam(nl.parameters(), lr=1e-3)
        inp = prep(cs, bs, norm_s, g, cp)
        first = None
        for _ in range(s5c.LEARN_STEPS):
            loss, _ = ds_loss(nl(inp, all_stages=True)[1], Os, R0b)
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nl.parameters(), CLIP)
            opt.step()
        nl.eval()
        with torch.no_grad():
            learn[g] = (first, float(ds_loss(nl(inp, all_stages=True)[1], Os, R0b)[0]), nl.alphas())
    check(f"可學性:{s5c.LEARN_N} 個樣本訓練 {s5c.LEARN_STEPS} 步,loss 降到初始的 50% 以下" + (";QUICK 只列出" if QUICK else ""),
          QUICK or all(b < 0.5 * a for a, b, _ in learn.values()),
          ";".join(f"{k} {a:.3f} → {b:.3f}" + (f"(α {' '.join(f'{x:.2f}' for x in al)})" if al else "")
                   for k, (a, b, al) in learn.items()))
    print(f"\n     裝置:{dev}")


# ============================================================================
# 訓練(協定 §18.3;迴圈同 scan_5c.train,加 warm start、deep supervision、梯度裁切)
# ============================================================================
def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def train(group, seed, dev, epochs=None):
    base, K, phys = VARIANTS[group]
    epochs = EPOCHS if epochs is None else epochs
    cfg, pr, cp, bs = s5c.setup(seed, PROBE, dev)
    R0 = s5c.r0_box(cfg, pr, dev)
    out = s5c.run_dir(group, PROBE, seed, N_TRAIN)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()

    init_dir = s5c.run_dir(base, PROBE, seed, N_TRAIN)
    init_meta = json.load(open(init_dir / "result.json"))
    norm = init_meta["norm"]                                              # 初始網路的輸入正規化常數(網路輸入不變)
    fields = s5c.make_train_fields(cfg, N_TRAIN, seed).to(dev)            # 與 B / NAF-B 相同的訓練場
    print(f"[init] {group} s{seed}:訓練場 {len(fields)} 個({time.time() - t0:.1f}s);初始網路 {init_dir.name}", flush=True)

    torch.manual_seed(seed)
    model = build_model(group, cfg, pr).to(dev)
    model.init.load_state_dict(torch.load(init_dir / "final.pt", map_location=dev))
    n_par = n_params(model)

    # 載入檢查(eval 模式):模型內的初始網路 = 獨立載入的初始網路;對照組初始化時 R0 上 = 初始網路
    ref_net, _, _ = s5c.load_net(base, PROBE, seed, N_TRAIN, cfg, pr, dev)
    model.eval()
    with torch.no_grad():
        sub = fields[:64]
        Osub = torch.polar(sub[:, 0], sub[:, 1])
        cnt = s5c.measure_box(sub, PROBE, pr, cp, bs, cfg, seed=seed + 98)
        inp = prep(cnt, bs, norm, group, cp)
        y_ref = ref_net(s5c.prep(cnt, bs, norm, base, cp))
        e_init = float((model.init(inp) - y_ref).abs().max())
        O0, outs = model(inp, all_stages=True)
        l_ref = float(s5c.nerr_loss(y_ref, Osub, R0))
        l_st = [float(s5c.nerr_loss(o, Osub, R0)) for o in outs]
        e_ctl = float(((torch.polar(outs[-1][:, 0], outs[-1][:, 1]) - torch.polar(y_ref[:, 0], y_ref[:, 1])).abs()
                       * R0).max())
    del ref_net
    check("載入檢查:模型內的初始網路 = 獨立載入的初始網路(< 1e-5)", e_init < 1e-5, f"最大差 {e_init:.1e}")
    if not phys:
        check("載入檢查:對照組初始化時 R0 上的輸出 = 初始網路(< 1e-5)", e_ctl < 1e-5, f"最大差 {e_ctl:.1e}")
    print(f"[init] params={n_par:,};64 個訓練場上 nerr:初始網路 {l_ref:.4f} → 各級(訓練前)"
          + " ".join(f"{v:.4f}" for v in l_st), flush=True)
    if not ok_all:
        raise SystemExit("❌ 載入檢查未通過,不訓練")

    p_init = list(model.init.parameters())
    ids = {id(p) for p in p_init}
    p_new = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.Adam([{"params": p_init, "lr": LR_INIT}, {"params": p_new, "lr": LR_NEW}])
    steps = len(fields) // BATCH
    total = epochs * steps + 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_INIT, LR_NEW], total_steps=total)
    meta = {"group": group, "probe": PROBE, "seed": seed, "n_train": N_TRAIN, "base": base, "stages": K, "phys": phys,
            "epochs": epochs, "batch": BATCH, "lr_init": LR_INIT, "lr_new": LR_NEW, "clip": CLIP, "hid": HID,
            "eps_e": EPS_E, "eps_p": EPS_P, "norm": norm, "params": n_par, "init_run": init_dir.name,
            "init_md5": _md5(init_dir / "final.pt"), "init_nerr64": l_ref, "stage_nerr64_before": l_st,
            "train_seed_base": s5c.train_chunk_seed(seed, 0), "quick": QUICK}
    json.dump(meta, open(out / "config_used.json", "w"), indent=2)

    ck = out / "ckpt.pt"
    start, hist = 0, []
    if ck.exists():
        d = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(d["model"])
        opt.load_state_dict(d["opt"])
        sched.load_state_dict(d["sched"])
        start, hist = d["epoch"] + 1, d["history"]
        print(f"[resume] 從 epoch {start} 續跑", flush=True)

    gstep = start * steps
    for ep in range(start, epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 100_003 + ep)              # = B 的前 20 個 epoch(同一組 batch 與雜訊)
        perm = torch.randperm(len(fields), generator=g)
        acc = {"loss": 0.0, "final": 0.0, "stages": [0.0] * K}
        n_ok, n_skip, n_clip, gns, te = 0, 0, 0, [], time.time()
        for i in range(steps):
            idx = perm[i * BATCH:(i + 1) * BATCH].to(dev)
            obj = fields[idx]
            with torch.no_grad():
                counts = s5c.measure_box(obj, PROBE, pr, cp, bs, cfg,
                                         seed=s5c.NOISE_SEED_BASE + seed * 10_000_019 + gstep)
                inp = prep(counts, bs, norm, group, cp)
            gstep += 1
            loss, ls = ds_loss(model(inp, all_stages=True)[1], torch.polar(obj[:, 0], obj[:, 1]), R0)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip += 1
                if sched.last_epoch + 1 < total:
                    sched.step()
                continue
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
            if not np.isfinite(gn):
                n_skip += 1
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch + 1 < total:
                    sched.step()
                continue
            opt.step()
            if sched.last_epoch + 1 < total:
                sched.step()
            n_ok += 1
            n_clip += gn > CLIP
            gns.append(gn)
            acc["loss"] += float(loss.detach())
            acc["final"] += float(ls[-1].detach())
            for k in range(K):
                acc["stages"][k] += float(ls[k].detach())
        acc["loss"] /= max(n_ok, 1)
        acc["final"] /= max(n_ok, 1)
        acc["stages"] = [v / max(n_ok, 1) for v in acc["stages"]]
        acc.update({"epoch": ep, "skipped": n_skip, "clip_frac": n_clip / max(n_ok, 1),
                    "grad_norm_median": float(np.median(gns)) if gns else None, "alphas": model.alphas(),
                    "sec": time.time() - te, "lr": sched.get_last_lr()})
        hist.append(acc)
        print(f"epoch {ep:03d}  loss={acc['loss']:.5f}  最後一級={acc['final']:.5f}  各級="
              + " ".join(f"{v:.4f}" for v in acc["stages"])
              + (f"  α=" + " ".join(f"{v:.3f}" for v in acc["alphas"]) if phys else "")
              + f"  梯度範數中位數 {acc['grad_norm_median'] or 0:.2e}(裁切 {100 * acc['clip_frac']:.0f}%)"
              + f"  lr={acc['lr'][0]:.1e}/{acc['lr'][1]:.1e}  跳過 {n_skip}  {acc['sec']:.1f}s", flush=True)
        tmp = out / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "history": hist}, tmp)
        os.replace(tmp, ck)                                               # 寫完才換名:時限砍在寫入中途也不會壞檔

    # 訓練場上的 nerr_ph(固定雜訊 seed;與 scan_5c.train 相同的做法)
    model.eval()
    with torch.no_grad():
        sub = fields[:s5c.TRAIN_EVAL_N]
        counts = s5c.measure_box(sub, PROBE, pr, cp, bs, cfg, seed=seed + 99)
        pred = predict(model, counts, bs, norm, group, cp)
        O = torch.polar(sub[:, 0], sub[:, 1])
        Et = ((O.abs() ** 2) * R0).sum((1, 2))
        tr = [float(s5c.nerr_loss(pred[i:i + 1], O[i:i + 1], R0)) for i in range(len(sub)) if Et[i] > 1e-12]
        pooled = float(s5c.nerr_loss(pred, O, R0))
    meta.update({"history": hist, "train_nerr_ph": float(np.mean(tr)), "train_nerr_pooled": pooled,
                 "alphas": model.alphas(), "train_sec": time.time() - t0})
    tmp = out / "final.tmp"
    torch.save(model.state_dict(), tmp)
    json.dump(meta, open(out / "result.json", "w"), indent=2)
    os.replace(tmp, out / "final.pt")                                     # final.pt 最後才出現(= 完成的標記)
    print(f"[done] {group} s{seed}:訓練場 nerr_ph {np.mean(tr):.4f}"
          + (f"  α {' '.join(f'{v:.3f}' for v in model.alphas())}" if phys else "")
          + f"  {time.time() - t0:.0f}s → {out}", flush=True)


# ============================================================================
# 評估(驗證場;協定 §18.5)
# ============================================================================
@torch.no_grad()
def evaluate(dev):
    import probe_4_2b as pb
    env = json.load(open(s5e.VAL_ENV))["envelope"]
    groups = BASES + ORDER
    cfg0 = s5.load_cfg(0)
    probes0 = {k: v for k, v in pb.build_probes(cfg0, dev).items() if k in s5.PROBES}
    print("  重新計時:迭代法與網路", flush=True)
    tim = s5b.timing(cfg0, probes0, dev)
    tg = time_groups(groups, dev)
    for B in tg:
        tim[B].update({f"net:{g}": v for g, v in tg[B].items()})
    res = {g: [] for g in groups}
    missing = []
    for s in SEEDS:
        d = s5d.setup_seed(s, s5e.val_seeds, dev)
        cfg, cp, bs, O, R0, F, n = d["cfg"], d["cp"], d["bs"], d["O"], d["R0"], d["F"], d["n"]
        for g in groups:
            net, norm, meta = s5c.load_net(g, PROBE, s, N_TRAIN, cfg, d["pr"], dev)
            if net is None:
                missing.append(f"{g}:s{s}")
                res[g].append(None)
                continue
            rec = {"params": meta["params"], "train_nerr_ph": meta["train_nerr_ph"],
                   "train_nerr_pooled": meta.get("train_nerr_pooled"), "train_sec": meta.get("train_sec"),
                   "history_last": meta["history"][-1]}
            if g in VARIANTS:
                O0, outs = predict(net, d["counts"], bs, norm, g, cp, all_stages=True)
                pred = outs[-1]
                rec["stages"] = [s5.field_metrics(s5c.to_field(O0, cfg, F), O, R0)["nerr_ph"]] + [
                    s5.field_metrics(s5c.to_field(torch.polar(o[:, 0], o[:, 1]), cfg, F), O, R0)["nerr_ph"] for o in outs]
                rec["alphas"] = net.alphas()
            else:
                pred = predict(net, d["counts"], bs, norm, g, cp)
            est = s5c.to_field(torch.polar(pred[:, 0], pred[:, 1]), cfg, F)
            rec["net"] = s5.field_metrics(est, O, R0)
            if g in ("B", NAF_B):
                # 「只有物理步、沒有 CNN、不訓練」:網路起點(R0 外常數,同 P6 的起點)+ 迭代法 1–3 次
                a0 = s5b.const_amp(d["counts"][s5c.CENTER], bs, cp, d["pr"])
                cst = torch.polar(a0[:, None, None].expand(n, F, F).contiguous(),
                                  torch.full((n, F, F), cfg.phase_max / 2, device=dev))
                O0f = torch.where(R0, est, cst)
                for meth in ["AP-C", "ePIE-C"]:
                    o = s5b.run_method(meth, d["counts"], bs, cp, d["pr"], d["st"], PHYS_ONLY_ITERS, O0f, d["init"])
                    rec[meth] = {str(it): s5.field_metrics(o[it], O, R0)["nerr_ph"] for it in PHYS_ONLY_ITERS}
                    del o
            res[g].append(rec)
            del net, pred, est
        print(f"  [seed {s}] 完成", flush=True)
    return env, res, tim, missing


def vals(res, g, field="nerr_ph"):
    if g not in res or any(r is None for r in res[g]):
        return None
    return np.array([r["net"][field] for r in res[g]], float)


def repro_check(res):
    """B、C、NAF-B 在驗證場的逐 seed nerr_ph = P7 評估(scan5e_p7_raw.json)。"""
    if not P7_RAW.exists():
        print(f"  ⚠️ 找不到 {P7_RAW} → 略過重現檢查")
        return
    old = json.load(open(P7_RAW))["results"]
    diffs = {}
    for g in BASES:
        a = vals(res, g)
        if a is None or g not in old or any(r is None for r in old[g]):
            continue
        diffs[g] = float(np.max(np.abs(a - np.array([r["net"]["nerr_ph"] for r in old[g]]))))
    check(f"重現:B、C、NAF-B 的驗證場 nerr_ph = P7 評估(§16.6;< {REPRO_TOL:g})",
          bool(diffs) and all(v < REPRO_TOL for v in diffs.values()), " / ".join(f"{k} {v:.1e}" for k, v in diffs.items()))


def report(env, res, tim, missing):
    v = {"missing": missing}
    Bs = ["64", "512"]
    groups = list(res)
    print("\n" + "=" * 100)
    print("計時(ms / 張;本 job 重新量;網路含輸入前處理)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {tim[B][k]:.4f}" for k in
                                         ["ePIE-C", "AP-C", f"K-HIO:{PROBE}"] + [f"net:{g}" for g in groups]))
    if missing:
        print(f"  ⚠️ 缺少的網路:{', '.join(missing)}(相關比較略過)")
    repro_check(res)

    print("\n" + "=" * 100)
    print(f"驗證場({PROBE},R0 的 nerr_ph,3 seeds)與 vs 包絡(對手 = 同時間內最好的迭代法設定)")
    print("=" * 100)
    for g in groups:
        a = vals(res, g)
        if a is None:
            print(f"  {g}:缺")
            continue
        tr = np.mean([r["train_nerr_ph"] for r in res[g]])
        secs = [r["train_sec"] for r in res[g] if r.get("train_sec")]
        print(f"  {g}:{a.mean():.4f}({' '.join(f'{x:.4f}' for x in a)})  參數 {res[g][0]['params']:,}  訓練場 {tr:.4f}"
              + (f"  訓練時間 {np.mean(secs) / 60:.0f} 分" if secs else ""))
        last = np.mean([r["history_last"].get("final", r["history_last"].get("sup", np.nan)) for r in res[g]])
        pooled = [r["train_nerr_pooled"] for r in res[g] if r.get("train_nerr_pooled") is not None]
        if pooled and np.mean(pooled) > 2 * last:
            print(f"     ⚠️ 訓練場 nerr(eval 模式,{np.mean(pooled):.4f})比最後一個 epoch 的訓練 loss({last:.4f})大 2 倍以上:"
                  "先檢查 BatchNorm 統計或訓練是否正常,再判讀")
        gv = {"nerr": list(a)}
        if g in VARIANTS:
            st = np.array([r["stages"] for r in res[g]])
            gv["stages"] = st.mean(0).tolist()
            gv["alphas"] = [r["alphas"] for r in res[g]]
            print("     (p5) 各級 nerr_ph(0 = 起點):" + " → ".join(f"{x:.4f}" for x in st.mean(0))
                  + ("" if not VARIANTS[g][2] else
                     ";學到的步長 α(逐 seed):" + " | ".join(" ".join(f"{x:.3f}" for x in r["alphas"]) for r in res[g])))
        for B in Bs:
            tn = tim[B][f"net:{g}"]
            e = s5b.envelope(env, PROBE, tn, tim[B])
            r = s5b.ratio(a, e[2])
            gv[B] = {"t_net": tn, "opp": [e[0], e[1], list(e[2])], "ratio": list(r)}
            print(f"     (n4) batch {B}:{tn:.4f} ms vs 對手 {s5b.fmt_conf(e[0])}({e[1]} 次){e[2].mean():.4f}"
                  f" → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}")
        gv["useful"] = gv["64"]["ratio"][0] == "較準"
        gv["robust"] = gv["useful"] and gv["512"]["ratio"][0] == "較準"
        print(f"     ▶ {'穩健有用(batch 64 與 512 都較準)' if gv['robust'] else ('有用(只在 batch 64)' if gv['useful'] else '未達有用')}")
        if g in ("B", NAF_B):
            po = {m: np.mean([[r[m][str(it)] for it in PHYS_ONLY_ITERS] for r in res[g]], 0) for m in ["AP-C", "ePIE-C"]}
            gv["phys_only"] = {m: po[m].tolist() for m in po}
            print(f"     (參考)只有物理步、不訓練:{g} 起點 + " + ";".join(
                f"{m} " + " / ".join(f"{k} 次 {x:.4f}" for k, x in zip(PHYS_ONLY_ITERS, po[m])) for m in po)
                + f"(時間 = 網路 + 次數 × 每次;batch 64 每次 AP-C {tim['64']['AP-C']:.4f}、ePIE-C {tim['64']['ePIE-C']:.4f} ms)")
        v[g] = gv

    def pair(a_g, b_g):
        a, b = vals(res, a_g), vals(res, b_g)
        if a is None or b is None:
            return None
        r = s5b.ratio(a, b)
        return r, tim["64"][f"net:{a_g}"] / tim["64"][f"net:{b_g}"], tim["512"][f"net:{a_g}"] / tim["512"][f"net:{b_g}"]

    print("\n" + "=" * 100)
    print("配對比較(協定 §18.5;比值 = 逐 seed 誤差比值的幾何平均,< 1 = 前者較準)")
    print("=" * 100)
    # (p1) 物理回饋的效果
    pp = pair("P6B", "P6Bctl")
    if pp:
        r, t64, t512 = pp
        lab = ("物理步有貢獻" if (r[1] < 1 and r[2] < -2) else
               ("物理步反而變差" if (r[1] > 1 and r[2] > 2) else "沒有統計上明確的差異"))
        print(f"  (p1) 物理回饋的效果:P6B vs P6Bctl 比值 {r[1]:.3f}(z {r[2]:+.1f});時間 × {t64:.2f}(b64)/ × {t512:.2f}(b512)"
              f" → {lab}")
        v["p1"] = {"ratio": list(r), "t64": t64, "t512": t512, "label": lab}
    else:
        print("  (p1) P6B vs P6Bctl:缺")
    # (p2)(p3)(p4)
    for tag, g, base in [("p2", "P6B", "B"), ("p3", "P6NAF", NAF_B), ("p4", "P6Bctl", "B")]:
        pp = pair(g, base)
        if pp is None:
            print(f"  ({tag}) {g} vs {base}:缺")
            continue
        r, t64, t512 = pp
        more_acc = r[1] < 1 and r[2] < -2
        thr = r[1] <= 0.9 and r[2] < -2
        env64, env512 = v[g]["64"]["ratio"][0] == "較準", v[g]["512"]["ratio"][0] == "較準"
        improved = more_acc and env64
        print(f"  ({tag}) {g} vs {base}{'(描述)' if tag == 'p4' else ''}:誤差比值 {r[1]:.3f}(z {r[2]:+.1f});"
              f"推論時間 × {t64:.2f}(b64)/ × {t512:.2f}(b512)")
        print(f"       更準:{'✅' if more_acc else '—'}{'(達 −10% 門檻)' if thr else ''}  "
              f"自己的時間下 vs 包絡:b64 {'較準 ✅' if env64 else '未達 ❌'}、b512 {'較準' if env512 else '未達'}"
              f"  → {'有改善' if improved else '無改善'}" + ("(且穩健有用)" if improved and env512 else ""))
        v[tag] = {"ratio": list(r), "t64": t64, "t512": t512, "more_acc": more_acc, "thr": thr,
                  "env64": env64, "env512": env512, "improved": improved}
    # 描述:與其他網路
    for g, h in [("P6B", "C"), ("P6B", NAF_B), ("P6NAF", "C")]:
        pp = pair(g, h)
        if pp:
            r, t64, t512 = pp
            print(f"  (描述) {g} vs {h}:誤差比值 {r[1]:.3f}(z {r[2]:+.1f});推論時間 × {t64:.2f}(b64)/ × {t512:.2f}(b512)")
            v[f"{g}_vs_{h}"] = {"ratio": list(r), "t64": t64, "t512": t512}
    print("\n  規則(優化方案 §7.2 的 P6 例外):對初始網路的比值 < 1 且 z < −2,且在自己的推論時間下仍贏包絡(至少 batch 64)→ 有改善;")
    print("  參考門檻 −10%。採用與否:回報後與使用者討論決定(§5.2b);採用者成為 P5 的老師 / 學生候選。")
    return v


def make_figure(env, res, tim):
    plt = s5._plt()
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    col = {"B": "#2a78d6", "C": "#eb6834", NAF_B: "#1baf7a", "P6B": "#0b3d91", "P6Bctl": "#8a8a8a", "P6NAF": "#0b6e4f"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, B in zip(axs, ["64", "512"]):
        tt = sorted({s5b.cfg_time(c, it, tim[B], PROBE) for c in s5b.configs() for it in s5b.stops(c)} - {0.0})
        ax.plot(tt, [s5b.envelope(env, PROBE, T, tim[B])[2].mean() for T in tt], color="#0b0b0b", lw=2,
                label="iterative envelope (validation)")
        for g in res:
            a = vals(res, g)
            if a is None:
                continue
            ax.plot([tim[B][f"net:{g}"]], [a.mean()], "s" if g in VARIANTS else "o", ms=7, color=col.get(g, "#8a5cd1"),
                    label=g)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{PROBE}, batch {B} (validation fields)", fontsize=10)
        ax.set_xlabel("time per sample (ms)")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    axs[0].set_ylabel("nerr on R0 (global phase only)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    p = FIG_DIR / "scan5g_p6.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


# ============================================================================
def check_files(need_models=True, need_env=False, need_p6=False):
    need = [Path(f) for f in ("scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py", "scan_5e.py", "probe_4_2b.py",
                              "probe_4_1.py", "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    if need_models:
        need += [s5c.run_dir(g, PROBE, s, N_TRAIN) / f for g in ("B", NAF_B) for s in SEEDS
                 for f in ("final.pt", "result.json")]
    if need_env:
        need.append(s5e.VAL_ENV)
        need += [s5c.run_dir("C", PROBE, s, N_TRAIN) / f for s in SEEDS for f in ("final.pt", "result.json")]
    if need_p6:
        need += [s5c.run_dir(g, PROBE, s, N_TRAIN) / f for g in ORDER for s in SEEDS for f in ("final.pt", "result.json")]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="單元測試(三組)")
    ap.add_argument("--train", action="store_true", help="訓練一個網路(配合 --task 0–8)")
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--eval", action="store_true", help="驗證場上評估")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    if a.check:
        if not check_files(need_models=True):
            sys.exit(1)
        unit_tests(dev)
        print("\n" + ("✅ 全部通過" if ok_all else "❌ 有項目未通過"))
        sys.exit(0 if ok_all else 1)
    if a.train:
        if a.task is None or not 0 <= a.task <= 8:
            raise SystemExit("--train 需要 --task 0–8")
        if not check_files(need_models=True):
            sys.exit(1)
        g, s = task_of(a.task)
        base, K, phys = VARIANTS[g]
        print(f"[task {a.task}] {g}(初始網路 {base}、{K} 級、{'有' if phys else '沒有'}物理步)、{PROBE}、seed {s}、"
              f"訓練場 {N_TRAIN}、{EPOCHS} epochs", flush=True)
        unit_tests(dev, [g])
        if not ok_all:
            print("\n❌ 單元測試未通過,不訓練")
            sys.exit(1)
        train(g, s, dev)
        print("✅ 訓練完成")
        return
    if a.eval:
        if not check_files(need_models=True, need_env=True, need_p6=True):
            sys.exit(1)
        env, res, tim, missing = evaluate(dev)
        raw = RUN_ROOT / f"scan5g_p6_raw{SUF}.json"
        json.dump({"results": res, "timing": tim, "missing": missing, "quick": QUICK}, open(raw, "w"))
        print(f"  原始結果先存檔:{raw}")
        verdict = report(env, res, tim, missing)
        json.dump({"results": res, "timing": tim, "verdict": verdict, "quick": QUICK},
                  open(RUN_ROOT / f"scan5g_p6{SUF}.json", "w"))
        make_figure(env, res, tim)
        print("\n" + ("✅ 內建檢查全部通過" if ok_all else "❌ 有項目未通過"))
        print("把完整輸出貼給 Claude")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
