#!/usr/bin/env python
"""優化第三步:P5 蒸餾(縮小版 P6 當學生)+ 探索性的 P6B4(階段五協定 §十九)。

組別(各 3 seeds):
  P5A     U-Net 寬度 16 + 3 級(小 CNN 32 通道),老師 P6B,從頭 60 epochs       正式
  P5Actl  同 P5A,沒有老師                                                    正式:蒸餾的對照
  P5B     U-Net 寬度 16 + 3 級(小 CNN 16 通道),老師 P6B                     正式:更激進的縮小
  P6B4    B + 4 級(warm start,同 P6B 的訓練)                                 探索性
學生的 loss:各級平均,每級 = (1 − w)·nerr(學生, 真值) + w·nerr(學生, 老師的最終輸出);w = 0.5(P5Actl:0)。
學生要先通過預先計時(batch 64 ≤ 0.83 × P6B、batch 512 ≤ 1.00 × P6B)才訓練。

用法(需在計算節點執行;需 scan_5.py … scan_5g.py 與其依賴、B / P6B / P6NAF 的 100k 模型、驗證場包絡):
    python scan_5h.py --check              # 單元測試
    python scan_5h.py --smoke              # 送件前的迷你全流程(dev 節點,約 10 分鐘;輸出到 scan5h_smoke/)
    python scan_5h.py --prep               # 單元測試 + 預先計時與選定(run_scan5h_prep.sh)
    python scan_5h.py --train-p6b4 --task 0-2   # P6B4(run_scan5h_p6b4.sh)
    python scan_5h.py --train --task 0-8   # 學生(run_scan5h_train.sh;0–2 = P5A、3–5 = P5Actl、6–8 = P5B)
    python scan_5h.py --eval               # 驗證場上評估(run_scan5h_eval.sh)
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_5g as g                                                  # noqa: E402  (也載入 scan_5 / 5b / 5c / 5d / 5e)
from src.config import Cfg                                          # noqa: E402

s5, s5b, s5c, s5d, s5e = g.s5, g.s5b, g.s5c, g.s5d, g.s5e
RUN_ROOT = s5.RUN_ROOT
QUICK = g.QUICK
PROBE = g.PROBE
SEEDS = g.SEEDS
N_TRAIN = g.N_TRAIN
TEACHER = "P6B"
EXPLORE = "P6B4"
STUDENTS = {"P5A": (16, 32, 0.5), "P5Actl": (16, 32, 0.0), "P5B": (16, 16, 0.5)}   # U-Net 寬度、小 CNN 通道、蒸餾權重 w
STUDENT_ORDER = ["P5A", "P5Actl", "P5B"]
NEW = [EXPLORE] + STUDENT_ORDER
BASE_EVAL = ["B", "P6B", "P6NAF"]
g.VARIANTS.update({EXPLORE: ("B", 4, True), **{k: ("B", 3, True) for k in STUDENTS}})   # 讓 scan_5g.prep 加上量測振幅
STUDENT_EPOCHS = 60
P6B4_EPOCHS = g.EPOCHS                       # 20
LR = s5c.LR                                  # 2e-3(同 B)
TIME_FRAC_64, TIME_FRAC_512 = 0.83, 1.00
TEACHER_TOL, NORM_TOL = 1e-4, 1e-3           # 正規化:合理性檢查(實際使用 B 的常數;不同 GPU 的加總順序可差約 1e-6)
TIME_LIMIT_H = {"student": 6.0, "p6b4": 4.0}
SUF = "_quick" if QUICK else ""
PREP_JSON = RUN_ROOT / f"scan5h_prep{SUF}.json"
P6_RAW = RUN_ROOT / f"scan5g_p6_raw{SUF}.json"
SMOKE_DIR = RUN_ROOT / f"scan5h_smoke{SUF}"
SMOKE_FIELDS = 2000                          # ≥ CAL_N 與 TRAIN_EVAL_N,且與正式訓練的前 2000 個場逐位元相同
SMOKE_EVAL_N = 64
FIG_DIR = Path("figs_scan5h")
LOG = "/work/elviss0915/runs/logs"
if QUICK:
    STUDENT_EPOCHS = 2
    SMOKE_FIELDS = s5c.N_TRAIN
    SMOKE_EVAL_N = 16

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and g.ok_all


def task_student(t):
    return STUDENT_ORDER[t // 3], SEEDS[t % 3]


def log_hint(grp, s):
    if grp == EXPLORE:
        return f"{LOG}/cdi_scan5h_p4_*_{SEEDS.index(s)}.out 與 .err"
    if grp in STUDENT_ORDER:
        return f"{LOG}/cdi_scan5h_tr_*_{STUDENT_ORDER.index(grp) * 3 + SEEDS.index(s)}.out 與 .err"
    return f"{LOG}/cdi_scan5h_ev_*.err(基準網路載入失敗)"


# ============================================================================
# 模型:學生 = 縮小版的 P6(前面的 U-Net 寬度 16;小 CNN 可縮小)
# ============================================================================
class StudentNet(g.P6Net):
    def __init__(self, group, cfg, pr, bs, R0):
        super().__init__(group, cfg, pr, bs, R0)                         # 物理步、起點、投影、步長同 P6
        width, hid, _ = STUDENTS[group]
        c = Cfg.from_dict(cfg.to_dict())
        c.base_channels = width
        self.init = s5c.PlainNet(c, "B")                                 # 同 B 的 U-Net(輸入 19 通道、下採樣 3 次),寬度 16
        self.refine = nn.ModuleList([g.Refine(hid) for _ in range(self.K)])


_g_build = g.build_model


def build_model(group, cfg, pr):
    if group in STUDENTS:
        from src.physics import beamstop_mask
        dev = pr["P"].device
        return StudentNet(group, cfg, pr, beamstop_mask(cfg, device=dev), s5c.r0_box(cfg, pr, dev))
    return _g_build(group, cfg, pr)


g.build_model = build_model                                          # scan_5g 的計時、單元測試也認得學生
s5c.build_model = build_model                                        # scan_5c.load_net 也認得


def kd_loss(outs, Ot, Tc, w, R0):
    """各級平均;每級 = (1 − w)·nerr(·, 真值) + w·nerr(·, 老師)。回傳 (loss, 真值項, 老師項, 各級真值項)。"""
    l_true, ls = g.ds_loss(outs, Ot, R0)
    if Tc is None or w == 0:
        return l_true, l_true, None, ls
    l_kd, _ = g.ds_loss(outs, Tc, R0)
    return (1 - w) * l_true + w * l_kd, l_true, l_kd, ls


def train_nerr(model, fields, grp, norm, pr, cp, bs, cfg, R0, seed):
    """訓練場上的 nerr_ph(前 TRAIN_EVAL_N 個、雜訊 seed + 99;與 scan_5c / scan_5g 的做法相同)。"""
    sub = fields[:s5c.TRAIN_EVAL_N]
    counts = s5c.measure_box(sub, PROBE, pr, cp, bs, cfg, seed=seed + 99)
    pred = g.predict(model, counts, bs, norm, grp, cp)
    O = torch.polar(sub[:, 0], sub[:, 1])
    Et = ((O.abs() ** 2) * R0).sum((1, 2))
    tr = [float(s5c.nerr_loss(pred[i:i + 1], O[i:i + 1], R0)) for i in range(len(sub)) if Et[i] > 1e-12]
    return float(np.mean(tr)), float(s5c.nerr_loss(pred, O, R0))


# ============================================================================
# 單元測試
# ============================================================================
def unit_tests(dev, groups):
    arch = [x for x in groups if x != "P5Actl"] or ["P5A"]              # P5Actl 與 P5A 同架構
    g.unit_tests(dev, arch)
    print("=" * 70)
    print("P5 / P6B4 另加的檢查")
    print("=" * 70)
    cfg, pr, cp, bs = s5c.setup(0, PROBE, dev)
    info, ok_a = [], True
    for grp in groups:
        torch.manual_seed(0)
        net = build_model(grp, cfg, pr)
        if grp in STUDENTS:
            width, hid, _ = STUDENTS[grp]
            ok_a &= (net.init.unet.inc.f[0].out_channels == width and net.refine[0].out.in_channels == hid
                     and net.K == 3 and net.phys and isinstance(net.init, s5c.PlainNet))
        else:
            ok_a &= (net.K == 4 and net.phys and g.n_params(net.init) == g.n_params(s5c.PlainNet(cfg, "B")))
        info.append(f"{grp} 參數 {g.n_params(net):,}(前面 {g.n_params(net.init):,} + 每級 {g.n_params(net.refine[0]):,} × {net.K}"
                    f" + 步長 {net.K})")
    check("架構:學生 = U-Net 寬度 16 + 3 級(小 CNN 32 / 16 通道);P6B4 = B + 4 級", ok_a, ";".join(info))
    # 蒸餾 loss
    b0, size, _ = s5c.geometry(cfg)
    fld = s5.make_fields(cfg, 8, seed=11, device=dev)[:, :, b0:b0 + size, b0:b0 + size].contiguous()
    Ot = torch.polar(fld[:, 0], fld[:, 1])
    cnt = s5c.measure_box(fld, PROBE, pr, cp, bs, cfg, seed=1)
    norm = s5c.calibrate_norm(cnt)
    R0 = s5c.r0_box(cfg, pr, dev)
    torch.manual_seed(0)
    net = build_model("P5A", cfg, pr).to(dev).eval()
    with torch.no_grad():
        outs = net(g.prep(cnt, bs, norm, "P5A", cp), all_stages=True)[1]
        a = kd_loss(outs, Ot, Ot, 0.5, R0)
        b = kd_loss(outs, Ot, None, 0.0, R0)
        ref = g.ds_loss(outs, Ot, R0)[0]
        other = torch.polar(fld[:, 0] * 0.5, fld[:, 1])
        c = kd_loss(outs, Ot, other, 0.5, R0)
    e1, e2 = float((a[0] - ref).abs()), float((b[0] - ref).abs())
    e3 = float((c[0] - (0.5 * ref + 0.5 * g.ds_loss(outs, other, R0)[0])).abs())
    check("蒸餾 loss:老師 = 真值 → = 真值 loss;w = 0 → 只剩真值項;一般情況 = 兩項各半(< 1e-6)",
          max(e1, e2, e3) < 1e-6, f"差 {e1:.1e} / {e2:.1e} / {e3:.1e}")
    print(f"\n     裝置:{dev}")


# ============================================================================
# 預先計時與選定(協定 §19.3)
# ============================================================================
def step_time(grp, dev, reps=None):
    """一個訓練步(batch 128;含量測模擬、前處理、老師的前向、反向與更新)的時間(秒)。權重與時間無關,用未訓練的權重。"""
    reps = reps or (3 if QUICK else 10)
    cfg, pr, cp, bs = s5c.setup(0, PROBE, dev)
    b0, size, _ = s5c.geometry(cfg)
    B = g.BATCH if not QUICK else 8
    obj = s5.make_fields(cfg, B, seed=cfg.test_seed + 7, device=dev)[:, :, b0:b0 + size, b0:b0 + size].contiguous()
    norm = s5c.calibrate_norm(s5c.measure_box(obj, PROBE, pr, cp, bs, cfg, seed=0))
    R0 = s5c.r0_box(cfg, pr, dev)
    model = build_model(grp, cfg, pr).to(dev).train()
    teacher = None
    if grp in STUDENTS and STUDENTS[grp][2] > 0:
        teacher = build_model(TEACHER, cfg, pr).to(dev).eval()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    w = STUDENTS[grp][2] if grp in STUDENTS else 0.0

    def one(k):
        with torch.no_grad():
            counts = s5c.measure_box(obj, PROBE, pr, cp, bs, cfg, seed=k)
            inp = g.prep(counts, bs, norm, grp, cp)
            Tc = None
            if teacher is not None:
                T = teacher(inp)
                Tc = torch.polar(T[:, 0], T[:, 1])
        loss = kd_loss(model(inp, all_stages=True)[1], torch.polar(obj[:, 0], obj[:, 1]), Tc, w, R0)[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), g.CLIP)
        opt.step()

    for k in range(2):
        one(k)
    s5._sync(dev)
    t0 = time.perf_counter()
    for k in range(reps):
        one(100 + k)
    s5._sync(dev)
    return (time.perf_counter() - t0) / reps


def prep_timing(dev, out_json):
    groups = ["B", TEACHER, EXPLORE, "P5A", "P5B"]
    print("\n  預先計時(未訓練權重;ms / 張;含輸入前處理)", flush=True)
    tim = g.time_groups(groups, dev)
    cfg = s5.load_cfg(0)
    pr_cpu = s5c.setup(0, PROBE, torch.device("cpu"))[1]
    params = {x: g.n_params(build_model(x, cfg, pr_cpu)) for x in groups}
    tT64, tT512 = tim["64"][TEACHER], tim["512"][TEACHER]
    for x in groups:
        print(f"    {x:<7} 參數 {params[x]:>10,}  batch 64 {tim['64'][x]:.4f}(× {tim['64'][x] / tT64:.2f} P6B)"
              f"  batch 512 {tim['512'][x]:.4f}(× {tim['512'][x] / tT512:.2f} P6B)")
    sel = {}
    for x in ["P5A", "P5B"]:
        sel[x] = bool(tim["64"][x] <= TIME_FRAC_64 * tT64 and tim["512"][x] <= TIME_FRAC_512 * tT512)
    sel["P5Actl"] = sel["P5A"]
    print(f"\n  選定規則:batch 64 ≤ {TIME_FRAC_64} × P6B 且 batch 512 ≤ {TIME_FRAC_512} × P6B(協定 §19.3)")
    for x in STUDENT_ORDER:
        print(f"    {x}:{'✅ 選定 → 訓練' if sel[x] else '— 未選定 → 不訓練'}")
    print("\n  預估訓練時間(一個訓練步的時間 × 步數;描述)", flush=True)
    steps = N_TRAIN // g.BATCH
    est = {}
    for x in [EXPLORE] + STUDENT_ORDER:
        st = step_time(x, dev)
        ep = STUDENT_EPOCHS if x in STUDENTS else P6B4_EPOCHS
        hours = st * steps * ep / 3600
        est[x] = {"step_sec": st, "hours": hours}
        lim = TIME_LIMIT_H["student" if x in STUDENTS else "p6b4"]
        warn = f"  ⚠️ 接近時限 {lim:.0f} 小時(超時會從存檔續跑,需重送)" if hours > 0.85 * lim else ""
        print(f"    {x:<7} 每步 {st:.3f} s × {steps} 步 × {ep} epochs ≈ {hours:.2f} 小時(另加產生訓練場約數分鐘){warn}")
    tmp = Path(str(out_json) + ".tmp")
    json.dump({"timing": tim, "params": params, "selected": sel, "train_estimate": est, "quick": QUICK},
              open(tmp, "w"), indent=2)
    os.replace(tmp, out_json)
    print(f"  存檔:{out_json}")
    return sel


# ============================================================================
# 訓練(協定 §19.4)
# ============================================================================
def train(group, seed, dev, n_fields=None, epochs=None, out=None):
    student = group in STUDENTS
    n_fields = N_TRAIN if n_fields is None else n_fields
    epochs = epochs or (STUDENT_EPOCHS if student else P6B4_EPOCHS)
    w = STUDENTS[group][2] if student else 0.0
    cfg, pr, cp, bs = s5c.setup(seed, PROBE, dev)
    R0 = s5c.r0_box(cfg, pr, dev)
    out = Path(out) if out is not None else s5c.run_dir(group, PROBE, seed, N_TRAIN)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()

    b_dir = s5c.run_dir("B", PROBE, seed, N_TRAIN)
    norm = json.load(open(b_dir / "result.json"))["norm"]                 # 與 B、老師 P6B 相同的輸入正規化
    fields = s5c.make_train_fields(cfg, n_fields, seed).to(dev)
    print(f"[init] {group} s{seed}:訓練場 {len(fields)} 個({time.time() - t0:.1f}s)", flush=True)
    if len(fields) >= s5c.CAL_N:
        with torch.no_grad():
            cal = s5c.calibrate_norm(s5c.measure_box(fields[:s5c.CAL_N], PROBE, pr, cp, bs, cfg, seed=seed + 7))
        e_n = max(abs(cal[k] - norm[k]) / max(abs(norm[k]), 1e-12) for k in norm)
        check("輸入正規化:以訓練場重新校準 ≈ B 的常數(相對差 < 1e-3;實際使用 B 的常數)", e_n < NORM_TOL, f"相對差 {e_n:.1e}")

    torch.manual_seed(seed)
    model = build_model(group, cfg, pr).to(dev)
    info = {}
    if not student:                                                       # P6B4:從 B 出發(同 P6B)
        model.init.load_state_dict(torch.load(b_dir / "final.pt", map_location=dev))
        ref, _, _ = s5c.load_net("B", PROBE, seed, N_TRAIN, cfg, pr, dev)
        model.eval()
        with torch.no_grad():
            cnt = s5c.measure_box(fields[:64], PROBE, pr, cp, bs, cfg, seed=seed + 98)
            e_i = float((model.init(g.prep(cnt, bs, norm, group, cp)) - ref(s5c.prep(cnt, bs, norm, "B", cp))).abs().max())
        check("載入檢查:模型內的初始網路 = B(< 1e-5)", e_i < 1e-5, f"最大差 {e_i:.1e}")
        info["init_md5"] = g._md5(b_dir / "final.pt")
        del ref
    teacher = None
    if w > 0:
        teacher, t_norm, t_meta = s5c.load_net(TEACHER, PROBE, seed, N_TRAIN, cfg, pr, dev)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        check("老師的輸入正規化 = 學生的(同一組常數)", t_norm == norm)
        if len(fields) >= s5c.TRAIN_EVAL_N:
            with torch.no_grad():
                tn, _ = train_nerr(teacher, fields, TEACHER, norm, pr, cp, bs, cfg, R0, seed)
            e_t = abs(tn - t_meta["train_nerr_ph"])
            check("老師載入:訓練場上的 nerr_ph = P6B 的 result.json(< 1e-4)", e_t < TEACHER_TOL,
                  f"{tn:.5f} vs {t_meta['train_nerr_ph']:.5f}(差 {e_t:.1e})")
        info["teacher_md5"] = g._md5(s5c.run_dir(TEACHER, PROBE, seed, N_TRAIN) / "final.pt")
    if not all_ok():
        raise SystemExit("❌ 載入檢查未通過,不訓練")

    if student:
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        max_lr = LR
    else:
        p_init = list(model.init.parameters())
        ids = {id(p) for p in p_init}
        opt = torch.optim.Adam([{"params": p_init, "lr": g.LR_INIT},
                                {"params": [p for p in model.parameters() if id(p) not in ids], "lr": g.LR_NEW}])
        max_lr = [g.LR_INIT, g.LR_NEW]
    steps = len(fields) // g.BATCH
    total = epochs * steps + 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total)
    n_par = g.n_params(model)
    print(f"[init] params={n_par:,};老師 {'P6B' if teacher is not None else '無'}(w = {w});{epochs} epochs × {steps} 步",
          flush=True)
    meta = {"group": group, "probe": PROBE, "seed": seed, "n_train": n_fields, "student": student,
            "width": STUDENTS[group][0] if student else 32, "hid": STUDENTS[group][1] if student else g.HID,
            "stages": model.K, "kd_w": w, "teacher": TEACHER if teacher is not None else None, "epochs": epochs,
            "batch": g.BATCH, "lr": max_lr, "clip": g.CLIP, "eps_e": g.EPS_E, "eps_p": g.EPS_P, "norm": norm,
            "params": n_par, "train_seed_base": s5c.train_chunk_seed(seed, 0), "quick": QUICK, **info}
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
        gen = torch.Generator().manual_seed(seed * 100_003 + ep)            # 同 B 的 batch 順序公式
        perm = torch.randperm(len(fields), generator=gen)
        acc = {"loss": 0.0, "true": 0.0, "kd": 0.0, "final": 0.0, "stages": [0.0] * model.K}
        n_ok, n_skip, n_clip, gns, te = 0, 0, 0, [], time.time()
        for i in range(steps):
            idx = perm[i * g.BATCH:(i + 1) * g.BATCH].to(dev)
            obj = fields[idx]
            with torch.no_grad():
                counts = s5c.measure_box(obj, PROBE, pr, cp, bs, cfg,
                                         seed=s5c.NOISE_SEED_BASE + seed * 10_000_019 + gstep)
                inp = g.prep(counts, bs, norm, group, cp)
                Tc = None
                if teacher is not None:
                    T = teacher(inp)
                    Tc = torch.polar(T[:, 0], T[:, 1])
            gstep += 1
            loss, l_true, l_kd, ls = kd_loss(model(inp, all_stages=True)[1], torch.polar(obj[:, 0], obj[:, 1]), Tc, w, R0)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip += 1
                if sched.last_epoch + 1 < total:
                    sched.step()
                continue
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), g.CLIP))
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
            n_clip += gn > g.CLIP
            gns.append(gn)
            acc["loss"] += float(loss.detach())
            acc["true"] += float(l_true.detach())
            acc["kd"] += float(l_kd.detach()) if l_kd is not None else 0.0
            acc["final"] += float(ls[-1].detach())
            for k in range(model.K):
                acc["stages"][k] += float(ls[k].detach())
        for k in ("loss", "true", "kd", "final"):
            acc[k] /= max(n_ok, 1)
        acc["stages"] = [v / max(n_ok, 1) for v in acc["stages"]]
        acc.update({"epoch": ep, "skipped": n_skip, "clip_frac": n_clip / max(n_ok, 1),
                    "grad_norm_median": float(np.median(gns)) if gns else None, "alphas": model.alphas(),
                    "sec": time.time() - te, "lr": sched.get_last_lr()})
        hist.append(acc)
        print(f"epoch {ep:03d}  loss={acc['loss']:.5f}  真值項={acc['true']:.5f}"
              + (f"  老師項={acc['kd']:.5f}" if teacher is not None else "")
              + "  各級(真值)=" + " ".join(f"{v:.4f}" for v in acc["stages"])
              + "  α=" + " ".join(f"{v:.3f}" for v in acc["alphas"])
              + f"  梯度範數中位數 {acc['grad_norm_median'] or 0:.2e}(裁切 {100 * acc['clip_frac']:.0f}%)"
              + f"  跳過 {n_skip}  {acc['sec']:.1f}s", flush=True)
        tmp = out / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "history": hist}, tmp)
        os.replace(tmp, ck)

    model.eval()
    with torch.no_grad():
        tr, pooled = train_nerr(model, fields, group, norm, pr, cp, bs, cfg, R0, seed)
    meta.update({"history": hist, "train_nerr_ph": tr, "train_nerr_pooled": pooled, "alphas": model.alphas(),
                 "train_sec": time.time() - t0})
    tmp = out / "final.tmp"
    torch.save(model.state_dict(), tmp)
    json.dump(meta, open(out / "result.json", "w"), indent=2)
    os.replace(tmp, out / "final.pt")                                     # final.pt 最後才出現(= 完成的標記)
    print(f"[done] {group} s{seed}:訓練場 nerr_ph {tr:.4f}  α {' '.join(f'{v:.3f}' for v in model.alphas())}"
          f"  {time.time() - t0:.0f}s → {out}", flush=True)


# ============================================================================
# 評估(協定 §19.8)
# ============================================================================
def model_dir(grp, s, smoke):
    return SMOKE_DIR / f"{grp}_s{s}" if (smoke and grp in NEW) else s5c.run_dir(grp, PROBE, s, N_TRAIN)


def new_status(prep_json, force_all):
    """每個新組的狀態:selected / not_selected / no_prep。"""
    if force_all:
        return {x: "selected" for x in NEW}
    st = {EXPLORE: "selected"}
    if not Path(prep_json).exists():
        st.update({x: "no_prep" for x in STUDENT_ORDER})
    else:
        sel = json.load(open(prep_json))["selected"]
        st.update({x: "selected" if sel[x] else "not_selected" for x in STUDENT_ORDER})
    return st


@torch.no_grad()
def evaluate(dev, seeds=None, n_sub=None, smoke=False, prep_json=PREP_JSON, force_all=False):
    import probe_4_2b as pb
    seeds = seeds or SEEDS
    env = json.load(open(s5e.VAL_ENV))["envelope"]
    status = new_status(prep_json, force_all)
    groups = BASE_EVAL + NEW
    cfg0 = s5.load_cfg(0)
    probes0 = {k: v for k, v in pb.build_probes(cfg0, dev).items() if k in s5.PROBES}
    print("  重新計時:迭代法與網路", flush=True)
    tim = s5b.timing(cfg0, probes0, dev)
    tg = g.time_groups(groups, dev)
    for B in tg:
        tim[B].update({f"net:{x}": v for x, v in tg[B].items()})
    res = {x: [] for x in groups}
    problems = []
    for s in seeds:
        d = s5d.setup_seed(s, s5e.val_seeds, dev)
        cfg, cp, bs, O, R0, F = d["cfg"], d["cp"], d["bs"], d["O"], d["R0"], d["F"]
        counts = d["counts"]
        if n_sub:
            counts, O = [c[:n_sub] for c in counts], O[:n_sub]
        for x in groups:
            if x in status and status[x] != "selected":
                res[x].append(None)
                continue
            md = model_dir(x, s, smoke)
            if not (md / "final.pt").exists():
                res[x].append(None)
                problems.append((x, s, "缺(訓練失敗、超時或未完成)"))
                continue
            try:
                meta = json.load(open(md / "result.json"))
                net = build_model(x, cfg, d["pr"]).to(dev)
                net.load_state_dict(torch.load(md / "final.pt", map_location=dev))
                net.eval()
                rec = {"params": meta["params"], "train_nerr_ph": meta["train_nerr_ph"],
                       "train_nerr_pooled": meta.get("train_nerr_pooled"), "train_sec": meta.get("train_sec"),
                       "history_last": meta["history"][-1]}
                if x == "B":
                    pred = g.predict(net, counts, bs, meta["norm"], x, cp)
                else:
                    O0, outs = g.predict(net, counts, bs, meta["norm"], x, cp, all_stages=True)
                    pred = outs[-1]
                    rec["stages"] = [s5.field_metrics(s5c.to_field(O0, cfg, F), O, R0)["nerr_ph"]] + [
                        s5.field_metrics(s5c.to_field(torch.polar(o[:, 0], o[:, 1]), cfg, F), O, R0)["nerr_ph"]
                        for o in outs]
                    rec["alphas"] = net.alphas()
                rec["net"] = s5.field_metrics(s5c.to_field(torch.polar(pred[:, 0], pred[:, 1]), cfg, F), O, R0)
                res[x].append(rec)
                del net, pred
            except Exception as e:                                        # noqa: BLE001  (記下來、繼續評估其他網路)
                res[x].append(None)
                problems.append((x, s, f"載入或推論失敗:{type(e).__name__}: {e}"))
        print(f"  [seed {s}] 完成", flush=True)
    return env, res, tim, status, problems


def vals(res, x):
    if x not in res or not res[x] or any(r is None for r in res[x]):
        return None
    return np.array([r["net"]["nerr_ph"] for r in res[x]], float)


def repro_check(res):
    if not P6_RAW.exists():
        print(f"  ⚠️ 找不到 {P6_RAW} → 略過重現檢查")
        return
    old = json.load(open(P6_RAW))["results"]
    diffs = {}
    for x in BASE_EVAL:
        a = vals(res, x)
        if a is None or x not in old or any(r is None for r in old[x]):
            continue
        diffs[x] = float(np.max(np.abs(a - np.array([r["net"]["nerr_ph"] for r in old[x]]))))
    check(f"重現:B、P6B、P6NAF 的驗證場 nerr_ph = P6 評估(§18.7;< {g.REPRO_TOL:g})",
          bool(diffs) and all(v < g.REPRO_TOL for v in diffs.values()), " / ".join(f"{k} {v:.1e}" for k, v in diffs.items()))


def report(env, res, tim, status, problems, smoke=False):
    v = {"status": status, "problems": [list(p) for p in problems]}
    Bs = ["64", "512"]
    if smoke:
        print("\n" + "!" * 100)
        print("迷你流程:網路只訓練 1 個 epoch、只用 1 個 seed 與少數驗證場 → 以下數字與判定都沒有意義,只用來確認程式能完整跑完")
        print("!" * 100)
    print("\n" + "=" * 100)
    print("狀態(先看這裡)")
    print("=" * 100)
    lab = {"selected": "", "not_selected": "預先計時未通過 → 依規則不訓練", "no_prep": "prep 沒有完成 → 沒有訓練(看 cdi_scan5h_prep_*.out)"}
    for x in NEW:
        extra = lab[status[x]]
        done = sum(r is not None for r in res[x])
        print(f"  {x:<7} {'(探索性)' if x == EXPLORE else '':<6} " + (extra if extra else f"完成 {done} / {len(res[x])} 個 seed"))
    if problems:
        print("  ⚠️ 有網路缺少或失敗(相關比較略過):")
        for x, s, why in problems:
            print(f"     {x} seed {s}:{why}" + ("" if smoke else f" → 看 {log_hint(x, s)}"))
    else:
        print("  ✅ 該有的網路都在")

    print("\n" + "=" * 100)
    print("計時(ms / 張;本 job 重新量;網路含輸入前處理)")
    print("=" * 100)
    for B in Bs:
        print(f"  batch {B}:" + "  ".join(f"{k} {tim[B][k]:.4f}" for k in
                                         ["ePIE-C", "AP-C", f"K-HIO:{PROBE}"] + [f"net:{x}" for x in res]))
    if not smoke:
        repro_check(res)

    print("\n" + "=" * 100)
    print(f"驗證場({PROBE},R0 的 nerr_ph,{len(next(iter(res.values())))} seeds)與 vs 包絡")
    print("=" * 100)
    for x in res:
        a = vals(res, x)
        if a is None:
            print(f"  {x}:—")
            continue
        tr = np.mean([r["train_nerr_ph"] for r in res[x]])
        secs = [r["train_sec"] for r in res[x] if r.get("train_sec")]
        print(f"  {x}{'(探索性)' if x == EXPLORE else ''}:{a.mean():.4f}({' '.join(f'{y:.4f}' for y in a)})"
              f"  參數 {res[x][0]['params']:,}  訓練場 {tr:.4f}" + (f"  訓練時間 {np.mean(secs) / 60:.0f} 分" if secs else ""))
        last = np.mean([r["history_last"].get("final", r["history_last"].get("sup", np.nan)) for r in res[x]])
        pooled = [r["train_nerr_pooled"] for r in res[x] if r.get("train_nerr_pooled") is not None]
        if pooled and np.mean(pooled) > 2 * last:
            print(f"     ⚠️ 訓練場 nerr(eval 模式,{np.mean(pooled):.4f})比最後一個 epoch 的訓練 loss({last:.4f})大 2 倍以上:"
                  "先檢查 BatchNorm 統計或訓練是否正常,再判讀")
        gv = {"nerr": list(a)}
        if "stages" in res[x][0]:
            st = np.array([r["stages"] for r in res[x]])
            gv["stages"] = st.mean(0).tolist()
            gv["alphas"] = [r["alphas"] for r in res[x]]
            print("     各級 nerr_ph(0 = 起點):" + " → ".join(f"{y:.4f}" for y in st.mean(0))
                  + ";步長 α(逐 seed):" + " | ".join(" ".join(f"{y:.3f}" for y in r["alphas"]) for r in res[x]))
        for B in Bs:
            tn = tim[B][f"net:{x}"]
            e = s5b.envelope(env, PROBE, tn, tim[B])
            r = s5b.ratio(a, e[2])
            gv[B] = {"t_net": tn, "opp": [e[0], e[1], list(e[2])], "ratio": list(r)}
            print(f"     batch {B}:{tn:.4f} ms vs 對手 {s5b.fmt_conf(e[0])}({e[1]} 次){e[2].mean():.4f}"
                  f" → 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {r[0]}")
        gv["useful"] = gv["64"]["ratio"][0] == "較準"
        gv["robust"] = gv["useful"] and gv["512"]["ratio"][0] == "較準"
        print(f"     ▶ {'穩健有用(batch 64 與 512 都較準)' if gv['robust'] else ('有用(只在 batch 64)' if gv['useful'] else '未達有用')}")
        v[x] = gv

    def pair(a_x, b_x):
        a, b = vals(res, a_x), vals(res, b_x)
        if a is None or b is None:
            return None
        return (s5b.ratio(a, b), tim["64"][f"net:{a_x}"] / tim["64"][f"net:{b_x}"],
                tim["512"][f"net:{a_x}"] / tim["512"][f"net:{b_x}"])

    print("\n" + "=" * 100)
    print("配對比較(協定 §19.8;比值 = 逐 seed 誤差比值的幾何平均,< 1 = 前者較準)")
    print("=" * 100)
    for x in ["P5A", "P5B"]:
        pp = pair(x, TEACHER)
        if pp is None:
            print(f"  (q1) {x} vs P6B:—")
            continue
        r, t64, t512 = pp
        fast = t64 <= TIME_FRAC_64 and t512 <= TIME_FRAC_512
        strict = fast and r[1] <= 1.03
        print(f"  (q1) {x} vs P6B:推論時間 × {t64:.2f}(b64)/ × {t512:.2f}(b512);誤差比值 {r[1]:.3f}(z {r[2]:+.1f})")
        print(f"       更快(時間條件):{'✅' if fast else '—'};嚴格的「更快」(誤差不高於 +3%):{'✅' if strict else '—'};"
              f"自己的時間下 vs 包絡:{'穩健有用' if v[x]['robust'] else ('只在 b64 有用' if v[x]['useful'] else '未達有用')}")
        v[f"q1_{x}"] = {"ratio": list(r), "t64": t64, "t512": t512, "fast": fast, "strict": strict}
    pp = pair("P5A", "P5Actl")
    if pp:
        r = pp[0]
        labk = ("蒸餾有幫助" if (r[1] < 1 and r[2] < -2) else ("蒸餾反而變差" if (r[1] > 1 and r[2] > 2) else "沒有統計上明確的差異"))
        print(f"  (q2) 蒸餾的效果:P5A vs P5Actl 比值 {r[1]:.3f}(z {r[2]:+.1f})→ {labk}")
        v["q2"] = {"ratio": list(r), "label": labk}
    else:
        print("  (q2) P5A vs P5Actl:—")
    for a_x, b_x in [("P5B", "P5A"), ("P5A", "B"), ("P5B", "B")]:
        pp = pair(a_x, b_x)
        if pp:
            r, t64, t512 = pp
            print(f"  (q3) {a_x} vs {b_x}:誤差比值 {r[1]:.3f}(z {r[2]:+.1f});推論時間 × {t64:.2f}(b64)/ × {t512:.2f}(b512)")
            v[f"q3_{a_x}_vs_{b_x}"] = {"ratio": list(r), "t64": t64, "t512": t512}
    pp = pair(EXPLORE, TEACHER)
    if pp:
        r, t64, t512 = pp
        more = r[1] < 1 and r[2] < -2
        print(f"  (q4,探索性) P6B4 vs P6B:誤差比值 {r[1]:.3f}(z {r[2]:+.1f});推論時間 × {t64:.2f}(b64)/ × {t512:.2f}(b512)"
              f" → {'更準' if more else '沒有明確更準'}")
        v["q4"] = {"ratio": list(r), "t64": t64, "t512": t512, "more_acc": more}
    else:
        print("  (q4,探索性) P6B4 vs P6B:—")
    second = bool(v.get("q2", {}).get("label") == "蒸餾有幫助" and v.get("q4", {}).get("more_acc"))
    v["second_batch"] = second
    print(f"\n  第二批(P6B4 教 P5A)的條件(蒸餾有幫助 且 P6B4 更準):{'✅ 符合 → 討論是否加做' if second else '— 不符合 → 不做'}")
    print("  P5 的判讀:回報「快多少、準度掉多少、是否仍穩健有用」,依優化方案 §5.2b 討論是否採用;")
    print("  P6B4 為探索性;要不要把任何組送去期中考,須在期中考之前另外決定(協定 §19.8)。")
    return v


def make_figure(env, res, tim, out_dir):
    plt = s5._plt()
    if plt is None:
        return
    out_dir.mkdir(exist_ok=True)
    col = {"B": "#2a78d6", "P6B": "#0b3d91", "P6NAF": "#0b6e4f", EXPLORE: "#8a5cd1", "P5A": "#d6452a",
           "P5Actl": "#9a9994", "P5B": "#eb6834"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, B in zip(axs, ["64", "512"]):
        tt = sorted({s5b.cfg_time(c, it, tim[B], PROBE) for c in s5b.configs() for it in s5b.stops(c)} - {0.0})
        ax.plot(tt, [s5b.envelope(env, PROBE, T, tim[B])[2].mean() for T in tt], color="#0b0b0b", lw=2,
                label="iterative envelope (validation)")
        for x in res:
            a = vals(res, x)
            if a is None:
                continue
            ax.plot([tim[B][f"net:{x}"]], [a.mean()], "o" if x == "B" else "s", ms=7, color=col.get(x, "#555"), label=x)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{PROBE}, batch {B} (validation fields)", fontsize=10)
        ax.set_xlabel("time per sample (ms)")
        ax.grid(True, color="#e4e3df", which="both", lw=0.5)
    axs[0].set_ylabel("nerr on R0 (global phase only)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    p = out_dir / "scan5h_p5.png"
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  圖:{p}")


# ============================================================================
def check_files(teacher=True, env=False):
    need = [Path(f) for f in ("scan_5.py", "scan_5b.py", "scan_5c.py", "scan_5d.py", "scan_5e.py", "scan_5g.py",
                              "probe_4_2b.py", "probe_4_1.py", "ambiguity_check.py")]
    need += [RUN_ROOT / f"{s5.BASE}_s{s}" / "config_used.json" for s in SEEDS]
    need += [s5c.run_dir("B", PROBE, s, N_TRAIN) / f for s in SEEDS for f in ("final.pt", "result.json")]
    if teacher:
        need += [s5c.run_dir(TEACHER, PROBE, s, N_TRAIN) / f for s in SEEDS for f in ("final.pt", "result.json")]
    if env:
        need.append(s5e.VAL_ENV)
        need += [s5c.run_dir("P6NAF", PROBE, s, N_TRAIN) / f for s in SEEDS for f in ("final.pt", "result.json")]
    missing = [str(p) for p in need if not p.exists()]
    check(f"需要的 {len(need)} 個檔案都存在", not missing, "缺:" + ", ".join(missing) if missing else "")
    return not missing


def run_eval(dev, smoke=False):
    kw = dict(seeds=[SEEDS[0]], n_sub=SMOKE_EVAL_N, smoke=True, prep_json=SMOKE_DIR / "prep.json", force_all=True) \
        if smoke else {}
    out = SMOKE_DIR if smoke else RUN_ROOT
    env, res, tim, status, problems = evaluate(dev, **kw)
    raw = out / f"scan5h_p5_raw{SUF}.json"
    json.dump({"results": res, "timing": tim, "status": status, "problems": problems, "quick": QUICK}, open(raw, "w"))
    print(f"  原始結果先存檔:{raw}")
    verdict = report(env, res, tim, status, problems, smoke=smoke)
    json.dump({"results": res, "timing": tim, "verdict": verdict, "quick": QUICK}, open(out / f"scan5h_p5{SUF}.json", "w"))
    make_figure(env, res, tim, (SMOKE_DIR / "figs") if smoke else FIG_DIR)
    return problems


def smoke(dev):
    print("#" * 70)
    print(f"迷你全流程(--smoke):輸出到 {SMOKE_DIR},不動正式結果")
    print("#" * 70)
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)                                          # 只刪這個 smoke 專用資料夾
    SMOKE_DIR.mkdir(parents=True)
    t0 = time.time()
    unit_tests(dev, NEW)
    if not all_ok():
        return False
    prep_timing(dev, SMOKE_DIR / "prep.json")
    for x in NEW:
        print(f"\n---- 迷你訓練:{x}(1 epoch、{SMOKE_FIELDS} 個訓練場)----", flush=True)
        train(x, SEEDS[0], dev, n_fields=SMOKE_FIELDS, epochs=1, out=SMOKE_DIR / f"{x}_s{SEEDS[0]}")
    print("\n---- 迷你評估(seed 0、驗證場前幾個)----", flush=True)
    problems = run_eval(dev, smoke=True)
    check("迷你評估:所有網路都成功載入與推論", not problems, "; ".join(f"{x} s{s}:{w}" for x, s, w in problems))
    print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒")
    return all_ok()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--train-p6b4", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    if a.check:
        ok = check_files()
        if ok:
            unit_tests(dev, NEW)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.smoke:
        ok = check_files(env=True) and smoke(dev)
        print("\n" + ("✅ 迷你全流程全部通過 → 可以送件" if ok else "❌ 有項目未通過 → 不要送件,把輸出貼給 Claude"))
        sys.exit(0 if ok else 1)
    if a.prep:
        PREP_JSON.unlink(missing_ok=True)                                 # 不讓舊的選定結果留下來(prep 失敗時學生一律不訓練)
        if not check_files():
            sys.exit(1)
        unit_tests(dev, NEW)
        if not all_ok():
            print("\n❌ 單元測試未通過,不做預先計時(學生的 task 會因此不訓練)")
            sys.exit(1)
        prep_timing(dev, PREP_JSON)
        print("\n✅ prep 完成")
        return
    if a.train_p6b4:
        if a.task is None or not 0 <= a.task <= 2:
            raise SystemExit("--train-p6b4 需要 --task 0–2")
        if not check_files(teacher=False):
            sys.exit(1)
        s = SEEDS[a.task]
        print(f"[task {a.task}] {EXPLORE}(探索性;B + 4 級、warm start、{P6B4_EPOCHS} epochs)、seed {s}", flush=True)
        unit_tests(dev, [EXPLORE])
        if not all_ok():
            print("\n❌ 單元測試未通過,不訓練")
            sys.exit(1)
        train(EXPLORE, s, dev)
        print("✅ 訓練完成")
        return
    if a.train:
        if a.task is None or not 0 <= a.task <= 8:
            raise SystemExit("--train 需要 --task 0–8")
        x, s = task_student(a.task)
        if not PREP_JSON.exists():
            print(f"❌ 找不到 {PREP_JSON}:prep 沒有完成(看 {LOG}/cdi_scan5h_prep_*.out)→ 不訓練")
            sys.exit(1)
        sel = json.load(open(PREP_JSON))["selected"]
        if not sel[x]:
            print(f"[task {a.task}] {x}:預先計時未通過(協定 §19.3)→ 依規則不訓練,正常結束")
            return
        if not check_files():
            sys.exit(1)
        width, hid, w = STUDENTS[x]
        print(f"[task {a.task}] {x}(U-Net 寬度 {width}、3 級、小 CNN {hid} 通道、{'老師 P6B、w = ' + str(w) if w else '沒有老師'};"
              f"從頭 {STUDENT_EPOCHS} epochs)、seed {s}", flush=True)
        unit_tests(dev, [x])
        if not all_ok():
            print("\n❌ 單元測試未通過,不訓練")
            sys.exit(1)
        train(x, s, dev)
        print("✅ 訓練完成")
        return
    if a.eval:
        if not check_files(env=True):
            sys.exit(1)
        run_eval(dev)
        print("\n" + ("✅ 內建檢查全部通過" if all_ok() else "❌ 有項目未通過"))
        print("把完整輸出貼給 Claude")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
