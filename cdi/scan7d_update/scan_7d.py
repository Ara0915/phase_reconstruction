#!/usr/bin/env python
"""階段七 7d(探索型支線):8 級的估計型 P6B4e8 與 8 級對照組 P6B4c8(階段七協定 §十七)。

背景:7c 的可行性檢查 no-go(固定公式從網路草稿出發,4 級只把位置誤差降到 0.57–0.72 倍)→ 主線不訓練;
      使用者決定把「級數加到時間與精度的最佳平衡點(8 級)」當作探索型成果送去訓練。
P6B4e8 = P6B4 + 每一級的位置步(順序 AP → 位置、每級 1 次),8 級;第 1–4 級 = P6B4 的權重,第 5–8 級 = 複製第 4 級。
P6B4c8 = 同樣 8 級、同樣起點與訓練,但沒有位置步(分開「位置步」與「級數變多」的貢獻)。
其餘(誤差配比、位置監督 λ = 0.1、學習率 2e-4 / 2e-3、20 epochs、評估的 13 個條件與判讀 e1–e8)同 7c(scan_7c.py,不修改)。

用法(需在計算節點執行):
    python scan_7d.py --check          # 內建檢查(含 scan_7c 的 7 項)
    python scan_7d.py --smoke          # 迷你全流程(dev 節點);通過才可正式執行
    python scan_7d.py --task 0..5      # run_scan7d_train.sh:0–2 = P6B4e8 seed 0–2;3–5 = P6B4c8 seed 0–2
    python scan_7d.py --final          # 彙整(run_scan7d_final.sh)
"""
import argparse
import copy
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
import scan_7c as c7                                                 # noqa: E402  (7c;不修改)

b7, a7, a6, h, g = c7.b7, c7.a7, c7.a6, c7.h, c7.g
s5, s5b, s5c = c7.s5, c7.s5b, c7.s5c
RUN_ROOT, QUICK, PROBE, SEEDS, SUF = c7.RUN_ROOT, c7.QUICK, c7.PROBE, c7.SEEDS, c7.SUF
C7_MD5 = "34575ffb68a7b5a17ce451ca89cdf0f7"  # 7c 的 scan_7c.py(位置步、評估與判讀的實作)
K8 = 8
ORDER, M_INNER = "ap", 1                     # §17.2:AP → 位置、每級 1 次
NEW_E, NEW_C = "P6B4e8", "P6B4c8"
NETS = ["P6B4", "P6B4r", NEW_C, NEW_E]
TASKS = [("e", 0), ("e", 1), ("e", 2), ("c", 0), ("c", 1), ("c", 2)]
SMOKE_DIR = RUN_ROOT / f"scan7d_smoke{SUF}"
PASSED = SMOKE_DIR / "PASSED"
SMOKE_TRAIN_N = c7.SMOKE_TRAIN_N
SMOKE_EPOCHS = 1
FIG_DIR = Path("figs_scan7d")
EQ_TOL = c7.EQ_TOL

ok_all = True


def check(label, passed, detail=""):
    global ok_all
    print(f"{'✅' if passed else '❌'} {label}")
    if detail:
        print(f"     {detail}")
    ok_all &= bool(passed)


def all_ok():
    return ok_all and c7.all_ok()


md5, dump_json, absdiff = a7.md5, a7.dump_json, a7.absdiff


def me_md5():
    return md5(Path(__file__).resolve())


def final_json(root=None):
    return (root or RUN_ROOT) / f"scan7d{SUF}.json"


def model_dir(kind, s, mroot=None):
    name = NEW_E if kind == "e" else NEW_C
    return (mroot / f"{name}_s{s}") if mroot is not None else s5c.run_dir(name, PROBE, s, a6.N_TRAIN)


# ============================================================================
# 等價檢查的比較方式
# ============================================================================
REL_TOL, FRAC_TOL, PIX_TOL = 1e-3, 1e-3, 1e-3


def eq_stats(a, b):
    """a、b:[N, 2, S, S](振幅、相位)。投影把相位截在 [0, φmax]:相位落在 ±π 附近的像素,極小的捨入差就可能讓截斷結果在 0 與 φmax
    之間跳動(不連續),所以只看最大差會誤判。改看:相對 L2 差、|差| > 1e-3 的像素比例;另列最大差像素的振幅與兩邊相位(判斷是否為跳動)。"""
    za, zb = torch.polar(*a.unbind(1)), torch.polar(*b.unbind(1))
    d = (za - zb).abs()
    rel = float(d.norm() / zb.abs().norm().clamp_min(1e-12))
    frac = float((d > PIX_TOL).float().mean())
    i = int(d.flatten().argmax())
    amp = float(b[:, 0].flatten()[i])
    pa, pb = float(a[:, 1].flatten()[i]), float(b[:, 1].flatten()[i])
    ok = rel < REL_TOL and frac < FRAC_TOL
    txt = (f"相對 L2 差 {rel:.1e}(< {REL_TOL:.0e})、|差| > {PIX_TOL:.0e} 的像素比例 {frac:.1e}(< {FRAC_TOL:.0e});"
           f"最大差 {float(d.max()):.1e}(該像素振幅 {amp:.3f}、相位 {pa:.3f} vs {pb:.3f})")
    return ok, txt


# ============================================================================
# 8 級的模型:第 5–8 級複製第 4 級
# ============================================================================
def expand_net(net, K=K8):
    """把 4 級的 P6Net 擴充成 K 級:新的級複製最後一級的 α 與小 CNN(就地修改)。"""
    k0 = net.K
    if K <= k0:
        return net
    with torch.no_grad():
        a = torch.cat([net.a.detach(), net.a.detach()[-1:].repeat(K - k0)])
    net.a = nn.Parameter(a)
    for _ in range(K - k0):
        net.refine.append(copy.deepcopy(net.refine[-1]))
    net.K = K
    return net


def build(kind, cfg, pr, dev, warm_seed=None):
    """kind e → P6B4e(8 級、位置步);kind c → P6B4e 外殼(use_pos = False,8 級)。warm_seed:先載入 P6B4 再擴充。"""
    model = c7.P6B4e(cfg, pr, dev, order=ORDER, m=M_INNER, use_pos=(kind == "e"))
    md = None
    if warm_seed is not None:
        md = c7.load_p6b4_into(model, warm_seed, dev)
    expand_net(model.net)
    model.K = K8
    model.b = nn.Parameter(torch.zeros(K8, device=dev))
    return model.to(dev), md


def load_e(s, cfg, pr, dev, mroot=None, kind="e"):
    """同 scan_7c.load_e 的介面。e → (P6B4e8, norm, meta);c → (8 級 P6Net, norm, meta)。"""
    md = model_dir(kind, s, mroot)
    meta = json.load(open(md / "result.json"))
    model, _ = build(kind, cfg, pr, dev)
    if kind == "c":
        model.net.load_state_dict(torch.load(md / "final.pt", map_location=dev))
        return model.net.eval(), meta["norm"], meta
    model.load_state_dict(torch.load(md / "final.pt", map_location=dev))
    return model.eval(), meta["norm"], meta


def bind_c7():
    """讓 scan_7c 的評估、判讀與圖使用本支線的網路(名稱、載入、資料夾)。"""
    c7.NEW_E, c7.NEW_C, c7.NETS = NEW_E, NEW_C, NETS
    c7.load_e, c7.model_dir = load_e, model_dir
    c7.COL[NEW_E], c7.COL[NEW_C] = "#1baf7a", "#2a78d6"


# ============================================================================
# 訓練(同 scan_7c.train_7c;起點 = P6B4 擴充成 8 級)
# ============================================================================
def train_7d(kind, seed, dev, n_fields=None, epochs=None, out=None):
    n_fields = a6.N_TRAIN if n_fields is None else n_fields
    epochs = epochs or g.EPOCHS
    cfg, pr, cp, bs = s5c.setup(seed, PROBE, dev)
    R0 = s5c.r0_box(cfg, pr, dev)
    out = Path(out) if out is not None else model_dir(kind, seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    norm = json.load(open(s5c.run_dir("B", PROBE, seed, a6.N_TRAIN) / "result.json"))["norm"]
    fields = s5c.make_train_fields(cfg, n_fields, seed).to(dev)
    name = NEW_E if kind == "e" else NEW_C
    print(f"[init] {name} s{seed}:訓練場 {len(fields)} 個({time.time() - t0:.1f}s)", flush=True)
    model, p_md = build(kind, cfg, pr, dev, warm_seed=seed)
    ref, rnorm, _ = a6.load_p6b4(seed, cfg, pr, dev)
    model.eval()
    with torch.no_grad(), c7.no_tf32():
        cnt = s5c.measure_box(fields[:32], PROBE, pr, cp, bs, cfg, seed=seed + 98)
        inp = g.prep(cnt, bs, norm, "P6B4", cp)
        model.beta_override = 0.0
        _, outs = model(inp, all_stages=True)
        model.beta_override = None
        eq_ok, eq_txt = eq_stats(outs[3], ref(inp))
    # 載入檢查用「權重逐位元相同」判定(穩定):輸出的比較會被相位截斷的不連續放大(見協定 §17.5 v3),只作描述
    sd, sr = model.net.state_dict(), ref.state_dict()
    w_ok = all((torch.equal(sd[k][:v.shape[0]], v) if k == "a" else torch.equal(sd[k], v)) for k, v in sr.items())
    c_ok = all(torch.equal(sd[k], sd[k.replace(f"refine.{j}.", "refine.3.", 1)])
               for j in range(4, K8) for k in sd if k.startswith(f"refine.{j}.")) and torch.equal(sd["a"][4:], sd["a"][3].expand(K8 - 4))
    check("載入檢查:U-Net 與第 1–4 級的權重 = P6B4(逐位元);第 5–8 級 = 複製第 4 級(逐位元);8 級;輸入正規化 = P6B4 的",
          w_ok and c_ok and rnorm == norm and model.K == K8,
          f"權重相同 {w_ok}、複製正確 {c_ok};描述:位置步關閉時第 4 級輸出 vs P6B4({'' if eq_ok else '未達等價門檻,'}{eq_txt})")
    del ref
    if not all_ok():
        raise SystemExit("❌ 載入檢查未通過,不訓練")
    tm = c7.TrainMeas7c(cfg, pr, cp, bs, dev)
    groups = [{"params": list(model.net.parameters()), "lr": c7.LR_OLD}]
    max_lr = [c7.LR_OLD]
    if kind == "e":
        groups.append({"params": [model.b], "lr": c7.LR_NEW})
        max_lr.append(c7.LR_NEW)
    else:
        model.b.requires_grad_(False)
    opt = torch.optim.Adam(groups)
    steps = len(fields) // g.BATCH
    total = epochs * steps + 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total)
    print(f"[init] {epochs} epochs × {steps} 步;{K8} 級;位置步 {'開(AP → 位置、每級 1 次)' if kind == 'e' else '關(對照組)'}", flush=True)
    meta = {"group": name, "kind": kind, "arch": "P6B4c8 (P6Net, 8 stages)" if kind == "c" else "P6B4e8", "K": K8, "order": ORDER,
            "m": M_INNER, "probe": PROBE, "seed": seed, "n_train": n_fields, "epochs": epochs, "batch": g.BATCH, "lr": max_lr,
            "clip": g.CLIP, "norm": norm, "lambda_pos": c7.LAMBDA_POS if kind == "e" else 0.0,
            "init": "P6B4 all weights; stages 5-8 = copies of stage 4", "init_md5": g._md5(p_md / "final.pt"),
            "p_pos_on": c7.P_POS_ON, "p_prb_on": c7.P_PRB_ON, "err_seed_base": c7.ERR_SEED_BASE, "quick": QUICK,
            "script_md5": me_md5(), "c7_md5": C7_MD5, "exploratory": True}
    json.dump(meta, open(out / "config_used.json", "w"), indent=2)
    ck = out / "ckpt.pt"
    start, hist = 0, []
    if ck.exists():
        dck = torch.load(ck, map_location=dev, weights_only=False)
        if dck.get("script_md5") != me_md5():
            raise SystemExit(f"❌ {ck} 是由不同版本的 scan_7d.py 存的:不續跑。先告訴 Claude")
        model.load_state_dict(dck["model"])
        opt.load_state_dict(dck["opt"])
        sched.load_state_dict(dck["sched"])
        start, hist = dck["epoch"] + 1, dck["history"]
        print(f"[resume] 從 epoch {start} 續跑", flush=True)
    gstep = start * steps
    K = model.K
    for ep in range(start, epochs):
        model.train()
        gen = torch.Generator().manual_seed(seed * 100_003 + ep)
        perm = torch.randperm(len(fields), generator=gen)
        acc = {"loss": 0.0, "obj": 0.0, "pos": 0.0, "final": 0.0, "stages": [0.0] * K, "pos_rms": [0.0] * K, "sat": [0.0] * K}
        n_ok, n_skip, n_clip, gns, te = 0, 0, 0, [], time.time()
        for i in range(steps):
            idx = perm[i * g.BATCH:(i + 1) * g.BATCH].to(dev)
            obj = fields[idx]
            with torch.no_grad():
                counts, err = tm(obj, c7.ERR_SEED_BASE + seed * 10_000_019 + gstep, s5c.NOISE_SEED_BASE + seed * 10_000_019 + gstep)
                inp = g.prep(counts, bs, norm, "P6B4", cp)
            gstep += 1
            want = err[0].to(dev)
            _, outs, ds, sats = model(inp, with_delta=True)
            l_obj, ls = g.ds_loss(outs, torch.polar(obj[:, 0], obj[:, 1]), R0)
            l_pos = torch.stack([(dk - want).pow(2).sum(-1).mean() for dk in ds]).mean()
            loss = l_obj + (c7.LAMBDA_POS * l_pos if kind == "e" else 0.0)
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
            acc["obj"] += float(l_obj.detach())
            acc["pos"] += float(l_pos.detach())
            acc["final"] += float(ls[-1].detach())
            for k in range(K):
                acc["stages"][k] += float(ls[k].detach())
                acc["pos_rms"][k] += float((ds[k].detach() - want).pow(2).sum(-1).mean(-1).sqrt().mean())
                acc["sat"][k] += float(sats[k])
        for k in ("loss", "obj", "pos", "final"):
            acc[k] /= max(n_ok, 1)
        for k in ("stages", "pos_rms", "sat"):
            acc[k] = [v / max(n_ok, 1) for v in acc[k]]
        acc.update({"epoch": ep, "skipped": n_skip, "clip_frac": n_clip / max(n_ok, 1),
                    "grad_norm_median": float(np.median(gns)) if gns else None, "alphas": model.net.alphas(),
                    "betas": model.betas(), "sec": time.time() - te, "lr": sched.get_last_lr()})
        hist.append(acc)
        print(f"epoch {ep:03d}  loss={acc['loss']:.5f}  物體={acc['obj']:.5f}  位置={acc['pos']:.4f}  各級物體="
              + " ".join(f"{v:.4f}" for v in acc["stages"]) + "  各級位置 RMS=" + " ".join(f"{v:.3f}" for v in acc["pos_rms"])
              + "  β=" + " ".join(f"{v:.2f}" for v in acc["betas"])
              + f"  飽和 {100 * np.mean(acc['sat']):.1f}%  梯度中位數 {acc['grad_norm_median'] or 0:.2e}(裁切 {100 * acc['clip_frac']:.0f}%)"
              + f"  跳過 {n_skip}  {acc['sec']:.1f}s", flush=True)
        tmp = out / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "history": hist, "script_md5": meta["script_md5"]}, tmp)
        os.replace(tmp, ck)
    model.eval()
    with torch.no_grad():
        sub = fields[:s5c.TRAIN_EVAL_N]
        c_id = s5c.measure_box(sub, PROBE, pr, cp, bs, cfg, seed=seed + 99)
        tr = c7.nerr_on(model, sub, c_id, norm, cp, bs, R0)
        c_er, _ = tm(sub, c7.EVAL_ERR_SEED + seed, seed + 99)
        tr_e = c7.nerr_on(model, sub, c_er, norm, cp, bs, R0)
    meta.update({"history": hist, "train_nerr_ph": tr, "train_nerr_ph_rand": tr_e, "alphas": model.net.alphas(),
                 "betas": model.betas(), "train_sec": time.time() - t0, "skipped_total": sum(x["skipped"] for x in hist)})
    tmp = out / "final.tmp"
    torch.save(model.net.state_dict() if kind == "c" else model.state_dict(), tmp)
    json.dump(meta, open(out / "result.json", "w"), indent=2)
    os.replace(tmp, out / "final.pt")
    print(f"[done] {name} s{seed}:訓練場 nerr_ph 理想 {tr:.4f} / 隨機誤差 {tr_e:.4f}  β {' '.join(f'{v:.2f}' for v in model.betas())}"
          f"  {time.time() - t0:.0f}s → {out}", flush=True)
    return meta


# ============================================================================
# 內建檢查
# ============================================================================
def checks(dev):
    c7.checks(dev)                                                     # 位置步、AP 步、可微分、抽樣、限幅(共用的實作)
    print("=" * 70)
    print("階段七 7d:8 級的檢查")
    print("=" * 70)
    n = 8 if not QUICK else 4
    cfg, pr, cp, bs = s5c.setup(0, PROBE, dev)
    ref, norm, _ = a6.load_p6b4(0, cfg, pr, dev)
    fb = s5c.make_train_fields(cfg, n, 0).to(dev)
    inp = g.prep(s5c.measure_box(fb, PROBE, pr, cp, bs, cfg, seed=3), bs, norm, "P6B4", cp)
    with torch.no_grad(), c7.no_tf32():
        me, _ = build("e", cfg, pr, dev, warm_seed=0)
        mc, _ = build("c", cfg, pr, dev, warm_seed=0)
        me.eval()
        mc.eval()
        y = ref(inp)
        me.beta_override = 0.0
        _, oe = me(inp, all_stages=True)
        me.beta_override = None
        _, oc = mc(inp, all_stages=True)
        eqs = [eq_stats(o[3], y) for o in (oe, oc)]
        d = a7.setup_cond(0, dev, "ideal", n)
        g8 = a6.run_pipe("G9", mc.net, norm, d)                           # 8 級的 P6Net 走 scan_6a 的整張流程
        ee, dl, _ = c7.run_g9e(me, norm, d, keep_delta=True)
    copies = all(torch.equal(p, q) for k in range(4, K8)
                 for p, q in zip(me.net.refine[k].parameters(), me.net.refine[3].parameters()))
    a_ok = bool(torch.allclose(me.net.a[4:], me.net.a[3].expand(4)))
    shapes = (len(oe) == K8 and len(oc) == K8 and dl.shape[0] == K8 and me.b.numel() == K8 and mc.net.K == K8
              and torch.isfinite(g8.real).all() and torch.isfinite(ee.real).all())
    check("8 級的模型:第 4 級的輸出 = P6B4(位置步關閉,關掉 TF32);第 5–8 級的小 CNN 與 α = 複製第 4 級;"
          "8 級的輸出、位置估計、b_k 的數量正確;8 級 P6Net 可走 scan_6a 的整張 G9",
          all(x[0] for x in eqs) and copies and a_ok and bool(shapes), "P6B4e8:" + eqs[0][1] + "\n     P6B4c8:" + eqs[1][1])
    # 存檔 → 載入 = 同一個模型
    import uuid
    tmpd = Path(os.environ.get("TMPDIR", "/tmp")) / f"scan7d_chk_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmpd.mkdir(parents=True, exist_ok=True)
    for kind, mod in (("e", me), ("c", mc)):
        md = tmpd / f"{NEW_E if kind == 'e' else NEW_C}_s0"
        md.mkdir(exist_ok=True)
        with torch.no_grad():
            mod.b.add_(0.1 * torch.arange(K8, device=dev, dtype=torch.float32))
        torch.save(mod.net.state_dict() if kind == "c" else mod.state_dict(), md / "final.pt")
        json.dump({"norm": norm}, open(md / "result.json", "w"))
    le, _, _ = load_e(0, cfg, pr, dev, tmpd, "e")
    lc, _, _ = load_e(0, cfg, pr, dev, tmpd, "c")
    with torch.no_grad():
        r_e = float((torch.polar(*le(inp).unbind(1)) - torch.polar(*me(inp).unbind(1))).abs().max())
        r_c = float((torch.polar(*lc(inp).unbind(1)) - torch.polar(*mc.net(inp).unbind(1))).abs().max())
    shutil.rmtree(tmpd, ignore_errors=True)
    check("存檔 → 載入:P6B4e8(含 b_k)與 P6B4c8(P6Net 格式)載入後輸出相同(差 0)", r_e == 0.0 and r_c == 0.0,
          f"P6B4e8 {r_e:.1e}、P6B4c8 {r_c:.1e}")
    print(f"\n     裝置:{dev}({a7.gpu_name(dev)})")


# ============================================================================
# 彙整(評估、判讀與圖沿用 scan_7c 的函式)
# ============================================================================
def load_inputs(old_root, env_root, r_root, mroot, iters):
    C, bad = {}, []
    for c in c7.CONDS:
        p = a7.cond_json(c, old_root) if c in b7.OLD_CONDS else b7.env_json(c, env_root)
        if not p.exists():
            bad.append(f"{p.name} 不存在")
            continue
        C[c] = json.load(open(p))
        m = C[c]["meta"]
        want = c7.A7_MD5 if c in b7.OLD_CONDS else c7.B7_MD5
        if m["script_md5"] != want or not C[c]["checks_ok"]:
            bad.append(f"{p.name} 版本或檢查不符")
    if not bad:
        n0 = C[c7.CONDS[0]]["meta"]["n"]
        bad += [f"{c} 的停止點或場數不同" for c in c7.CONDS if C[c]["meta"]["iters"] != list(iters) or C[c]["meta"]["n"] != n0]
    check("12 個條件檔都在、版本正確、檢查全過、設定一致", not bad, ";".join(bad))
    r7 = b7.final_json(r_root)
    J7 = json.load(open(r7)) if r7.exists() else None
    okr = J7 is not None and J7.get("script_md5") == c7.B7_MD5 and J7.get("complete") is True
    check(f"7b 的結果檔 {r7.name} 在、版本正確、完整", okr)
    mb = []
    for kind in ("e", "c"):
        for s in SEEDS:
            md = model_dir(kind, s, mroot)
            if not ((md / "final.pt").exists() and (md / "result.json").exists()):
                mb.append(f"{md.name} 缺 final.pt / result.json")
            elif json.load(open(md / "result.json")).get("script_md5") != me_md5():
                mb.append(f"{md.name} 由不同版本的 scan_7d.py 訓練")
    for s in SEEDS:
        md = b7.model_dir(s, None if r_root == RUN_ROOT else r_root)
        if not ((md / "final.pt").exists() and (md / "result.json").exists()):
            mb.append(f"{md.name}(P6B4r)缺 final.pt / result.json")
    check("6 個新模型(P6B4e8、P6B4c8 各 3 seeds)與 P6B4r 都在、由同一版程式訓練", not mb, ";".join(mb))
    return (C, J7) if not bad and okr and not mb else None


def run_final(dev, old_root, env_root, r_root, mroot, fig_dir, iters, smoke):
    bind_c7()
    where = "validation fields (smoke)" if smoke else "validation fields, 7x7 scan"
    L = load_inputs(old_root, env_root, r_root, mroot, iters)
    if L is None:
        return None
    C, J7 = L
    t0 = time.time()
    print("\n  計時(ms / 整張)", flush=True)
    tim = a7.timing(dev)
    tim["net"] = c7.time_nets(dev, mroot, None if r_root == RUN_ROOT else r_root)
    print(f"  (計時完成,經過 {time.time() - t0:.0f} 秒;GPU {tim['gpu']})", flush=True)
    print("\n  網路評估(13 個條件 × 3 seeds × 4 個網路,G9)", flush=True)
    R, P, keep = c7.eval_all(dev, C, mroot, r_root, log=lambda s: print(s, flush=True))
    d4 = max(absdiff(R[c]["P6B4"]["nerr_ph"][i], C[c]["nets"]["G9"]["nerr_ph"][i]) for c in c7.CONDS for i in range(len(SEEDS)))
    dr = max(absdiff(R[c]["P6B4r"]["nerr_ph"][i], J7["nets"][c]["P6B4r"]["G9"]["nerr_ph"][i]) for c in c7.CONDS for i in range(len(SEEDS)))
    check(f"重現:凍結 P6B4 的 G9 = 條件檔、P6B4r 的 G9 = 7b 的結果(nerr_ph < {a7.REPRO_TOL:.0e})",
          d4 < a7.REPRO_TOL and dr < a7.REPRO_TOL, f"P6B4 最大差 {d4:.1e}、P6B4r 最大差 {dr:.1e}")
    print("\n" + "#" * 100)
    print(f"探索型支線(協定 §十七):{NEW_E} = 8 級估計型、{NEW_C} = 8 級對照組;以下判讀標準同 §16.4(e1–e8)。")
    print(f"報表中「P6B4e」「P6B4c」的字樣分別指 {NEW_E}、{NEW_C}。探針步的規則(§16.6)只作描述:本專題不再改結構。")
    print("#" * 100)
    V = c7.report(C, R, P, tim, iters, smoke)
    c7.ssim_rank(V, R, J7.get("iter_ssim"))
    final = final_json(mroot)
    res = {"verdict": V, "nets": R, "pos": P, "timing": tim, "checks_ok": all_ok(), "smoke": smoke, "script_md5": me_md5(),
           "c7_md5": C7_MD5, "exploratory": True, "complete": False}
    dump_json(res, final)
    print("\n" + "=" * 100)
    print("圖(描述,不改變判讀)")
    print("=" * 100)
    try:
        c7.figures(V, R, P, tim, keep, mroot, fig_dir, where)
    except Exception as e:                                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"畫圖完成(判讀不受影響,已存於 {final.name})", False, f"{type(e).__name__}: {e}")
    res["checks_ok"] = all_ok()
    res["complete"] = bool(all_ok())                                     # 只有全部檢查通過才標為完整(否則可重跑)
    dump_json(res, final)
    print(f"  結果:{final}")
    return res


def check_files(kind):
    ok = c7.check_files("final" if kind == "final" else ("smoke" if kind == "smoke" else "check"))
    m = md5(Path(c7.__file__).resolve())
    check("scan_7c.py = 7c 的版本(位置步與評估的實作)", m == C7_MD5, f"md5 {m[:8]}…")
    return ok and m == C7_MD5


def need_passed(me):
    if not PASSED.exists():
        raise SystemExit(f"❌ 找不到 {PASSED}:先在 dev 節點跑 python scan_7d.py --smoke,全部通過才可送件")
    pm = json.load(open(PASSED))["script_md5"]
    if pm != me:
        raise SystemExit(f"❌ smoke 通過時的 scan_7d.py(md5 {pm})與現在的({me})不同:用現在的版本重跑 --smoke")
    print(f"✅ smoke 已通過(同一版本,{json.load(open(PASSED))['time']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", type=int, choices=range(len(TASKS)))
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not QUICK:
        raise SystemExit("❌ 沒有 GPU:請在計算節點執行")
    me = me_md5()
    print(f"scan_7d.py md5 {me};GPU {a7.gpu_name(dev)}")
    if not check_files("final" if a.final else ("smoke" if a.smoke else "check")):
        print("\n❌ 缺檔案或版本不同,停下來")
        sys.exit(1)
    if a.check:
        checks(dev)
        print("\n" + ("✅ 全部通過" if all_ok() else "❌ 有項目未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.smoke:
        print("#" * 70)
        print(f"迷你全流程(--smoke):檢查、P6B4e8 / P6B4c8 各 3 seeds × {SMOKE_EPOCHS} epoch × {SMOKE_TRAIN_N} 個場、彙整;輸出到 {SMOKE_DIR}")
        print("#" * 70)
        if SMOKE_DIR.exists():
            shutil.rmtree(SMOKE_DIR)
        SMOKE_DIR.mkdir(parents=True)
        t0 = time.time()
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 不要送件,把輸出貼給 Claude")
            sys.exit(1)
        tt = time.time()
        for kind, s in TASKS:
            train_7d(kind, s, dev, SMOKE_TRAIN_N, SMOKE_EPOCHS, model_dir(kind, s, SMOKE_DIR))
        per = (time.time() - tt) / len(TASKS)
        run_final(dev, a7.SMOKE_DIR, b7.SMOKE_DIR, b7.SMOKE_DIR, SMOKE_DIR, SMOKE_DIR / "figs", a7.SMOKE_ITERS, smoke=True)
        print(f"\n  迷你全流程耗時 {time.time() - t0:.0f} 秒(每個迷你訓練平均 {per:.0f} 秒,含載入的固定開銷)")
        if all_ok():
            json.dump({"script_md5": me, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(PASSED, "w"))
            print("\n✅ 迷你全流程全部通過 → 可以送出 run_scan7d_train.sh,再送 run_scan7d_final.sh")
            sys.exit(0)
        print("\n❌ 有項目未通過 → 不要送件,把輸出貼給 Claude")
        sys.exit(1)
    if a.task is not None:
        kind, s = TASKS[a.task]
        need_passed(me)
        md = model_dir(kind, s)
        if (md / "final.pt").exists():
            raise SystemExit(f"❌ {md}/final.pt 已存在:這個模型已經訓練完。為避免覆蓋,先告訴 Claude")
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有訓練。把輸出貼給 Claude")
            sys.exit(1)
        train_7d(kind, s, dev)
        print("\n" + (f"✅ {md.name} 訓練完成、檢查全過" if all_ok() else f"❌ {md.name} 有檢查未通過"))
        sys.exit(0 if all_ok() else 1)
    if a.final:
        out = final_json()
        if out.exists() and json.load(open(out)).get("complete"):
            raise SystemExit(f"❌ {out} 已存在且完整:正式結果已經有了。為避免覆蓋,先告訴 Claude")
        need_passed(me)
        checks(dev)
        if not all_ok():
            print("\n❌ 內建檢查未通過 → 沒有彙整。把輸出貼給 Claude")
            sys.exit(1)
        res = run_final(dev, RUN_ROOT, RUN_ROOT, RUN_ROOT, None, FIG_DIR, a7.ITERS, smoke=False)
        print("\n" + ("✅ 內建檢查全部通過" if all_ok() and res else "❌ 有項目未通過(判讀先不要採信)"))
        print("把完整輸出與 figs_scan7d/ 的圖傳給 Claude")
        sys.exit(0 if all_ok() and res else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
