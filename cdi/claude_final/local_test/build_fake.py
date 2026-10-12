"""本機可行性測試用:建一個假的 RUN_ROOT(QUICK、CPU、隨機權重的模型),結構同國網。
驗證資料的包絡檔與階段八的網路數字,一律用「原本的階段七 / 八程式」算出 → scan_final.py 的重現檢查是真的在比對兩套程式。
用法:cd stage && CDI_QUICK=1 CDI_RUN_ROOT=../runs python ../build_fake.py"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
import scan_8c as c8                                                  # noqa: E402

s8 = c8.s8
d7, c7, b7, a7, a6, s5, s5b, s5c, h, g = s8.d7, s8.c7, s8.b7, s8.a7, s8.a6, s8.s5, s8.s5b, s8.s5c, s8.h, s8.g
R = s8.RUN_ROOT
assert s8.QUICK, "CDI_QUICK=1"
dev = torch.device("cpu")
torch.manual_seed(0)
t0 = time.time()


def log(x):
    print(f"[{time.time() - t0:6.0f}s] {x}", flush=True)


# 1. config
from src.config import Cfg                                            # noqa: E402
from src.physics import calibrate_flux, beamstop_mask                 # noqa: E402
import probe_4_1 as p41                                               # noqa: E402
import probe_4_2b as pb                                               # noqa: E402

cfg = Cfg.load("configs/ideal_base.yaml")
cfg.ref_energy = float(calibrate_flux(p41.make_home(cfg, 256, seed=12345)))
for s in s8.SEEDS:
    d = R / f"{s5.BASE}_s{s}"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(cfg.to_dict(), open(d / "config_used.json", "w"))
log("config")

# 2. 模型(隨機權重)
for s in s8.SEEDS:
    cfg = s5.load_cfg(s)
    pr = pb.build_probes(cfg, dev)[s8.PROBE]
    cp = s5.scan_cfg(cfg, s8.PROBE, pr, s5.SCANS["3x3s8"][2])
    bs = beamstop_mask(cfg, device=dev)
    fb = s5c.make_train_fields(cfg, 32, s)
    norm = s5c.calibrate_norm(s5c.measure_box(fb, s8.PROBE, pr, cp, bs, cfg, seed=7))
    for grp in ("B", "P6B4", "P6B4r"):
        md = s5c.run_dir(grp, s8.PROBE, s, a6.N_TRAIN)
        md.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(100 + s)
        net = h.build_model("B" if grp == "B" else "P6B4", cfg, pr)
        torch.save(net.state_dict(), md / "final.pt")
        json.dump({"norm": norm}, open(md / "result.json", "w"))
    me, _ = d7.build("e", cfg, pr, dev, warm_seed=s)
    with torch.no_grad():
        me.b.add_(0.1)
    md = d7.model_dir("e", s)
    md.mkdir(parents=True, exist_ok=True)
    torch.save(me.state_dict(), md / "final.pt")
    json.dump({"norm": norm, "init_md5": g._md5(s5c.run_dir("P6B4", s8.PROBE, s, a6.N_TRAIN) / "final.pt")}, open(md / "result.json", "w"))
    for kind in ("L", "C"):
        mk = s8.model_dir(kind, s)
        mk.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for p in me.parameters():
                p.add_(0.001 * torch.randn_like(p))
        torch.save(me.state_dict(), mk / "final.pt")
        json.dump({"norm": norm, "init_md5": g._md5(md / "final.pt"), "script_md5": s8.me_md5()}, open(mk / "result.json", "w"))
log("模型")

# 3. 劑量
json.dump({**s8.dose_all(dev), "script_md5": s8.me_md5()}, open(s8.DOSE_JSON, "w"))
log("劑量")

# 4. 驗證的包絡檔(原本的程式)
n = s5b.N_FIELDS
for c in c7.CONDS:
    env, nets, stats = a7.measure_condition(dev, c, n, a7.ITERS, log=log)
    md5 = c7.A7_MD5 if c in b7.OLD_CONDS else c7.B7_MD5
    p = a7.cond_json(c) if c in b7.OLD_CONDS else b7.env_json(c)
    json.dump({"meta": {"cond": c, "n": n, "iters": list(a7.ITERS), "script_md5": md5}, "env": env, "nets": nets, "stats": stats,
               "checks_ok": True}, open(p, "w"))
    log(f"包絡 {c}")
for e in s8.EXAMS:
    env, stats = s8.measure_exam(dev, e, n, a7.ITERS, smoke=False, log=log)
    json.dump({"meta": {"exam": e, "n": n, "iters": list(a7.ITERS), "script_md5": s8.S8_MD5 if hasattr(s8, "S8_MD5") else s8.me_md5()},
               "env": env, "stats": stats, "checks_ok": True}, open(s8.env_json(e), "w"))
    log(f"包絡 {e}")
for p in (b7.final_json(), d7.final_json()):
    json.dump({"complete": True}, open(p, "w"))

# 5. 階段八的結果檔(原本的程式算網路的數字)
roots = s8.roots_of(False)
Rx, keep, info = s8.eval_exams(dev, n, False, roots, log=log)
E = {e: json.load(open(s8.env_json(e)))["env"] for e in s8.EXAMS}
fc = {e: s8.fail_value(E[e], Rx, e) for e in s8.EXAMS}
R13 = s8.eval_orig(dev, n, roots, log=log)
J8 = {"script_md5": s8.me_md5(), "complete": True, "smoke": False,
      "timing": json.load(open(sys.argv[1]))["timing"],
      "verdict": {"fail_c": fc, "exam": {e: {"nets": {nm: {"nerr_c": s8.net_c(Rx, e, nm, fc[e]).mean(1).tolist()} for nm in s8.NETS}}
                                         for e in s8.EXAMS}},
      "orig": {c: {nm: {"nerr_sr": R13[c][nm]["nerr_sr"], "nerr_ph": R13[c][nm]["nerr_ph"]} for nm in s8.NETS} for c in R13}}
json.dump(J8, open(s8.final_json(), "w"))
log("scan8.json")

# 6. 8c 的結果檔(原本的程式算前 2 個場的 CRLB)
d = s8.setup_exam(0, dev, "L-ideal", 2, False)
ph = c8.Phys.of(d)
rows = []
for i in range(2):
    r = c8.crlb_field(ph, d["O"][i])
    rows.append({"i": i, "T": r["T"], "Ec": float(d["Ec"][i])})
json.dump({"meta": {"md5": c8.me_md5(), "smoke": False}, "complete": True, "parts": {"A": {"T1": {"ideal": rows}}},
           "report": {"crlb": {"L-paper": {"crlb_c": 0.06}}}}, open(c8.OUT_JSON, "w"))
log("scan8c.json;完成")
