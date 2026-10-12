"""本機可行性測試(快速版):設定、模型、劑量沿用 build_fake.py 已建好的;驗證的包絡檔只用原程式算重現檢查會比對的值
(seed 0、REPRO_CONFS、第 20 次),其餘(設定, 停止點, seed)填格式正確、隨次數遞減的假值。階段八的網路數字用原程式算。
用法:cd stage && CDI_QUICK=1 CDI_RUN_ROOT=../runs PYTHONUTF8=1 python ../build_fast.py <階段八的 scan8.json>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
import scan_8c as c8                                                  # noqa: E402

s8 = c8.s8
d7, c7, b7, a7, a6, s5, s5b = s8.d7, s8.c7, s8.b7, s8.a7, s8.a6, s8.s5, s8.s5b
assert s8.QUICK
dev = torch.device("cpu")
t0 = time.time()
REPRO = ["AP|one", "ePIE-C|rand", "AP-C:both|const", "ePIE-C:prb|zero", "AP-C:posA|khio20", "K-HIO"]
n = s5b.N_FIELDS


def log(x):
    print(f"[{time.time() - t0:6.0f}s] {x}", flush=True)


def fake(it, s, k):
    return float(0.4 / (1.0 + it / 5.0) * (1.0 + 0.05 * s) * (1.0 + 0.01 * (k % 7)))


for c in c7.CONDS:
    d = a7.setup_cond(0, dev, c, n)
    real = {conf: a7.metrics7(b7.iter_est(d, conf, 20, dev), d["O"], d["geo"].U) for conf in REPRO}
    env = {}
    for k, conf in enumerate(a7.configs7()):
        env[conf] = {}
        for it in a7.stops7(conf, a7.ITERS):
            rec = {q: [] for q in a7.MKEYS}
            for s in s8.SEEDS:
                if s == 0 and it == 20 and conf in real:
                    for q in a7.MKEYS:
                        rec[q].append(real[conf][q])
                else:
                    v = fake(it, s, k)
                    for q, val in (("nerr_sr", v), ("nerr_sh", v * 1.1), ("nerr_ph", v * 1.2), ("shift", 0.1), ("ramp", 0.0), ("n_fail", 0)):
                        rec[q].append(val)
            env[conf][str(it)] = rec
        if ":" in conf:
            env[conf]["diag"] = {"pos_rms": [0.1] * 3, "prb_err": [0.01] * 3}
    md5 = c7.A7_MD5 if c in b7.OLD_CONDS else c7.B7_MD5
    p = a7.cond_json(c) if c in b7.OLD_CONDS else b7.env_json(c)
    json.dump({"meta": {"cond": c, "n": n, "iters": list(a7.ITERS), "script_md5": md5}, "env": env, "checks_ok": True}, open(p, "w"))
    log(f"包絡 {c}")

for e in s8.EXAMS:
    d = s8.setup_exam(0, dev, e, n, False)
    mats = d["mats"]
    real = {conf: s8.summarize(s8.lat_metrics(s8.iter_est_L(d, conf, 20, dev), d), mats) for conf in REPRO}
    base = s8.summarize(s8.lat_metrics(a6.make_start("rand", d, None, dev), d), mats)
    env = {}
    for k, conf in enumerate(a7.configs7()):
        env[conf] = {}
        for it in a7.stops7(conf, a7.ITERS):
            rec = {}
            for s in s8.SEEDS:
                src = real[conf] if (s == 0 and it == 20 and conf in real) else base
                f = 1.0 if src is not base else fake(it, s, k) / 0.4
                for q, v in src.items():
                    vv = v * f if q in ("c_sum", "c_sum_W", "c_sum_Mo", "c_max", "c_max_W", "c_max_Mo", "nerr_sr", "nerr_ph") else v
                    rec.setdefault(q, []).append(vv)
            env[conf][str(it)] = rec
    json.dump({"meta": {"exam": e, "n": n, "iters": list(a7.ITERS), "script_md5": s8.me_md5()}, "env": env, "checks_ok": True},
              open(s8.env_json(e), "w"))
    log(f"包絡 {e}")
for p in (b7.final_json(), d7.final_json()):
    json.dump({"complete": True}, open(p, "w"))

roots = s8.roots_of(False)
Rx, _, _ = s8.eval_exams(dev, n, False, roots, log=log)
E = {e: json.load(open(s8.env_json(e)))["env"] for e in s8.EXAMS}
fc = {e: s8.fail_value(E[e], Rx, e) for e in s8.EXAMS}
R13 = s8.eval_orig(dev, n, roots, log=log)
J8 = {"script_md5": s8.me_md5(), "complete": True, "smoke": False, "timing": json.load(open(sys.argv[1]))["timing"],
      "verdict": {"fail_c": fc, "exam": {e: {"nets": {nm: {"nerr_c": s8.net_c(Rx, e, nm, fc[e]).mean(1).tolist()} for nm in s8.NETS}}
                                         for e in s8.EXAMS}},
      "orig": {c: {nm: {"nerr_sr": R13[c][nm]["nerr_sr"], "nerr_ph": R13[c][nm]["nerr_ph"]} for nm in s8.NETS} for c in R13}}
json.dump(J8, open(s8.final_json(), "w"))
log("scan8.json")
d = s8.setup_exam(0, dev, "L-ideal", 2, False)
ph = c8.Phys.of(d)
rows = [{"i": i, "T": c8.crlb_field(ph, d["O"][i], extra=False)["T"]} for i in range(2)]
json.dump({"meta": {"md5": c8.me_md5(), "smoke": False}, "complete": True, "parts": {"A": {"T1": {"ideal": rows}}},
           "report": {"crlb": {"L-paper": {"crlb_c": 0.06}}}}, open(c8.OUT_JSON, "w"))
log("scan8c.json;完成")
