"""本機測試(QUICK、驗證資料):跑 1 個 C 的包絡 job(L-ideal),其餘 15 個包絡檔以已跑完的 ideal / L-ideal 複製(只為了走過彙整的程式路徑),
再跑彙整 run_final(smoke 的設定)。用法:cd stage && CDI_QUICK=1 CDI_RUN_ROOT=../runs python ../test_agg.py"""
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
import scan_final as sf                                               # noqa: E402

dev = torch.device("cpu")
plan = sf.plan_of(True, sf.timing512())
n = sf.a7.SMOKE_FIELDS
root = sf.SMOKE_DIR
if not sf.env_json("L-ideal", root).exists():
    sf.run_task(dev, sf.TASKS.index(("c", "L-ideal")), n, sf.VAL_C, plan, root, smoke=True)
for part, name in sf.TASKS:
    src = sf.env_json("ideal" if part == "ab" else "L-ideal", root)
    dst = sf.env_json(name, root)
    if not dst.exists():
        shutil.copy(src, dst)
ctx = {"smoke": True, "root": root, "n": n, "n_val": n, "plan": plan, "base_ab": sf.VAL_AB, "base_c": sf.VAL_C,
       "fig_dir": root / "figs", "n_crlb": 2, "manifest": sf.manifest_key(sf.manifest())}
res = sf.run_final(dev, ctx)
print("\n結果:", "完成" if res is not None else "None", ";all_ok =", sf.all_ok())
