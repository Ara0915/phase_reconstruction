#!/usr/bin/env python
"""給了繞射相位的 oracle,在各原型上表現如何?—— 區分網路失敗的兩種原因。

背景(實驗設計 §十六):
    拿掉平凡歧異性後,網路在 polygon / bandpass 上仍 ≈ 平庸基準(0.04 / 0.06),
    lattice 則 0.50;純 HIO(對齊後)在三者上為 0.38 / 0.65 / 0.73。

兩種解釋:
    (甲)相位回復本身(非凸、需迭代)對前饋網路太難
         -> oracle(相位已給,任務退化為線性的反傅立葉轉換)在 polygon / bandpass 上表現好
    (乙)此 U-Net 連線性的反傅立葉轉換都無法在這類結構上學好(架構問題)
         -> oracle 在 polygon / bandpass 上也失敗

比較(皆為既有 checkpoint,不重新訓練;各 3 seeds;分數皆報未對齊與對齊後):

    條件   相位未知的網路       oracle(給相位)            純 HIO 5000(參考上限)
    現況   ideal_base          v2_proc_oracle             隨機起點
    理想   ideal_bs0_ph1e5     ideal_bs0_ph1e5_oracle     隨機起點

用法(需在計算節點執行;需 realign_eval.py、ambiguity_check.py 在同一資料夾):
    python oracle_proto.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from realign_eval import load, net_out, proto_labels, score_all
from src.data import generalization_suites
from src.hio import hio, random_init
from src.physics import beamstop_mask, forward_measure

RUN_ROOT = Path("/work/elviss0915/runs")
OUT_JSON = RUN_ROOT / "oracle_proto.json"
SEEDS = [0, 1, 2]
CONDITIONS = {
    "現況 bs=3 ph=1e3": {"normal": "ideal_base", "oracle": "v2_proc_oracle"},
    "理想 bs=0 ph=1e5": {"normal": "ideal_bs0_ph1e5", "oracle": "ideal_bs0_ph1e5_oracle"},
}
HIO_N = 5000
SANITY_TOL = 0.01


def run_condition(names, device):
    """回傳 {方法: [每個 seed 的 {原型: score_all 結果}]}。"""
    res = {"normal": [], "oracle": [], "hio": []}
    for s in SEEDS:
        per_method = {}
        for role in ["normal", "oracle"]:
            run_dir = RUN_ROOT / f"{names[role]}_s{s}"
            cfg, model = load(run_dir, device)
            assert cfg.oracle_phase == (role == "oracle"), f"{run_dir.name} 的 oracle_phase 不符"
            bs = beamstop_mask(cfg, device=device)
            home = generalization_suites(cfg, cfg.eval_n)["procedural"].to(device)
            torch.manual_seed(cfg.test_seed + s)
            counts = forward_measure(home, bs, cfg)
            pred = net_out(model, home, counts, bs, cfg)

            whole, _ = score_all(model, home, bs, cfg, pred)
            m = json.load(open(run_dir / "metrics.json"))["test"]["frc_gain"]
            if abs(whole["raw"]["frc_gain"] - m) > SANITY_TOL:
                raise SystemExit(f"{run_dir.name}:未對齊 {whole['raw']['frc_gain']:+.4f} "
                                 f"與 metrics.json {m:+.4f} 對不上,停下來查")
            labels, kinds = proto_labels(cfg.eval_n, cfg, home)
            rec = {"all": whole}
            for i, k in enumerate(kinds):
                msk = (labels == i).to(device)
                rec[k], _ = score_all(model, home[msk], bs, cfg, pred[msk])
            per_method[role] = rec
            print(f"  [{run_dir.name}] 完成(未對齊 {whole['raw']['frc_gain']:+.4f},"
                  f"metrics.json {m:+.4f})", flush=True)

            if role == "normal":
                # 純 HIO:不經網路,用相位未知那組的量測(與 realign_eval 相同的種子與起點)
                init = random_init(counts, cfg, seed=cfg.test_seed + s, device=device)
                ref = hio(init, counts, bs, cfg, n_iter=HIO_N, beta=cfg.hio_beta)
                rec_h = {"all": score_all(model, home, bs, cfg, ref)[0]}
                for i, k in enumerate(kinds):
                    msk = (labels == i).to(device)
                    rec_h[k], _ = score_all(model, home[msk], bs, cfg, ref[msk])
                per_method["hio"] = rec_h
        for role in res:
            res[role].append(per_method[role])
    return res, kinds


def ms(v):
    a = np.array(v, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def report(allres):
    for cond, (res, kinds) in allres.items():
        print("\n" + "=" * 92)
        print(f"{cond}:FRC gain(對齊後),mean ± std(3 seeds);括號內為未對齊")
        print("=" * 92)
        print(f"  {'原型':<10}{'相位未知的網路':>24}{'oracle(給相位)':>24}{'純 HIO ' + str(HIO_N):>22}")
        for k in ["all"] + kinds:
            row = f"  {('全部' if k == 'all' else k):<10}"
            for role in ["normal", "oracle", "hio"]:
                a = ms([r[k]["aligned"]["frc_gain"] for r in res[role]])
                raw = np.mean([r[k]["raw"]["frc_gain"] for r in res[role]])
                row += f"{a[0]:>+12.4f}±{a[1]:.4f}({raw:+.3f})"
            print(row)
        print(f"\n  材料 MAE(對齊後;平庸基準約 0.30)")
        for k in ["all"] + kinds:
            row = f"  {('全部' if k == 'all' else k):<10}"
            for role in ["normal", "oracle", "hio"]:
                row += f"{np.mean([r[k]['aligned']['material_mae'] for r in res[role]]):>24.4f}"
            print(row)
        print(f"\n  oracle − 相位未知(對齊後 FRC gain,逐 seed 相減)")
        for k in kinds:
            d = [o[k]["aligned"]["frc_gain"] - n[k]["aligned"]["frc_gain"]
                 for o, n in zip(res["oracle"], res["normal"])]
            m, s = ms(d)
            print(f"    {k:<10}{m:>+.4f} ± {s:.4f}")
    print("\n判讀準則見 實驗設計_1b6 §十七(結果出來前已寫定)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allres = {}
    for cond, names in CONDITIONS.items():
        print(f"{cond}")
        allres[cond] = run_condition(names, device)
    json.dump({c: r for c, (r, _) in allres.items()}, open(OUT_JSON, "w"))
    report(allres)


if __name__ == "__main__":
    main()
