#!/bin/bash
# 把 report_figs_update/collect_report_figs.py 安裝到 ~/cdi —— 在計算節點內執行。
# 只新增 1 個檔案(collect_report_figs.py),不修改任何既有檔案;腳本本身也只讀結果檔。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
R=/work/elviss0915/runs
declare -A DEPEND=(
  ["scan_5c.py"]="161df412d2771c611ec75490a1ea019f"
  ["scan_5b.py"]="f3ce06c0b08c1775dad81e5635c7ed0e"
  ["scan_5.py"]="68a6924fc6f4eb589ea15372cd18f761"
  ["probe_4_1.py"]="b5ebcef9ce546b480c27ca38ba22443f"
  ["realign_eval.py"]="f826432269d473b39f3e71bae2643e5f"
  ["ambiguity_check.py"]="9e9aafc246e43788ba34369621bece4e"
  ["src/hio.py"]="a5eac4662f326b269ad1f122760f607b"
  ["src/physics.py"]="f7d6902e4cb27345f643c7b3e11bd507"
)
NEWFILES=(collect_report_figs.py)
bad=0
echo "---- 1. 依賴的既有檔案(不修改,只比對版本)----"
for f in "${!DEPEND[@]}"; do
  if [ ! -f "$DST/$f" ]; then echo "  ❌ $f 不存在"; bad=1; continue; fi
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${DEPEND[$f]}" ]; then echo "  ✅ $f"; else echo "  ❌ $f 與預期版本不同($now)"; bad=1; fi
done
echo "---- 2. 要讀的結果檔(缺了只會少那一張圖,不擋安裝)----"
for f in probe_4_1.json probe42b_raw.json scan5b_raw.json scan5c_raw_n20000.json scan5c_raw_n100000.json scan5d_raw.json; do
  [ -f "$R/$f" ] && echo "  ✅ $f" || echo "  ⚠️ 缺 $R/$f"
done
for g in ideal_base ideal_bs0_ph1e5 ideal_bs0_ph1e5_oracle; do for s in 0 1 2; do
  [ -f "$R/${g}_s$s/metrics.json" ] && echo "  ✅ ${g}_s$s/metrics.json" || echo "  ⚠️ 缺 $R/${g}_s$s/metrics.json"
done; done
for d in figs_realign figs_probe_4_1 figs_probe42b figs_scan5c figs_scan5c_n20000 figs_scan5d; do
  [ -d "$DST/$d" ] && echo "  ✅ 原圖資料夾 $d" || echo "  ⚠️ 沒有 $DST/$d(orig/ 會少這一份)"
done
[ -e "$DST/report_figs" ] && echo "  ⚠️ $DST/report_figs 已存在 —— 重跑會覆寫裡面同名的圖(只影響這個輸出資料夾)"
echo "---- 3. 新檔案不可覆蓋不同內容 ----"
for f in "${NEWFILES[@]}"; do
  if [ -e "$DST/$f" ]; then
    if cmp -s <(sed 's/\r$//' "$SRC/$f") "$DST/$f"; then echo "  ✅ $f 已存在且相同"; else echo "  ❌ $f 已存在且內容不同 —— 不覆蓋"; bad=1; fi
  else
    echo "  ✅ $f 尚不存在,可新增"
  fi
done
[ "$bad" -ne 0 ] && { echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"; exit 1; }
added=0
for f in "${NEWFILES[@]}"; do
  if [ ! -e "$DST/$f" ]; then
    cp "$SRC/$f" "$DST/$f"
    sed -i 's/\r$//' "$DST/$f"
    added=$((added+1))
  fi
done
echo "  完成。新增 $added 個(不修改既有檔案)。"
md5sum "$DST/collect_report_figs.py"
