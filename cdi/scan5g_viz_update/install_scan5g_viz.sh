#!/bin/bash
# 把 scan_5g_viz.py 安裝到 ~/cdi —— 在計算節點內執行。只新增 1 個檔案,不修改任何既有檔案。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
R=/work/elviss0915/runs
bad=0
echo "---- 1. 依賴(不修改,只比對版本)----"
now=$(md5sum "$DST/scan_5g.py" 2>/dev/null | cut -d' ' -f1 || true)
[ "$now" = "18238c4ebda1da7bae86b051f35b9587" ] && echo "  ✅ scan_5g.py" || { echo "  ❌ scan_5g.py 不存在或版本不同($now)"; bad=1; }
[ -f "$R/scan5g_p6_raw.json" ] && echo "  ✅ scan5g_p6_raw.json(P6 評估的計時)" || { echo "  ❌ 缺 $R/scan5g_p6_raw.json"; bad=1; }
[ -f "$R/scan5e_val_envelope.json" ] && echo "  ✅ scan5e_val_envelope.json" || { echo "  ❌ 缺 $R/scan5e_val_envelope.json"; bad=1; }
for g in B NAFw32b2 P6B; do
  [ -f "$R/scan5c_${g}_DEF2_n100000_s0/final.pt" ] && echo "  ✅ 模型 ${g} s0" || { echo "  ❌ 缺模型 scan5c_${g}_DEF2_n100000_s0"; bad=1; }
done
echo "---- 2. 新檔案(不覆蓋不同內容;只替換本對話前一版且先備份)----"
f=scan_5g_viz.py
OLD1=520e7492be0711ecccbd8e612316eecd      # 前一版(沒有全景圖與類型分組)
NEW=6df64d5ffc3e832d5886f80f0975bf53
act=add
if [ -e "$DST/$f" ]; then
  cur=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$cur" = "$NEW" ]; then echo "  ✅ $f 已是新版"; act=none
  elif [ "$cur" = "$OLD1" ]; then echo "  ✅ $f 是前一版 → 備份後換成新版"; act=replace
  else echo "  ❌ $f 已存在且不是已知版本($cur)—— 不覆蓋"; bad=1; fi
else
  echo "  ✅ $f 尚不存在,可新增"
fi
[ "$bad" -ne 0 ] && { echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"; exit 1; }
if [ "$act" = replace ]; then cp "$DST/$f" "$DST/$f.bak_$(date +%Y%m%d_%H%M%S)"; fi
if [ "$act" != none ]; then cp "$SRC/$f" "$DST/$f"; sed -i 's/\r$//' "$DST/$f"; echo "  完成(不修改其他檔案)。"; fi
md5sum "$DST/$f"
