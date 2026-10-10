#!/bin/bash
# 把 scan8d_update/ 的檔案安裝到 ~/cdi —— 在登入節點或計算節點執行皆可(只複製檔案、不計算)。
# 只新增 2 個檔案,不修改任何既有檔案:scan_8d.py、run_scan8d.sh
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
R=/work/elviss0915/runs
NEWFILES=(scan_8d.py run_scan8d.sh)
bad=0
echo "---- 1. 依賴的既有檔案(不修改,只比對版本)----"
declare -A DEPEND=(
  ["scan_8.py"]="4821d2607f6e8aa381f0e6b5c5e72513"
  ["scan_7d.py"]="b6e14b742fbf45615f87046f5618bb4c"
  ["scan_8c.py"]="3d0915fffaff52a5b4b2b749a9e102dd"
)
for f in "${!DEPEND[@]}"; do
  if [ ! -f "$DST/$f" ]; then echo "  ❌ $f 不存在"; bad=1; continue; fi
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${DEPEND[$f]}" ]; then echo "  ✅ $f"; else echo "  ❌ $f 與預期版本不同($now)"; bad=1; fi
done
echo "---- 2. 需要的模型與劑量檔(正式與 smoke)----"
[ -f "$R/scan8_dose.json" ] && echo "  ✅ scan8_dose.json" || { echo "  ❌ 缺 $R/scan8_dose.json"; bad=1; }
for s in 0 1 2; do
  for g in P6B4e8L P6B4e8; do
    [ -f "$R/scan5c_${g}_DEF2_n100000_s$s/final.pt" ] && [ -f "$R/scan5c_${g}_DEF2_n100000_s$s/result.json" ] || { echo "  ❌ 缺模型 scan5c_${g}_DEF2_n100000_s$s"; bad=1; }
  done
  [ -f "$R/scan8_smoke/P6B4e8L_s$s/final.pt" ] || { echo "  ❌ 缺 $R/scan8_smoke/P6B4e8L_s$s(smoke 用)"; bad=1; }
  [ -f "$R/scan7d_smoke/P6B4e8_s$s/final.pt" ] || { echo "  ❌ 缺 $R/scan7d_smoke/P6B4e8_s$s(smoke 用)"; bad=1; }
done
echo "---- 3. 8d 的輸出不可已存在(避免覆蓋)----"
[ -e "$R/scan8d.json" ] && { echo "  ❌ $R/scan8d.json 已存在 —— 先告訴 Claude"; bad=1; }
[ -e "$R/scan8d.json.partial" ] && echo "  ⚠️ $R/scan8d.json.partial 存在(中斷過的正式執行;同一版程式會續跑)"
[ -d "$DST/figs_scan8d" ] && echo "  ⚠️ $DST/figs_scan8d 已存在 —— 正式執行會覆寫裡面同名的圖"
echo "---- 4. 新檔案不可覆蓋不同內容 ----"
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
chmod +x "$DST/run_scan8d.sh"
mkdir -p "$R/logs"
echo "  完成。新增 $added 個(不修改既有檔案)。"
md5sum "$DST/scan_8d.py" "$DST/run_scan8d.sh"
