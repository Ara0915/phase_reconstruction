#!/bin/bash
# 把 hio_budget_update/ 裡的檔案安裝到 ~/cdi —— 在計算節點內執行。
#
# 只「新增」兩個檔案,不修改任何既有檔案:
#     src/hio_sw.py    HIO 擴充版(可換 support、shrinkwrap、ER 收尾);src/hio.py 不動
#     hio_budget.py    §二十三 的量測腳本
# 先比對它依賴的既有檔案是否為預期版本(與本地測試時相同),不同就停。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi

declare -A EXPECT=(
  ["src/hio.py"]="a5eac4662f326b269ad1f122760f607b"
  ["src/model.py"]="06a6f59825bdf967b0f1832738e7dba2"
  ["src/config.py"]="f7dba52dbf30398f1dfbd9a496f20af6"
  ["src/metrics.py"]="796c05881ba0a09b96a8aa497bf92ae4"
  ["src/physics.py"]="07e82213808c95fbef33cbad0e985c4c"
  ["src/procedural.py"]="adca05ecfa98ef63c75bfe952dc76194"
  ["realign_eval.py"]="f826432269d473b39f3e71bae2643e5f"
  ["ambiguity_check.py"]="9e9aafc246e43788ba34369621bece4e"
)

echo "---- 1. 比對依賴的既有檔案 ----"
bad=0
for f in "${!EXPECT[@]}"; do
  if [ ! -f "$DST/$f" ]; then
    echo "  ❌ $f 不存在"; bad=1; continue
  fi
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${EXPECT[$f]}" ]; then
    echo "  ✅ $f"
  else
    echo "  ❌ $f 與預期版本不同($now)"
    bad=1
  fi
done
[ -f "$DST/configs/ideal_base.yaml" ] && echo "  ✅ configs/ideal_base.yaml 存在" \
  || { echo "  ❌ configs/ideal_base.yaml 不存在"; bad=1; }

echo "---- 2. 新檔案不可覆蓋既有檔案 ----"
for pair in "src/hio_sw.py:src/hio_sw.py" "hio_budget.py:hio_budget.py"; do
  s="${pair%%:*}"; d="${pair##*:}"
  if [ -e "$DST/$d" ]; then
    if cmp -s "$SRC/$s" "$DST/$d"; then
      echo "  ✅ $d 已存在且內容相同(之前裝過)"
    else
      echo "  ❌ $d 已存在且內容不同 —— 不覆蓋"
      bad=1
    fi
  else
    echo "  ✅ $d 尚不存在,可新增"
  fi
done

if [ "$bad" -ne 0 ]; then
  echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"
  exit 1
fi

echo "---- 3. 安裝(只新增)----"
[ -e "$DST/src/hio_sw.py" ] || cp "$SRC/src/hio_sw.py" "$DST/src/hio_sw.py"
[ -e "$DST/hio_budget.py" ] || cp "$SRC/hio_budget.py" "$DST/hio_budget.py"
for f in src/hio_sw.py hio_budget.py; do
  if grep -q $'\r' "$DST/$f"; then sed -i 's/\r$//' "$DST/$f"; fi
done
md5sum "$DST/src/hio_sw.py" "$DST/hio_budget.py"
echo "  完成。新增 2 個檔案,未修改任何既有檔案。"
