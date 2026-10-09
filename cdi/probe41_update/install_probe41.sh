#!/bin/bash
# 把 probe41_update/ 裡的檔案安裝到 ~/cdi —— 在計算節點內執行。
#
# 只新增或更新 probe_4_1.py(國網上若是第一版,先備份再取代),不修改其他任何檔案。
# 先比對它依賴的既有檔案是否為預期版本(與本地測試時相同),不同就停。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi

declare -A EXPECT=(
  ["src/hio_sw.py"]="3b2a58b9f487279cc78b2556dd18949c"
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
if [ -f /work/elviss0915/runs/hio_budget.json ]; then
  echo "  ✅ /work/elviss0915/runs/hio_budget.json 存在"
else
  echo "  ❌ /work/elviss0915/runs/hio_budget.json 不存在(需先跑過 hio_budget.py)"; bad=1
fi

echo "---- 2. probe_4_1.py 的現況 ----"
# 第一版(劑量檢查有誤)的 md5;若國網上是這一版,備份後以新版取代
OLD_V1="aba0a6ff2b41925f8751e7e854f5f017"
upgrade=0
if [ -e "$DST/probe_4_1.py" ]; then
  now=$(md5sum "$DST/probe_4_1.py" | cut -d' ' -f1)
  if cmp -s "$SRC/probe_4_1.py" "$DST/probe_4_1.py"; then
    echo "  ✅ probe_4_1.py 已是新版(之前裝過)"
  elif [ "$now" = "$OLD_V1" ]; then
    echo "  ✅ probe_4_1.py 是第一版 → 備份後更新為新版"
    upgrade=1
  else
    echo "  ❌ probe_4_1.py 已存在且不是任何已知版本($now)—— 不覆蓋"; bad=1
  fi
else
  echo "  ✅ probe_4_1.py 尚不存在,可新增"
fi

if [ "$bad" -ne 0 ]; then
  echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"
  exit 1
fi

echo "---- 3. 安裝 ----"
if [ "$upgrade" -eq 1 ]; then
  mkdir -p "$DST/backup_pre_probe41v2"
  cp "$DST/probe_4_1.py" "$DST/backup_pre_probe41v2/probe_4_1.py"
  echo "  已備份第一版到 $DST/backup_pre_probe41v2/"
  cp "$SRC/probe_4_1.py" "$DST/probe_4_1.py"
fi
[ -e "$DST/probe_4_1.py" ] || cp "$SRC/probe_4_1.py" "$DST/probe_4_1.py"
if grep -q $'\r' "$DST/probe_4_1.py"; then sed -i 's/\r$//' "$DST/probe_4_1.py"; fi
md5sum "$DST/probe_4_1.py"
echo "  完成。只動了 probe_4_1.py,其他既有檔案都沒有修改。"
