#!/bin/bash
# 把 arch2_update/ 裡的檔案安裝到 ~/cdi —— 在計算節點內執行。
#
# 會修改兩個既有檔案(src/config.py、src/model.py)。
# 先比對 md5:兩者須為上次 arch_update 安裝後的版本,一致才覆蓋;
# 覆蓋前另存備份到 ~/cdi/backup_pre_arch2/。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi

declare -A EXPECT=(
  ["src/config.py"]="e8b13312b94b3c38896f712f10245ee2"
  ["src/model.py"]="f4a693d3be871de02037c09bc1147304"
)

echo "---- 1. 比對國網上的現有版本 ----"
bad=0
for f in "${!EXPECT[@]}"; do
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${EXPECT[$f]}" ]; then
    echo "  ✅ $f 與預期版本相同"
  else
    echo "  ❌ $f 與預期版本不同 —— 不覆蓋"
    bad=1
  fi
done
if [ "$bad" -ne 0 ]; then
  echo "停止。把上面的訊息貼給 Claude。"
  exit 1
fi

echo "---- 2. 備份 ----"
mkdir -p "$DST/backup_pre_arch2/src"
cp "$DST/src/config.py" "$DST/src/model.py" "$DST/backup_pre_arch2/src/"
echo "  已備份到 $DST/backup_pre_arch2/"

echo "---- 3. 安裝 ----"
cp "$SRC/config.py" "$SRC/model.py" "$DST/src/"
cp "$SRC/check_arch2.py" "$SRC/run_arch2.sh" "$DST/"
cp "$SRC/arch_fft_amb.yaml" "$SRC/arch_unroll_amb.yaml" "$DST/configs/"
sed -i 's/\r$//' "$DST/run_arch2.sh"
echo "  完成。修改 2 個、新增 4 個:"
echo "    修改 src/config.py  src/model.py"
echo "    新增 configs/arch_fft_amb.yaml  configs/arch_unroll_amb.yaml  check_arch2.py  run_arch2.sh"
