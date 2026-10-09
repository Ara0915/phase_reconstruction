#!/bin/bash
# 把 amb_update/ 裡的檔案安裝到 ~/cdi —— 在計算節點內執行。
#
# 會修改三個既有檔案(src/config.py、src/losses.py、train.py)。
# 為避免蓋掉你在國網上改過的版本:先比對 md5,與交給 Claude 的原始版本一致才覆蓋;
# 覆蓋前另存備份到 ~/cdi/backup_pre_amb/。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi

declare -A ORIG=(
  ["src/config.py"]="559bd32595cffb739d0a62135e1f787b"
  ["src/losses.py"]="2290916ed89353adafdc235d2ec53904"
  ["train.py"]="7a96515eda912ad8215bc9e532a7aed4"
)

echo "---- 1. 比對國網上的現有版本 ----"
bad=0
for f in "${!ORIG[@]}"; do
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${ORIG[$f]}" ]; then
    echo "  ✅ $f 與原始版本相同"
  else
    echo "  ❌ $f 與原始版本不同(你在國網上改過?)—— 不覆蓋"
    bad=1
  fi
done
if [ "$bad" -ne 0 ]; then
  echo "停止。把上面的訊息貼給 Claude。"
  exit 1
fi

echo "---- 2. 備份 ----"
mkdir -p "$DST/backup_pre_amb/src"
cp "$DST/src/config.py" "$DST/src/losses.py" "$DST/backup_pre_amb/src/"
cp "$DST/train.py" "$DST/backup_pre_amb/"
echo "  已備份到 $DST/backup_pre_amb/"

echo "---- 3. 安裝 ----"
cp "$SRC/config.py" "$SRC/losses.py" "$DST/src/"
cp "$SRC/train.py" "$SRC/check_amb.py" "$SRC/run_amb.sh" "$DST/"
cp "$SRC/amb_base.yaml" "$DST/configs/"
sed -i 's/\r$//' "$DST/run_amb.sh"
echo "  完成。修改 3 個、新增 3 個:"
echo "    修改 src/config.py  src/losses.py  train.py"
echo "    新增 configs/amb_base.yaml  check_amb.py  run_amb.sh"
