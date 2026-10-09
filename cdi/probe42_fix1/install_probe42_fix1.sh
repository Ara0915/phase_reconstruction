#!/bin/bash
# 更新 ~/cdi/check_probe42.py(只修正「未訓練的展開 = hio.py」這一項測試的設計)—— 在計算節點內執行。
# 先比對 md5:國網上須為 probe42_update 附的版本,一致才備份並取代。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
OLD="778c19708b7e96b924d23feaa033e0a9"
NEW="4eb6a0230035d17a25cc68901c5be2dc"
now=$(md5sum "$DST/check_probe42.py" | cut -d' ' -f1)
if [ "$now" = "$NEW" ]; then
  echo "  ✅ check_probe42.py 已是修正版(之前裝過)"; exit 0
elif [ "$now" != "$OLD" ]; then
  echo "  ❌ check_probe42.py 不是預期版本($now)—— 不覆蓋。把這段訊息貼給 Claude。"; exit 1
fi
mkdir -p "$DST/backup_pre_probe42fix1"
cp "$DST/check_probe42.py" "$DST/backup_pre_probe42fix1/"
cp "$SRC/check_probe42.py" "$DST/check_probe42.py"
sed -i 's/\r$//' "$DST/check_probe42.py"
echo "  已備份舊版到 $DST/backup_pre_probe42fix1/"
md5sum "$DST/check_probe42.py"
echo "  完成。只更新了 check_probe42.py。"
