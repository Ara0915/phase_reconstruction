#!/bin/bash
# 5-3 修正 1:只替換 ~/cdi/scan_5c.py(單元測試「資料不重疊」排除全零的空白物體塊)—— 在計算節點內執行。
# 先比對目前的版本、備份,再替換;其他檔案都不動。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
OLD=2b3057a199db5c3493edd199ef8c24a9
NEW=161df412d2771c611ec75490a1ea019f
now=$(md5sum "$DST/scan_5c.py" | cut -d' ' -f1)
if [ "$now" = "$NEW" ]; then echo "  ✅ scan_5c.py 已是修正版,不需要動"; exit 0; fi
if [ "$now" != "$OLD" ]; then echo "  ❌ scan_5c.py 不是預期的舊版($now)—— 不替換,把這行貼給 Claude"; exit 1; fi
src_md5=$(md5sum "$SRC/scan_5c.py" | cut -d' ' -f1)
[ "$src_md5" = "$NEW" ] || { echo "  ❌ 上傳的新檔 md5 不對($src_md5)—— 不替換"; exit 1; }
cp -p "$DST/scan_5c.py" "$DST/scan_5c.py.bak_${OLD:0:6}"
cp "$SRC/scan_5c.py" "$DST/scan_5c.py"
echo "  ✅ 已備份舊版為 scan_5c.py.bak_${OLD:0:6},並換成修正版"
md5sum "$DST/scan_5c.py"
