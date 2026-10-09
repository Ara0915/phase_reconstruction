#!/bin/bash
# 把 diag42_v3_update/ 的診斷腳本(diag_nan_v3.py、diag_nan_v3b.py、diag_nan_v3c.py)複製到 ~/cdi —— 只新增檔案,不修改任何既有檔案。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
bad=0
for f in realign_eval.py src/model.py src/losses.py src/data.py; do
  [ -f "$DST/$f" ] && echo "  ✅ $f 存在" || { echo "  ❌ $f 不存在"; bad=1; }
done
now=$(md5sum "$DST/src/model.py" | cut -d' ' -f1)
[ "$now" = "5187ec38cf0c49fae3267e90026aca7a" ] && echo "  ✅ src/model.py 為 v3 版本" || { echo "  ❌ src/model.py 不是 v3 版本($now)"; bad=1; }
[ -f /work/elviss0915/runs/probe_v3_unroll_s0/final.pt ] && echo "  ✅ probe_v3_unroll_s0/final.pt 存在" || { echo "  ❌ probe_v3_unroll_s0/final.pt 不存在"; bad=1; }
for f in diag_nan_v3.py diag_nan_v3b.py diag_nan_v3c.py; do
  if [ -e "$DST/$f" ] && ! cmp -s "$SRC/$f" "$DST/$f"; then
    echo "  ❌ $f 已存在且內容不同 —— 不覆蓋"; bad=1
  fi
done
[ "$bad" -ne 0 ] && { echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"; exit 1; }
for f in diag_nan_v3.py diag_nan_v3b.py diag_nan_v3c.py; do
  if [ -e "$DST/$f" ]; then echo "  ✅ $f 已存在且相同"; else cp "$SRC/$f" "$DST/$f"; sed -i 's/\r$//' "$DST/$f"; echo "  ✅ 新增 $f"; fi
done
md5sum "$DST/diag_nan_v3.py" "$DST/diag_nan_v3b.py" "$DST/diag_nan_v3c.py"
echo "  完成。"
