#!/bin/bash
# 把 diag42_update/diag_unroll.py 複製到 ~/cdi —— 只新增 1 個檔案,不修改任何既有檔案。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
bad=0
for f in check_probe42_ds.py realign_eval.py ambiguity_check.py; do
  [ -f "$DST/$f" ] && echo "  ✅ $f 存在" || { echo "  ❌ $f 不存在"; bad=1; }
done
[ -f /work/elviss0915/runs/probe42_ds.json ] && echo "  ✅ probe42_ds.json 存在" || { echo "  ❌ probe42_ds.json 不存在(先跑完評分)"; bad=1; }
for s in 0 1 2; do
  [ -f /work/elviss0915/runs/probe_ds_unroll_s$s/final.pt ] && echo "  ✅ probe_ds_unroll_s$s" || { echo "  ❌ probe_ds_unroll_s$s/final.pt 不存在"; bad=1; }
done
if [ -e "$DST/diag_unroll.py" ] && ! cmp -s "$SRC/diag_unroll.py" "$DST/diag_unroll.py"; then
  echo "  ❌ diag_unroll.py 已存在且內容不同 —— 不覆蓋"; bad=1
fi
[ "$bad" -ne 0 ] && { echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"; exit 1; }
[ -e "$DST/diag_unroll.py" ] || cp "$SRC/diag_unroll.py" "$DST/diag_unroll.py"
sed -i 's/\r$//' "$DST/diag_unroll.py"
md5sum "$DST/diag_unroll.py"
echo "  完成。新增 1 個(diag_unroll.py)。"
