#!/bin/bash
# 把 probe42_v3c_update/ 的檔案安裝到 ~/cdi —— 在計算節點內執行。
# 只新增 4 個檔案,不修改任何既有檔案(v3c 只改設定檔的學習率,程式與 v3b 相同):
#     check_probe42_v3c.py、run_probe42_v3c.sh、run_check_probe42_v3c.sh、configs/probe_v3c_unroll.yaml
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
R=/work/elviss0915/runs
declare -A DEPEND=(
  ["train.py"]="4233b73a82f87ecba6e4d81e7ee66460"
  ["src/model.py"]="5187ec38cf0c49fae3267e90026aca7a"
  ["src/data.py"]="e6f66a5e669bf6a5381649ee52a3c7de"
  ["src/config.py"]="d9d63f6b0c8c46033c1a21b189c3a724"
  ["src/losses.py"]="75dac35af528fca9f4a7ad830f7dc09a"
  ["src/physics.py"]="f7d6902e4cb27345f643c7b3e11bd507"
  ["src/hio.py"]="a5eac4662f326b269ad1f122760f607b"
  ["eval.py"]="e6e7d2c4700e4144dc393bad55c99e9b"
  ["realign_eval.py"]="f826432269d473b39f3e71bae2643e5f"
)
NEWFILES=(check_probe42_v3c.py run_probe42_v3c.sh run_check_probe42_v3c.sh configs/probe_v3c_unroll.yaml)
bad=0
echo "---- 1. 依賴的既有檔案(須為 v3b 安裝後的版本,不修改)----"
for f in "${!DEPEND[@]}"; do
  if [ ! -f "$DST/$f" ]; then echo "  ❌ $f 不存在"; bad=1; continue; fi
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${DEPEND[$f]}" ]; then echo "  ✅ $f"; else echo "  ❌ $f 與預期版本不同($now)"; bad=1; fi
done
[ -f "$DST/configs/probe_v3b_unroll.yaml" ] && echo "  ✅ configs/probe_v3b_unroll.yaml" || { echo "  ❌ configs/probe_v3b_unroll.yaml 不存在"; bad=1; }
for s in 0 1 2; do
  [ -e "$R/probe_v3c_unroll_s$s" ] && echo "  ⚠️ $R/probe_v3c_unroll_s$s 已存在 —— train.py 會從 ckpt 續跑;若不是有意續跑,先告訴 Claude"
done
echo "---- 2. 新檔案不可覆蓋不同內容 ----"
for f in "${NEWFILES[@]}"; do
  if [ -e "$DST/$f" ]; then
    if cmp -s "$SRC/$f" "$DST/$f"; then echo "  ✅ $f 已存在且相同"; else echo "  ❌ $f 已存在且內容不同 —— 不覆蓋"; bad=1; fi
  else
    echo "  ✅ $f 尚不存在,可新增"
  fi
done
[ "$bad" -ne 0 ] && { echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"; exit 1; }
for f in "${NEWFILES[@]}"; do
  [ -e "$DST/$f" ] || cp "$SRC/$f" "$DST/$f"
  sed -i 's/\r$//' "$DST/$f"
done
echo "  完成。新增 4 個(不修改既有檔案)。"
