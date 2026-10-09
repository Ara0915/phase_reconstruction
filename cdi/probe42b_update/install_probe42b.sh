#!/bin/bash
# 把 probe42b_update/ 的檔案安裝到 ~/cdi —— 在計算節點內執行。
# 只新增 2 個檔案,不修改任何既有檔案:probe_4_2b.py、run_probe42b.sh
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
R=/work/elviss0915/runs
declare -A DEPEND=(
  ["probe_4_1.py"]="b5ebcef9ce546b480c27ca38ba22443f"
  ["src/hio_sw.py"]="3b2a58b9f487279cc78b2556dd18949c"
  ["src/hio.py"]="a5eac4662f326b269ad1f122760f607b"
  ["src/physics.py"]="f7d6902e4cb27345f643c7b3e11bd507"
  ["src/metrics.py"]="796c05881ba0a09b96a8aa497bf92ae4"
  ["src/procedural.py"]="adca05ecfa98ef63c75bfe952dc76194"
  ["src/config.py"]="d9d63f6b0c8c46033c1a21b189c3a724"
  ["src/model.py"]="5187ec38cf0c49fae3267e90026aca7a"
  ["src/data.py"]="e6f66a5e669bf6a5381649ee52a3c7de"
  ["realign_eval.py"]="f826432269d473b39f3e71bae2643e5f"
  ["ambiguity_check.py"]="9e9aafc246e43788ba34369621bece4e"
)
NEWFILES=(probe_4_2b.py run_probe42b.sh)
bad=0
echo "---- 1. 依賴的既有檔案(不修改,只比對版本)----"
for f in "${!DEPEND[@]}"; do
  if [ ! -f "$DST/$f" ]; then echo "  ❌ $f 不存在"; bad=1; continue; fi
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${DEPEND[$f]}" ]; then echo "  ✅ $f"; else echo "  ❌ $f 與預期版本不同($now)"; bad=1; fi
done
[ -s "$R/probe_4_1.json" ] && echo "  ✅ $R/probe_4_1.json" || { echo "  ❌ $R/probe_4_1.json 不存在(需 4-1 的結果)"; bad=1; }
for s in 0 1 2; do
  [ -f "$R/ideal_base_s$s/final.pt" ] && echo "  ✅ ideal_base_s$s" || { echo "  ❌ $R/ideal_base_s$s/final.pt 不存在"; bad=1; }
done
[ -e "$R/probe42b.json" ] && echo "  ⚠️ $R/probe42b.json 已存在 —— 完整量測會覆寫它;若不是有意重跑,先告訴 Claude"
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
chmod +x "$DST/run_probe42b.sh"
mkdir -p "$R/logs"
echo "  完成。新增 2 個(不修改既有檔案)。"
md5sum "$DST/probe_4_2b.py" "$DST/run_probe42b.sh"
