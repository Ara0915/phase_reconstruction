#!/bin/bash
# 把 scan5g_update/ 的檔案安裝到 ~/cdi —— 在計算節點內執行。
# 只新增 3 個檔案,不修改任何既有檔案:scan_5g.py、run_scan5g_train.sh、run_scan5g_eval.sh
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi
R=/work/elviss0915/runs
declare -A DEPEND=(
  ["scan_5e.py"]="cecd24eb06ec1d22abc7f6f56adc6e8a"
  ["scan_5d.py"]="f2f9d1acba473926831537c8c4f5fb49"
  ["scan_5c.py"]="161df412d2771c611ec75490a1ea019f"
  ["scan_5b.py"]="f3ce06c0b08c1775dad81e5635c7ed0e"
  ["scan_5.py"]="68a6924fc6f4eb589ea15372cd18f761"
  ["probe_4_2b.py"]="5c98526596de2d854efaab0d0c5da190"
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
NEWFILES=(scan_5g.py run_scan5g_train.sh run_scan5g_eval.sh)
bad=0
echo "---- 1. 依賴的既有檔案(不修改,只比對版本)----"
for f in "${!DEPEND[@]}"; do
  if [ ! -f "$DST/$f" ]; then echo "  ❌ $f 不存在"; bad=1; continue; fi
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${DEPEND[$f]}" ]; then echo "  ✅ $f"; else echo "  ❌ $f 與預期版本不同($now)"; bad=1; fi
done
for s in 0 1 2; do
  [ -f "$R/ideal_base_s$s/config_used.json" ] && echo "  ✅ ideal_base_s$s" || { echo "  ❌ $R/ideal_base_s$s/config_used.json 不存在"; bad=1; }
done
for g in B NAFw32b2 C; do for s in 0 1 2; do
  [ -f "$R/scan5c_${g}_DEF2_n100000_s$s/final.pt" ] && [ -f "$R/scan5c_${g}_DEF2_n100000_s$s/result.json" ] && echo "  ✅ 模型 ${g} DEF2 s$s" || { echo "  ❌ 缺模型 scan5c_${g}_DEF2_n100000_s$s"; bad=1; }
done; done
[ -f "$R/scan5e_val_envelope.json" ] && echo "  ✅ 驗證場包絡 scan5e_val_envelope.json" || { echo "  ❌ 缺 $R/scan5e_val_envelope.json"; bad=1; }
[ -f "$R/scan5e_p7_raw.json" ] && echo "  ✅ P7 的評估結果 scan5e_p7_raw.json(評估時做重現檢查)" || echo "  ⚠️ 找不到 scan5e_p7_raw.json —— 評估時會略過重現檢查(不影響訓練)"
ls -d "$R"/scan5c_P6* >/dev/null 2>&1 && echo "  ⚠️ 已有 P6 的訓練資料夾 —— 重送會從存檔續跑;若不是有意,先告訴 Claude"
echo "---- 2. 新檔案不可覆蓋不同內容 ----"
for f in "${NEWFILES[@]}"; do
  if [ -e "$DST/$f" ]; then
    if cmp -s <(sed 's/\r$//' "$SRC/$f") "$DST/$f"; then echo "  ✅ $f 已存在且相同"; else echo "  ❌ $f 已存在且內容不同 —— 不覆蓋"; bad=1; fi
  else
    echo "  ✅ $f 尚不存在,可新增"
  fi
done
[ "$bad" -ne 0 ] && { echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"; exit 1; }
added=0
for f in "${NEWFILES[@]}"; do
  if [ ! -e "$DST/$f" ]; then
    cp "$SRC/$f" "$DST/$f"
    sed -i 's/\r$//' "$DST/$f"
    added=$((added+1))
  fi
done
chmod +x "$DST"/run_scan5g_*.sh
mkdir -p "$R/logs"
echo "  完成。新增 $added 個(不修改既有檔案)。"
md5sum "$DST/scan_5g.py" "$DST"/run_scan5g_*.sh
