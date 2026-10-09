#!/bin/bash
# 把 probe42_update/ 裡的檔案安裝到 ~/cdi —— 在計算節點內執行。
#
# 修改 4 個既有檔案(先比對 md5,須為目前國網上的版本,一致才覆蓋;覆蓋前備份到 ~/cdi/backup_pre_probe42/):
#     src/config.py    新增 probe、probe_r、unroll_refine、unroll_every(預設值 = 原行為)
#     src/physics.py   support_mask 在探針設定時回傳圓盤;新增 probe_amplitude、apply_probe
#     src/data.py      訓練池與所有測試集套上探針(probe = none 時不變)
#     src/model.py     展開版可選輕量修正網路與修正頻率(預設值 = 原版,既有 checkpoint 不受影響)
# 新增 5 個檔案:select_unroll.py、check_probe42.py、run_probe42.sh、
#     configs/probe_unet.yaml、configs/probe_unet_oracle.yaml
#     (configs/probe_unroll.yaml 由 select_unroll.py 在國網上計時後產生)
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi

declare -A MODIFY=(
  ["src/config.py"]="f7dba52dbf30398f1dfbd9a496f20af6"
  ["src/physics.py"]="07e82213808c95fbef33cbad0e985c4c"
  ["src/data.py"]="557be0645fb61e23a72199e7d34262e9"
  ["src/model.py"]="06a6f59825bdf967b0f1832738e7dba2"
)
declare -A NEWVER=(
  ["src/config.py"]="a5b6abc6db1783a20ec47b1df20cb052"
  ["src/physics.py"]="f7d6902e4cb27345f643c7b3e11bd507"
  ["src/data.py"]="e6a5f86a858a276ae77536a91e5550e1"
  ["src/model.py"]="f6ec2cf95f27e759053a080ddd7fc49a"
)
declare -A DEPEND=(
  ["src/hio.py"]="a5eac4662f326b269ad1f122760f607b"
  ["src/hio_sw.py"]="3b2a58b9f487279cc78b2556dd18949c"
  ["src/metrics.py"]="796c05881ba0a09b96a8aa497bf92ae4"
  ["src/procedural.py"]="adca05ecfa98ef63c75bfe952dc76194"
  ["src/losses.py"]="75dac35af528fca9f4a7ad830f7dc09a"
  ["train.py"]="bb8a27be597ebb9189bbb95ddde9d486"
  ["eval.py"]="e6e7d2c4700e4144dc393bad55c99e9b"
  ["realign_eval.py"]="f826432269d473b39f3e71bae2643e5f"
  ["ambiguity_check.py"]="9e9aafc246e43788ba34369621bece4e"
  ["probe_4_1.py"]="b5ebcef9ce546b480c27ca38ba22443f"
)
NEWFILES=(select_unroll.py check_probe42.py run_probe42.sh configs/probe_unet.yaml configs/probe_unet_oracle.yaml)

bad=0
echo "---- 1. 要修改的既有檔案(須為目前版本)----"
upgraded=0
for f in "${!MODIFY[@]}"; do
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${MODIFY[$f]}" ]; then
    echo "  ✅ $f 為預期的舊版本,將備份後更新"
  elif [ "$now" = "${NEWVER[$f]}" ]; then
    echo "  ✅ $f 已是新版本(之前裝過)"; upgraded=$((upgraded+1))
  else
    echo "  ❌ $f 不是任何已知版本($now)—— 不覆蓋"; bad=1
  fi
done

echo "---- 2. 依賴的既有檔案(不修改)----"
for f in "${!DEPEND[@]}"; do
  if [ ! -f "$DST/$f" ]; then echo "  ❌ $f 不存在"; bad=1; continue; fi
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${DEPEND[$f]}" ]; then echo "  ✅ $f"; else echo "  ❌ $f 與預期版本不同($now)"; bad=1; fi
done
[ -f "$DST/configs/amb_base.yaml" ] && echo "  ✅ configs/amb_base.yaml 存在" || { echo "  ❌ configs/amb_base.yaml 不存在"; bad=1; }
for f in /work/elviss0915/runs/probe_4_1.json /work/elviss0915/runs/amb_base_s0/final.pt \
         /work/elviss0915/runs/arch_unroll_amb_s0/final.pt /work/elviss0915/runs/ideal_base_s0/final.pt; do
  [ -f "$f" ] && echo "  ✅ $f" || { echo "  ❌ $f 不存在"; bad=1; }
done

echo "---- 3. 新檔案不可覆蓋不同內容 ----"
for f in "${NEWFILES[@]}"; do
  if [ -e "$DST/$f" ]; then
    if cmp -s "$SRC/$f" "$DST/$f"; then echo "  ✅ $f 已存在且相同"; else echo "  ❌ $f 已存在且內容不同 —— 不覆蓋"; bad=1; fi
  else
    echo "  ✅ $f 尚不存在,可新增"
  fi
done
if [ -e "$DST/configs/probe_unroll.yaml" ]; then
  echo "  ⚠️ configs/probe_unroll.yaml 已存在(之前跑過 select_unroll.py);select_unroll.py 會檢查是否一致"
fi

if [ "$bad" -ne 0 ]; then
  echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"
  exit 1
fi

echo "---- 4. 備份與安裝 ----"
mkdir -p "$DST/backup_pre_probe42/src"
for f in "${!MODIFY[@]}"; do
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${MODIFY[$f]}" ]; then
    cp "$DST/$f" "$DST/backup_pre_probe42/$f"
    cp "$SRC/$f" "$DST/$f"
  fi
done
for f in "${NEWFILES[@]}"; do
  [ -e "$DST/$f" ] || cp "$SRC/$f" "$DST/$f"
done
for f in "${!MODIFY[@]}" "${NEWFILES[@]}"; do
  if grep -q $'\r' "$DST/$f"; then sed -i 's/\r$//' "$DST/$f"; fi
done
echo "  備份在 $DST/backup_pre_probe42/"
md5sum "$DST/src/config.py" "$DST/src/physics.py" "$DST/src/data.py" "$DST/src/model.py"
echo "  完成。修改 4 個、新增 5 個。"
