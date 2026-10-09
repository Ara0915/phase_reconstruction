#!/bin/bash
# 把 probe42_v3_update/ 裡的檔案安裝到 ~/cdi —— 在計算節點內執行。
#
# 修改 2 個既有檔案(先比對 md5,須為目前國網上的版本,一致才覆蓋;覆蓋前備份到 ~/cdi/backup_pre_probe42v3/):
#     src/model.py     展開版新增自然讀出 + 有界殘差、修正上限(協定 §8.10);預設值 = 原行為(本地已驗證逐位元相同)
#     src/config.py    新增 unroll_readout: head、unroll_fix_bound: None、unroll_out_bound: None
# 新增 4 個檔案:check_probe42_v3.py、run_probe42_v3.sh、run_check_probe42_v3.sh、configs/probe_v3_unroll.yaml
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi

declare -A MODIFY=(
  ["src/model.py"]="d82207a2fb10c6bce3ae38118087bcb1"
  ["src/config.py"]="17a7c74793e81c9db60b514392e8dc58"
)
declare -A NEWVER=(
  ["src/model.py"]="5187ec38cf0c49fae3267e90026aca7a"
  ["src/config.py"]="172146e587938644d70bc5657803ad8d"
)
declare -A DEPEND=(
  ["train.py"]="4233b73a82f87ecba6e4d81e7ee66460"
  ["src/physics.py"]="f7d6902e4cb27345f643c7b3e11bd507"
  ["src/data.py"]="e6a5f86a858a276ae77536a91e5550e1"
  ["src/hio.py"]="a5eac4662f326b269ad1f122760f607b"
  ["src/hio_sw.py"]="3b2a58b9f487279cc78b2556dd18949c"
  ["src/metrics.py"]="796c05881ba0a09b96a8aa497bf92ae4"
  ["src/procedural.py"]="adca05ecfa98ef63c75bfe952dc76194"
  ["src/losses.py"]="75dac35af528fca9f4a7ad830f7dc09a"
  ["eval.py"]="e6e7d2c4700e4144dc393bad55c99e9b"
  ["realign_eval.py"]="f826432269d473b39f3e71bae2643e5f"
  ["ambiguity_check.py"]="9e9aafc246e43788ba34369621bece4e"
  ["probe_4_1.py"]="b5ebcef9ce546b480c27ca38ba22443f"
)
NEWFILES=(check_probe42_v3.py run_probe42_v3.sh run_check_probe42_v3.sh configs/probe_v3_unroll.yaml)
R=/work/elviss0915/runs

bad=0
echo "---- 1. 要修改的既有檔案(須為目前版本)----"
for f in "${!MODIFY[@]}"; do
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${MODIFY[$f]}" ]; then
    echo "  ✅ $f 為預期的舊版本,將備份後更新"
  elif [ "$now" = "${NEWVER[$f]}" ]; then
    echo "  ✅ $f 已是新版本(之前裝過)"
  else
    echo "  ❌ $f 不是任何已知版本($now)—— 不覆蓋"; bad=1
  fi
done

echo "---- 2. 依賴的既有檔案與首批結果(不修改)----"
for f in "${!DEPEND[@]}"; do
  if [ ! -f "$DST/$f" ]; then echo "  ❌ $f 不存在"; bad=1; continue; fi
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${DEPEND[$f]}" ]; then echo "  ✅ $f"; else echo "  ❌ $f 與預期版本不同($now)"; bad=1; fi
done
for f in "$DST/configs/amb_base.yaml" "$DST/configs/probe_unet.yaml" "$DST/configs/probe_unet_oracle.yaml" \
         "$DST/configs/probe_unroll.yaml" "$DST/configs/probe_ds_unroll.yaml" "$R/probe42_select.json" "$R/probe_4_1.json" \
         "$R/amb_base_s0/final.pt" "$R/arch_unroll_amb_s0/final.pt" \
         "$R/probe_unet_s0/final.pt" "$R/probe_unet_s1/final.pt" "$R/probe_unet_s2/final.pt" \
         "$R/probe_unet_oracle_s0/final.pt" "$R/probe_unet_oracle_s1/final.pt" "$R/probe_unet_oracle_s2/final.pt"; do
  [ -f "$f" ] && echo "  ✅ $f" || { echo "  ❌ $f 不存在"; bad=1; }
done
for s in 0 1 2; do
  if [ -e "$R/probe_v3_unroll_s$s" ]; then
    echo "  ⚠️ $R/probe_v3_unroll_s$s 已存在 —— train.py 會從 ckpt 續跑;若不是有意續跑,先告訴 Claude"
  fi
done

echo "---- 3. 新檔案不可覆蓋不同內容 ----"
for f in "${NEWFILES[@]}"; do
  if [ -e "$DST/$f" ]; then
    if cmp -s "$SRC/$f" "$DST/$f"; then echo "  ✅ $f 已存在且相同"; else echo "  ❌ $f 已存在且內容不同 —— 不覆蓋"; bad=1; fi
  else
    echo "  ✅ $f 尚不存在,可新增"
  fi
done

if [ "$bad" -ne 0 ]; then
  echo "停止,沒有安裝任何東西。把上面的訊息貼給 Claude。"
  exit 1
fi

echo "---- 4. 備份與安裝 ----"
mkdir -p "$DST/backup_pre_probe42v3/src"
for f in "${!MODIFY[@]}"; do
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${MODIFY[$f]}" ]; then
    cp "$DST/$f" "$DST/backup_pre_probe42v3/$f"
    cp "$SRC/$f" "$DST/$f"
  fi
done
for f in "${NEWFILES[@]}"; do
  [ -e "$DST/$f" ] || cp "$SRC/$f" "$DST/$f"
done
for f in "${!MODIFY[@]}" "${NEWFILES[@]}"; do
  if grep -q $'\r' "$DST/$f"; then sed -i 's/\r$//' "$DST/$f"; fi
done
echo "  備份在 $DST/backup_pre_probe42v3/"
fail=0
for f in "${!MODIFY[@]}"; do
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  [ "$now" = "${NEWVER[$f]}" ] && echo "  ✅ $f = 新版本" || { echo "  ❌ $f 安裝後 md5 不符($now)"; fail=1; }
done
[ "$fail" -eq 0 ] && echo "  完成。修改 2 個、新增 4 個。" || { echo "  安裝後檢查失敗,把上面的訊息貼給 Claude。"; exit 1; }
