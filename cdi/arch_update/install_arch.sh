#!/bin/bash
# 把 arch_update/ 裡的檔案安裝到 ~/cdi —— 在計算節點內執行。
#
# 會修改兩個既有檔案(src/config.py、src/model.py)。
# 先比對 md5:config.py 須為上次 amb_update 安裝後的版本、model.py 須為國網上的現行版本(不含 dilation),
# 一致才覆蓋;覆蓋前另存備份到 ~/cdi/backup_pre_arch/。
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST=/home/elviss0915/cdi

declare -A EXPECT=(
  ["src/config.py"]="358e72cf8364da91c54977eba9e9ab3f"
  ["src/model.py"]="43b0efd597af5742f3f62ea43c94a4a0"
)

echo "---- 1. 比對國網上的現有版本 ----"
bad=0
for f in "${!EXPECT[@]}"; do
  now=$(md5sum "$DST/$f" | cut -d' ' -f1)
  if [ "$now" = "${EXPECT[$f]}" ]; then
    echo "  ✅ $f 與預期版本相同"
  else
    echo "  ❌ $f 與預期版本不同 —— 不覆蓋"
    bad=1
  fi
done
if [ "$bad" -ne 0 ]; then
  echo "停止。把上面的訊息貼給 Claude。"
  exit 1
fi

echo "---- 2. 備份 ----"
mkdir -p "$DST/backup_pre_arch/src"
cp "$DST/src/config.py" "$DST/src/model.py" "$DST/backup_pre_arch/src/"
echo "  已備份到 $DST/backup_pre_arch/"

echo "---- 3. 安裝 ----"
cp "$SRC/config.py" "$SRC/model.py" "$DST/src/"
cp "$SRC/check_arch.py" "$SRC/run_arch.sh" "$DST/"
cp "$SRC/arch_fft_oracle.yaml" "$SRC/arch_attn_oracle.yaml" "$DST/configs/"
sed -i 's/\r$//' "$DST/run_arch.sh"
echo "  完成。修改 2 個、新增 4 個:"
echo "    修改 src/config.py  src/model.py"
echo "    新增 configs/arch_fft_oracle.yaml  configs/arch_attn_oracle.yaml  check_arch.py  run_arch.sh"
