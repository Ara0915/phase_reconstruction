# §1b.6 理想條件驗證 —— 操作說明

> ## ⚠️ 執行規則(絕對,無例外)
>
> **所有 python / 訓練 / 評估一律在計算節點內執行。**
>
> 登入節點**只做兩件事**:
> - `sbatch` 送件
> - `squeue` / `sacct` 查狀態
>
> 不論工作多輕、跑多快,都不在登入節點執行任何 python。
> 本文件的每一個 `python` 指令都已包在 `srun` 之內。

---

## 檔案

| 檔案 | 放到哪 |
|---|---|
| `configs/ideal_base.yaml` | `~/cdi/configs/` |
| `configs/ideal_bs0_ph1e5.yaml` | `~/cdi/configs/` |
| `configs/ideal_bs0_ph1e5_oracle.yaml` | `~/cdi/configs/` |
| `smoke_ideal_conditions.sh` | `~/cdi/` |
| `run_ideal_conditions.sh` | `~/cdi/` |
| `check_ideal_conditions.py` | `~/cdi/` |
| `hio_ideal_conditions.py` | `~/cdi/`(追加實驗:近理想量測下的 HIO,見實驗設計 §九) |
| `ambiguity_check.py` | `~/cdi/`(平凡歧異性檢查,見實驗設計 §十一;需先跑過 `hio_ideal_conditions.py`) |
| `realign_eval.py` | `~/cdi/`(以對齊方式重新評估既有結論,見實驗設計 §十三;需 `ambiguity_check.py` 在同一資料夾) |

### 命名說明

前綴 `ideal_*` 為**新增**,已登記到研究日誌 §B3 的前綴對照表。
命名可組合、自我說明,`1e5` 的寫法對齊既有的 `dose_1e4`:

| config | beamstop | photons | 用途 |
|---|---|---|---|
| `ideal_base` | 3 | 1e3 | 基準 + 重現性檢查 |
| `ideal_bs0_ph1e5` | **0** | **1e5** | **4-0a 主臂** |
| `ideal_bs0_ph1e5_oracle` | 0 | 1e5 | 4-0a 天花板(+ 相位) |
| `ideal_bs0` | **0** | 1e3 | 4-0b 歸因(之後才做) |
| `ideal_ph1e5` | 3 | **1e5** | 4-0b 歸因(之後才做) |

---

## 步驟

### 步驟 0 — 進入計算節點

**後面的步驟 1–3 全部在這個 session 裡面做。**

```bash
srun -A mst114378 -p dev --gres=gpu:1 -c 8 -t 01:00:00 --pty bash
```

**確認提示字元變成 `hgpn` 開頭再往下。** 沒變就是還在登入節點,停。

```bash
cd ~/cdi
module purge
module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi
```

---

### 步驟 1 —(計算節點內)驗證 config 的變因控制

```bash
python check_ideal_conditions.py --configs-only
```

**預期輸出**:三項 ✅,並印出「劑量比 = 100x,預期 log_mean 平移 = 4.605」。

出現 ❌ 就是 config 放錯或欄位打錯,先修再往下。

---

### 步驟 2 —(計算節點內)煙霧測試

**這步不要跳過。`photons_per_pix = 100000` 從未跑過**
(既有最高是 `dose_1e4` 的 10,000,中心像素期望光子數會由 7,720 → 772,000)。

```bash
bash smoke_ideal_conditions.sh
```

**要看到四件事**:

1. 兩組都跑完不報錯
2. `log_mean` 平移約 **+4.6** → 正規化常數確實重新校正(檢查 2)
3. `ref_energy` 兩組一致 → 符合「只由物體決定,與劑量無關」
4. 沒有 NaN/Inf,並記下單 epoch 秒數

> **若 log_mean 平移量 < 1.0 → 停下來。**
> 代表常數沒有重新校正,正式跑會在最重要的實驗上產生假陰性。

---

### 步驟 3 —(計算節點內)估時間

煙霧測試是 `subset_n=2000, epochs=3`;正式跑是 `20000 × 40`。

```
單 epoch 秒數 × 40 × 10 × 1.5(餘裕) = 需要的秒數
```

實測 0.6 s/epoch → 約 6 分鐘,腳本已可改為 `-t 01:00:00`(較易被 backfill 排入)。
**不用離開計算節點**,直接接步驟 4。

---

### 步驟 4 —(計算節點內)送正式 batch(9 jobs)

**直接在同一個計算節點 session 裡送**,不必回登入節點。
送出的 batch job 與這個互動 session 互相獨立:session 結束或逾時,batch job 照跑。

```bash
cd ~/cdi
sbatch run_ideal_conditions.sh
squeue -u elviss0915
```

`--array=0-8%2` 已內建併發上限 2。

腳本內部第一步會**再跑一次 config 驗證**(在計算節點內),
不通過時**只有 task 0 會中止**,其餘 8 個 task 不受影響照跑 ——
故真正的把關是送件前的 `--configs-only`(步驟⑧),job 內這次只是留紀錄。

**partition**:腳本用 `normal2`(H200),與既有 `run_*.sh` 一致。
`normal`(H100)排隊通常較嚴重,不使用。

---

### 步驟 5 —(計算節點內)跑完後的健全性檢查與結果

若原本的互動 session 還在,直接在裡面跑;已逾時才重開:

```bash
srun -A mst114378 -p dev --gres=gpu:1 -c 4 -t 00:30:00 --pty bash
cd ~/cdi
module purge && module load miniconda3/24.11.1 cuda/12.4
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate /work/elviss0915/envs/cdi

python check_ideal_conditions.py
```

輸出包含:

- **檢查 2** 正規化常數重新校正
- **檢查 3** 重現性 —— `ideal_base` 是否重現 `v2_proc`
- **主結果表** 三組 × 六個指標,mean ± std
- **主判準** 正常路徑的改善幅度、與 Oracle 落差的收斂程度
- **交叉評估矩陣** 2×2 條件下的表現

> **檢查 3 的參考值已填好**:`V2_PROC_FRC_GAIN = 0.2106`(`v2_proc` 實測 3 seeds 平均),容忍 ±0.02。
> 取值依據寫在 `check_ideal_conditions.py` 頂端的註解。

---

## 怎麼判讀

### 主判準

> **`ideal_bs0_ph1e5` 相對 `ideal_base` 的 FRC gain 改善幅度,
> 以及它與 `ideal_bs0_ph1e5_oracle` 之間落差的收斂程度。**

現況條件下的落差已知約 **+0.08**(`v2_proc` ~0.21 vs `v2_proc_oracle` 0.2924)。

| 結果 | 解讀 | 後續 |
|---|---|---|
| 大幅改善 + 落差顯著收斂 | **核心診斷坐實** | 照原計畫進階段四 |
| 小幅改善 | 部分成立 | 跑 4-0b 歸因(`ideal_bs0` / `ideal_ph1e5`) |
| 差異 < 約 0.02 | **無法與 seed 波動區分** | 視同幾乎沒改善;單 seed 標準差 0.0093,3-seed 平均之差的標準誤 0.0076 |
| 幾乎沒改善 | **診斷可能是錯的** | **停下來重想**,階段四、五的動機須重新論證 |

**事先聲明**:若主判準顯示無改善,這是有價值的負面結果,
**不應以「換個指標再看看」處理**。

### ⚠️ 交叉評估矩陣的正確讀法

`eval_photons × eval_beamstops = 2×2` 產生四個格子。

**只有主場格子(訓練條件 = 評估條件)可以拿來比較模型好壞。**

離開主場的格子(例如 `ideal_base` 在 ph=1e5 下評估)帶有
**§12b.4 的正規化假影** —— `log_mean` 是在該模型的訓練劑量下校正的,
換劑量會使輸入整體平移。**那些格子的退化不可解讀為物理效應。**

反過來說,那些格子正好是**待辦 2(輸入正規化對劑量偏移的敏感度)的
直接量測**,而且劑量差 100 倍(§12b.4 當時只有 3 倍)。
等於順手把一個 open question 關掉。

---

## 這批**不包含**過取樣,是刻意的

§1b.6 原文寫「beamstop=0、劑量提高、**過取樣顯著提高**」三項一起推。
本設計把過取樣拆出去,理由:

1. **過取樣 2.29 並未違反定理** —— 唯一性門檻是 2,已經滿足,只是餘裕薄。
   真正字面上違反的只有 beamstop(缺資料)與 Poisson 雜訊(非精確量測)。

2. **過取樣對主指標的作用方向與假設相反** —— §12.3 實測
   過取樣率 2.29 → 4.57 時 FRC gain 由 0.1400 降到 0.1147。
   三項綁一起推,可能出現 beamstop+劑量帶來改善、過取樣帶來惡化,
   **互相抵消而看似沒動** → 在本專案最重要的實驗上產生**假陰性**。

過取樣改列獨立的第三臂(4-0c),與既有的 `ovs64` / `ovs128` 配對比較。
注意那兩組用 `match_support_to: null`,與本批的 `0.0318` 不同,
**不可直接並列** —— 4-0c 需要自己的 canvas-64 對照。

---

## 已查核過的事

送出前逐條查了程式碼,原本擔心的兩個地雷**都已經修好了**:

| 原本的擔憂 | 查核結果 |
|---|---|
| `eval.py` 是否用各組自己的物理條件評估(§8.1 / §12b.1 **踩過兩次**) | ✅ **已修**。`eval.py:51` 讀該 run 的 `config_used.json`,`bs_mask` 與 `photons` 皆來自其自身設定 |
| 正規化常數是否在新劑量下重新校正(§12b.4) | ✅ **已修**。`train.py:72` 在 `log_mean is None` 時呼叫 `calibrate_input_norm`,後者用該 run 的 `cfg.photons_per_pix` |
| `config_used.json` 是否在校正後才寫出 | ✅ `train.py` 先校正、再 dump |
| `beamstop_r: 0.0` 是否支援 | ✅ `bs00.yaml` 已用過 |
| `Cfg.load` 是否擋得下打錯的欄位 | ✅ `config.py:195` 會 raise |

煙霧測試仍然要跑,因為 **ph=1e5 這個量級是新的**。
