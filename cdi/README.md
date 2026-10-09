# CDI 相位回復 — Slurm 版

由 `cdi_v4.ipynb` 拆分而來。設定進 YAML、邏輯進 `src/`、結果寫檔案。

環境:國網中心晶創25(Nano 5),帳號 `elviss0915`,計畫 `mst114378`。

---

## 目前要跑的實驗

**先跑 Oracle 控制實驗**(`run_oracle.sh`,4 configs × 3 seeds = 12 jobs),
再視結果決定要不要跑完整的 beamstop 掃描(`run_array.sh`,15 jobs)。

### Oracle 控制實驗 — 先跑這個

Beamstop 假設已在 Colab 驗證:方向對了,四個指標都顯著改善(3.7σ~5.2σ),
**但幅度很小** —— AUC gain 只從 −0.0042 擺到 +0.0191,而「完美 vs 平庸」的
尺度是 0.50;R-factor 幾乎沒動(16.3 → 15.5 倍離雜訊底線)。
就算完全沒有 beamstop,解析度也只是**追平**平庸預測器。

所以 beamstop 是真實但**次要**的因素,主因還在別的地方。

目前排除完的清單:劑量 → 輸出頭 → 目標函數 → 誤差分配 → 輸入表示 → 低頻缺失(次要)。
剩下兩個候選:**架構感受野不足** vs **任務本身資訊不足**。
但我們無法區分它們,因為從來沒有建立過**上界**。

**做法**:把繞射場的真實相位交給網路(`oracle_phase: true`),其餘完全不變 ——
光子數、Poisson 雜訊、beamstop 全部相同,通道 0 是同一張 log 繞射圖,
只額外給 cos/sin(繞射場相位)。唯一的差別就是那個偵測器記不下來的相位。

2×2 設計:

| | 相位未知 | 相位已知 |
|---|---|---|
| **bs=0** | `bs00` | `oracle_bs00` ← 純上界,應該接近完美 |
| **bs=3** | `bs03` | `oracle_bs03` |

- 直行相減 = 相位遺失的代價
- 橫列相減 = beamstop 低頻缺失的代價

判讀:

| 結果 | 意義 |
|---|---|
| Oracle 幾乎完美 | 架構沒問題 → 瓶頸是相位遺失本身,要往更強的先驗或 ptychography 走 |
| Oracle 也做不好 | **架構就是不夠** → attention 有明確理由 |

**這是唯一能把最後兩個假設分開的方法。** 沒有它,之後不管加什麼架構,
你都不知道自己在補哪個洞。

```bash
sbatch run_oracle.sh
```

### Beamstop 掃描 — 之後再跑

`run_array.sh`,5 個設定 × 3 個 seed = 15 個 job。
即使 Oracle 給出結論,這張「解析度 vs beamstop 大小」的曲線
仍然是專題的核心結果之一 —— 它回答真實工程問題「我的 beamstop 能做多大」。

### 已經跑過、不要重跑的

這三個假設已在 Colab 跑完,全是負面結果 —— **不要用付費算力重跑**:

| 實驗 | 結果 |
|---|---|
| 提高 `w_dc`(1.0 / 3.0) | R 大幅改善,但 PSNR 惡化 18.9σ / 11.8σ,解析度從 3.76 掉到 12.80 px |
| 頻率加權 loss | 完全無顯著差異 |
| 徑向壓平輸入 | 相位惡化 3.7σ,其餘無差異 |

`w_dc` 那組的失敗方式給了 beamstop 的線索:FRC 在 f ≈ 0.05~0.08 直墜,
而 `beamstop_r=3` 對應的頻率正好是 3/64 ≈ 0.047。
原因是資料一致性項**只在有量測的區域計算**,beamstop 內完全沒有約束 ——
模型一被要求「顧好量得到的地方」就放棄了低頻。

`configs/` 底下沒有這三個設定的 config,是刻意的。

---

## 檔案

```
/home/elviss0915/cdi/              程式碼(小、可版控)
├── train.py                       訓練進入點
├── eval.py                        評估進入點(讀 final.pt,寫 metrics.json)
├── collect.py                     彙整所有 metrics.json 成表(本機跑)
├── plot.py                        畫圖(本機跑)
├── prefetch_data.py               在登入節點先抓資料集
├── validate_generator.py          程序生成器的量化驗證(不需 GPU)
├── run_oracle.sh                  Slurm:Oracle 控制實驗(先跑這個)
├── run_norm.sh                    Slurm:輸入正規化對照(6 jobs)
├── run_sources.sh                 Slurm:訓練資料來源對照(12 jobs)← 目前要跑的
├── run_array.sh                   Slurm:beamstop 掃描
├── smoke_test.sh                  dev 佇列的縮小版測試
├── configs/
│   ├── src_*.yaml                 訓練資料來源對照(目前要跑的)
│   ├── oracle_bs00/03.yaml        Oracle 控制實驗(已完成)
│   ├── bs00.yaml ~ bs10.yaml      beamstop 掃描
│   └── dose_*.yaml                劑量掃描(第三階段,先不送)
└── src/
    ├── config.py                  Cfg dataclass,所有超參數的唯一來源
    ├── physics.py                 樣品模型、前向模型、光通量校正、輸入表示
    ├── data.py                    資料集、GPUSampler、泛化測試樣品
    ├── model.py                   U-Net(support 約束在輸出層)
    ├── losses.py                  L1 + 相位 + 資料一致性 (+ 頻率加權)
    └── metrics.py                 PSNR / phase RMSE / FRC / R-factor + 參考線

/work/elviss0915/                  資料與產出(大、不版控)
├── envs/cdi/                      conda 環境
├── data/                          MNIST / Fashion-MNIST(prefetch 抓好)
└── runs/{config}_s{seed}/
    ├── ckpt.pt                    續跑用(含 model/opt/sched/RNG 狀態)
    ├── final.pt                   最後一個 epoch 的權重
    ├── config_used.json           實際生效的設定,**含 ref_energy**
    ├── history.json               每個 epoch 的 loss 與 LR
    └── metrics.json               完整評估結果
```

---

## 執行流程

### 1. 上傳

在**你自己的終端機**(不是連線中的 SSH session):

```powershell
scp -P 2222 -r C:\你的路徑\cdi elviss0915@nano5.nchc.org.tw:/home/elviss0915/
```

注意 `-P 2222`(大寫 P)。檔案不多的話用 git 更順,之後改程式只要 `git pull`。

### 2. 建目錄 + 抓資料 ⚠️ 不能跳過

```bash
mkdir -p /work/elviss0915/runs/logs
cd /home/elviss0915/cdi
python prefetch_data.py --data-root /work/elviss0915/data
```

**計算節點通常沒有對外網路。** 所有 `load_raw` 都是 `download=False`,
沒先抓好的話 15 個 job 會全部在第一行卡死,錯誤訊息是連線逾時,很難看懂。

`runs/logs` 不建的話,job 會因為寫不出日誌直接失敗,錯誤訊息同樣很難懂。

### 3. 確認 module 名稱

```bash
module avail 2>&1 | grep -i -E "conda|cuda"
sinfo -s                      # 確認 partition 名稱
```

`run_array.sh` 和 `smoke_test.sh` 裡的 `miniconda3/24.11.1`、`cuda/12.4`
是照規格文件寫的,實際版本請自己對一次。

### 4. 在 dev 佇列跑煙霧測試 ⚠️ 不要直接送 15 個 job

```bash
srun -A mst114378 -p dev --gres=gpu:1 -c 8 -t 01:00:00 --pty bash
bash smoke_test.sh
exit                          # 記得離開,互動式 session 掛著也在計費
```

腳本會自己產一個縮小版 config(2000 張、3 epochs)並印出單 epoch 秒數。

要確認四件事:

1. 跑完不報錯
2. `runs/smoke_s0/` 底下出現 `ckpt.pt`、`final.pt`、`metrics.json`
3. **中途 Ctrl+C 再跑一次,日誌要出現 `[resume] 從 epoch N 續跑`**
4. 記下單 epoch 秒數

**估算時限**:正式跑是 `subset_n=20000`(10 倍資料)× 12 epochs。
把煙霧測試的單 epoch 秒數 × 10 × 12 × 1.5,填進 `run_array.sh` 的 `-t`。
目前預設 `01:30:00`,實測後請自行調整 —— 設太短會被砍,設太長在滿載佇列排更久。

### 5. 送件

```bash
sbatch run_oracle.sh      # 先跑這個:2x2 Oracle 控制實驗,12 jobs
# 看完結果再決定
sbatch run_array.sh       # beamstop 完整掃描,15 jobs
```

### 6. 監看

```bash
squeue --me                                    # PD=排隊 R=執行中 CG=收尾
tail -f /work/elviss0915/runs/logs/*_0.out     # 看第 0 號 job
scancel <jobid>                                # 取消整個 array
scancel <jobid>_3                              # 只取消其中一個
```

### 7. 收結果(本機)

```powershell
scp -P 2222 -r elviss0915@nano5.nchc.org.tw:/work/elviss0915/runs C:\你的路徑\
```

```bash
python collect.py --runs-root ./runs --csv summary.csv
python plot.py --runs-root ./runs --out figs
```

**分析留在本機做** —— 不需要 GPU,沒必要花叢集的錢。

---

## 相對於原規格文件修正的地方

| 項目 | 原規格 | 這裡 |
|---|---|---|
| `download=True` | 有 | **改成 False + prefetch 腳本**(計算節點沒網路) |
| 實驗內容 | 重跑已知負面結果的三個實驗 | **改成 beamstop 掃描**(真正待答的問題) |
| `lr` | 3.0e-4 | **2e-3**(對齊 notebook) |
| `epochs` | 100 | **12** |
| `batch_size` | 64 | **128** |
| `w_dc` baseline | 1.0(實測會惡化 18.9σ) | **0.1** |
| `photon_budget: 1e6` | 命名誤導 | **`photons_per_pix: 1000`**(每像素,對應劑量) |
| `pad_factor: 2` | 對不上 | **`canvas: 64` / `digit: 28`**(比例 2.29) |
| LR scheduler | 骨架完全沒有 | **OneCycleLR,狀態存進 ckpt** |
| `ref_energy` | 沒處理 | **寫進 config_used.json,eval 讀回不重算** |
| DataLoader | 建議使用 | **GPUSampler**(資料常駐 GPU,原本就是為了避開 DataLoader 開銷) |
| val split | 骨架假設存在 | **目前沒有 val**,改存 `final.pt` |
| `cudnn.deterministic` | True | **False**(實測 seed 波動僅 ±0.11 dB,不值得犧牲速度) |
| `conda activate` | 直接呼叫 | **先 `source conda.sh`**(非互動 shell 會失敗) |
| `-t` | 08:00:00 | **01:30:00**(小模型,實測後再調) |
| Oracle 控制實驗 | 無 | **新增**(唯一能分開「架構不足」與「相位太難」的實驗) |

---

## 已驗證的行為

- 續跑後的 loss 與 LR 軌跡跟未中斷的執行**完全相同**(RNG 狀態也存了)
- `ref_energy` 在 train 與 eval 之間一致
- `collect.py` 正確算出跨 seed 的 mean ± std 與 3σ 顯著性
- 全部 `print` 都有 `flush=True`;沒有 `tqdm`;沒有 `plt.show()`
- Oracle 路徑實測比正常路徑好(PSNR 21.11 vs 16.28、R 0.628 vs 0.996),
  確認多給的相位通道真的有被用到,不是接錯線

---

## 三個要記得的坑

**登入節點不能跑訓練。** `cbi-lgn01` 是共用的,直接 `python train.py` 會被警告甚至停權。
計算一律經過 `srun` 或 `sbatch`。`pip install`、改程式、發呆都在登入節點做。

**計費看佔用時間,不看使用率。** 互動式 session 掛著不用也在扣。

**錢包是共用的。** 送 15 個 job 前先用煙霧測試的實測時間換算金額,跟老師確認。

---

## 一個提醒

目前規模(193 萬參數、64×64、20000 張)在 H100 上是幾分鐘的事。
**單就算力來說你不需要 HPC** —— 你要的是「送出去就不用顧」:
斷線不掉進度、15 個 job 自動排完、checkpoint 續跑。這個理由完全成立。

但別因為有了 H100 就急著把模型放大。目前的瓶頸是問題定義與診斷,不是算力。
真正需要算力是之後放大到 256×256 或跑大規模參數掃描的時候。
