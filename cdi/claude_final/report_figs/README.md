# 第三次報告的圖(驗證資料)

由 `make_report_figs.py` 產生(只讀結果檔)。需求清單:`../docs/報告圖表需求_第三次.md`。

| 檔案 | 來源 |
|---|---|
| `E2_errors_bar.png` | scan7d.json nets[條件]['P6B4']['nerr_sr'](3 seeds 平均) |
| `E3_dose_per_pattern.png` | scan8.json verdict.dose(N_inc、f_W、f_Mo;3 seeds 平均) |
| `R1_three_nets.png` | scan7d.json nets(3 seeds 平均) |
| `R2_position_rms.png` | scan7d.json pos[條件](rms0、各級 rms;3 seeds 平均) |
| `P1_Lpaper_nerr_c.png` | scan8.json verdict.exam['L-paper'](網路 3 seeds 平均、iter_best);scan8c.json report.crlb['L-paper'] |
| `P6_attribution.png` | scan8.json verdict.h5(比值) |
| `P7_costs.png` | (a) scan8.json verdict.h6;(b)(c) scan8c.json report.B3.table(劑量掃描)、report.crlb['L-ideal'](CRLB ∝ 1/劑量) |
| `P2_recon_Lpaper_WS2.png` | 複製既有圖 claude_stage8\results\figs_scan8\c_recon_L-paper_W.png;階段八 8b 的重建圖(百分位樣本、SV / DV 圈與判讀顏色) |
| `P2_recon_Lpaper_MoS2.png` | 複製既有圖 claude_stage8\results\figs_scan8\c_recon_L-paper_Mo.png;同上 |
| `P8_frc.png` | 複製既有圖 claude_stage8\results\figs_scan8\c_frc.png;階段八 8b 的 FRC |
| `P7_extra_ceiling_8d.png` | 複製既有圖 claude_stage8\results_8d\figs_scan8d\d8_ceiling.png;8d 精度天花板(參考) |
| `R3_recon_stage7d.png` | 複製既有圖 claude_stage7_figs\figs_scan7d\c_recon.png;7d 的重建圖:樣本 #228 = P6B4e8 在 combo 的中位數(清單寫 posL 的中位數 → 不同,見 README) |

## 第二批(`make_report_figs2.py`,需要包絡檔)

包絡檔(國網 `/work/elviss0915/runs/`)放在 `cdi/claude_stage7_figs/`:`scan8_env_L-paper.json`、`scan8_env_L-combo.json`、`scan7a_cond_combo.json`。
時間、包絡、達到時間與下界的算法同 scan_7a / 7b / 8c;重算的 L-paper 迭代法 500 次內最佳 = scan8.json(ePIE-C:posA|one ×20、0.11255)。

| 檔案 | 內容 |
|---|---|
| `P3_time_nerr_c.png` | L-paper 時間 vs nerr_c(b64 / b512)+ 1.5 / 2 × CRLB;k = 1.5:b64 ≥ × 7.3、b512 ≥ × 5.5;k = 2:迭代法快 × 4.5–6.7(b64)/ 8.1–11.3(b512) |
| `P4_speed_vs_k.png` | 加速倍數 vs k(1.25–3) |
| `P5_time_svf1.png` | 時間 vs SV-F1(L-paper、L-combo);L-combo 網路 0.893 |
| `R5_combo_time.png` | 階段七 combo 時間 vs nerr_sr;Q = 0.05 的加速區間 |

## 第三批(`make_report_figs3.py`,示意圖;用產生器與量測函式,CPU)

| 檔案 | 內容 |
|---|---|
| `E1_error_types.png` | 四種誤差示意:位置(σ = 0.5 px,放大 5 倍)、探針(名目 vs 離焦 +30%)、劑量(基準 vs ÷100 的同一張繞射圖)、部分同調(cohH);程序生成驗證場 80M、seed 0 |
| `A1_two_strategies.png` | P6B4r(容忍)vs P6B4e8(估計)的架構示意 |
| `L1_lattice.png` | L-paper 驗證場(87M,第 0 個 WS₂、第 1 個 MoS₂)的真值相位;S 單空缺(紅圈)、雙空缺(白虛線圈);10 Å 比例尺 |
| `L2_newdef.png` | L-newdef 驗證場的線缺陷(紅)、缺 S 區域(黃)、一般空缺(白) |
| `L4_metric.png` | 同一個 L-paper 驗證場:真值 vs 空白平面 —— 舊指標 nerr_sr 0.0155(< 0.02)、新指標 nerr_c 1.02 |

本機以 `claude_final/local_test/` 的假 RUN_ROOT 執行(config 由 `configs/ideal_base.yaml` 產生;Δx = 0.3577 Å/px 與國網相同,晶格與物體由 seed 決定 → 與國網相同)。E1 的計數尺度依 ref_energy,只作示意(圖上不標絕對電子數)。U2(選做)未做。

## R4(文字,不需產圖)
combo、Q = 0.05:P6B4e8 穩健加速 b64 × 2.08–2.79、b512 × 1.70–2.28;時間(b512)P6B4 0.89 ms、P6B4e8 1.64 ms(scan7d.json)。

## 數字核對
E2、R1、R2、R4、P1、P6、P7 的數字皆直接讀自結果檔,與清單一致(scan7d.json、scan8.json、scan8c.json)。

## 與清單不同之處
- R3:沿用 7d 的既有重建圖,樣本是 P6B4e8 在 combo(all three)的中位數 #228,不是清單寫的 posL 中位數;圖中已含 posL 列(同一個場)。若要嚴格依清單,需在國網重算。
- P7(b)(c):依 8c 的劑量掃描表重畫(清單原寫沿用 c8_dose.png);d8_ceiling.png 另附為參考。
