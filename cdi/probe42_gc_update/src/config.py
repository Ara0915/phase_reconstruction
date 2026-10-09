"""設定物件 —— 所有超參數的唯一來源。

原則:.py 裡不出現任何硬編碼的實驗數字,全部來自 YAML。
"""
from dataclasses import dataclass, asdict
import math
import yaml


@dataclass
class Cfg:
    # ---- 幾何 ----
    canvas: int = 64          # 畫布邊長;物體置中 -> support 佔 19%,滿足過取樣唯一性條件
    digit: int = 28           # 物體邊長
    support_pad: int = 2      # 已知 support 比物體外擴幾格(實驗上由自相關估出的餘裕)

    # ---- 樣品(多材料模型) ----
    phase_max: float = math.pi / 2
    mat_abs: tuple = (0.30, 1.00)   # (輕元素, 重元素) 吸收係數
    mat_pha: tuple = (1.00, 0.70)   # (輕元素, 重元素) 相位係數
    amp_floor: float = 0.05

    # ---- 量測 ----
    beamstop_r: float = 3.0         # beamstop 半徑(pixel)
    photons_per_pix: float = 1e3    # 每像素平均光子數 = 劑量(不是總預算)
    add_poisson: bool = True

    # ---- 輸入表示 ----
    input_transform: str = "none"   # none | radial_flatten

    # 輸入正規化方式。
    #   per_sample: 逐張減平均除標準差(舊行為)
    #   global:     用全資料集共用的常數
    #
    # 這一項比看起來重要。逐張標準化會把「這個樣品散射了多少光子」的
    # 絕對尺度資訊洗掉 —— 而那正是 v4 修正 #2(固定光通量)刻意保留的東西。
    # 等於在前向模型裡小心保留,又在輸入層丟掉。
    # 網路收不到絕對亮度,只能靠訓練分布的先驗去猜,這會直接製造泛化落差。
    input_norm: str = "per_sample"  # per_sample | global

    # ---- Oracle 控制實驗 ----
    # True 時把「繞射場的真實相位」一併交給網路。
    # 這是唯一能把「架構不足」和「相位遺失本身太難」分開的實驗:
    #   Oracle 幾乎完美 -> 架構沒問題,瓶頸是相位遺失
    #   Oracle 也做不好 -> 架構就是不夠,attention 才有理由
    oracle_phase: bool = False

    # ---- 模型 ----
    base_channels: int = 32
    n_down: int = 3

    # 編碼器卷積的 dilation。用於在**參數量不變**的前提下擴大感受野。
    #
    # 動機:過取樣實驗(canvas 64/96/128,物體固定 28)原本要測
    # 「增加已知為零的區域」的效果,但畫布放大時感受野涵蓋率同時下降,
    # 兩個效應方向相反且混在一起,使該實驗無法歸因。
    #
    # **以梯度追蹤實測**的感受野與涵蓋率(非公式估算 —— 公式會高估,
    # 因其未計入 decoder 與 skip connection 的實際傳遞):
    #   canvas  dil=1        dil=2        dil=3
    #      64   64 (100.0%)  63 (98.4%)   64 (100.0%)
    #      96   85 ( 88.5%)  95 (99.0%)   96 (100.0%)
    #     128   85 ( 66.4%)  127(99.2%)  128 (100.0%)
    # 參數量三者皆為 1,928,450,完全不變。
    #
    # ⚠️(實驗設計 §19.4)上表係以另一份支援 dilation 的 model.py 量得;
    # 國網上實際執行的 model.py 不支援 dilation,ovs96 / ovs128 / ctl_d2 / ctl_d3
    # 的 dilation 設定當時皆被靜默忽略。現行 build_model 遇 dilation != 1 會直接報錯。
    #
    # 注意:改變 dilation 本身會引入 gridding 效應(kernel 取樣變稀疏),
    # 故需以「固定 canvas、只改 dilation」的對照組隔離此效應,
    # 方能確認主實驗的差異確實來自過取樣率。
    dilation: int = 1

    # ---- 損失 ----
    # 振幅損失在「樣品內」與「空白區」分開計算並各自正規化。
    #
    # 動機:舊版以 F.l1_loss 在整個輸出區域上平均,使兩區的相對權重由
    # **面積比例**隱含決定,而非由設計者指定。兩資料集的空白佔比不同
    # (MNIST 46.4%、程序生成 63.3%),故振幅與相位的有效權重比
    # 會隨資料集而變,這本身即不利於跨設定比較。
    # 相位損失原本就只在樣品內計算,兩通道的正規化並不一致。
    #
    # 待釐清的現象:實測模型振幅存在系統性低估
    #   MNIST -0.026、程序生成 -0.193(後者約 7.5 倍)
    # 該偏差為有號量,PSNR 與 L1 皆取絕對值或平方而無法偵測;
    # 僅材料反演因需計算 φ/A 而對其極度敏感,方將其暴露出來。
    #
    # **注意**:「空白區主導損失促使模型壓低振幅」曾是本項修改的假設,
    # 但受控測試(整體縮放、含邊界溢出的模擬輸出)顯示舊版損失的
    # 最小值仍落在正確的 k=1.0,並未重現該誘因。故低估的成因尚未確認,
    # 本修改的正當性來自「兩通道正規化應一致」,而非該假設。
    # 修改後是否消除低估,須以實際訓練驗證。
    w_amp_out: float = 1.0      # 空白區振幅損失的權重
    w_phase: float = 1.0
    w_dc: float = 0.1
    w_freq: float = 0.0
    freq_power: float = 1.0

    # ---- 平凡歧異性不變的訓練目標(研究日誌 §十四) ----
    # 物體平移、或共軛翻轉再加全域相位 φ_max,繞射強度完全相同;
    # 本專案的約束(32x32 方框 support、φ ∈ [0, φ_max])排除不了這些等價解。
    # 監督式訓練每張輸入只給其中一個作為答案,網路猜不到時只能取平均(模糊)。
    #
    # True 時,算 loss 前先把真值換成「與網路輸出最接近的等價版本」:
    #   {原解, 共軛翻轉 + 全域相位 φ_max} x {|平移| <= amb_max_shift 的循環平移}
    # 最接近 = 複數場 L2 距離最小(以 FFT 互相關一次算完所有平移)。
    # 選擇不回傳梯度;資料一致性項(w_dc)本就對等價解不變,不受影響。
    # False(預設)時 loss 與先前完全相同。
    amb_invariant: bool = False
    amb_max_shift: int = 4          # 實測純 HIO 相對真值的平移中位數 2 px

    # ---- 網路架構(實驗設計 §十八、§十九) ----
    # 實測(§十八):即使給了繞射相位(oracle),U-Net 在非週期結構上仍學不會
    # 反傅立葉轉換(polygon 0.11、bandpass 0.04;同一量測下純 HIO 0.75 / 0.71)。
    #
    #   unet  原架構(預設,行為與先前完全相同)
    #   fft   偵測器端小網路 -> 固定的反傅立葉轉換 -> 實空間 U-Net
    #         繞射振幅直接取自量測(不用學),網路只需在偵測器端估相位、
    #         在實空間做修正。轉換本身由程式計算。
    #   attn  U-Net 的 16x16 與 8x8 兩層各加一個 self-attention 區塊,
    #         讓每個位置都能直接看到整張圖;轉換須由網路自己學。
    arch: str = "unet"
    fft_k: int = 4                  # fft:偵測器端同時估計的相位假設數(K 個複數場)
    attn_heads: int = 4             # attn:multi-head 數

    # ---- 架構 unroll:把 HIO 展開成網路(實驗設計 §二十一) ----
    # 第 0 輪:同 fft 架構的偵測器端估相位 -> 固定反轉換 -> 學習的修正。
    # 第 1..T 輪:固定的傅立葉約束(換上量測振幅)-> 固定的 HIO 實空間更新
    #             -> 學習的修正(殘差,初始化為 0 —— 未訓練時即等於 T 次 HIO)。
    # 最後以小卷積頭輸出振幅 / 相位(與 U-Net 相同的三個輸出約束)。
    unroll_t: int = 5               # 展開輪數
    unroll_base: int = 12           # 每輪修正網路(小 U-Net)的基底通道數;總參數約 1.66M,略少於 U-Net 的 1.93M
    unroll_beta: float = 0.9        # HIO 負回饋強度,與 hio.py 相同
    # 修正網路的種類與頻率(階段四協定 §3.5;預設值 = 原版,舊 checkpoint 不受影響)
    #   unet  小 U-Net(基底 unroll_base,3 次下採樣)
    #   c16   4 層 3x3 卷積、16 通道      c8  3 層 3x3 卷積、8 通道
    unroll_refine: str = "unet"
    unroll_every: int = 1           # 每幾輪做一次學習的修正(中間只跑物理步驟);1 = 每輪

    # ---- 探針照明(階段四,協定 §3.2) ----
    # none  平面波(psi = O),既有所有實驗
    # disk  平頂圓盤探針(實數、內部 1、外部 0),圓心在畫布中心、半徑 probe_r px。
    #       出射波 psi = P * O 取代物體,成為量測與重建的目標;
    #       support_mask 回傳圓盤(模型輸出遮罩與 HIO 的 support 同時改變)。
    probe: str = "none"
    probe_r: float = 12.0

    # ---- 訓練資料來源 ----
    # 可組合:mnist | fashion | shapes | texture | procedural
    # 已確認的病因(研究日誌 §6.2):MNIST 只有 10 類,
    # 網路能靠「辨識類別 -> 由記憶重繪」的捷徑取得高分。
    # 實測 PCA 有效維度:MNIST 69,程序生成 227(相同 support 密度下)。
    train_sources: tuple = ("mnist",)

    # 訓練池中「相異物體」的數量。None = 全部相異(等於 subset_n)。
    #
    # 設為較小值時,只生成該數量的相異物體,再重複填滿至 subset_n,
    # 使**梯度更新步數、batch size、資料總量皆維持不變**,
    # 唯一變動的是「模型看過幾種不同的結構」。
    # 這是「多樣性劑量曲線」實驗的單一變因。
    #
    # 背景:程序生成資料使 Oracle 泛化落差由 16.39 降至 8.17 dB,
    # 但兩點之間相差 2000 倍(MNIST 10 類 vs 程序生成 20000 個相異),
    # 中間的行為未知 —— 效果是持續遞減或存在飽和點,
    # 決定了「繼續增加資料」是否仍為有效的改進方向。
    n_unique: int = None
    proc_kinds: tuple = ("polygon", "lattice", "bandpass")
    proc_weights: tuple = (1, 2, 2)
    proc_contrast_gamma: float = 0.42   # 對齊 MNIST 的 support 內平均振幅 0.405
    match_support_to: float = None      # 對齊 support 佔比;None = 各來源自然值

    # ---- 訓練時的物理參數隨機化 ----
    # 固定值可以被網路當成常數背下來。對照組(Ophus, Stanford/LBNL)於訓練時
    # 隨機化探針、像差、雜訊與部分同調性,明確以 parameter-free generalization
    # 為目標。評估時一律使用 cfg 的固定值,以確保跨設定可比較。
    randomize_photons: tuple = None     # (lo, hi),對數均勻取樣
    randomize_beamstop: tuple = None    # (lo, hi),均勻取樣

    # 評估時要掃描的實驗條件。
    #
    # **這是先前評估方式的缺陷**:訓練時劑量於 100-10000 間隨機、
    # beamstop 於 0-8 間隨機,但評估只用固定的一組(劑量 1000、beamstop 3),
    # 等同「範圍練得廣、只考其中一題」——
    # 一個涵蓋廣泛條件的模型,在單一條件上本就可能輸給專精該條件者。
    # 據此得出的「參數隨機化使表現下降」結論並不成立。
    #
    # parameter-free generalization 的真正含意是
    # 「換不同實驗條件仍能運作」,故須在多重條件下評估。
    # 此即計畫 benchmark 的 probe robustness 軸。
    eval_photons: tuple = ()            # 空 tuple = 只用 cfg.photons_per_pix
    eval_beamstops: tuple = ()          # 空 tuple = 只用 cfg.beamstop_r

    # ---- HIO 後處理 ----
    # 網路輸出當初始解,再以 HIO 迭代強制資料一致性與支撐約束。
    #
    # 依據(交接簡報 §4.4):跨分布能力僅在繞射相位已知時出現
    # (v2_proc_oracle 2.0/4 對 v2_proc 0.0/4),且資料多樣性推進至
    # 50000 種相異物體仍無法帶來該能力 —— 瓶頸為物理資訊而非訓練資料。
    # R-factor 高於雜訊底線約 20 倍,顯示輸出在物理上仍不一致。
    #
    # **迭代數必須掃描而非固定**:實測(合成劣化 + Poisson 雜訊)
    #   迭代  0    PSNR 28.14  FRC 0.849  材料 0.196
    #   迭代 10    PSNR 36.60  FRC 0.967  材料 0.053  <- 最佳
    #   迭代 300   PSNR 33.54  FRC 0.926  材料 0.083  <- 開始擬合雜訊
    # 過多迭代會擬合 Poisson 雜訊;R-factor 的最小值與 FRC 最佳點接近,
    # 且不需 ground truth,故可作為真實資料上的停止準則。
    hio_iters: tuple = ()        # 要評估的迭代數,空 tuple = 不做 HIO
    hio_beta: float = 0.9        # 負回饋強度,文獻常用 0.9

    # 純 HIO 對照(隨機起點,不使用網路):傳統方法的基準。
    # 用於量化「網路提供的初始解」在速度軸上值多少 ——
    # 若網路 + 200 次迭代可達純 HIO 數千次的品質,
    # 則網路初始化有實質貢獻,即使跨分布能力未改善。
    hio_random_init: bool = False

    # ---- 訓練 ----
    epochs: int = 40
    batch_size: int = 128
    lr: float = 2e-3
    # 梯度範數裁切(階段四協定 §3.9)。None = 不裁切(既有所有實驗的行為,逐位元不變)。
    # 設定時另啟用「非有限值防護」:loss 或梯度出現 nan / inf 的 step 直接跳過並計數,
    # 不讓單一壞 step 污染權重(probe_unroll 首批曾於 epoch 4 暴衝、epoch 10 變 nan)。
    grad_clip: float = None
    subset_n: int = 20000

    # ---- 評估 ----
    test_seed: int = 1234           # 測試集固定,讓不同 job 的數字可直接比較
    eval_n: int = 512
    gen_n: int = 256

    # ---- 路徑 ----
    data_root: str = "/work/elviss0915/data"

    # ---- 執行 ----
    deterministic: bool = False     # True 會關掉 cudnn.benchmark 變慢;
                                    # 實測 seed 波動僅 ±0.11 dB,通常不需要

    # ---- 由程式填入,不要手寫 ----
    ref_energy: float = None        # 光通量校正常數:訓練時算好存檔,評估時讀回
    log_mean: float = None          # input_norm="global" 用的全域常數
    log_std: float = None
    ac_scale: float = None

    @property
    def crop(self):
        return self.digit + 2 * self.support_pad

    @classmethod
    def load(cls, path):
        d = yaml.safe_load(open(path)) or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path} 有未知欄位: {sorted(unknown)}")
        for k in ("mat_abs", "mat_pha"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("mat_abs", "mat_pha"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)
