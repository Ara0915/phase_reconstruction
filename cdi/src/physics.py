"""前向模型:樣品 -> 偵測器上的光子計數 -> 網路輸入。

核心觀念:偵測器只記錄強度 I = |FFT(psi)|^2。
繞射場的相位「沒有被記錄」(不是被破壞、也不是被隨機打亂)。
樣品自身的相位 phi 才是我們要解的答案,它的資訊被編碼在 I 的分布裡。
"""
import torch


# ==============================================================================
# 樣品:幾何 x 材料
# ==============================================================================
def center_pad(img, cfg):
    """把 digit x digit 置中放進 canvas。

    padding 的作用:相位恢復唯一性要求物體 support <= 總面積一半。
    置中則把物體釘死,打掉平移歧異性。
    """
    out = torch.zeros(*img.shape[:-2], cfg.canvas, cfg.canvas,
                      device=img.device, dtype=img.dtype)
    s = (cfg.canvas - cfg.digit) // 2
    out[..., s:s + cfg.digit, s:s + cfg.digit] = img
    return out


def build_object(thickness, material, cfg):
    """由幾何 t 與材料 m 合成複數樣品的 (振幅, 相位)。

        A   = t * a(m)                被吸收多少
        phi = phase_max * t * p(m)    被拖慢多少

    單一材料的話 A 和 phi 都只是 t 的函數,問題退化成 1 個自由度。
    真實樣品是多材料的(Si / high-k / 金屬 / SiO2),
    不同材料的吸收與折射率不成比例,必須兩個通道才描述得完。

    可逆性:phi/A = p(m)/a(m) 只跟 m 有關 -> 解出 m -> 再由 A 解出 t。
    """
    a = cfg.mat_abs[0] + (cfg.mat_abs[1] - cfg.mat_abs[0]) * material
    p = cfg.mat_pha[0] + (cfg.mat_pha[1] - cfg.mat_pha[0]) * material
    amp = (thickness * a).clamp(0, 1)
    pha = (cfg.phase_max * thickness * p).clamp(0, cfg.phase_max)
    pha = pha * (thickness > 0).float()      # 沒有材料就沒有光程延遲
    return torch.stack([amp, pha], dim=-3)


def support_mask(cfg, device=None):
    """已知的寬鬆 support。

    相位恢復裡最強的約束 —— HIO 每次迭代都在用它。
    真實實驗可從自相關圖估出(自相關範圍恰為物體 support 的兩倍),
    是免費的資訊。

    探針照明(cfg.probe != "none")時,出射波只存在於探針範圍內,
    已知的 support 即探針圓盤(階段四協定 §3.2);圓盤必須完全落在方框內。
    """
    m = torch.zeros(cfg.canvas, cfg.canvas)
    s = (cfg.canvas - cfg.digit) // 2 - cfg.support_pad
    e = s + cfg.digit + 2 * cfg.support_pad
    m[max(s, 0):min(e, cfg.canvas), max(s, 0):min(e, cfg.canvas)] = 1.0
    if getattr(cfg, "probe", "none") != "none":
        p = probe_amplitude(cfg)
        if bool(((p > 0) & (m <= 0)).any()):
            raise ValueError(f"探針範圍超出方框 support(probe_r={cfg.probe_r})")
        m = (p > 0).float()
    return m.to(device) if device is not None else m


def probe_amplitude(cfg, device=None):
    """探針振幅 [H, W]。disk:平頂圓盤,圓心在畫布中心 ((N-1)/2),半徑 probe_r px。"""
    kind = getattr(cfg, "probe", "none")
    n = cfg.canvas
    if kind == "none":
        p = torch.ones(n, n)
    elif kind == "disk":
        ax = torch.arange(n, dtype=torch.float32)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        c = (n - 1) / 2.0
        p = ((yy - c) ** 2 + (xx - c) ** 2 <= float(cfg.probe_r) ** 2).float()
    else:
        raise ValueError(f"未知的 probe: {kind}")
    return p.to(device) if device is not None else p


def apply_probe(objs, cfg):
    """物體 [N, 2, H, W] -> 出射波 psi = P * O(同樣以振幅 / 相位表示)。

    平頂實數探針:圓盤內振幅與相位即物體本身,圓盤外為 0。
    probe = none 時原樣回傳(既有所有實驗的行為不變)。
    """
    if getattr(cfg, "probe", "none") == "none":
        return objs
    p = probe_amplitude(cfg, device=objs.device)
    return torch.stack([objs[:, 0] * p, objs[:, 1] * (p > 0).float()], dim=1)


def beamstop_mask(cfg, radius=None, device=None):
    """圓形 beamstop:擋住中心直射光,該區沒有任何量測資料。

    真實系統必須裝 —— 直射光比繞射訊號強數個數量級,會讓偵測器飽和,
    也逼你縮短曝光而收不到高角度的弱訊號。代價是中心低頻永久缺失。
    """
    r0 = cfg.beamstop_r if radius is None else radius
    n = cfg.canvas
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32),
                            torch.arange(n, dtype=torch.float32),
                            indexing="ij")
    c = n / 2.0
    r = torch.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    m = (r > r0).float()
    return m.to(device) if device is not None else m


# ==============================================================================
# 光通量校正
# ==============================================================================
def calibrate_flux(objects):
    """算出參考能量,讓「平均樣品」拿到 photons_per_pix 的劑量。

    這模擬實驗前的光通量校正。**不可以逐樣本正規化** ——
    材料多的樣品本來就散射比較多光子,總計數是「樣品有多少東西」的直接量測。
    抹掉它等於把秤歸零之後才秤重,並且會逼網路改用訓練分布的先驗去猜絕對尺度。

    校正一次後就固定;換樣品時光源不會跟著變,所以泛化測試也用同一個常數。
    """
    return float((objects[:, 0] ** 2).sum(dim=(-2, -1)).mean().item())


def forward_measure(obj, bs_mask, cfg, photons=None, add_poisson=None):
    """樣品 -> 偵測器上的光子計數。這是實驗裡唯一拿得到的東西。"""
    if cfg.ref_energy is None:
        raise RuntimeError("cfg.ref_energy 未設定,請先跑 calibrate_flux")
    photons = cfg.photons_per_pix if photons is None else photons
    poisson = cfg.add_poisson if add_poisson is None else add_poisson

    psi = torch.polar(obj[:, 0], obj[:, 1])
    # norm='ortho' 讓 Parseval 成立:sum|E|^2 == sum|psi|^2
    E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    I = E.abs() ** 2                       # <<< 繞射場的相位在此永久消失

    npix = I.shape[-1] * I.shape[-2]
    I = I * (photons * npix / cfg.ref_energy)   # 固定光通量,非逐樣本正規化

    if poisson:
        # 光是離散粒子,計數服從 Poisson。變異數 = 期望值,
        # 所以弱訊號區(高角度 = 高解析度)的 SNR 天生就差。
        I = torch.poisson(I.clamp_min(0))
    return I * bs_mask


# ==============================================================================
# 網路輸入
# ==============================================================================
_RAD_CACHE = {}


def _radial_index(size, device):
    key = (size, str(device))
    if key not in _RAD_CACHE:
        yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                                torch.arange(size, dtype=torch.float32),
                                indexing="ij")
        c = size / 2.0
        r = torch.sqrt((yy - c) ** 2 + (xx - c) ** 2)
        idx = r.round().long().clamp(max=size // 2).to(device)
        _RAD_CACHE[key] = (idx.reshape(-1), int(idx.max().item()) + 1)
    return _RAD_CACHE[key]


def radial_flatten(x):
    """減掉每一圈的平均、除以每一圈的標準差。

    繞射強度的徑向趨勢跨兩個數量級(實測 7720 -> 88.6),
    即使取 log 仍主宰整張圖的變化。但徑向平均本身不帶結構資訊 ——
    那只是功率譜包絡,網路從像素座標就能免費算出來。
    真正的結構藏在偏離徑向平均的微小起伏裡。
    """
    B, H, W = x.shape
    i, nbin = _radial_index(H, x.device)
    flat = x.reshape(B, -1)
    ones = torch.ones(H * W, device=x.device)
    cnt = torch.zeros(nbin, device=x.device).index_add_(0, i, ones).clamp_min(1)
    s1 = torch.zeros(B, nbin, device=x.device).index_add_(1, i, flat)
    s2 = torch.zeros(B, nbin, device=x.device).index_add_(1, i, flat ** 2)
    mean = s1 / cnt
    std = (s2 / cnt - mean ** 2).clamp_min(0).sqrt().clamp_min(1e-4)
    return ((flat - mean[:, i]) / std[:, i]).reshape(B, H, W)


def _norm_channels(counts, cfg):
    """回傳 (log 強度通道, 自相關通道),依 cfg.input_norm 決定正規化方式。"""
    logI = torch.log1p(counts)
    ac = torch.fft.ifft2(torch.fft.ifftshift(counts, dim=(-2, -1)),
                         norm="ortho").real
    ac = torch.fft.fftshift(ac, dim=(-2, -1))

    if cfg.input_norm == "global":
        if cfg.log_mean is None:
            raise RuntimeError("input_norm='global' 需要先跑 calibrate_input_norm")
        ch0 = (logI - cfg.log_mean) / cfg.log_std
        ch1 = ac / cfg.ac_scale
    elif cfg.input_norm == "per_sample":
        m = logI.mean(dim=(-2, -1), keepdim=True)
        s = logI.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        ch0 = (logI - m) / s
        ch1 = ac / ac.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(f"未知的 input_norm: {cfg.input_norm}")

    if cfg.input_transform == "radial_flatten":
        ch0 = radial_flatten(logI) * (counts > 0).float()
    elif cfg.input_transform != "none":
        raise ValueError(f"未知的 input_transform: {cfg.input_transform}")
    return ch0, ch1


def calibrate_input_norm(objects, bs_mask, cfg, photons=None):
    """算出全域的輸入正規化常數(與 ref_energy 一樣,只算一次然後固定)。"""
    counts = forward_measure(objects, bs_mask, cfg, photons)
    logI = torch.log1p(counts)
    ac = torch.fft.ifft2(torch.fft.ifftshift(counts, dim=(-2, -1)),
                         norm="ortho").real
    cfg.log_mean = float(logI.mean())
    cfg.log_std = float(logI.std().clamp_min(1e-6))
    cfg.ac_scale = float(ac.abs().amax(dim=(-2, -1)).mean().clamp_min(1e-12))
    return cfg.log_mean, cfg.log_std, cfg.ac_scale


def measurement_to_input(counts, bs_mask, cfg):
    """量測 -> 網路輸入。關鍵是**確定性**:同一樣品永遠對應同一輸入。

    ch0: log(1+counts)(或徑向壓平)。繞射圖動態範圍跨數個量級,
         不壓縮的話網路只看得到中心幾個像素。
    ch1: 自相關 = IFFT{I}。唯一能從純強度確定性算出的實空間量,
         其 support 恰為物體 support 的兩倍。
    ch2: beamstop 遮罩,明確告訴網路哪裡沒有資料。
    """
    ch0, ch1 = _norm_channels(counts, cfg)
    ch2 = bs_mask.expand_as(ch0)
    return torch.stack([ch0, ch1, ch2], dim=1)


# ==============================================================================
# Oracle 控制實驗
# ==============================================================================
def oracle_input(obj, counts, bs_mask, cfg):
    """把繞射場的**真實相位**交給網路,其餘完全不變。

    這是嚴格的「只多給相位」實驗:
      - 光子數、Poisson 雜訊、beamstop 全部跟正常路徑相同
      - 通道 0 是同一張 log 繞射圖
      - 只額外給 cos/sin(繞射場相位)

    所以它與正常路徑的唯一差別就是那個「偵測器記不下來的相位」。
    這樣才能乾淨地回答:相位遺失到底讓我們損失多少?

    通道:[log 強度, cos(相位), sin(相位), beamstop 遮罩] -> 4 channels
    """
    ch0, _ = _norm_channels(counts, cfg)

    psi = torch.polar(obj[:, 0], obj[:, 1])
    E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    ang = torch.angle(E)
    ch1 = torch.cos(ang) * bs_mask
    ch2 = torch.sin(ang) * bs_mask

    ch3 = bs_mask.expand_as(ch0)
    return torch.stack([ch0, ch1, ch2, ch3], dim=1)


def input_channels(cfg):
    return 4 if cfg.oracle_phase else 3


def build_input(obj, counts, bs_mask, cfg):
    """統一入口。訓練與評估都走這裡,確保兩邊表示法一致。"""
    if cfg.oracle_phase:
        return oracle_input(obj, counts, bs_mask, cfg)
    return measurement_to_input(counts, bs_mask, cfg)
