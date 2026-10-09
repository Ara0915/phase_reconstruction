"""HIO（Hybrid Input-Output）相位回復迭代，用於精煉網路輸出。

動機（見交接簡報 §4.4）：跨分布能力僅在繞射相位已知時出現
（`v2_proc_oracle` 2.0/4 對 `v2_proc` 0.0/4），且資料多樣性推進至
50000 種相異物體仍無法帶來該能力。這指出瓶頸為**物理資訊**而非訓練資料。

HIO 每次迭代所強制的正是資料一致性與支撐約束 —— 而 R-factor 目前
高於雜訊底線約 20 倍（0.4359 對 0.020），顯示網路輸出在物理上仍不一致。

此路線未被 Oracle 實驗排除，因其增加的是**物理迭代**而非模型容量。

---

演算法（Fienup 1982）。每次迭代兩步：

1. **傅立葉約束**：把重建的繞射振幅換成量測值，保留自己算出的相位
2. **實空間約束**：support 內接受新值；support 外（違反約束處）施加負回饋

    g_{k+1}(r) = g'_k(r)                    r ∈ 滿足約束
                 g_k(r) - beta * g'_k(r)     r ∉ 滿足約束

負回饋項是 HIO 與單純交替投影（ER）的關鍵差異，
用於跳出局部極小（特別是 twin image 停滯）。

---

本專案的三項特化，與訓練時的物理設定嚴格一致：

- **beamstop 內無量測**：該區保留模型自己的估計，不可強制為零
  （曾實測：提高 w_dc 時模型放棄 beamstop 區的低頻，FRC 在
   f≈0.047 崩潰，恰為 beamstop 半徑 3 對應的 3/64）
- **相位非負**：φ ∈ [0, phase_max]，此約束消除共軛翻轉（twin image）
- **振幅有界**：A ∈ [0, 1]

Poisson 計數需先轉回振幅：measured_amp = sqrt(counts / scale)，
其中 scale 與 forward_measure 完全相同。
"""
import torch

from .physics import support_mask


def _measured_amp(counts, cfg, photons=None):
    """由光子計數還原繞射振幅，尺度與 forward_measure 一致。

    forward_measure 中 I = |E|^2 * (photons * npix / ref_energy)，
    故 |E| = sqrt(counts / scale)。此處必須用同一組常數，
    否則 HIO 的傅立葉約束會與訓練時的物理不一致。
    """
    photons = cfg.photons_per_pix if photons is None else photons
    npix = counts.shape[-1] * counts.shape[-2]
    scale = photons * npix / cfg.ref_energy
    return (counts.clamp_min(0) / scale).sqrt()


@torch.no_grad()
def random_init(counts, cfg, seed=0, device=None):
    """純 HIO 用的隨機初始解 —— 傳統方法的基準,不使用任何學習成分。

    做法：以量測到的繞射振幅配上隨機相位，反轉換回實空間，
    再套用 support 與物理約束。這是相位回復的標準起始方式。

    用途：對照「網路提供的初始解」值多少。若網路 + 200 次迭代
    即可達到純 HIO 需數千次才能到的品質，則網路初始化在
    reconstruction speed 軸上有實質貢獻，即使跨分布能力未改善。
    """
    dev = counts.device if device is None else device
    sup = support_mask(cfg, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    amp = _measured_amp(counts, cfg).to(dev)
    pha = (torch.rand(amp.shape, generator=g).to(dev) * 2 - 1) * torch.pi
    E = torch.polar(amp, pha)
    x = torch.fft.ifft2(torch.fft.ifftshift(E, dim=(-2, -1)), norm="ortho")
    return torch.stack([x.abs().clamp(0, 1) * sup,
                        torch.angle(x).clamp(0, cfg.phase_max) * sup], dim=1)


@torch.no_grad()
def hio(init_obj, counts, bs_mask, cfg, n_iter=50, beta=0.9,
        photons=None, return_trace=False):
    """以 HIO 精煉初始猜測。

    參數
    ----
    init_obj : [N, 2, H, W]，通道 0 為振幅、通道 1 為相位（網路輸出）
    counts   : [N, H, W]    量測到的光子計數
    bs_mask  : [H, W]       beamstop 遮罩，1 = 有量測
    n_iter   : 迭代次數。**必須掃描而非固定** —— HIO 可能破壞
               網路提供的良好初始解，且迭代過多會開始擬合 Poisson 雜訊
    beta     : 負回饋強度，文獻常用 0.9

    回傳
    ----
    [N, 2, H, W] 精煉後的物體；return_trace=True 時另回傳每次迭代的
    R-factor（不需 ground truth，可用於選擇停止點）
    """
    dev = init_obj.device
    sup = support_mask(cfg, device=dev)                    # [H, W]
    bs = bs_mask.to(dev)
    meas_amp = _measured_amp(counts, cfg, photons).to(dev)  # [N, H, W]

    # 目前的實空間估計（複數）
    g = torch.polar(init_obj[:, 0], init_obj[:, 1])
    trace = []

    for _ in range(n_iter):
        # ---- 傅立葉約束 ----
        E = torch.fft.fftshift(torch.fft.fft2(g, norm="ortho"), dim=(-2, -1))
        mag, pha = E.abs(), torch.angle(E)

        # beamstop 內沒有量測，保留自身估計；其餘換成量測振幅
        new_mag = torch.where(bs > 0, meas_amp, mag)
        E_proj = torch.polar(new_mag, pha)

        g_prime = torch.fft.ifft2(
            torch.fft.ifftshift(E_proj, dim=(-2, -1)), norm="ortho")

        # ---- 實空間約束 ----
        amp_p = g_prime.abs()
        pha_p = torch.angle(g_prime)

        # 違反約束處：support 外、相位為負、或振幅超出 [0,1]
        # （相位非負是消除 twin image 的物理約束，見模組說明）
        viol = (sup <= 0) | (pha_p < 0) | (pha_p > cfg.phase_max) | (amp_p > 1.0)

        amp_ok = amp_p.clamp(0, 1)
        pha_ok = pha_p.clamp(0, cfg.phase_max)
        g_ok = torch.polar(amp_ok, pha_ok)

        # HIO 的負回饋：違反處以 g - beta*g' 取代，而非直接投影
        g = torch.where(viol, g - beta * g_prime, g_ok)

        if return_trace:
            trace.append(_r_factor_amp(g, meas_amp, bs))

    amp = g.abs().clamp(0, 1) * sup
    pha = torch.angle(g).clamp(0, cfg.phase_max) * sup
    out = torch.stack([amp, pha], dim=1)
    return (out, torch.tensor(trace)) if return_trace else out


@torch.no_grad()
def _r_factor_amp(g, meas_amp, bs):
    """以繞射振幅計算的 R-factor，**不需 ground truth**。

    用途：HIO 迭代過多會開始擬合 Poisson 雜訊，需要停止準則，
    而真實情境下沒有 ground truth 可參考。R-factor 只比對
    「重建算回去的繞射圖」與「實際量到的」，故可作為客觀的停止依據。
    """
    E = torch.fft.fftshift(torch.fft.fft2(g, norm="ortho"), dim=(-2, -1))
    num = ((E.abs() - meas_amp).abs() * bs).sum()
    den = (meas_amp * bs).sum().clamp_min(1e-12)
    return float(num / den)


@torch.no_grad()
def er(init_obj, counts, bs_mask, cfg, n_iter=50, photons=None):
    """Error Reduction：HIO 的 beta=0 特例（純交替投影）。

    收斂較穩定但易停在局部極小，作為 HIO 的對照組。
    """
    dev = init_obj.device
    sup = support_mask(cfg, device=dev)
    bs = bs_mask.to(dev)
    meas_amp = _measured_amp(counts, cfg, photons).to(dev)
    g = torch.polar(init_obj[:, 0], init_obj[:, 1])

    for _ in range(n_iter):
        E = torch.fft.fftshift(torch.fft.fft2(g, norm="ortho"), dim=(-2, -1))
        E_proj = torch.polar(torch.where(bs > 0, meas_amp, E.abs()),
                             torch.angle(E))
        gp = torch.fft.ifft2(
            torch.fft.ifftshift(E_proj, dim=(-2, -1)), norm="ortho")
        amp = gp.abs().clamp(0, 1) * sup
        pha = torch.angle(gp).clamp(0, cfg.phase_max) * sup
        g = torch.polar(amp, pha)

    return torch.stack([g.abs().clamp(0, 1) * sup,
                        torch.angle(g).clamp(0, cfg.phase_max) * sup], dim=1)
