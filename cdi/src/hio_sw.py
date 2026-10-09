"""HIO 的擴充版:可換 support、shrinkwrap、ER 收尾(實驗設計 §二十三)。

**不取代 `hio.py`。** `hio.py` 是既有結果所用的版本,一行都不改;
本檔只為 §二十三 的量測新增兩種對照:

    S*  support 換成真實物體範圍(診斷用上限,實驗上拿不到)
    B   shrinkwrap(Marchesini et al., PRB 68, 140101, 2003)+ 最後 10% 改跑 ER

共用的一步與 `hio.py` 逐行相同(傅立葉約束、實空間約束、HIO 負回饋),
並以單元測試保證:關閉 shrinkwrap 與 ER 時,輸出與 `hio.hio()` 相同;
全程 ER 時,輸出與 `hio.er()` 相同。

shrinkwrap(每 sw_every 次更新一次 support):
    1. 取目前實空間估計(已套約束)的振幅,限於目前 support 內
    2. 以標準差 sigma 的高斯模糊(FFT 實作,與空間卷積等價,已測)
    3. 取「模糊後 >= thresh × 該張最大值」的像素
    4. 與原本的方框 support 取交集 —— 只縮不擴,用到的先驗資訊不多於簡單版
    5. sigma *= sigma_decay,下限 sigma_min
    若某張更新後為空(數值上幾乎不可能),保留原 support。

參數預設值即 §二十三 事先寫定的值:
    sw_every = 20、sigma 3 -> 1.5(每次 x0.99)取自該文;
    thresh = 0.04:以「由真值算出的 support 須包住物體 99.9% 以上」的規則,
    在與測試集不同 seed 的校準集上決定(該文預設 0.2 會切掉本專案分散物體的邊緣)。
"""
import math

import torch

from .hio import _measured_amp
from .physics import support_mask

SW_EVERY = 20
SIGMA0 = 3.0
SIGMA_MIN = 1.5
SIGMA_DECAY = 0.99
THRESH = 0.04
ER_FRAC = 0.10


def gaussian_blur(x, sigma):
    """實數影像 [N, H, W] 的高斯模糊(循環邊界),核的總和為 1。

    標準差 sigma(像素)的正規化高斯,其傅立葉轉換為 exp(-2 pi^2 sigma^2 f^2),
    f 以 cycles/pixel 計。
    """
    h, w = x.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device)[:, None]
    fx = torch.fft.fftfreq(w, device=x.device)[None, :]
    k = torch.exp(-2 * math.pi ** 2 * sigma ** 2 * (fy ** 2 + fx ** 2))
    return torch.fft.ifft2(torch.fft.fft2(x) * k).real


def shrinkwrap_support(amp, cur_sup, box, sigma, thresh=THRESH):
    """回傳新的 support [N, H, W](float 0/1)。amp 為目前估計的振幅 [N, H, W]。"""
    b = gaussian_blur(amp * cur_sup, sigma)
    new = ((b >= thresh * b.amax(dim=(-2, -1), keepdim=True)) & (box > 0)).float()
    empty = new.sum(dim=(-2, -1), keepdim=True) == 0
    return torch.where(empty, cur_sup, new)


def true_support(objs):
    """真實物體範圍(振幅 > 0),含位置。只供診斷用的 S*。"""
    return (objs[:, 0] > 0).float()


@torch.no_grad()
def hio_ext(init_obj, counts, bs_mask, cfg, n_iter, beta=0.9, support=None,
            shrinkwrap=False, er_frac=0.0, photons=None, return_support=False,
            sw_every=SW_EVERY, sigma0=SIGMA0, sigma_min=SIGMA_MIN,
            sigma_decay=SIGMA_DECAY, thresh=THRESH):
    """HIO(可選 shrinkwrap、ER 收尾、自訂 support)。

    參數
    ----
    support    : None -> 方框(= hio.py);或 [N, H, W] 逐張的 support(S* 用)
    shrinkwrap : True -> 依模組說明每 sw_every 次縮緊 support(與方框取交集)
    er_frac    : 最後 round(er_frac * n_iter) 次改跑 ER(純投影)
    """
    dev = init_obj.device
    box = support_mask(cfg, device=dev)                                 # [H, W]
    sup = (box.expand(init_obj.shape[0], *box.shape) if support is None
           else support.to(dev).float()).clone()                        # [N, H, W]
    bs = bs_mask.to(dev)
    meas_amp = _measured_amp(counts, cfg, photons).to(dev)
    n_er = int(round(er_frac * n_iter)) if n_iter >= 10 else 0
    n_hio = n_iter - n_er
    sigma = sigma0

    g = torch.polar(init_obj[:, 0], init_obj[:, 1])
    for k in range(n_iter):
        # ---- 傅立葉約束(與 hio.py 相同)----
        E = torch.fft.fftshift(torch.fft.fft2(g, norm="ortho"), dim=(-2, -1))
        mag, pha = E.abs(), torch.angle(E)
        new_mag = torch.where(bs > 0, meas_amp, mag)
        g_prime = torch.fft.ifft2(
            torch.fft.ifftshift(torch.polar(new_mag, pha), dim=(-2, -1)), norm="ortho")

        # ---- 實空間約束(與 hio.py 相同,support 可逐張)----
        amp_p = g_prime.abs()
        pha_p = torch.angle(g_prime)
        viol = (sup <= 0) | (pha_p < 0) | (pha_p > cfg.phase_max) | (amp_p > 1.0)
        g_ok = torch.polar(amp_p.clamp(0, 1), pha_p.clamp(0, cfg.phase_max))
        if k < n_hio:
            g = torch.where(viol, g - beta * g_prime, g_ok)             # HIO
        else:
            g = torch.where(sup > 0, g_ok, torch.zeros_like(g_ok))      # ER(= hio.er)

        # ---- shrinkwrap(最後一次迭代後不更新)----
        if shrinkwrap and (k + 1) % sw_every == 0 and (k + 1) < n_iter:
            sup = shrinkwrap_support(amp_p.clamp(0, 1), sup, box, sigma, thresh)
            sigma = max(sigma * sigma_decay, sigma_min)

    amp = g.abs().clamp(0, 1) * sup
    pha = torch.angle(g).clamp(0, cfg.phase_max) * sup
    out = torch.stack([amp, pha], dim=1)
    return (out, sup) if return_support else out
