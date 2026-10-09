"""損失函式。

三項:
  1. 振幅 L1
  2. 相位 L1,**只在樣品內算** —— 真空區的相位沒有意義,硬要學是在學雜訊
  3. 資料一致性 —— 把輸出丟回前向模型,看算出的繞射圖跟量到的合不合。
     這是物理進入 loss 的地方,也是真實實驗(無 ground truth)時唯一算得出來的項。

選配:頻率加權的複數場誤差。未加權時由 Parseval 定理它等同實空間 MSE;
乘上 r^power 之後高頻誤差被放大,模型不能靠「把細節抹平」降低 loss。
"""
import torch
import torch.nn.functional as F


def radial_weight(cfg, device=None):
    n = cfg.canvas
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32),
                            torch.arange(n, dtype=torch.float32),
                            indexing="ij")
    c = n / 2.0
    r = torch.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    w = (r / r.max()) ** cfg.freq_power
    w = w / w.mean()
    return w.to(device) if device is not None else w


def _roll_batch(x, sy, sx):
    """逐樣本循環平移:out[b] = torch.roll(x[b], (sy[b], sx[b]), dims=(-2, -1))。"""
    b, h, w = x.shape
    ar_h = torch.arange(h, device=x.device)
    ar_w = torch.arange(w, device=x.device)
    rows = (ar_h[None, :] - sy[:, None]) % h          # roll(x, s)[i] = x[(i - s) % N]
    cols = (ar_w[None, :] - sx[:, None]) % w
    bi = torch.arange(b, device=x.device)[:, None, None]
    return x[bi, rows[:, :, None], cols[:, None, :]]


@torch.no_grad()
def align_target(pred, target, cfg):
    """把真值換成與網路輸出最接近的平凡等價版本(見 config.amb_invariant)。

    候選:
      原解                        (A(r),       φ(r))
      共軛翻轉 + 全域相位 φ_max    (A(-r),      φ_max − φ(-r))   仍落在 [0, φ_max]
    各自搭配 |sy|, |sx| <= amb_max_shift 的循環平移。

    兩候選的全域相位皆已固定、範數相同,且循環平移不改變範數,故
    「複數場 L2 距離最小」等價於「Re<pred, 候選> 最大」,可用 FFT 一次算完所有平移。
    翻轉以畫布中心為軸(i -> N-1-i),與嚴格的 i -> -i mod N 只差 1 px 平移,皆為等價解。

    回傳 (對齊後的真值 [N,2,H,W], 選翻轉的比例, 有平移的比例)。
    """
    n, _, h, w = target.shape
    p = torch.polar(pred[:, 0].float(), pred[:, 1].float())
    Fp = torch.fft.fft2(p)
    A, ph = target[:, 0], target[:, 1]
    cands = [(A, ph),
             (torch.flip(A, (-2, -1)), cfg.phase_max - torch.flip(ph, (-2, -1)))]

    # 平移視窗:有號平移 |s| <= amb_max_shift
    k = int(cfg.amb_max_shift)
    s_h = (torch.arange(h, device=target.device) + h // 2) % h - h // 2
    s_w = (torch.arange(w, device=target.device) + w // 2) % w - w // 2
    win = ((s_h.abs() <= k)[:, None] & (s_w.abs() <= k)[None, :]).reshape(-1)

    best_val = best_i = best_c = None
    for ci, (a, f) in enumerate(cands):
        c = torch.polar(a, f)
        # cc[s] = sum_r p(r) conj(c(r - s)):c 平移 s 後與 p 的內積
        cc = torch.fft.ifft2(Fp * torch.conj(torch.fft.fft2(c))).real.reshape(n, -1)
        cc = cc.masked_fill(~win[None, :], float("-inf"))
        val, idx = cc.max(dim=1)
        if best_val is None:
            best_val, best_i = val, idx
            best_c = torch.zeros(n, dtype=torch.long, device=target.device)
        else:
            better = val > best_val
            best_val = torch.where(better, val, best_val)
            best_i = torch.where(better, idx, best_i)
            best_c = torch.where(better, torch.full_like(best_c, ci), best_c)

    sy, sx = best_i // w, best_i % w
    a_sel = torch.where(best_c[:, None, None] == 1, cands[1][0], cands[0][0])
    f_sel = torch.where(best_c[:, None, None] == 1, cands[1][1], cands[0][1])
    out = torch.stack([_roll_batch(a_sel, sy, sx), _roll_batch(f_sel, sy, sx)], dim=1)
    moved = ((sy != 0) | (sx != 0)).float().mean()
    return out, float(best_c.float().mean()), float(moved)


def build_loss(cfg, device):
    wmap = radial_weight(cfg, device) if cfg.w_freq > 0 else None

    def loss_fn(pred, target, counts, bs_mask, photons=None):
        photons = cfg.photons_per_pix if photons is None else photons
        amb_twin = amb_shift = 0.0
        if cfg.amb_invariant:
            # 真值換成最接近網路輸出的等價解;以下所有項照原樣計算
            target, amb_twin, amb_shift = align_target(pred.detach(), target, cfg)
        amp_p, ph_p = pred[:, 0], pred[:, 1]
        amp_t, ph_t = target[:, 0], target[:, 1]

        # 振幅:樣品內與空白區分開計算,各自除以自己的像素數,
        # 使兩區權重由 cfg.w_amp_out 明確指定,而非由面積比例隱含決定,
        # 並與相位損失(原本即只在樣品內計算)的正規化方式一致。
        # 詳見 config.py 中 w_amp_out 的說明,含尚未確認的成因。
        sup = (amp_t > cfg.amp_floor).float()
        out = 1.0 - sup
        n_in = sup.sum().clamp_min(1.0)
        n_out = out.sum().clamp_min(1.0)
        d_amp = (amp_p - amp_t).abs()
        l_amp_in = (d_amp * sup).sum() / n_in
        l_amp_out = (d_amp * out).sum() / n_out
        l_amp = l_amp_in + cfg.w_amp_out * l_amp_out

        l_pha = ((ph_p - ph_t).abs() * sup).sum() / sup.sum().clamp_min(1.0)

        psi_p = torch.polar(amp_p, ph_p)
        E = torch.fft.fftshift(torch.fft.fft2(psi_p, norm="ortho"), dim=(-2, -1))
        npix = E.shape[-1] * E.shape[-2]
        I_p = (E.abs() ** 2) * (photons * npix / cfg.ref_energy)
        num = ((I_p.clamp_min(0).sqrt() - counts.clamp_min(0).sqrt()).abs()
               * bs_mask).sum()
        den = (counts.clamp_min(0).sqrt() * bs_mask).sum().clamp_min(1e-12)
        l_dc = num / den

        if wmap is not None:
            psi_t = torch.polar(amp_t, ph_t)
            D = torch.fft.fftshift(torch.fft.fft2(psi_p - psi_t, norm="ortho"),
                                   dim=(-2, -1))
            l_freq = ((D.abs() ** 2) * wmap).mean()
        else:
            l_freq = torch.zeros((), device=pred.device)

        total = (l_amp + cfg.w_phase * l_pha
                 + cfg.w_dc * l_dc + cfg.w_freq * l_freq)
        return total, dict(amp=l_amp.item(), amp_in=l_amp_in.item(),
                           amp_out=l_amp_out.item(), phase=l_pha.item(),
                           dc=l_dc.item(), freq=float(l_freq),
                           amb_twin=amb_twin, amb_shift=amb_shift)

    return loss_fn
