"""U-Net,輸出層被三個物理約束綁住。"""
import torch
import torch.nn as nn

from .physics import input_channels, support_mask


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    """三個物理約束:

      1. 振幅 >= 0            沒有負的材料量
      2. 相位 >= 0            材料只會讓光變慢,延遲只有一個方向。
                              這一條同時打掉 twin image 歧異性 ——
                              共軛解的相位是負的,網路結構上就輸不出來。
      3. 只在已知 support 內   樣品是有限大小的實體,外面是真空
    """

    def __init__(self, cfg, cin=None):
        super().__init__()
        cin = input_channels(cfg) if cin is None else cin
        b, nd = cfg.base_channels, cfg.n_down
        self.phase_max = cfg.phase_max
        self.inc = DoubleConv(cin, b)
        self.downs = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 2 ** i, b * 2 ** (i + 1)))
            for i in range(nd)])
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(b * 2 ** (i + 1), b * 2 ** i, 2, stride=2)
            for i in reversed(range(nd))])
        self.convs = nn.ModuleList([
            DoubleConv(b * 2 ** (i + 1), b * 2 ** i) for i in reversed(range(nd))])
        self.out = nn.Conv2d(b, 2, 1)
        self.register_buffer("sup", support_mask(cfg))

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        y = skips[-1]
        for up, conv, skip in zip(self.ups, self.convs, reversed(skips[:-1])):
            y = conv(torch.cat([up(y), skip], 1))
        z = self.out(y)
        amp = torch.sigmoid(z[:, 0:1]) * self.sup
        pha = self.phase_max * torch.sigmoid(z[:, 1:2]) * self.sup
        return torch.cat([amp, pha], dim=1)


# ==============================================================================
# 架構 fft:內建反傅立葉轉換(實驗設計 §十九)
# ==============================================================================
class FFTUNet(nn.Module):
    """偵測器端小網路 -> 固定的反傅立葉轉換 -> 實空間 U-Net。

    動機(§十八):U-Net 的輸入在偵測器(傅立葉)空間、輸出在實空間,
    兩者之間是全域的反傅立葉轉換;卷積只看局部,實測連 oracle 都學不會。
    本架構把轉換交給程式,網路只做兩件局部的事:

      1. 偵測器端:對每個頻率估計相位(K 個假設)。
         振幅**直接取自量測** —— 由輸入通道 0 反推正規化得到 sqrt(counts / scale),
         與 forward_measure 的尺度完全一致(同 hio.py 的 _measured_amp)。
         beamstop 內沒有量測,振幅改由網路估計(同 HIO 的處理)。
      2. 固定轉換:psi_k = IFFT(A · e^{i θ_k}),與 forward_measure 互為逆運算。
      3. 實空間:原本的 U-Net(含三個輸出約束)把 K 個複數場整合成最終輸出。

    oracle 時,偵測器端只需學會「相位 = 輸入給的 cos / sin」;
    相位未知時,偵測器端要自己估相位 —— 同一架構,兩條路徑皆適用。
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.input_norm == "global", "arch=fft 需要 input_norm=global(才能反推量測振幅)"
        assert cfg.input_transform == "none", "arch=fft 需要 input_transform=none"
        assert cfg.randomize_photons is None, "arch=fft 目前假設固定劑量"
        for k in ("log_mean", "log_std", "ref_energy"):
            assert getattr(cfg, k) is not None, f"arch=fft 需要先校正 {k}"
        k, cin = int(cfg.fft_k), input_channels(cfg)
        self.k = k
        self.det = nn.Sequential(
            nn.Conv2d(cin, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 3 * k, 1))                      # 每個假設:cos 分量、sin 分量、beamstop 內振幅
        self.unet = UNet(cfg, cin=2 * k)
        npix = cfg.canvas * cfg.canvas
        self.register_buffer("log_mean", torch.tensor(float(cfg.log_mean)))
        self.register_buffer("log_std", torch.tensor(float(cfg.log_std)))
        self.register_buffer("scale", torch.tensor(
            float(cfg.photons_per_pix) * npix / float(cfg.ref_energy)))

    def measured_amp(self, x):
        """由輸入通道 0 反推量測振幅 |E|;通道 0 = (log1p(counts) − log_mean) / log_std。"""
        counts = torch.expm1(x[:, 0] * self.log_std + self.log_mean).clamp_min(0)
        return torch.sqrt(counts / self.scale)

    def field(self, x, a, b, c):
        """組出 K 個複數繞射場並做固定的反傅立葉轉換。a, b, c: [N, K, H, W]。"""
        bsm = x[:, -1:]                                   # 兩種輸入的最後一個通道皆為 beamstop 遮罩
        amp = torch.where(bsm > 0, self.measured_amp(x)[:, None], c)
        nrm = torch.sqrt(a * a + b * b + 1e-6)
        E = torch.complex(amp * a / nrm, amp * b / nrm)
        return torch.fft.ifft2(torch.fft.ifftshift(E, dim=(-2, -1)), norm="ortho")

    def forward(self, x):
        h = self.det(x)
        k = self.k
        psi = self.field(x, h[:, :k], h[:, k:2 * k],
                         nn.functional.softplus(h[:, 2 * k:]))
        return self.unet(torch.cat([psi.real, psi.imag], dim=1))


# ==============================================================================
# 架構 attn:U-Net + self-attention(實驗設計 §十九)
# ==============================================================================
class AttnBlock(nn.Module):
    """標準 transformer 區塊(pre-norm):attention + MLP,皆為殘差。

    位置編碼為可學習的絕對位置 —— 反傅立葉轉換的權重取決於**絕對**頻率位置,
    不能只靠相對位置。
    """

    def __init__(self, dim, heads, n_tokens):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2) + self.pos       # [B, HW, C]
        y = self.n1(t)
        t = t + self.attn(y, y, y, need_weights=False)[0]
        t = t + self.mlp(self.n2(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class AttnUNet(UNet):
    """U-Net,在編碼器最深兩層(canvas 64 時為 16x16、8x8)的輸出後各加一個 AttnBlock。

    與原 U-Net 共用所有卷積層的結構;唯一差別是這兩個區塊。
    在這兩層做而非全解析度,是因為 attention 的計算量隨像素數平方成長
    (64x64 = 4096 個位置,16x16 = 256 個)。
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        b, nd, n = cfg.base_channels, cfg.n_down, cfg.canvas
        self.attn_at = [i for i in range(nd) if i >= nd - 2]
        self.blocks = nn.ModuleDict({
            str(i): AttnBlock(b * 2 ** (i + 1), cfg.attn_heads, (n // 2 ** (i + 1)) ** 2)
            for i in self.attn_at})

    def forward(self, x):
        skips = [self.inc(x)]
        for i, d in enumerate(self.downs):
            y = d(skips[-1])
            if str(i) in self.blocks:
                y = self.blocks[str(i)](y)
            skips.append(y)
        y = skips[-1]
        for up, conv, skip in zip(self.ups, self.convs, reversed(skips[:-1])):
            y = conv(torch.cat([up(y), skip], 1))
        z = self.out(y)
        amp = torch.sigmoid(z[:, 0:1]) * self.sup
        pha = self.phase_max * torch.sigmoid(z[:, 1:2]) * self.sup
        return torch.cat([amp, pha], dim=1)


def build_model(cfg):
    arch = getattr(cfg, "arch", "unet")
    if arch == "unet":
        return UNet(cfg)
    if arch == "fft":
        return FFTUNet(cfg)
    if arch == "attn":
        return AttnUNet(cfg)
    raise ValueError(f"未知的 arch: {arch}")
