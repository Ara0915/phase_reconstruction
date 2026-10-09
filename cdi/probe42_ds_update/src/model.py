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


# ==============================================================================
# 架構 unroll:把 HIO 展開成網路(實驗設計 §二十一)
# ==============================================================================
class RefineNet(nn.Module):
    """小 U-Net,無輸出約束;最後一層初始化為 0(未訓練時輸出恆為 0)。"""

    def __init__(self, cin, cout, base, n_down):
        super().__init__()
        self.inc = DoubleConv(cin, base)
        self.downs = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2 ** i, base * 2 ** (i + 1)))
            for i in range(n_down)])
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(base * 2 ** (i + 1), base * 2 ** i, 2, stride=2)
            for i in reversed(range(n_down))])
        self.convs = nn.ModuleList([
            DoubleConv(base * 2 ** (i + 1), base * 2 ** i) for i in reversed(range(n_down))])
        self.out = nn.Conv2d(base, cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        y = skips[-1]
        for up, conv, skip in zip(self.ups, self.convs, reversed(skips[:-1])):
            y = conv(torch.cat([up(y), skip], 1))
        return self.out(y)


class ConvRefine(nn.Module):
    """輕量修正網路:n_layers 層 3x3 卷積(全解析度,無下採樣);最後一層初始化為 0。"""

    def __init__(self, cin, cout, width, n_layers):
        super().__init__()
        layers, c = [], cin
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.ReLU(True)]
            c = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(c, cout, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.body(x))


def make_refine(kind, cin, cout, base):
    """修正網路的工廠(階段四協定 §3.5 的候選)。"""
    if kind == "unet":
        return RefineNet(cin, cout, base, 3)
    if kind == "c16":
        return ConvRefine(cin, cout, 16, 4)
    if kind == "c8":
        return ConvRefine(cin, cout, 8, 3)
    raise ValueError(f"未知的 unroll_refine: {kind}")


class UnrollNet(FFTUNet):
    """把 HIO 展開成 T 輪的網路;整體仍是單次前向的獨立網路,不需外接 HIO。

      第 0 輪  偵測器端小網路估相位(同 fft 架構,K = 1)
               g0 = IFFT(A · e^{iθ}) + R_0(...)                       學習的修正
      第 t 輪  G  = FFT(g_{t-1})                                       固定
               E' = 量測振幅 A · G / |G|(beamstop 內保留 G)           固定:傅立葉約束
               g' = IFFT(E')                                           固定
               h  = HIO(g_{t-1}, g')                                   固定:同 hio.py 的實空間更新
               g_t = h + R_t(g', g_{t-1}, h)                           學習的修正(殘差)
      輸出     小卷積頭 -> 振幅 / 相位(與 U-Net 相同的三個輸出約束)

    R_t 的最後一層初始化為 0:未訓練時 g_T 恰為「網路起點 + T 次 HIO」,訓練只能在此之上改進。
    FFT 慣例與 forward_measure 一致:E = fftshift(fft2(ψ, ortho))。
    """

    def __init__(self, cfg):
        cfg_k1 = type(cfg).from_dict({**cfg.to_dict(), "fft_k": 1})
        super().__init__(cfg_k1)
        del self.unet                                     # 不用 fft 架構的大 U-Net
        self.t = int(cfg.unroll_t)
        self.beta = float(cfg.unroll_beta)
        self.phase_max = cfg.phase_max
        b = int(cfg.unroll_base)
        kind = getattr(cfg, "unroll_refine", "unet")
        self.every = int(getattr(cfg, "unroll_every", 1))
        if self.every < 1:
            raise ValueError(f"unroll_every 須 >= 1(得到 {self.every})")
        # 做修正的輪次:第 every, 2·every, ... 輪(every = 1 時每輪,與原版相同)
        self.fix_rounds = [t for t in range(self.t) if (t + 1) % self.every == 0]
        self.refine = nn.ModuleList(
            [make_refine(kind, 2, 2, b)] +                                   # 第 0 輪:輸入 g0
            [make_refine(kind, 6, 2, b) for _ in self.fix_rounds])          # 修正輪:g', g_{t-1}, h
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 2, 1))
        self.register_buffer("sup", support_mask(cfg))
        # 分段反傳(協定 §8.5):0 = 不分段;K > 0 時訓練用 forward_segments,每 K 輪切斷梯度
        self.trunc = int(getattr(cfg, "unroll_trunc", 0) or 0)
        if self.trunc < 0 or (self.trunc > 0 and self.t % self.trunc != 0):
            raise ValueError(f"unroll_trunc 須為 0 或整除 unroll_t(得到 {self.trunc}、T = {self.t})")

    @staticmethod
    def _c2r(z):
        return torch.stack([z.real, z.imag], dim=1)

    @staticmethod
    def _r2c(x):
        return torch.complex(x[:, 0], x[:, 1])

    def fourier_step(self, g, A, bsm):
        """傅立葉約束:把振幅換成量測值、保留自己的相位;beamstop 內保留自身估計。"""
        G = torch.fft.fftshift(torch.fft.fft2(g, norm="ortho"), dim=(-2, -1))
        mag = torch.sqrt(G.real ** 2 + G.imag ** 2 + 1e-12)
        E = torch.where(bsm > 0, G * (A / mag), G)
        return torch.fft.ifft2(torch.fft.ifftshift(E, dim=(-2, -1)), norm="ortho")

    def hio_step(self, g_prev, g_new):
        """與 hio.py 相同的實空間更新:違反約束處施加負回饋 g - β g',其餘接受並截斷至約束內。"""
        amp_p, pha_p = g_new.abs(), torch.angle(g_new)
        viol = (self.sup <= 0) | (pha_p < 0) | (pha_p > self.phase_max) | (amp_p > 1.0)
        g_ok = torch.polar(amp_p.clamp(0, 1), pha_p.clamp(0, self.phase_max))
        return torch.where(viol, g_prev - self.beta * g_new, g_ok)

    def rounds(self, g, A, bsm, seg=0):
        """跑 T 輪。seg = 0:回傳最後的 g(原行為)。
        seg = K > 0:每 K 輪以輸出頭讀出一次並切斷梯度,回傳各段輸出的 list(最後一個即第 T 輪)。"""
        k, outs = 0, []
        for t in range(self.t):
            g_new = self.fourier_step(g, A, bsm)
            h = self.hio_step(g, g_new)
            if (t + 1) % self.every == 0:
                k += 1
                feat = torch.cat([self._c2r(g_new), self._c2r(g), self._c2r(h)], dim=1)
                g = h + self._r2c(self.refine[k](feat))
            else:
                g = h                                   # 只跑物理步驟
            if seg and (t + 1) % seg == 0:
                outs.append(self.readout(g))
                g = g.detach()                          # 段界:梯度不再往前一段傳(數值不變)
        return outs if seg else g

    def readout(self, g):
        """輸出頭:複數場 -> 振幅 / 相位(與 U-Net 相同的三個輸出約束)。"""
        z = self.head(torch.cat([self._c2r(g), g.abs()[:, None]], dim=1))
        amp = torch.sigmoid(z[:, 0:1]) * self.sup
        pha = self.phase_max * torch.sigmoid(z[:, 1:2]) * self.sup
        return torch.cat([amp, pha], dim=1)

    def start(self, x):
        bsm = x[:, -1]
        A = self.measured_amp(x)
        h0 = self.det(x)
        g0 = self.field(x, h0[:, 0:1], h0[:, 1:2], nn.functional.softplus(h0[:, 2:3]))[:, 0]
        g0 = g0 + self._r2c(self.refine[0](self._c2r(g0)))
        return g0, A, bsm

    def forward(self, x):
        g0, A, bsm = self.start(x)
        return self.readout(self.rounds(g0, A, bsm))

    def forward_segments(self, x):
        """訓練用(unroll_trunc > 0):回傳 T / K 段的輸出;最後一段與 forward(x) 數值相同。"""
        assert self.trunc > 0, "forward_segments 需要 unroll_trunc > 0"
        g0, A, bsm = self.start(x)
        return self.rounds(g0, A, bsm, seg=self.trunc)


def build_model(cfg):
    # 國網上的 DoubleConv 不支援 dilation;過去 dilation 設定曾被靜默忽略(實驗設計 §19.4)。
    if getattr(cfg, "dilation", 1) != 1:
        raise ValueError(f"本版 model.py 不支援 dilation={cfg.dilation}(設定會被忽略),請改回 1")
    arch = getattr(cfg, "arch", "unet")
    if arch == "unet":
        return UNet(cfg)
    if arch == "fft":
        return FFTUNet(cfg)
    if arch == "attn":
        return AttnUNet(cfg)
    if arch == "unroll":
        return UnrollNet(cfg)
    raise ValueError(f"未知的 arch: {arch}")
