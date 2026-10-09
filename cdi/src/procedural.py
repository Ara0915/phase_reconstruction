"""程序生成的樣品產生器。

目的不是「讓題目更難」,而是**讓捷徑失效**。

已確認的病因(見研究日誌 §6.2):網路在訓練分布上達到理論極限,
卻在未見過的分布上完全喪失優勢 —— 它學到的是
「辨識類別 -> 由記憶重繪」,而非通用反演。
MNIST 只有 10 類,這條捷徑一直存在。

設計要求:
  1. 每個樣本獨一無二 —— 沒有「類別」可以認
  2. 頻譜可控且真的含高頻 —— 不是 upsample 的帶限訊號
  3. 任意解析度 —— 擺脫寫死的 28x28
  4. 維持「幾何 x 材料」雙變數,且兩者共用邊界(物理上正確)
  5. 複雜度可調 —— 才能做「多樣性要多少才夠」的參數研究

三種原型:
  polygon  隨機凸多邊形疊加  -> 銳利直邊與角點,最乾淨的高頻來源
  lattice  隨機晶格 + 缺陷    -> 週期性,頻域離散峰,最接近材料/半導體結構
  bandpass 帶通濾波雜訊       -> 頻譜可精確指定,唯一能直接掃描高頻能量的類型
"""
import math

import torch

from .physics import build_object, center_pad


# ==============================================================================
# 共用工具
# ==============================================================================
def _grid(d, device):
    """回傳 [d, d] 的座標網格,值域 [-1, 1]。"""
    v = torch.linspace(-1, 1, d, device=device)
    yy, xx = torch.meshgrid(v, v, indexing="ij")
    return yy, xx


def _match_support(t, cfg, target):
    """把厚度圖依分位數裁切,使 support 佔畫布的比例達到 target。

    為什麼需要:support 佔比會同時污染「相對高頻能量」與「樣本相似度」
    兩個指標 —— 稀疏物體天生就有較高的相對高頻(孤立細結構在頻域展得開)
    與較低的樣本相關係數(兩張稀疏圖重疊少)。
    要跟 MNIST 做公平比較,必須先對齊這個變因。
    """
    n, d, _ = t.shape
    flat = t.reshape(n, -1).float()
    keep = target * cfg.canvas ** 2 / (d * d)      # 換算成框內的填充率

    # 分位數裁切後還會經過正規化與 amp_floor,實際 support 會比 keep 小。
    # 用幾步修正逼近目標,避免寫死一個經驗係數。
    for _ in range(6):
        k = min(max(keep, 0.02), 0.98)
        q = torch.quantile(flat, 1 - k, dim=1, keepdim=True)
        tt = (flat - q).clamp_min(0)
        tt = tt / tt.amax(dim=1, keepdim=True).clamp_min(1e-6)
        got = float((tt * cfg.mat_abs[1] > cfg.amp_floor).float().mean()
                    * d * d / cfg.canvas ** 2)
        if abs(got - target) < 0.1 * target or got < 1e-6:
            break
        keep *= target / max(got, 1e-6)
    return tt.reshape(n, d, d)


def _finalize(t, m, cfg, target_support=None, contrast_gamma=1.0):
    """把厚度與材料整理成標準形式並放進畫布。

    厚度逐張正規化到 [0,1](形狀本身才是資訊,絕對厚度不是),
    材料只在有材料的地方有意義。
    target_support 不為 None 時,先裁切到指定的 support 佔比。
    """
    if target_support is not None:
        t = _match_support(t, cfg, target_support)
    t = t / t.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    if contrast_gamma != 1.0:
        # gamma < 1 會把中間值往上推,提高 support 內的平均振幅。
        # 分位數裁切後大部分存活的值都貼近閾值(很暗),
        # 用它把對比度拉回接近 MNIST 那種「筆畫接近飽和」的分布。
        t = t.clamp_min(0) ** contrast_gamma
    m = m * (t > 0).float()
    return build_object(center_pad(t, cfg), center_pad(m, cfg), cfg)


# ==============================================================================
# 類型 A:隨機凸多邊形疊加
# ==============================================================================
def make_polygons(n, cfg, seed=0, n_shapes=(3, 8), n_edges=(3, 7),
                  radius=(0.15, 0.45), target_support=None, contrast_gamma=1.0,
                  device="cpu"):
    """凸多邊形 = 半平面的交集,因此可以完全向量化。

    每個多邊形隨機指派一種材料;重疊處厚度累加,材料取最後蓋上的那個 ——
    這讓幾何與材料**共用同一組邊界**,物理上正確
    (相對於 MNIST 版本:材料圖是另一張數字,邊界與厚度完全無關)。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = cfg.digit
    yy, xx = _grid(d, device)
    t = torch.zeros(n, d, d, device=device)
    m = torch.zeros(n, d, d, device=device)

    for i in range(n):
        k = int(torch.randint(n_shapes[0], n_shapes[1] + 1, (1,), generator=g))
        for _ in range(k):
            ne = int(torch.randint(n_edges[0], n_edges[1] + 1, (1,), generator=g))
            cy, cx = (torch.rand(2, generator=g) * 1.2 - 0.6).tolist()
            r0 = float(torch.rand(1, generator=g)) * (radius[1] - radius[0]) + radius[0]
            ang = torch.rand(ne, generator=g) * 2 * math.pi
            # 每個邊的支撐距離隨機,形狀才不會都是正多邊形
            rr = r0 * (0.55 + 0.45 * torch.rand(ne, generator=g))

            inside = torch.ones(d, d, dtype=torch.bool, device=device)
            for j in range(ne):
                a = float(ang[j])
                nx, ny = math.cos(a), math.sin(a)
                inside &= ((xx - cx) * nx + (yy - cy) * ny) <= float(rr[j])

            t[i] += inside.float()
            m[i] = torch.where(inside, float(torch.rand(1, generator=g)), m[i])
    return _finalize(t, m, cfg, target_support, contrast_gamma)


# ==============================================================================
# 類型 B:隨機晶格 + 缺陷
# ==============================================================================
def make_lattice(n, cfg, seed=0, spacing=(3.0, 7.0), atom_r=(0.6, 1.6),
                 defect_p=(0.0, 0.25), n_species=3, target_support=None,
                 contrast_gamma=1.0, device="cpu"):
    """在隨機晶格上放置原子,含隨機的晶格常數、方向、原子大小與缺陷率。

    週期性使高頻能量集中於離散的布拉格峰,是很好的解析度測試,
    也最接近材料科學與半導體樣品的結構。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = cfg.digit
    ys = torch.arange(d, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    t = torch.zeros(n, d, d, device=device)
    m = torch.zeros(n, d, d, device=device)

    for i in range(n):
        a = float(torch.rand(1, generator=g)) * (spacing[1] - spacing[0]) + spacing[0]
        th = float(torch.rand(1, generator=g)) * math.pi / 2
        sig = float(torch.rand(1, generator=g)) * (atom_r[1] - atom_r[0]) + atom_r[0]
        pdef = float(torch.rand(1, generator=g)) * (defect_p[1] - defect_p[0]) + defect_p[0]
        # 第二軸可以不正交,產生非立方晶格
        shear = float(torch.rand(1, generator=g)) * 0.4 - 0.2

        c = d / 2.0
        k = int(d / a) + 2
        pts, spec = [], []
        for iy in range(-k, k + 1):
            for ix in range(-k, k + 1):
                if float(torch.rand(1, generator=g)) < pdef:
                    continue
                u, v = ix * a, iy * a
                u = u + shear * v
                px = c + u * math.cos(th) - v * math.sin(th)
                py = c + u * math.sin(th) + v * math.cos(th)
                if -3 * sig <= px <= d + 3 * sig and -3 * sig <= py <= d + 3 * sig:
                    pts.append((py, px))
                    spec.append(float(torch.randint(0, n_species, (1,),
                                                    generator=g)) / max(n_species - 1, 1))
        if not pts:
            pts, spec = [(c, c)], [0.5]

        P = torch.tensor(pts, device=device)
        S = torch.tensor(spec, device=device)
        # [n_pts, d, d] 高斯原子
        dist2 = ((yy[None] - P[:, 0, None, None]) ** 2
                 + (xx[None] - P[:, 1, None, None]) ** 2)
        blobs = torch.exp(-dist2 / (2 * sig ** 2))
        t[i] = blobs.sum(0)
        m[i] = (blobs * S[:, None, None]).sum(0) / blobs.sum(0).clamp_min(1e-6)

    # 晶格填滿整個框,裁掉弱尾巴讓 support 有限
    t = torch.where(t < 0.15 * t.amax(dim=(-2, -1), keepdim=True),
                    torch.zeros_like(t), t)
    return _finalize(t, m, cfg, target_support, contrast_gamma)


# ==============================================================================
# 類型 C:帶通濾波雜訊
# ==============================================================================
def make_bandpass(n, cfg, seed=0, f_lo=(0.05, 0.20), f_hi=(0.25, 0.50),
                  fill=(0.25, 0.55), target_support=None, contrast_gamma=1.0,
                  device="cpu"):
    """白雜訊經隨機帶通濾波後閾值化。

    這是唯一能**精確指定頻譜內容**的類型:f_lo / f_hi 直接控制
    高頻能量落在哪個範圍,因此可以直接掃描「高頻能量 vs 重建品質」。
    也是檢查「有沒有不小心做出帶限訊號」的對照。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = cfg.digit
    fy = torch.fft.fftfreq(d).to(device)
    r = torch.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)

    noise = torch.randn(n, 2, d, d, generator=g).to(device)
    lo = torch.rand(n, generator=g).to(device) * (f_lo[1] - f_lo[0]) + f_lo[0]
    hi = torch.rand(n, generator=g).to(device) * (f_hi[1] - f_hi[0]) + f_hi[0]
    hi = torch.maximum(hi, lo + 0.05)
    fill_t = torch.rand(n, generator=g).to(device) * (fill[1] - fill[0]) + fill[0]

    band = ((r[None] >= lo[:, None, None]) & (r[None] <= hi[:, None, None])).float()
    out = []
    for c in range(2):
        F = torch.fft.fft2(noise[:, c]) * band
        x = torch.fft.ifft2(F).real
        x = (x - x.mean(dim=(-2, -1), keepdim=True))
        x = x / x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        out.append(x)

    # 依目標填充率取分位數當閾值
    flat = out[0].reshape(n, -1)
    q = torch.quantile(flat, 1 - fill_t.clamp(0.05, 0.9), dim=1)
    q = q.diagonal() if q.dim() == 2 else q
    t = (out[0] - q[:, None, None]).clamp_min(0)
    m = (out[1] * 0.5 + 0.5).clamp(0, 1)
    # 全零的樣本補一個中央方塊,避免除以零
    dead = t.amax(dim=(-2, -1)) < 1e-6
    if dead.any():
        s = d // 4
        t[dead, s:d - s, s:d - s] = 1.0
    return _finalize(t, m, cfg, target_support, contrast_gamma)


# ==============================================================================
# 統一入口
# ==============================================================================
GENERATORS = {
    "polygon": make_polygons,
    "lattice": make_lattice,
    "bandpass": make_bandpass,
}


def make_procedural(n, cfg, seed=0, kinds=("polygon", "lattice", "bandpass"),
                    weights=(1, 2, 2), target_support=None,
                    contrast_gamma=1.0, device="cpu"):
    """等比例混合各原型,回傳 [n, 2, canvas, canvas]。

    weights:各原型的混合比例(預設 1:2:2)。
      實測(target_support=0.05, n=384)的 PCA 有效維度:
        1:1:1 -> 165    1:2:2 -> 178    0:1:1 -> 196    1:1:3 -> 190
      polygon 的多樣性最低,但保留它是為了銳利直邊、角點與遮擋關係 ——
      這些是另外兩種原型給不了的結構特性。1:2:2 為折衷。

    target_support:指定 support 佔畫布的比例。做與其他資料集的對照時
      務必設成相同的值,否則指標會被稀疏度污染。
      注意此機制只能**減少** support(以分位數裁切),無法增加。
    """
    kinds = list(kinds)
    w = [1.0] * len(kinds) if weights is None else list(weights)[:len(kinds)]
    tot = sum(w)
    per = [int(n * x / tot) for x in w]
    per[-1] += n - sum(per)
    outs = [GENERATORS[k](p, cfg, seed=seed + 1000 * i,
                          target_support=target_support,
                          contrast_gamma=contrast_gamma, device=device)
            for i, (k, p) in enumerate(zip(kinds, per)) if p > 0]
    out = torch.cat(outs, 0)
    perm = torch.randperm(len(out), generator=torch.Generator().manual_seed(seed))
    return out[perm]
