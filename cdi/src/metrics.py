"""評估指標。每個數字都必須配一條參考線,否則沒有意義。

教訓:FRC = 2.03 px 看起來像完美重建,但一個什麼都不學的假模型也拿 2.03。
所以這裡的 FRC 一律裁切到物體區域,並同時回報平庸基準與曲線下面積。
"""
import numpy as np
import torch

from .physics import build_input, forward_measure


# ------------------------------------------------------------------ 基本指標
def psnr(pred, target):
    mse = ((pred - target) ** 2).mean(dim=(-2, -1)).clamp_min(1e-12)
    return float((10 * torch.log10(1.0 / mse)).mean())


def phase_rmse(pred, target, sup):
    """只在樣品內算,並先移除全域相位偏移 ——
    psi * exp(i*theta) 的繞射強度完全一樣,直接比絕對值會錯怪模型。"""
    d = (pred - target) * sup
    n = sup.sum(dim=(-2, -1)).clamp_min(1.0)
    off = d.sum(dim=(-2, -1)) / n
    d = d - off[:, None, None] * sup
    return float(torch.sqrt((d ** 2).sum(dim=(-2, -1)) / n).mean())


def frc(pred_c, target_c, n_bins=None):
    A = torch.fft.fftshift(torch.fft.fft2(pred_c, norm="ortho"), dim=(-2, -1))
    B = torch.fft.fftshift(torch.fft.fft2(target_c, norm="ortho"), dim=(-2, -1))
    H, W = A.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2).to(A.device)
    rmax = min(H, W) // 2
    n_bins = n_bins or rmax
    edges = torch.linspace(0, rmax, n_bins + 1, device=A.device)
    fs, vs = [], []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum() < 2:
            continue
        a, b = A[..., m], B[..., m]
        num = (a * b.conj()).sum(-1).abs()
        den = torch.sqrt((a.abs() ** 2).sum(-1)
                         * (b.abs() ** 2).sum(-1)).clamp_min(1e-12)
        vs.append(float((num / den).mean()))
        fs.append(float((edges[i] + edges[i + 1]) / 2 / min(H, W)))
    return np.array(fs), np.array(vs)


def frc_cropped(pred, target, cfg):
    """只在物體區域算。畫布 81% 是 padding,重建與答案在那裡都接近 0,
    會讓傅立葉係數天生相關、FRC 被嚴重高估。"""
    c = cfg.crop
    s = (cfg.canvas - c) // 2
    pc = torch.polar(pred[:, 0], pred[:, 1])[:, s:s + c, s:s + c]
    tc = torch.polar(target[:, 0], target[:, 1])[:, s:s + c, s:s + c]
    return frc(pc, tc)


def frc_at(fs, vs, frac=0.5):
    i = int(np.argmin(np.abs(np.asarray(fs) - frac * fs[-1])))
    return float(vs[i])


def frc_resolution(fs, vs, threshold=0.5):
    """FRC 曲線跌破門檻處的解析度(pixel),以線性插值取得連續值。

    為什麼要插值:FRC 只有約 15 個頻段,若直接回傳「跌破的那一格」,
    解析度就只能落在 15 個離散值上(21.33 / 12.80 / 9.14 / 7.11 / 5.82 ...),
    相鄰兩格在 7-9 px 區間相差 22-29%。模型要進步超過兩成才會跳格,
    否則不同設定會顯示完全相同的數字(實測 os64/os96/os128 三者
    皆為 7.1111±0.0000,並非表現相同,而是刻度太粗)。

    另外改為取「最後一次由上往下穿越門檻」的位置:
    只看第一次跌破會受曲線抖動影響,曾出現「更模糊卻解析度更好」的
    矛盾排序(模糊 1 次 -> 2.78 px,模糊 2 次 -> 2.06 px)。
    """
    fs, vs = np.asarray(fs, dtype=float), np.asarray(vs, dtype=float)
    cross = np.where((vs[:-1] >= threshold) & (vs[1:] < threshold))[0]
    if len(cross) == 0:
        # 從未跌破 -> 解析度優於最高頻段(回傳 Nyquist 極限)
        return float(1.0 / fs[-1]) if vs[-1] >= threshold else float(1.0 / fs[0])
    i = int(cross[-1])          # 最後一次穿越,避免曲線抖動造成誤判
    v0, v1 = vs[i], vs[i + 1]
    t = (v0 - threshold) / max(v0 - v1, 1e-12)
    f_cross = fs[i] + t * (fs[i + 1] - fs[i])
    return float(1.0 / f_cross) if f_cross > 0 else float("inf")


def frc_crossover_resolution(fs, v_model, v_trivial, frac=0.1):
    """模型「還贏得過平庸基準」的最細尺度,回傳解析度(pixel)。

    為什麼需要這個:原本的 frc_resolution 用固定門檻 0.5,
    但門檻是否適用取決於平庸基準曲線的形狀,而形狀隨資料集而變:

        MNIST 風格 平庸曲線: 0.69 0.60 0.55 0.50 0.39 0.29 0.18 ...(會掉)
        procedural 平庸曲線: 0.79 0.67 0.59 0.57 0.59 0.64 0.70 ...(平的)

    MNIST 的樣本彼此相似,平均圖是個有結構的形狀,曲線起伏大會掉下去;
    procedural 每張都不同,平均起來是一團均勻模糊,跟任何樣本都
    「低但穩定地相似」,全程維持在 0.5 之上、從不跌破 ——
    於是固定門檻對它完全失效(實測回報 2.065 px,並非真有此解析度)。

    本指標改問:**模型的優勢在多細的尺度之前還存在?**
    以 adv(f) = 模型 - 平庸,取 adv 最後一次跌破 (frac x 峰值) 的頻率。
    由於基準線被減掉,其形狀差異不再影響結果,故可跨資料集比較。

    回傳 inf 表示模型在任何尺度上都沒贏過平庸基準。
    """
    fs = np.asarray(fs, dtype=float)
    adv = np.asarray(v_model, dtype=float) - np.asarray(v_trivial, dtype=float)
    peak = adv.max()
    if peak <= 0:
        return float("inf")
    thr = frac * peak
    above = np.where(adv >= thr)[0]
    i = int(above[-1])
    if i >= len(fs) - 1:
        return float(1.0 / fs[-1])          # 到最高頻仍有優勢
    a0, a1 = adv[i], adv[i + 1]
    t = (a0 - thr) / max(a0 - a1, 1e-12)
    f_cross = fs[i] + t * (fs[i + 1] - fs[i])
    return float(1.0 / f_cross) if f_cross > 0 else float("inf")


def frc_auc(fs, vs):
    """曲線下平均值。比單點取樣穩健 ——
    平庸基準的 FRC 曲線是非單調的,half-Nyquist 剛好落在隆起處。"""
    fs, vs = np.asarray(fs), np.asarray(vs)
    return float(np.trapezoid(vs, fs) / (fs[-1] - fs[0]))


def r_factor(pred, counts, bs_mask, cfg, photons=None):
    """晶體學借來的資料一致性指標。**不需要 ground truth** ——
    真實實驗沒有人知道正確答案,這是唯一還算得出來的品質指標。"""
    photons = cfg.photons_per_pix if photons is None else photons
    psi = torch.polar(pred[:, 0], pred[:, 1])
    E = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
    npix = E.shape[-1] * E.shape[-2]
    I = (E.abs() ** 2) * (photons * npix / cfg.ref_energy)
    num = ((I.sqrt() - counts.clamp_min(0).sqrt()).abs() * bs_mask).sum()
    den = (counts.clamp_min(0).sqrt() * bs_mask).sum().clamp_min(1e-12)
    return float(num / den)


def r_factor_radial(pred, counts, bs_mask, cfg, photons=None, n_bins=12):
    """把 R-factor 按空間頻率拆開 —— **不需要 ground truth 的解析度曲線**。

    這是 FRC 的無真值版本。FRC 需要正確答案(或兩次獨立重建)才能算,
    但真實實驗兩者都沒有;而 R-factor 只比對「重建產生的繞射圖」與
    「實際量到的繞射圖」,因此在真實資料上照樣算得出來。

    判讀方式跟 FRC 相同:
      - 低頻環的 R 小 -> 粗結構重建得好
      - R 隨頻率上升 -> 細節開始失守
      - **一定要同時畫雜訊底線那條曲線** —— 它是 Poisson 統計決定的下限,
        兩條線開始分離的頻率,就是重建失去可信度的地方。

    實測特性(合成資料校準):R-factor 遠比 PSNR 嚴格,
    物體域 PSNR 需達約 48 dB,整體 R 才會落在雜訊底線的 3 倍以內。
    所以 R ≈ 0.3 不代表「很差」,而是「相當於 PSNR 約 30 dB」。
    """
    photons = cfg.photons_per_pix if photons is None else photons
    scale = photons * cfg.canvas ** 2 / cfg.ref_energy

    def to_I(o):
        E = torch.fft.fftshift(torch.fft.fft2(torch.polar(o[:, 0], o[:, 1]),
                                              norm="ortho"), dim=(-2, -1))
        return (E.abs() ** 2) * scale

    I_p = to_I(pred)
    H = cfg.canvas
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    r = torch.sqrt((yy - H / 2) ** 2 + (xx - H / 2) ** 2).to(I_p.device)
    edges = torch.linspace(0, H // 2, n_bins + 1)

    fs, rs, cnt = [], [], []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1]) & (bs_mask > 0)
        if m.sum() < 2:
            continue
        num = (I_p[:, m].clamp_min(0).sqrt()
               - counts[:, m].clamp_min(0).sqrt()).abs().sum()
        den = counts[:, m].clamp_min(0).sqrt().sum().clamp_min(1e-12)
        fs.append(float((edges[i] + edges[i + 1]) / 2 / H))
        rs.append(float(num / den))
        cnt.append(float(counts[:, m].mean()))
    return np.array(fs), np.array(rs), np.array(cnt)


def recover_material(obj, cfg, eps=1e-6):
    """由 (振幅, 相位) 反解材料分布 m。

    物體模型為 A = t·a(m)、φ = PHASE_MAX·t·p(m),其中
        a(m) = a0 + (a1-a0)·m
        p(m) = p0 + (p1-p0)·m
    因此 φ/A = PHASE_MAX · p(m)/a(m),**與厚度 t 無關**,只取決於材料。
    令 r = φ/(A·PHASE_MAX),反解得

        m = (p0 - r·a0) / (r·(a1-a0) - (p1-p0))

    (已驗證:代入真值反解的最大誤差為 1.2e-07。)

    為什麼需要這個指標:先前以「相對於抄振幅基準的進步幅度」比較不同
    訓練資料,但該基準的難度取決於資料集本身的振幅-相位相關性 ——
    實測 MNIST 為 0.783、procedural 僅 0.118(差 6.6 倍),
    procedural 的基準天生較差,使相對進步虛高、不可跨資料集比較。

    材料反演不依賴任何基準線,直接問「有沒有解出材料」,
    故可公平比較。物理上這也正是相位成像的價值所在 ——
    輕元素幾乎不吸收、在振幅上看不見,只能靠相位分辨。
    """
    a0, a1 = cfg.mat_abs
    p0, p1 = cfg.mat_pha
    amp, pha = obj[:, 0], obj[:, 1]
    r = pha / (amp.clamp_min(eps) * cfg.phase_max)
    denom = r * (a1 - a0) - (p1 - p0)
    m = (p0 - r * a0) / torch.where(denom.abs() < eps,
                                    torch.full_like(denom, eps), denom)
    return m.clamp(0, 1)


def material_error(pred, target, cfg):
    """材料反演的平均絕對誤差,只在樣品內計算。

    真空區沒有材料可言;且 A→0 時 φ/A 會發散,必須排除。
    回傳 (模型誤差, 平庸基準誤差):平庸基準為「一律猜資料集平均材料」。

    **必須先移除全域相位偏移**,理由與 phase_rmse 相同:
    ψ·exp(iθ) 的繞射強度完全相同,全域相位在物理上不可觀測,
    模型沒有理由、也沒有依據把它定出來。

    未做此校正時的後果(實測):整體相位偏移 0.20 rad 這種物理上無害的
    情形,phase_rmse 僅 0.0152(幾乎無感),材料 MAE 卻達 0.1390。
    這曾使 src_proc 的材料 MAE 被誤判為 0.4164(劣於平庸基準),
    而相同相位 RMSE 的對照組實際只有 0.1098 —— 差距 3.8 倍全來自此。
    """
    sup = (target[:, 0] > cfg.amp_floor).float()
    n = sup.sum().clamp_min(1)

    # 逐樣本移除全域相位偏移,與 phase_rmse 的處理一致
    d = (pred[:, 1] - target[:, 1]) * sup
    ns = sup.sum(dim=(-2, -1)).clamp_min(1.0)
    off = (d.sum(dim=(-2, -1)) / ns)[:, None, None]
    pred = torch.stack([pred[:, 0],
                        (pred[:, 1] - off * sup).clamp(0, cfg.phase_max)], 1)

    m_true = recover_material(target, cfg)
    m_pred = recover_material(pred, cfg)
    err = float(((m_pred - m_true).abs() * sup).sum() / n)
    m_bar = (m_true * sup).sum() / n
    err_triv = float(((m_bar - m_true).abs() * sup).sum() / n)
    return err, err_triv


# ------------------------------------------------------------------ 參考線
def _amp_linear_phase(amp, pha, sup):
    """由振幅線性預測相位的最小平方解 ——
    「完全不解相位、只把重建好的振幅換算過去」能拿到的最好成績。
    模型必須贏過它,才代表真的在解獨立的相位資訊。"""
    x, y, w = (amp * sup).flatten(), (pha * sup).flatten(), sup.flatten()
    n = w.sum().clamp_min(1)
    mx, my = (x * w).sum() / n, (y * w).sum() / n
    vx = ((x - mx) ** 2 * w).sum() / n
    cov = ((x - mx) * (y - my) * w).sum() / n
    return ((cov / vx.clamp_min(1e-12)) * (amp - mx) + my) * sup


@torch.no_grad()
def evaluate(model, objects, bs_mask, cfg, photons=None, pred_override=None):
    """回傳一個純量 dict(可直接 json.dump)加上曲線。

    pred_override:直接指定重建結果,跳過模型前向。
    用於評估 HIO 精煉後的輸出 —— 所有指標與參考線的計算方式完全相同,
    確保精煉前後可直接比較。
    """
    device = next(model.parameters()).device
    model.eval()
    obj = objects.to(device)
    counts = forward_measure(obj, bs_mask, cfg, photons)
    pred = (model(build_input(obj, counts, bs_mask, cfg))
            if pred_override is None else pred_override.to(device))

    p, t = pred.cpu(), obj.cpu()
    sup = (t[:, 0] > cfg.amp_floor).float()

    # 平庸預測器只能使用**資料集層級**的資訊,不得觸及單一樣本的 ground truth。
    #
    # 先前的版本寫成 const * sup,而 sup 是逐樣本的 support 遮罩 ——
    # 等於免費把每個樣本的正確支撐形狀交給平庸預測器。FRC 算的是複數場
    # A·exp(iφ) 的相關性,支撐形狀會透過相位通道進入,於是基準線被灌高。
    #
    # 影響且不對稱:procedural 每張樣本形狀都不同,知道正確 support 極有價值,
    # AUC 被灌 +0.276;MNIST 樣本彼此相似,只被灌 +0.023(差 12 倍)。
    # 這使得 procedural 上的平庸曲線變成一條不會跌破 0.5 的平線,
    # 連帶讓 src_proc / os* 的 FRC gain 全部虛假地變成負值。
    triv_a = t[:, 0].mean(0, keepdim=True).expand_as(t[:, 0])
    mean_sup = sup.mean(0, keepdim=True).expand_as(sup)
    const = (t[:, 1] * sup).sum() / sup.sum().clamp_min(1)
    triv = torch.stack([triv_a, const * mean_sup], 1)
    copy_p = _amp_linear_phase(p[:, 0], t[:, 1], sup)

    fm, vm = frc_cropped(p, t, cfg)
    ft, vt = frc_cropped(triv, t, cfg)

    mat_err, mat_err_triv = material_error(p, t, cfg)

    # 無真值的解析度曲線:模型 vs 雜訊底線
    fr, rr, ph = r_factor_radial(pred, counts, bs_mask, cfg, photons)
    _, rr0, _ = r_factor_radial(obj, counts, bs_mask, cfg, photons)

    return dict(
        amp_psnr=psnr(p[:, 0], t[:, 0]),
        amp_psnr_trivial=psnr(triv_a, t[:, 0]),
        phase_rmse=phase_rmse(p[:, 1], t[:, 1], sup),
        phase_rmse_trivial=phase_rmse(triv[:, 1], t[:, 1], sup),
        phase_rmse_copy_amp=phase_rmse(copy_p, t[:, 1], sup),
        frc_half=frc_at(fm, vm),
        frc_half_trivial=frc_at(ft, vt),
        frc_res=frc_resolution(fm, vm),
        frc_res_trivial=frc_resolution(ft, vt),
        frc_auc=frc_auc(fm, vm),
        frc_auc_trivial=frc_auc(ft, vt),
        frc_gain=frc_auc(fm, vm) - frc_auc(ft, vt),
        material_mae=mat_err,
        material_mae_trivial=mat_err_triv,
        r_factor=r_factor(pred, counts, bs_mask, cfg, photons),
        r_factor_noise_floor=r_factor(obj, counts, bs_mask, cfg, photons),
        _curves=dict(freq=fm.tolist(), model=vm.tolist(), trivial=vt.tolist()),
        _r_radial=dict(freq=fr.tolist(), model=rr.tolist(), floor=rr0.tolist(),
                       photons=ph.tolist()),
    )
