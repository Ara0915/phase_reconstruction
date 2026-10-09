"""資料集。

注意:訓練刻意**不使用 DataLoader**。這個模型只有約 190 萬參數、輸入 64x64,
瓶頸不在算力而在資料搬運 —— DataLoader worker、PIL 轉換、CPU->GPU 拷貝的
固定開銷比實際運算還貴。MNIST 只有 47 MB,整份丟進 GPU 記憶體即可。
"""
import torch
from torchvision import datasets

from .physics import apply_probe, build_object, center_pad
from .procedural import make_procedural


def load_raw(cfg, train=True, fashion=False):
    """讀取 MNIST / Fashion-MNIST 的原始 uint8 tensor。

    download=False —— HPC 計算節點通常沒有對外網路。
    請先在登入節點跑 prefetch_data.py 把資料抓好。
    """
    ds_cls = datasets.FashionMNIST if fashion else datasets.MNIST
    try:
        ds = ds_cls(root=cfg.data_root, train=train, download=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"在 {cfg.data_root} 找不到資料集。計算節點沒有對外網路,"
            f"請先在登入節點執行:python prefetch_data.py --data-root {cfg.data_root}"
        ) from e
    return ds.data


class SampleDataset(torch.utils.data.Dataset):
    """MNIST 的兩張圖分別擔任幾何(厚度)與材料(成分)。

    跟「兩張無關的圖當實部虛部」不同:兩張圖描述的是**同一個樣品**的兩個面向,
    共用同一個 support,合成方式有物理依據。
    """

    def __init__(self, cfg, raw, seed=None):
        self.cfg = cfg
        self.raw = raw
        self.n = len(raw)
        g = torch.Generator()
        g.manual_seed(cfg.test_seed if seed is None else seed)
        self.pair_idx = torch.randperm(self.n, generator=g)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        t = center_pad(self.raw[i].float() / 255.0, self.cfg)
        m = center_pad(self.raw[self.pair_idx[i]].float() / 255.0, self.cfg)
        return build_object(t, m, self.cfg)


def stack_dataset(ds, n=None):
    n = len(ds) if n is None else min(n, len(ds))
    return torch.stack([ds[i] for i in range(n)])


def build_pool(cfg, n, seed=0, sources=None):
    """依 cfg.train_sources 組出訓練用的物體池 [N, 2, canvas, canvas]。

    每個來源分到 n // len(sources) 個樣本。
    程序生成較慢(112x112 約 73 ms/樣本),故一次生成後常駐,不即時算。
    """
    sources = list(sources or cfg.train_sources)

    # n_unique:只生成這麼多相異物體,之後重複填滿至 n。
    # 如此梯度步數與資料總量不變,唯一變因是「看過幾種不同結構」。
    n_gen = n if cfg.n_unique is None else min(cfg.n_unique, n)

    per = [n_gen // len(sources)] * len(sources)
    per[-1] += n_gen - sum(per)
    ts = cfg.match_support_to
    out = []
    for i, (src, k) in enumerate(zip(sources, per)):
        sd = seed + 977 * i
        if src == "mnist":
            out.append(stack_dataset(
                SampleDataset(cfg, load_raw(cfg, train=True)[:k], seed=sd), k))
        elif src == "fashion":
            out.append(make_fashion(cfg, k))
        elif src == "shapes":
            out.append(make_shapes(cfg, k, seed=sd))
        elif src == "texture":
            out.append(make_texture(cfg, k, seed=sd))
        elif src == "procedural":
            out.append(make_procedural(
                k, cfg, seed=sd, kinds=tuple(cfg.proc_kinds),
                weights=tuple(cfg.proc_weights), target_support=ts,
                contrast_gamma=cfg.proc_contrast_gamma))
        else:
            raise ValueError(f"未知的 train_source: {src}")
    pool = torch.cat(out, 0)
    g = torch.Generator().manual_seed(seed)
    pool = pool[torch.randperm(len(pool), generator=g)]

    if len(pool) < n:
        # 重複填滿至 n,使各設定的梯度步數與資料總量一致
        reps = -(-n // len(pool))          # 無條件進位
        pool = pool.repeat(reps, 1, 1, 1)[:n]
        pool = pool[torch.randperm(len(pool), generator=g)]
    # 探針照明:出射波 psi = P * O 取代物體(probe = none 時不變)。
    # 在此套用,ref_energy 與輸入正規化常數便以 psi 校正(階段四協定 §2.1)。
    return apply_probe(pool, cfg)


def count_unique(pool, chunk=2000):
    """實際相異樣本數。

    程序生成的隨機參數偶爾碰撞,故名目 n_unique 與實際值可能略有出入
    (實測 n_unique=None、20000 張時實際為 19980,差 0.1%)。
    報告多樣性時應採用實測值。

    以雜湊比對,避免 torch.unique 在大張量上的記憶體開銷。
    """
    seen = set()
    for i in range(0, len(pool), chunk):
        blk = pool[i:i + chunk].reshape(len(pool[i:i + chunk]), -1)
        for row in blk:
            seen.add(hash(row.numpy().tobytes()))
    return len(seen)


class PoolSampler:
    """整份物體池常駐 GPU,訓練時直接切 index。

    不走 DataLoader:此模型約 190 萬參數、輸入 64x64,
    瓶頸在資料搬運而非算力,worker 與拷貝的固定開銷比運算還貴。
    記憶體:20000 x 2 x 64 x 64 x 4 B = 655 MB,H200 綽綽有餘。
    """

    def __init__(self, cfg, pool, device):
        self.cfg = cfg
        self.pool = pool.to(device)
        self.n = len(self.pool)
        self.device = device

    def batch(self, idx):
        return self.pool[idx]

    def epoch_indices(self, batch_size, generator=None):
        order = torch.randperm(self.n, device=self.device)
        for i in range(0, self.n - batch_size + 1, batch_size):
            yield order[i:i + batch_size]


# ==============================================================================
# 泛化測試用的樣品(模型沒看過的分布)
# ==============================================================================
def make_shapes(cfg, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = cfg.digit
    t = torch.zeros(n, d, d)
    m = torch.zeros(n, d, d)
    for i in range(n):
        for _ in range(4):
            h, w = torch.randint(4, 12, (2,), generator=g)
            y = torch.randint(0, d - int(h), (1,), generator=g).item()
            x = torch.randint(0, d - int(w), (1,), generator=g).item()
            t[i, y:y+int(h), x:x+int(w)] = torch.rand(1, generator=g).item()*.8+.2
            m[i, y:y+int(h), x:x+int(w)] = torch.rand(1, generator=g).item()
    return build_object(center_pad(t, cfg), center_pad(m, cfg), cfg)


def make_texture(cfg, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = cfg.digit
    t = torch.rand(n, d, d, generator=g)
    m = torch.rand(n, d, d, generator=g)
    return build_object(center_pad(t, cfg), center_pad(m, cfg), cfg)


def make_fashion(cfg, n):
    raw = load_raw(cfg, train=False, fashion=True)
    g = torch.Generator().manual_seed(cfg.test_seed)
    i = torch.randperm(len(raw), generator=g)[:n]
    j = torch.randperm(len(raw), generator=g)[:n]
    return build_object(center_pad(raw[i].float() / 255.0, cfg),
                        center_pad(raw[j].float() / 255.0, cfg), cfg)


def generalization_suites(cfg, n):
    """回傳 {名稱: [N,2,H,W] tensor}。

    測試集固定用 TEST_SEED,與訓練池不同 seed,確保沒有重疊。
    procedural 那組用固定 seed 生成,對 MNIST 訓練的模型是全新分布;
    對 procedural 訓練的模型則是同分布但不同樣本 —— 兩者都需要。

    探針照明時,每個測試集都套上同一個探針(階段四協定 §2.2:評估端必須跟著訓練端一起改)。
    """
    suites = {
        "mnist_test":     stack_dataset(
            SampleDataset(cfg, load_raw(cfg, train=False)), n),
        "fashion_mnist":  make_fashion(cfg, n),
        "random_shapes":  make_shapes(cfg, n),
        "random_texture": make_texture(cfg, n),
        "procedural":     make_procedural(
            n, cfg, seed=cfg.test_seed, kinds=tuple(cfg.proc_kinds),
            weights=tuple(cfg.proc_weights),
            target_support=cfg.match_support_to,
            contrast_gamma=cfg.proc_contrast_gamma),
    }
    return {k: apply_probe(v, cfg) for k, v in suites.items()}
