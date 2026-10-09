#!/usr/bin/env python
"""在**登入節點**先把資料集抓好。

計算節點通常沒有對外網路,download=True 會讓所有 job 在第一行卡死,
而且錯誤訊息是連線逾時,很難一眼看懂。

用法(在登入節點,不需要 GPU):
    python prefetch_data.py --data-root /work/elviss0915/data
"""
import argparse
from pathlib import Path
from torchvision import datasets

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="/work/elviss0915/data")
a = ap.parse_args()

Path(a.data_root).mkdir(parents=True, exist_ok=True)
for cls in (datasets.MNIST, datasets.FashionMNIST):
    for train in (True, False):
        d = cls(root=a.data_root, train=train, download=True)
        print(f"{cls.__name__} train={train}: {len(d.data)} 張 -> {a.data_root}")
print("\n完成。之後 job 用 download=False 讀這個目錄即可。")
