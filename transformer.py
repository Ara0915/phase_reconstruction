import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 資料前處理：傅立葉轉換與捨棄相位
# ==========================================
class PhaseRetrievalDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # 取得原始影像 (空間域)，大小為 [1, 28, 28]
        img, _ = self.mnist[idx] #MNIST 資料集回傳的格式是 (圖片, 標籤)
        
        # 執行 2D FFT (轉換至頻域)
        fft_img = torch.fft.fft2(img)
        fft_shifted = torch.fft.fftshift(fft_img) # 將低頻移至中心
        
        # 捨棄相位，僅提取振幅 (Magnitude)
        magnitude = torch.abs(fft_shifted)
        
        # 對數轉換以壓縮動態範圍 (幫助神經網路更容易學習)
        magnitude = torch.log(magnitude + 1e-8)
        
        # 將振幅正規化到大約 [-1, 1] 或 [0, 1] 之間 (此處簡化處理)，Z-score 正規化。它將這張頻譜圖的所有數值平移並縮放，使其平均值變為 0、標準差變為 1
        magnitude = (magnitude - magnitude.mean()) / magnitude.std()

        # 輸入為僅含振幅的頻譜，目標為原始空間域影像，移除第0維(頻道數)
        return magnitude.squeeze(0), img.squeeze(0)

# ==========================================
# 2. Transformer 模型架構
# ==========================================
class TransformerPhaseRetriever(nn.Module):
    def __init__(self, input_dim=28, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        # 將每一列 (28維) 投影到更高的嵌入維度
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # 位置編碼 (Positional Encoding)，對於序列模型非常重要
        self.pos_embedding = nn.Parameter(torch.randn(1, 28, embed_dim))
        
        # Transformer 編碼器層
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 輸出層：將特徵重新投影回 28 維的空間像素
        self.output_layer = nn.Linear(embed_dim, input_dim)
        
    def forward(self, x):
        # x 的維度: [Batch, 28, 28] -> 視為長度為28的序列，每個元素維度28
        x = self.embedding(x) # [Batch, 28, embed_dim]
        x = x + self.pos_embedding # 加上位置資訊
        
        # 通過 Transformer
        x = self.transformer(x) # [Batch, 28, embed_dim]
        
        # 重建回空間域的影像 [Batch, 28, 28]
        out = self.output_layer(x)
        return out

# ==========================================
# 3. 視覺化結果功能 (新增)
# ==========================================
def visualize_results(model, dataset, device, num_samples=5):
    """
    隨機挑選幾張圖片，畫出：原始圖片 vs 頻譜振幅(輸入) vs 重建圖片(輸出)
    """
    model.eval() # 切換到評估模式
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    
    # 設定標題 (只在第一列顯示)
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Magnitude (Input)")
    axes[0, 2].set_title("Reconstructed")

    with torch.no_grad(): # 推論時不計算梯度，節省記憶體
        for i in range(num_samples):
            # 依序拿取前幾筆資料
            magnitude, original = dataset[i]
            
            # 將輸入資料送入模型 (需要增加 Batch 維度)
            mag_input = magnitude.unsqueeze(0).to(device)
            
            # 模型預測
            reconstructed = model(mag_input).cpu().squeeze(0)
            
            # 畫出 1. 原始圖片
            axes[i, 0].imshow(original.numpy(), cmap='gray')
            axes[i, 0].axis('off')
            
            # 畫出 2. 頻譜振幅 (只有振幅，無相位)
            axes[i, 1].imshow(magnitude.numpy(), cmap='gray')
            axes[i, 1].axis('off')
            
            # 畫出 3. 模型重建的圖片 (限制數值範圍在 0~1 之間)
            recon_img = torch.clamp(reconstructed, 0, 1).numpy()
            axes[i, 2].imshow(recon_img, cmap='gray')
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. 執行與訓練邏輯 (已整合視覺化)
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 載入 MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    raw_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 封裝成我們的相位恢復資料集
    train_dataset = PhaseRetrievalDataset(raw_train_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、損失函數與優化器
    model = TransformerPhaseRetriever().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 訓練迴圈 (示範 5 個 Epoch)
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (magnitude, target_img) in enumerate(train_loader):
            magnitude = magnitude.to(device)
            target_img = target_img.to(device)

            optimizer.zero_grad()
            
            # 模型預測 (從振幅預測原始影像)
            reconstructed_img = model(magnitude)
            
            # 計算損失並反向傳播
            loss = criterion(reconstructed_img, target_img)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx} Loss: {loss.item():.6f}")
                
        print(f"--- Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.6f} ---")

    # 訓練結束後，呼叫畫圖功能 
    print("訓練完成！準備顯示結果圖...")
    visualize_results(model, train_dataset, device, num_samples=5)

# ==========================================
# 5. 執行腳本
# ==========================================
if __name__ == "__main__":
    train_model()