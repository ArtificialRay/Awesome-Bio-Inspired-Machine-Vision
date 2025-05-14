import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# === Siamese 相似度网络（输出 0~1） ===
class SiameseNet(nn.Module):
    def __init__(self, input_dim=128):
        
        )
        self.output_layer = nn.Sequential(
            
        )

    def forward(self, x1, x2):
        out1 = 
        out2 = 
        diff =
        sim = 
        return sim


class PairDataset(Dataset):
    def __init__(self, pairs, labels):
       

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]


# === 加载检测数据 ===
def load_detections(detection_array, vis_threshold=0.0):
    id_to_feats = {}
    

def build_pairs(id_to_feats):
    
    return np.array(pairs), np.array(labels)


def main(vis_threshold=0.3, epochs=50):
    

        # 验证阶段
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = model(x1, x2)
                val_loss += criterion(outputs, y).item()
                preds = (outputs > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"[{epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "similarity_model/best_model.pth")
            print("📌 Best model saved!")

    print("✅ 完成训练，模型已保存")


if __name__ == '__main__':
    main(0.001, 80)
