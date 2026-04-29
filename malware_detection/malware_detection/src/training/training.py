"""
模型训练模块 - 优化版
支持基于注意力机制的组合优化模型 (处理多输出)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
from tqdm import tqdm
import json

# ================= 1. EMBER 数据集类 =================
class EmberDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, use_mmap: bool = False):
        """
        Args:
            features: 特征数组（可以是内存映射）
            labels: 标签数组
            use_mmap: 是否使用内存映射模式（大数据时需要）
        """
        self.features = features
        self.labels = labels
        self.use_mmap = use_mmap or hasattr(features, 'filename')  # 检测是否是 mmap

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        if self.use_mmap:
            # 从内存映射按需读取并转换
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            # 原有方式
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ================= 2. 增强型训练器 (核心修改点) =================
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        is_optimized: bool = True # 新增：标识是否使用带注意力的优化模型
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.is_optimized = is_optimized
        self.history = {'train_loss': [], 'val_acc': [], 'attn_avg': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # --- 适配修改点 ---
            if self.is_optimized:
                # 接收双输出：预测值和注意力权重
                outputs, attn_weights = self.model(inputs)
            else:
                outputs = self.model(inputs)
            # -----------------

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, np.ndarray]:
        self.model.eval()
        correct = 0
        total = 0
        all_weights = [] # 用于分析场景适配情况

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.is_optimized:
                    outputs, attn_weights = self.model(inputs)
                    all_weights.append(attn_weights.cpu().numpy())
                else:
                    outputs = self.model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_weights = np.mean(np.concatenate(all_weights), axis=0) if all_weights else np.array([])
        
        return accuracy, avg_weights

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_dir: str
    ):
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc, avg_attn = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc)
            self.history['attn_avg'].append(avg_attn.tolist())

            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            # 场景适配体现：打印当前轮次模型最关注的特征组
            if self.is_optimized:
                group_names = ['Histogram', 'Entropy', 'Strings', 'Meta']
                top_group = group_names[np.argmax(avg_attn)]
                print(f"🔍 场景适配分析: 当前模型主要依赖 [{top_group}] 特征进行决策")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'attn_history': avg_attn
                }, os.path.join(save_dir, 'best_model.pth'))

        # 保存训练历史
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)
            
        return self.history