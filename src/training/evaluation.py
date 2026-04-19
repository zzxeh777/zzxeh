"""
模型评估模块 - 增强版
支持注意力权重分析与特征贡献度可视化
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

class ModelEvaluator:
    def __init__(self, model, device: str = 'cuda', class_names: List[str] = None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names or ["Benign", "Malicious"]
        # 特征组名称，对应 models.py 中的分组
        self.group_names = ['字节直方图', '字节熵', '字符串分布', '通用元数据']

    def evaluate(self, data_loader, is_optimized: bool = True) -> Dict:
        """
        增强评估函数：增加注意力权重收集
        """
        all_preds = []
        all_labels = []
        all_probs = []
        all_weights = [] # 新增：收集注意力权重

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                
                if is_optimized:
                    # 获取模型双输出
                    logits, weights = self.model(inputs)
                    all_weights.append(weights.cpu().numpy())
                else:
                    logits = self.model(inputs)
                
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(probs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        # 基础指标计算
        metrics = self._calculate_basic_metrics(all_labels, all_preds, all_probs)
        
        # 创新点评估：特征组贡献度分析
        if is_optimized and all_weights:
            metrics['attention_analysis'] = self._analyze_attention(all_weights, all_labels)
            
        return metrics

    def _analyze_attention(self, weights_list: List[np.ndarray], labels: List[int]) -> Dict:
        """
        分析不同类别（良性 vs 恶意）下，模型关注点的差异
        这是证明“场景适配”能力的关键数据
        """
        weights = np.concatenate(weights_list, axis=0)
        labels = np.array(labels)
        
        analysis = {}
        for i, name in enumerate(self.class_names):
            # 计算该类别下的平均注意力分布
            class_weights = weights[labels == i]
            if len(class_weights) > 0:
                avg_w = np.mean(class_weights, axis=0)
                analysis[name] = {self.group_names[j]: float(avg_w[j]) for j in range(len(self.group_names))}
        
        return analysis

    def plot_attention_heatmap(self, metrics: Dict, save_path: str):
        """
        可视化：生成特征组贡献度热力图
        这是论文里非常核心的实验图表
        """
        if 'attention_analysis' not in metrics:
            print("⚠️ 无注意力数据可供绘图")
            return

        data = metrics['attention_analysis']
        classes = list(data.keys())
        groups = self.group_names
        
        matrix = np.array([[data[c][g] for g in groups] for c in classes])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, annot=True, cmap='YlGnBu', xticklabels=groups, yticklabels=classes)
        plt.title("不同类型样本的特征组注意力权重分布 (场景适配分析)")
        plt.xlabel("特征逻辑组")
        plt.ylabel("样本类别")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _calculate_basic_metrics(self, y_true, y_pred, y_prob) -> Dict:
        # (保留原有的评估逻辑，但建议加上 AUC)
        y_prob = np.array(y_prob)
        auc = roc_auc_score(y_true, y_prob[:, 1]) if len(self.class_names) == 2 else 0
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }