"""
深度学习模型模块 - 优化版
引入特征组注意力机制 (Feature Group Attention)
实现特征组合优化与场景适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class FeatureGroupAttention(nn.Module):
    """
    特征组注意力模块：实现组合优化
    将高维特征分为不同的逻辑组，动态计算每组的权重
    """
    def __init__(self, group_dims: Dict[str, int], embed_dim: int = 64):
        super(FeatureGroupAttention, self).__init__()
        self.groups = nn.ModuleDict()
        
        # 为每个特征组建立一个编码层
        for name, dim in group_dims.items():
            self.groups[name] = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.ReLU()
            )
        
        # 注意力计算层：根据各组编码后的特征生成权重
        num_groups = len(group_dims)
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim * num_groups, num_groups),
            nn.Softmax(dim=1)  # 保证各组权重之和为1
        )

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        group_embeddings = []
        for name, net in self.groups.items():
            group_embeddings.append(net(x_dict[name]))
        
        # 拼接所有组的嵌入向量
        combined = torch.cat(group_embeddings, dim=1) # [batch, embed_dim * num_groups]
        
        # 计算组权重 (场景适配的核心：不同样本生成的权重不同)
        weights = self.attention_net(combined) # [batch, num_groups]
        
        # 加权融合
        weighted_embeddings = []
        for i, (name, _) in enumerate(self.groups.items()):
            # 将权重作用于对应组的嵌入
            w = weights[:, i:i+1] 
            weighted_embeddings.append(group_embeddings[i] * w)
            
        final_embedding = torch.sum(torch.stack(weighted_embeddings, dim=2), dim=2)
        return final_embedding, weights

class OptimizedMalwareDetector(nn.Module):
    """
    基于注意力机制的组合优化恶意软件检测模型
    支持PE文件特征和图像特征的多模态输入
    """
    def __init__(self, num_classes: int = 9, use_image: bool = False):
        super(OptimizedMalwareDetector, self).__init__()
        self.use_image = use_image
        
        # 定义 PE 特征的逻辑分组
        self.group_dims = {
            'histogram': 256,
            'byte_entropy': 256,
            'strings': 34,  # 新的字符串特征
            'general': 1741  # 剩余特征
        }
        
        # 1. 组合优化层
        self.attention_module = FeatureGroupAttention(self.group_dims)
        
        # 2. 图像特征处理（如果使用）
        if use_image:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(64 * 56 * 56, 128),
                nn.ReLU()
            )
            # 融合特征维度
            fusion_dim = 64 + 128
        else:
            fusion_dim = 64
        
        # 3. 深度分类层 (场景适配推理)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor, image: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将原始向量拆分回逻辑组
        # 注意：这里的切片索引需要根据实际的特征提取顺序调整
        x_dict = {
            'histogram': x[:, 0:256],
            'byte_entropy': x[:, 256:512],
            'strings': x[:, 512:546],  # 字符串特征 (34维)
            'general': x[:, 546:]  # 剩余特征
        }
        
        # 组合优化与特征融合
        feat_embedding, attn_weights = self.attention_module(x_dict)
        
        # 如果使用图像特征，进行融合
        if self.use_image and image is not None:
            image_features = self.image_encoder(image)
            combined_features = torch.cat([feat_embedding, image_features], dim=1)
        else:
            combined_features = feat_embedding
        
        # 分类
        logits = self.classifier(combined_features)
        return logits, attn_weights


class OptimizedAPKDetector(nn.Module):
    """
    基于注意力机制的组合优化恶意软件检测模型
    """
    def __init__(self, num_classes: int = 2):
        super(OptimizedAPKDetector, self).__init__()
        
        # 定义 EMBER/APK 特征的逻辑分组 (基于你 jsonl 中的字段)
        # 2287 维分布：Histogram(256), ByteEntropy(256), Strings(34), Meta(1741)
        self.group_dims = {
            'histogram': 256,
            'byte_entropy': 256,
            'strings': 34,
            'general': 1741  # 剩余特征
        }
        
        # 1. 组合优化层
        self.attention_module = FeatureGroupAttention(self.group_dims)
        
        # 2. 深度分类层 (场景适配推理)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将原始 2287 维向量拆分回逻辑组
        # 注意：这里的切片索引需要根据你实际的 feature_extractor 输出顺序调整
        x_dict = {
            'histogram': x[:, 0:256],
            'byte_entropy': x[:, 256:512],
            'strings': x[:, 512:546],
            'general': x[:, 546:2287]
        }
        
        # 组合优化与特征融合
        feat_embedding, attn_weights = self.attention_module(x_dict)
        
        # 分类
        logits = self.classifier(feat_embedding)
        return logits, attn_weights

# 保持与你原有 app.py 的兼容性，保留基础 MLP 但建议在 app.py 中切换为 OptimizedAPKDetector
class EmberMLP(nn.Module):
    def __init__(self, input_dim=2381, num_classes=2):
        super(EmberMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# 你原有的 CNN 模型保持不变...
class MalwareDetectionCNN(nn.Module):
    def __init__(self, num_classes: int = 9, dropout_rate: float = 0.5):
        super(MalwareDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 112 * 112, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EfficientNetMalwareDetector(nn.Module):
    """
    基于EfficientNet的恶意软件检测模型
    适用于二进制文件转换的图像输入
    """
    def __init__(self, num_classes: int = 9):
        super(EfficientNetMalwareDetector, self).__init__()
        
        # 简化版EfficientNet结构
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # MBConv1
            self._mbconv_block(32, 16, 1),
            
            # MBConv6
            self._mbconv_block(16, 24, 2),
            self._mbconv_block(24, 24, 1),
            
            # MBConv6
            self._mbconv_block(24, 40, 2),
            self._mbconv_block(40, 40, 1),
            
            # MBConv6
            self._mbconv_block(40, 80, 2),
            self._mbconv_block(80, 80, 1),
            self._mbconv_block(80, 80, 1),
            
            # MBConv6
            self._mbconv_block(80, 112, 1),
            self._mbconv_block(112, 112, 1),
            self._mbconv_block(112, 112, 1),
            
            # MBConv6
            self._mbconv_block(112, 192, 2),
            self._mbconv_block(192, 192, 1),
            self._mbconv_block(192, 192, 1),
            self._mbconv_block(192, 192, 1),
            
            # MBConv6
            self._mbconv_block(192, 320, 1),
            
            # 最后一层
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )
    
    def _mbconv_block(self, in_channels, out_channels, stride):
        """Mobile Inverted Bottleneck Conv Block"""
        expand_ratio = 6
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # 扩展卷积
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ])
        
        # 深度可分离卷积
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        ])
        
        # 投影卷积
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ])
        
        # 短接连接
        if stride == 1 and in_channels == out_channels:
            return nn.Sequential(*layers)
        else:
            return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ViTMalwareDetector(nn.Module):
    """
    基于Vision Transformer的恶意软件检测模型
    适用于二进制文件转换的图像输入
    """
    def __init__(self, num_classes: int = 9, image_size: int = 224, patch_size: int = 16, dim: int = 768, depth: int = 12, heads: int = 12, mlp_dim: int = 3072):
        super(ViTMalwareDetector, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 1 * patch_size * patch_size  # 1 channel
        
        # 补丁嵌入
        self.patch_embedding = nn.Linear(self.patch_dim, dim)
        
        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # 分类标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=0.1),
            num_layers=depth
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: [batch, 1, 224, 224]
        batch_size = x.shape[0]
        
        # 分割补丁
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_dim)
        
        # 补丁嵌入
        x = self.patch_embedding(x)
        
        # 添加分类标记
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置嵌入
        x = x + self.pos_embedding
        
        # 转置以适应Transformer输入格式
        x = x.transpose(0, 1)  # [seq_len, batch, dim]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 取分类标记的输出
        x = x[0]
        
        # 分类
        x = self.classifier(x)
        return x