"""
创新点 1: 多异构深度学习模型集成融合
实现 MLP + CNN1D + Transformer 异构集成
利用不同网络对二进制特征、序列特征、统计特征的差异化提取能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class EmberTransformer(nn.Module):
    """
    MBER 特征的轻量 Transformer 模型
    将2381维特征划分为多个patch，用Transformer捕捉长距离依赖关系

    设计思路:
    - 将特征序列划分为多个patch (每个patch约50维)
    - 使用轻量Transformer编码器 (2层, 64维)
    - 捕捉特征之间的全局关系
    """
    def __init__(self, input_dim: int = 2381, patch_size: int = 50,
                 embed_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, num_classes: int = 2):
        super(EmberTransformer, self).__init__()

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.num_patches = math.ceil(input_dim / patch_size)

        # Patch embedding: 将每个patch映射到embed_dim维度
        self.patch_embed = nn.Linear(patch_size, embed_dim)

        # 位置编码 (可学习)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

        # 分类token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer编码器 (轻量版)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  # FFN维度
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, num_classes)
        )

        # 参数量统计
        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EmberTransformer: 总参数={total:,}, 可训练={trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 2381] 输入特征
        Returns:
            logits: [batch, num_classes] 分类输出
        """
        batch_size = x.shape[0]

        # 将特征划分为patches
        # [batch, 2381] -> [batch, num_patches, patch_size]
        # 需要padding到完整的patch数
        pad_size = self.num_patches * self.patch_size - self.input_dim
        if pad_size > 0:
            x = F.pad(x, (0, pad_size))

        x = x.view(batch_size, self.num_patches, self.patch_size)

        # Patch embedding
        x = self.patch_embed(x)  # [batch, num_patches, embed_dim]

        # 添加cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_patches+1, embed_dim]

        # 添加位置编码
        x = x + self.pos_embedding

        # Transformer编码
        x = self.transformer(x)

        # 取cls token的输出进行分类
        cls_output = x[:, 0]  # [batch, embed_dim]

        # 分类
        logits = self.classifier(cls_output)
        return logits


class MultiHeadFusionLayer(nn.Module):
    """
    多头融合层
    对多个分支的预测概率进行注意力加权融合

    支持三种融合策略:
    1. probability_attention: 对预测概率加权
    2. feature_attention: 对特征embedding加权
    3. gating: 门控机制融合
    """
    def __init__(self, num_branches: int = 3, fusion_type: str = 'probability_attention',
                 embed_dim: int = 64, num_classes: int = 2):
        super(MultiHeadFusionLayer, self).__init__()

        self.fusion_type = fusion_type
        self.num_branches = num_branches
        self.num_classes = num_classes

        if fusion_type == 'probability_attention':
            # 对概率向量进行注意力加权
            self.prob_attention = nn.Sequential(
                nn.Linear(num_classes * num_branches, num_branches),
                nn.Softmax(dim=1)
            )

        elif fusion_type == 'feature_attention':
            # 对特征embedding进行注意力加权 (需要各分支输出embedding)
            self.feature_attention = nn.Sequential(
                nn.Linear(embed_dim * num_branches, num_branches),
                nn.Softmax(dim=1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(embed_dim * 2, num_classes)
            )

        elif fusion_type == 'gating':
            # 门控融合: 学习每个分支的重要性
            self.gate_net = nn.Sequential(
                nn.Linear(num_classes * num_branches, num_branches),
                nn.Sigmoid()  # 输出0-1的门控值
            )

        elif fusion_type == 'meta_attention':
            # 元注意力: 用一个小的attention网络学习融合权重
            self.meta_attention = nn.Sequential(
                nn.Linear(num_classes * num_branches, 32),
                nn.ReLU(),
                nn.Linear(32, num_branches),
                nn.Softmax(dim=1)
            )

    def forward(self, branch_probs: torch.Tensor, branch_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            branch_probs: [batch, num_branches, num_classes] 各分支的预测概率
            branch_features: [batch, num_branches, embed_dim] 各分支的特征embedding (可选)
        Returns:
            fused_probs: [batch, num_classes] 融合后的预测概率
        """
        if self.fusion_type == 'probability_attention':
            # Flatten概率向量
            flat_probs = branch_probs.view(branch_probs.shape[0], -1)  # [batch, num_branches * num_classes]
            weights = self.prob_attention(flat_probs)  # [batch, num_branches]

            # 加权融合概率
            weights = weights.unsqueeze(-1)  # [batch, num_branches, 1]
            fused_probs = torch.sum(branch_probs * weights, dim=1)  # [batch, num_classes]

        elif self.fusion_type == 'feature_attention':
            if branch_features is None:
                raise ValueError("feature_attention fusion需要branch_features")
            flat_features = branch_features.view(branch_features.shape[0], -1)
            weights = self.feature_attention(flat_features).unsqueeze(-1)
            fused_features = torch.sum(branch_features * weights, dim=1)
            fused_logits = self.classifier(fused_features)
            fused_probs = F.softmax(fused_logits, dim=1)

        elif self.fusion_type == 'gating':
            flat_probs = branch_probs.view(branch_probs.shape[0], -1)
            gates = self.gate_net(flat_probs)  # [batch, num_branches]
            gates = gates.unsqueeze(-1)
            fused_probs = torch.sum(branch_probs * gates, dim=1)

        elif self.fusion_type == 'meta_attention':
            flat_probs = branch_probs.view(branch_probs.shape[0], -1)
            weights = self.meta_attention(flat_probs).unsqueeze(-1)
            fused_probs = torch.sum(branch_probs * weights, dim=1)

        return fused_probs


class HeterogeneousEnsembleDetector(nn.Module):
    """
    多异构深度学习模型集成融合

    架构:
    ├── MLP分支: 全连接网络 → 捕捉统计特征
    ├── CNN1D分支: 卷积网络 → 捕捉局部模式特征
    ├── Transformer分支: 自注意力 → 捕捉长距离依赖特征
    └── 多头融合层: Attention加权 → 最终分类

    创新点:
    1. 异构网络结构，充分利用不同网络的特征提取能力
    2. 注意力融合机制，动态调整各分支权重
    3. 轻量化设计，总参数量控制在合理范围
    """
    def __init__(self, input_dim: int = 2381, num_classes: int = 2,
                 embed_dim: int = 64, fusion_type: str = 'probability_attention'):
        super(HeterogeneousEnsembleDetector, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        print("=" * 60)
        print("构建异构集成模型")
        print("=" * 60)

        # ============ 分支1: MLP (统计特征提取) ============
        print("\n[分支1] MLP - 统计特征提取")
        self.mlp_branch = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim)  # 输出embedding用于融合
        )
        self.mlp_classifier = nn.Linear(embed_dim, num_classes)

        mlp_params = sum(p.numel() for p in self.mlp_branch.parameters()) + \
                     sum(p.numel() for p in self.mlp_classifier.parameters())
        print(f"  参数量: {mlp_params:,}")

        # ============ 分支2: CNN1D (局部模式特征) ============
        print("\n[分支2] CNN1D - 局部模式特征")
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.cnn_classifier = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

        cnn_params = sum(p.numel() for p in self.cnn_branch.parameters()) + \
                     sum(p.numel() for p in self.cnn_classifier.parameters())
        print(f"  参数量: {cnn_params:,}")

        # ============ 分支3: Transformer (长距离依赖) ============
        print("\n[分支3] Transformer - 长距离依赖特征")
        self.transformer_branch = EmberTransformer(
            input_dim=input_dim,
            patch_size=50,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            num_classes=num_classes
        )

        # ============ 融合层 ============
        print("\n[融合层] 多头注意力融合")
        self.fusion_layer = MultiHeadFusionLayer(
            num_branches=3,
            fusion_type=fusion_type,
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        # 总参数量统计
        self._print_total_params()

    def _print_total_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n" + "=" * 60)
        print(f"异构集成模型总参数量")
        print(f"  总计: {total:,}")
        print(f"  可训练: {trainable:,}")
        print("=" * 60)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch, 2381] 输入特征

        Returns:
            fused_probs: [batch, num_classes] 融合后的预测概率
            fused_logits: [batch, num_classes] 融合后的logits
            branch_probs_dict: 各分支的预测概率字典
        """
        batch_size = x.shape[0]

        # ============ 分支1: MLP ============
        mlp_embed = self.mlp_branch(x)  # [batch, embed_dim]
        mlp_logits = self.mlp_classifier(mlp_embed)  # [batch, num_classes]
        mlp_probs = F.softmax(mlp_logits, dim=1)

        # ============ 分支2: CNN1D ============
        cnn_input = x.unsqueeze(1)  # [batch, 1, 2381]
        cnn_feat = self.cnn_branch(cnn_input).squeeze(-1)  # [batch, 128]
        cnn_logits = self.cnn_classifier(cnn_feat)  # [batch, num_classes]
        cnn_probs = F.softmax(cnn_logits, dim=1)

        # ============ 分支3: Transformer ============
        trans_logits = self.transformer_branch(x)  # [batch, num_classes]
        trans_probs = F.softmax(trans_logits, dim=1)

        # ============ 融合 ============
        # 堆叠各分支概率: [batch, num_branches, num_classes]
        branch_probs = torch.stack([mlp_probs, cnn_probs, trans_probs], dim=1)

        # 注意力融合
        fused_probs = self.fusion_layer(branch_probs)  # [batch, num_classes]
        fused_logits = torch.log(fused_probs + 1e-8)  # 避免log(0)

        # 返回各分支概率用于分析
        branch_probs_dict = {
            'mlp': mlp_probs,
            'cnn': cnn_probs,
            'transformer': trans_probs
        }

        return fused_probs, fused_logits, branch_probs_dict

    def get_branch_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取各分支的独立预测结果 (用于分析)
        """
        _, _, branch_probs_dict = self.forward(x)

        predictions = {}
        for name, probs in branch_probs_dict.items():
            predictions[name] = {
                'probs': probs,
                'preds': torch.argmax(probs, dim=1)
            }
        return predictions

    def get_fusion_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取融合层的注意力权重 (用于可视化)
        支持所有融合类型
        """
        batch_size = x.shape[0]

        # 在eval+no_grad模式下计算，确保与forward()一致且稳定
        was_training = self.training
        self.eval()
        with torch.no_grad():
            # 计算各分支概率
            mlp_embed = self.mlp_branch(x)
            mlp_logits = self.mlp_classifier(mlp_embed)
            mlp_probs = F.softmax(mlp_logits, dim=1)

            cnn_input = x.unsqueeze(1)
            cnn_feat = self.cnn_branch(cnn_input).squeeze(-1)
            cnn_logits = self.cnn_classifier(cnn_feat)
            cnn_probs = F.softmax(cnn_logits, dim=1)

            trans_logits = self.transformer_branch(x)
            trans_probs = F.softmax(trans_logits, dim=1)

            # 堆叠各分支概率
            branch_probs = torch.stack([mlp_probs, cnn_probs, trans_probs], dim=1)
            flat_probs = branch_probs.view(batch_size, -1)

            # 根据融合类型计算权重
            fusion_type = self.fusion_layer.fusion_type
            if fusion_type == 'probability_attention':
                weights = self.fusion_layer.prob_attention(flat_probs)
            elif fusion_type == 'gating':
                weights = self.fusion_layer.gate_net(flat_probs)
            elif fusion_type == 'meta_attention':
                weights = self.fusion_layer.meta_attention(flat_probs)
            elif fusion_type == 'feature_attention':
                # feature_attention 需要 branch_features，这里用概率近似
                weights = self.fusion_layer.feature_attention(flat_probs)
            else:
                # 默认均匀权重
                weights = torch.ones(batch_size, 3, device=x.device) / 3.0

            # 替换NaN为均匀权重，防止传播
            nan_mask = torch.isnan(weights).any(dim=1)
            if nan_mask.any():
                weights[nan_mask] = torch.tensor([1/3, 1/3, 1/3], device=x.device, dtype=weights.dtype)

        if was_training:
            self.train()

        return weights  # [batch, 3]


class LightweightHeterogeneousEnsemble(nn.Module):
    """
    轻量版异构集成模型
    进一步压缩参数量，适合资源受限场景

    特点:
    - MLP: 1层隐藏层
    - CNN: 2层卷积
    - Transformer: 1层
    - 总参数量控制在100K以内
    """
    def __init__(self, input_dim: int = 2381, num_classes: int = 2):
        super(LightweightHeterogeneousEnsemble, self).__init__()

        embed_dim = 32  # 更小的embedding维度

        # MLP分支 (轻量)
        self.mlp_branch = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.mlp_head = nn.Linear(embed_dim, num_classes)

        # CNN分支 (轻量)
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.cnn_head = nn.Linear(32, num_classes)

        # Transformer分支 (轻量: 1层)
        self.trans_embed = nn.Linear(50, embed_dim)
        self.num_patches = math.ceil(input_dim / 50)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.trans_attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.trans_head = nn.Linear(embed_dim, num_classes)

        # 简单加权平均融合
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"LightweightHeterogeneousEnsemble: 总参数={total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # MLP
        mlp_feat = self.mlp_branch(x)
        mlp_logits = self.mlp_head(mlp_feat)
        mlp_probs = F.softmax(mlp_logits, dim=1)

        # CNN
        cnn_feat = self.cnn_branch(x.unsqueeze(1)).squeeze(-1)
        cnn_logits = self.cnn_head(cnn_feat)
        cnn_probs = F.softmax(cnn_logits, dim=1)

        # Transformer (简化版)
        pad_size = self.num_patches * 50 - x.shape[1]
        if pad_size > 0:
            x_pad = F.pad(x, (0, pad_size))
        else:
            x_pad = x
        x_patches = x_pad.view(batch_size, self.num_patches, 50)
        trans_feat = self.trans_embed(x_patches) + self.pos_embed
        trans_feat, _ = self.trans_attn(trans_feat, trans_feat, trans_feat)
        trans_logits = self.trans_head(trans_feat.mean(dim=1))
        trans_probs = F.softmax(trans_logits, dim=1)

        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_probs = weights[0] * mlp_probs + weights[1] * cnn_probs + weights[2] * trans_probs

        return fused_probs


# ============ 模型工厂函数 ============
def create_heterogeneous_ensemble(model_type: str = 'standard',
                                   input_dim: int = 2381,
                                   num_classes: int = 2,
                                   fusion_type: str = 'probability_attention',
                                   **kwargs):
    """
    创建异构集成模型

    Args:
        model_type: 'standard' (完整版) 或 'lightweight' (轻量版)
        input_dim: 输入特征维度
        num_classes: 分类数量
        fusion_type: 融合策略 ('probability_attention', 'gating', 'meta_attention')

    Returns:
        异构集成模型实例
    """
    if model_type == 'standard':
        return HeterogeneousEnsembleDetector(
            input_dim=input_dim,
            num_classes=num_classes,
            fusion_type=fusion_type,
            **kwargs
        )
    elif model_type == 'lightweight':
        return LightweightHeterogeneousEnsemble(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")


# ============ 测试代码 ============
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("异构集成模型测试")
    print("=" * 60)

    # 测试数据
    batch_size = 4
    input_dim = 2381
    x = torch.randn(batch_size, input_dim)

    # 测试完整版
    print("\n测试 HeterogeneousEnsembleDetector (完整版):")
    model = create_heterogeneous_ensemble('standard', fusion_type='probability_attention')
    fused_probs, fused_logits, branch_probs = model(x)

    print(f"\n输入: {x.shape}")
    print(f"融合输出: probs={fused_probs.shape}, logits={fused_logits.shape}")
    print(f"各分支概率:")
    for name, probs in branch_probs.items():
        print(f"  {name}: {probs.shape}, 预测={torch.argmax(probs, dim=1)}")

    # 获取融合权重
    weights = model.get_fusion_weights(x)
    print(f"\n融合权重: MLP={weights[0].mean():.3f}, CNN={weights[1].mean():.3f}, Transformer={weights[2].mean():.3f}")

    # 测试轻量版
    print("\n" + "-" * 60)
    print("测试 LightweightHeterogeneousEnsemble (轻量版):")
    model_lite = create_heterogeneous_ensemble('lightweight')
    probs_lite = model_lite(x)
    print(f"输出: {probs_lite.shape}")

    print("\n" + "=" * 60)
    print("✓ 异构集成模型测试通过!")
    print("=" * 60)