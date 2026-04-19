"""
对抗训练模块
实现FGSM、PGD等对抗攻击算法，用于增强模型鲁棒性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AdversarialTrainer:
    """
    对抗训练器
    支持FGSM、PGD等对抗攻击算法
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03, attack_type: str = 'fgsm'):
        """
        初始化对抗训练器
        
        Args:
            model: 要训练的模型
            epsilon: 扰动大小
            attack_type: 攻击类型，支持 'fgsm' 和 'pgd'
        """
        self.model = model
        self.epsilon = epsilon
        self.attack_type = attack_type
    
    def fgsm_attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        FGSM (Fast Gradient Sign Method) 攻击
        
        Args:
            images: 输入图像
            labels: 真实标签
            
        Returns:
            对抗样本
        """
        # 确保模型处于训练模式
        self.model.train()
        
        # 复制输入图像
        images = images.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # 反向传播计算梯度
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        attack_images = images + self.epsilon * images.grad.sign()
        attack_images = torch.clamp(attack_images, 0, 1)  # 确保像素值在[0,1]范围内
        
        return attack_images
    
    def pgd_attack(self, images: torch.Tensor, labels: torch.Tensor, steps: int = 10, alpha: float = 0.003) -> torch.Tensor:
        """
        PGD (Projected Gradient Descent) 攻击
        
        Args:
            images: 输入图像
            labels: 真实标签
            steps: 迭代步数
            alpha: 每步的扰动大小
            
        Returns:
            对抗样本
        """
        # 确保模型处于训练模式
        self.model.train()
        
        # 复制输入图像
        attack_images = images.clone().detach()
        
        for _ in range(steps):
            attack_images.requires_grad_(True)
            outputs = self.model(attack_images)
            loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # 计算扰动并更新对抗样本
            attack_images = attack_images + alpha * attack_images.grad.sign()
            
            # 确保扰动不超过epsilon
            perturbation = torch.clamp(attack_images - images, -self.epsilon, self.epsilon)
            attack_images = torch.clamp(images + perturbation, 0, 1)
            
            #  detach to prevent gradients from accumulating
            attack_images = attack_images.detach()
        
        return attack_images
    
    def feature_fgsm_attack(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        针对特征向量的FGSM攻击
        
        Args:
            features: 输入特征向量
            labels: 真实标签
            
        Returns:
            对抗特征向量
        """
        # 确保模型处于训练模式
        self.model.train()
        
        # 复制输入特征
        features = features.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs, _ = self.model(features)
        loss = F.cross_entropy(outputs, labels)
        
        # 反向传播计算梯度
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗特征
        attack_features = features + self.epsilon * features.grad.sign()
        
        return attack_features
    
    def feature_pgd_attack(self, features: torch.Tensor, labels: torch.Tensor, steps: int = 10, alpha: float = 0.003) -> torch.Tensor:
        """
        针对特征向量的PGD攻击
        
        Args:
            features: 输入特征向量
            labels: 真实标签
            steps: 迭代步数
            alpha: 每步的扰动大小
            
        Returns:
            对抗特征向量
        """
        # 确保模型处于训练模式
        self.model.train()
        
        # 复制输入特征
        attack_features = features.clone().detach()
        
        for _ in range(steps):
            attack_features.requires_grad_(True)
            outputs, _ = self.model(attack_features)
            loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # 计算扰动并更新对抗特征
            attack_features = attack_features + alpha * attack_features.grad.sign()
            
            # 确保扰动不超过epsilon
            perturbation = torch.clamp(attack_features - features, -self.epsilon, self.epsilon)
            attack_features = features + perturbation
            
            #  detach to prevent gradients from accumulating
            attack_features = attack_features.detach()
        
        return attack_features
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer, use_adv: bool = True) -> float:
        """
        执行一个训练步骤，包括对抗训练
        
        Args:
            batch: 包含输入和标签的批次
            optimizer: 优化器
            use_adv: 是否使用对抗训练
            
        Returns:
            训练损失
        """
        inputs, labels = batch
        
        # 标准训练
        optimizer.zero_grad()
        if isinstance(inputs, tuple):
            # 多模态输入
            outputs, _ = self.model(*inputs)
        else:
            # 单输入
            try:
                outputs, _ = self.model(inputs)
            except ValueError:
                # 对于不返回注意力权重的模型
                outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 对抗训练
        if use_adv:
            optimizer.zero_grad()
            if isinstance(inputs, tuple):
                # 多模态输入，只对图像部分进行攻击
                features, images = inputs
                if self.attack_type == 'fgsm':
                    adv_images = self.fgsm_attack(images, labels)
                else:
                    adv_images = self.pgd_attack(images, labels)
                adv_outputs, _ = self.model(features, adv_images)
            else:
                # 单输入
                model_name = self.model.__class__.__name__
                if model_name in ['EfficientNetMalwareDetector', 'ViTMalwareDetector']:
                    # 图像输入模型
                    if self.attack_type == 'fgsm':
                        adv_inputs = self.fgsm_attack(inputs, labels)
                    else:
                        adv_inputs = self.pgd_attack(inputs, labels)
                    adv_outputs = self.model(adv_inputs)
                else:
                    # 特征输入模型
                    if self.attack_type == 'fgsm':
                        adv_inputs = self.feature_fgsm_attack(inputs, labels)
                    else:
                        adv_inputs = self.feature_pgd_attack(inputs, labels)
                    adv_outputs, _ = self.model(adv_inputs)
            
            adv_loss = F.cross_entropy(adv_outputs, labels)
            adv_loss.backward()
            optimizer.step()
            
            # 返回总损失
            return (loss.item() + adv_loss.item()) / 2
        
        return loss.item()


def generate_adversarial_examples(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, epsilon: float = 0.03, attack_type: str = 'fgsm') -> torch.Tensor:
    """
    生成对抗样本
    
    Args:
        model: 目标模型
        inputs: 输入样本
        labels: 真实标签
        epsilon: 扰动大小
        attack_type: 攻击类型
        
    Returns:
        对抗样本
    """
    trainer = AdversarialTrainer(model, epsilon, attack_type)
    
    model_name = model.__class__.__name__
    if model_name in ['EfficientNetMalwareDetector', 'ViTMalwareDetector']:
        # 图像输入模型
        if attack_type == 'fgsm':
            return trainer.fgsm_attack(inputs, labels)
        else:
            return trainer.pgd_attack(inputs, labels)
    else:
        # 特征输入模型
        if attack_type == 'fgsm':
            return trainer.feature_fgsm_attack(inputs, labels)
        else:
            return trainer.feature_pgd_attack(inputs, labels)