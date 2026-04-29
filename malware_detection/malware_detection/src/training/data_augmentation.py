"""
数据增强模块
提供多种数据增强技术以提升模型泛化能力
"""

import numpy as np
from typing import Tuple
import random


class DataAugmentation:
    """数据增强类"""
    
    def __init__(self, augmentation_prob: float = 0.5):
        """
        初始化数据增强器
        
        Args:
            augmentation_prob: 应用增强的概率
        """
        self.augmentation_prob = augmentation_prob
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        对图像进行增强
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, C)
            
        Returns:
            增强后的图像
        """
        augmented = image.copy()
        
        # 随机应用各种增强
        if random.random() < self.augmentation_prob:
            augmented = self._add_noise(augmented)
        
        if random.random() < self.augmentation_prob:
            augmented = self._adjust_brightness(augmented)
        
        if random.random() < self.augmentation_prob:
            augmented = self._horizontal_flip(augmented)
        
        if random.random() < self.augmentation_prob:
            augmented = self._vertical_flip(augmented)
        
        if random.random() < self.augmentation_prob:
            augmented = self._rotate(augmented)
        
        return augmented
    
    def augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        对特征向量进行增强
        
        Args:
            features: 输入特征向量
            
        Returns:
            增强后的特征向量
        """
        augmented = features.copy()
        
        # 添加高斯噪声
        if random.random() < self.augmentation_prob:
            noise_scale = 0.01
            noise = np.random.normal(0, noise_scale, features.shape)
            augmented = augmented + noise
        
        # 特征缩放
        if random.random() < self.augmentation_prob:
            scale_factor = random.uniform(0.95, 1.05)
            augmented = augmented * scale_factor
        
        # 特征遮挡（随机将部分特征置0）
        if random.random() < self.augmentation_prob:
            mask_ratio = 0.1
            mask_size = int(len(augmented) * mask_ratio)
            mask_indices = random.sample(range(len(augmented)), mask_size)
            augmented[mask_indices] = 0
        
        return augmented
    
    @staticmethod
    def _add_noise(image: np.ndarray, noise_level: float = 5.0) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(image.dtype)
    
    @staticmethod
    def _adjust_brightness(image: np.ndarray, factor: float = None) -> np.ndarray:
        """调整亮度"""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        
        adjusted = image * factor
        return np.clip(adjusted, 0, 255).astype(image.dtype)
    
    @staticmethod
    def _horizontal_flip(image: np.ndarray) -> np.ndarray:
        """水平翻转"""
        return np.fliplr(image)
    
    @staticmethod
    def _vertical_flip(image: np.ndarray) -> np.ndarray:
        """垂直翻转"""
        return np.flipud(image)
    
    @staticmethod
    def _rotate(image: np.ndarray, angle: int = None) -> np.ndarray:
        """旋转图像"""
        if angle is None:
            angle = random.choice([90, 180, 270])
        
        k = angle // 90
        return np.rot90(image, k)


class BalancedSampler:
    """平衡采样器，用于处理不平衡数据集"""
    
    def __init__(self, labels: np.ndarray):
        """
        初始化平衡采样器
        
        Args:
            labels: 标签数组
        """
        self.labels = labels
        self.class_counts = {}
        self.class_indices = {}
        
        # 统计各类样本
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            self.class_counts[label] = len(indices)
            self.class_indices[label] = indices
    
    def get_balanced_indices(self, num_samples: int = None) -> np.ndarray:
        """
        获取平衡的样本索引
        
        Args:
            num_samples: 每类采样数量，None则使用最小类的样本数
            
        Returns:
            平衡后的索引数组
        """
        if num_samples is None:
            num_samples = min(self.class_counts.values())
        
        balanced_indices = []
        
        for label, indices in self.class_indices.items():
            if len(indices) >= num_samples:
                # 随机采样
                sampled_indices = np.random.choice(indices, num_samples, replace=False)
            else:
                # 过采样
                sampled_indices = np.random.choice(indices, num_samples, replace=True)
            
            balanced_indices.extend(sampled_indices)
        
        # 打乱顺序
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        return balanced_indices
    
    def get_class_weights(self) -> dict:
        """
        计算类别权重，用于损失函数
        
        Returns:
            类别权重字典
        """
        total_samples = len(self.labels)
        num_classes = len(self.class_counts)
        
        class_weights = {}
        for label, count in self.class_counts.items():
            class_weights[label] = total_samples / (num_classes * count)
        
        return class_weights


class MixUp:
    """MixUp数据增强"""
    
    def __init__(self, alpha: float = 0.2):
        """
        初始化MixUp
        
        Args:
            alpha: Beta分布参数
        """
        self.alpha = alpha
    
    def mixup(self, x1: np.ndarray, x2: np.ndarray, 
              y1: int, y2: int) -> Tuple[np.ndarray, Tuple[int, int, float]]:
        """
        执行MixUp
        
        Args:
            x1, x2: 两个样本
            y1, y2: 两个标签
            
        Returns:
            混合后的样本和标签信息
        """
        lambda_ = np.random.beta(self.alpha, self.alpha)
        
        # 混合样本
        x_mixed = lambda_ * x1 + (1 - lambda_) * x2
        
        # 返回混合样本和标签信息
        return x_mixed, (y1, y2, lambda_)


if __name__ == "__main__":
    # 测试数据增强
    print("数据增强模块测试")
    
    # 测试图像增强
    augmentor = DataAugmentation(augmentation_prob=0.8)
    test_image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    augmented_image = augmentor.augment_image(test_image)
    print(f"原始图像形状: {test_image.shape}")
    print(f"增强后图像形状: {augmented_image.shape}")
    
    # 测试特征增强
    test_features = np.random.randn(512)
    augmented_features = augmentor.augment_features(test_features)
    print(f"原始特征形状: {test_features.shape}")
    print(f"增强后特征形状: {augmented_features.shape}")
    
    # 测试平衡采样
    test_labels = np.array([0]*100 + [1]*50 + [2]*200)
    sampler = BalancedSampler(test_labels)
    balanced_indices = sampler.get_balanced_indices()
    print(f"原始标签分布: {sampler.class_counts}")
    print(f"平衡后样本数: {len(balanced_indices)}")
    
    class_weights = sampler.get_class_weights()
    print(f"类别权重: {class_weights}")