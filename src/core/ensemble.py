"""
集成学习模块
实现多个模型的集成，提高检测性能
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import os
import sys

# 导入模型
from .models import (
    EmberMLP, 
    MalwareDetectionCNN, 
    OptimizedMalwareDetector, 
    EfficientNetMalwareDetector, 
    ViTMalwareDetector
)
from ..training.data_preprocessing import PEFeatureExtractor, BinaryToImage


class EnsembleModel:
    """
    集成模型
    结合多个模型的预测结果
    """
    
    def __init__(self, models: List[Dict], device: torch.device = torch.device('cpu')):
        """
        初始化集成模型
        
        Args:
            models: 模型列表，每个元素是一个字典，包含 'model_path', 'model_type', 'weight' 键
            device: 设备
        """
        self.models = []
        self.weights = []
        self.device = device
        
        # 加载模型
        for model_info in models:
            model = self._load_model(model_info['model_path'], model_info['model_type'])
            self.models.append(model)
            self.weights.append(model_info.get('weight', 1.0))
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"✅ 集成模型初始化完成，共加载 {len(self.models)} 个模型")
    
    def _load_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        """
        加载单个模型
        
        Args:
            model_path: 模型路径
            model_type: 模型类型
            
        Returns:
            加载的模型
        """
        print(f"📦 加载模型: {model_path} (类型: {model_type})")
        
        # 根据模型类型创建模型实例
        if model_type == 'ember':
            model = EmberMLP(input_dim=2381, num_classes=9)
        elif model_type == 'cnn':
            model = MalwareDetectionCNN(num_classes=9)
        elif model_type == 'optimized':
            model = OptimizedMalwareDetector(num_classes=9)
        elif model_type == 'efficientnet':
            model = EfficientNetMalwareDetector(num_classes=9)
        elif model_type == 'vit':
            model = ViTMalwareDetector(num_classes=9)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        
        print(f"✅ 模型加载完成: {model_path}")
        return model
    
    def predict(self, file_path: str) -> Dict:
        """
        对单个文件进行预测
        
        Args:
            file_path: 文件路径
            
        Returns:
            预测结果字典
        """
        # 提取特征
        extractor = PEFeatureExtractor()
        features = extractor.extract_features(file_path)
        if features is None:
            return {'success': False, 'error': 'Failed to extract features from file'}
        
        feature_vector = extractor.create_feature_vector(features)
        # 确保特征向量维度为2381
        if len(feature_vector) < 2381:
            feature_vector = np.pad(feature_vector, (0, 2381 - len(feature_vector)))
        elif len(feature_vector) > 2381:
            feature_vector = feature_vector[:2381]
        
        input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
        
        # 转换为图像（用于图像输入模型）
        converter = BinaryToImage()
        image = converter.convert(file_path)
        if image is not None:
            image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            image_tensor = None
        
        # 收集所有模型的预测结果
        all_probs = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                model_name = model.__class__.__name__
                
                if model_name in ['EfficientNetMalwareDetector', 'ViTMalwareDetector']:
                    # 图像输入模型
                    if image_tensor is not None:
                        outputs = model(image_tensor)
                    else:
                        continue
                elif hasattr(model, 'use_image') and model.use_image:
                    # 多模态模型
                    if image_tensor is not None:
                        outputs, _ = model(input_tensor, image_tensor)
                    else:
                        outputs, _ = model(input_tensor)
                else:
                    # 特征输入模型
                    try:
                        outputs, _ = model(input_tensor)
                    except ValueError:
                        # 对于不返回注意力权重的模型
                        outputs = model(input_tensor)
                
                # 计算概率
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                all_probs.append(probs * self.weights[i])
        
        # 集成预测结果
        ensemble_probs = np.sum(all_probs, axis=0)
        pred_idx = np.argmax(ensemble_probs)
        confidence = ensemble_probs[pred_idx]
        
        # 恶意软件家族标签
        malware_families = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda', 'Tracur', 'Kelihos_ver1', 'Obfuscator.ACY', 'Gatak']
        predicted_family = malware_families[pred_idx]
        
        # 获取top 5预测结果
        top_indices = np.argsort(ensemble_probs)[::-1][:5]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'family': malware_families[idx],
                'probability': round(float(ensemble_probs[idx]), 4)
            })
        
        return {
            'success': True,
            'predicted_family': predicted_family,
            'confidence': float(confidence),
            'top_5_predictions': top_predictions,
            'ensemble_probs': ensemble_probs.tolist()
        }
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        评估集成模型
        
        Args:
            test_data: 测试数据，每个元素是一个字典，包含 'file_path' 和 'label' 键
            
        Returns:
            评估结果字典
        """
        correct = 0
        total = len(test_data)
        
        for data in test_data:
            file_path = data['file_path']
            true_label = data['label']
            
            result = self.predict(file_path)
            if result['success']:
                predicted_label = result['predicted_family']
                if predicted_label == true_label:
                    correct += 1
        
        accuracy = correct / total
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }


def create_ensemble_model(model_configs: List[Dict], device: Optional[torch.device] = None) -> EnsembleModel:
    """
    创建集成模型
    
    Args:
        model_configs: 模型配置列表
        device: 设备
        
    Returns:
        集成模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return EnsembleModel(model_configs, device)