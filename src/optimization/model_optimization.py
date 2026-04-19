"""
模型优化模块
实现模型量化、推理优化等功能，提高模型的推理速度和部署效率
"""

import torch
import torch.nn as nn
import torch.quantization
from torch.utils.mobile_optimizer import optimize_for_mobile
import os
from typing import Optional, Tuple


class ModelOptimizer:
    """
    模型优化器
    支持模型量化、推理优化等功能
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化模型优化器
        
        Args:
            model: 要优化的模型
        """
        self.model = model
    
    def quantize_dynamic(self, backend: str = 'fbgemm') -> torch.nn.Module:
        """
        动态量化
        
        Args:
            backend: 量化后端，可选 'fbgemm' (x86) 或 'qnnpack' (ARM)
            
        Returns:
            量化后的模型
        """
        print(f"🔄 开始动态量化模型，使用后端: {backend}")
        
        # 设置量化后端
        torch.backends.quantized.engine = backend
        
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d, nn.BatchNorm2d},
            dtype=torch.qint8
        )
        
        print("✅ 动态量化完成")
        return quantized_model
    
    def quantize_static(self, calibration_data: torch.Tensor, backend: str = 'fbgemm') -> torch.nn.Module:
        """
        静态量化
        
        Args:
            calibration_data: 用于校准的数据集
            backend: 量化后端，可选 'fbgemm' (x86) 或 'qnnpack' (ARM)
            
        Returns:
            量化后的模型
        """
        print(f"🔄 开始静态量化模型，使用后端: {backend}")
        
        # 设置量化后端
        torch.backends.quantized.engine = backend
        
        # 准备模型
        self.model.eval()
        
        # 添加量化和反量化节点
        quantized_model = torch.quantization.prepare(self.model, inplace=False)
        
        # 校准模型
        print("📊 正在校准模型...")
        with torch.no_grad():
            for i in range(len(calibration_data)):
                sample = calibration_data[i:i+1]
                if isinstance(sample, tuple):
                    quantized_model(*sample)
                else:
                    try:
                        quantized_model(sample)
                    except ValueError:
                        # 对于返回注意力权重的模型
                        quantized_model(sample)[0]
        
        # 转换模型
        quantized_model = torch.quantization.convert(quantized_model, inplace=False)
        
        print("✅ 静态量化完成")
        return quantized_model
    
    def optimize_for_mobile(self, quantized: bool = False) -> torch.jit.ScriptModule:
        """
        优化模型以在移动设备上运行
        
        Args:
            quantized: 是否是量化模型
            
        Returns:
            优化后的TorchScript模型
        """
        print("🔄 开始优化模型以在移动设备上运行")
        
        #  traced_model = torch.jit.trace(self.model, torch.randn(1, 2381))
        # 对于不同类型的模型，需要不同的输入
        model_name = self.model.__class__.__name__
        if model_name in ['EfficientNetMalwareDetector', 'ViTMalwareDetector']:
            # 图像输入模型
            traced_model = torch.jit.trace(self.model, torch.randn(1, 1, 224, 224))
        else:
            # 特征输入模型
            traced_model = torch.jit.trace(self.model, torch.randn(1, 2381))
        
        # 优化模型
        optimized_model = optimize_for_mobile(traced_model)
        
        print("✅ 模型优化完成")
        return optimized_model
    
    def save_optimized_model(self, model: torch.nn.Module, save_path: str, format: str = 'pth'):
        """
        保存优化后的模型
        
        Args:
            model: 优化后的模型
            save_path: 保存路径
            format: 保存格式，可选 'pth', 'pt', 'torchscript'
        """
        print(f"💾 保存优化后的模型到: {save_path}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if format == 'torchscript':
            torch.jit.save(model, save_path)
        else:
            torch.save(model.state_dict(), save_path)
        
        print("✅ 模型保存完成")
    
    def load_optimized_model(self, load_path: str, model_type: str, num_classes: int = 9) -> torch.nn.Module:
        """
        加载优化后的模型
        
        Args:
            load_path: 加载路径
            model_type: 模型类型
            num_classes: 类别数量
            
        Returns:
            加载的模型
        """
        print(f"📦 从: {load_path} 加载优化后的模型")
        
        # 导入模型
        from models import (
            EmberMLP, 
            MalwareDetectionCNN, 
            OptimizedMalwareDetector, 
            EfficientNetMalwareDetector, 
            ViTMalwareDetector
        )
        
        # 根据模型类型创建模型实例
        if model_type == 'ember':
            model = EmberMLP(input_dim=2381, num_classes=num_classes)
        elif model_type == 'cnn':
            model = MalwareDetectionCNN(num_classes=num_classes)
        elif model_type == 'optimized':
            model = OptimizedMalwareDetector(num_classes=num_classes)
        elif model_type == 'efficientnet':
            model = EfficientNetMalwareDetector(num_classes=num_classes)
        elif model_type == 'vit':
            model = ViTMalwareDetector(num_classes=num_classes)
        else:
            model = OptimizedMalwareDetector(num_classes=num_classes)
        
        # 加载模型
        if load_path.endswith('.pt'):
            model = torch.jit.load(load_path)
        else:
            model.load_state_dict(torch.load(load_path))
        
        model.eval()
        print("✅ 模型加载完成")
        return model


def benchmark_model(model: nn.Module, input_data: torch.Tensor, iterations: int = 100) -> dict:
    """
    基准测试模型性能
    
    Args:
        model: 要测试的模型
        input_data: 输入数据
        iterations: 测试迭代次数
        
    Returns:
        性能指标字典
    """
    import time
    
    print("📈 开始基准测试...")
    
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            if isinstance(input_data, tuple):
                model(*input_data)
            else:
                try:
                    model(input_data)
                except ValueError:
                    # 对于返回注意力权重的模型
                    model(input_data)[0]
    
    # 测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            if isinstance(input_data, tuple):
                model(*input_data)
            else:
                try:
                    model(input_data)
                except ValueError:
                    # 对于返回注意力权重的模型
                    model(input_data)[0]
    end_time = time.time()
    
    # 计算性能指标
    avg_inference_time = (end_time - start_time) / iterations * 1000  # 转换为毫秒
    throughput = iterations / (end_time - start_time)  # 每秒处理样本数
    
    # 计算模型大小
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        model_size = os.path.getsize(f.name) / (1024 * 1024)  # 转换为MB
    os.unlink(f.name)
    
    performance = {
        'average_inference_time_ms': avg_inference_time,
        'throughput_samples_per_second': throughput,
        'model_size_mb': model_size
    }
    
    print(f"✅ 基准测试完成: 平均推理时间 = {avg_inference_time:.4f} ms, 吞吐量 = {throughput:.2f} 样本/秒, 模型大小 = {model_size:.2f} MB")
    
    return performance


def optimize_model_pipeline(model: nn.Module, calibration_data: Optional[torch.Tensor] = None, output_dir: str = './optimized_models') -> dict:
    """
    模型优化 pipeline
    
    Args:
        model: 要优化的模型
        calibration_data: 用于静态量化的校准数据
        output_dir: 输出目录
        
    Returns:
        优化结果字典
    """
    print("🚀 开始模型优化 pipeline...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = ModelOptimizer(model)
    results = {}
    
    # 1. 基准测试原始模型
    print("\n1. 测试原始模型性能")
    model_name = model.__class__.__name__
    if model_name in ['EfficientNetMalwareDetector', 'ViTMalwareDetector']:
        test_input = torch.randn(1, 1, 224, 224)
    else:
        test_input = torch.randn(1, 2381)
    
    results['original'] = benchmark_model(model, test_input)
    
    # 2. 动态量化
    print("\n2. 执行动态量化")
    dynamic_quantized_model = optimizer.quantize_dynamic()
    results['dynamic_quantized'] = benchmark_model(dynamic_quantized_model, test_input)
    optimizer.save_optimized_model(dynamic_quantized_model, os.path.join(output_dir, 'dynamic_quantized.pth'))
    
    # 3. 静态量化（如果提供了校准数据）
    if calibration_data is not None:
        print("\n3. 执行静态量化")
        static_quantized_model = optimizer.quantize_static(calibration_data)
        results['static_quantized'] = benchmark_model(static_quantized_model, test_input)
        optimizer.save_optimized_model(static_quantized_model, os.path.join(output_dir, 'static_quantized.pth'))
    
    # 4. 优化为移动设备
    print("\n4. 优化模型以在移动设备上运行")
    mobile_model = optimizer.optimize_for_mobile()
    results['mobile_optimized'] = benchmark_model(mobile_model, test_input)
    optimizer.save_optimized_model(mobile_model, os.path.join(output_dir, 'mobile_optimized.pt'), format='torchscript')
    
    print("\n📊 优化结果总结:")
    for key, value in results.items():
        print(f"{key}:")
        print(f"  - 平均推理时间: {value['average_inference_time_ms']:.4f} ms")
        print(f"  - 吞吐量: {value['throughput_samples_per_second']:.2f} 样本/秒")
        print(f"  - 模型大小: {value['model_size_mb']:.2f} MB")
    
    return results