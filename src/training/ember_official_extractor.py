"""
EMBER 官方特征提取器封装
基于 elastic/ember 官方实现，确保特征与 EMBER 数据集完全兼容

使用方法:
    from src.training.ember_official_extractor import EMBERFeatureExtractor

    extractor = EMBERFeatureExtractor()
    features = extractor.extract('sample.exe')  # 返回 2381 维 numpy 数组
"""

import os
import sys
import numpy as np
from typing import Optional

# 添加 ember_official 到 Python 路径
EMBER_OFFICIAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ember_official')
if os.path.exists(EMBER_OFFICIAL_PATH):
    sys.path.insert(0, EMBER_OFFICIAL_PATH)

try:
    from ember.features import PEFeatureExtractor as OfficialPEFeatureExtractor
    EMBER_AVAILABLE = True
except ImportError:
    EMBER_AVAILABLE = False
    print("警告: EMBER 官方模块未找到，请确保 ember_official 目录存在")


class EMBERFeatureExtractor:
    """
    EMBER 官方特征提取器封装

    特征维度: 2381 (EMBER v2)
    特征分组:
        - ByteHistogram: 256维 [0:256]
        - ByteEntropyHistogram: 256维 [256:512]
        - StringExtractor: 104维 [512:616]
        - GeneralFileInfo: 10维
        - HeaderFileInfo: 62维
        - SectionInfo: 255维
        - ImportsInfo: 1280维
        - ExportsInfo: 128维
        - DataDirectories: 30维
    """

    def __init__(self, feature_version=2, print_warning=False):
        """
        初始化 EMBER 特征提取器

        Args:
            feature_version: EMBER 特征版本 (1 或 2，默认 2)
            print_warning: 是否打印 LIEF 版本警告
        """
        if not EMBER_AVAILABLE:
            raise ImportError("EMBER 官方模块未安装")

        self.extractor = OfficialPEFeatureExtractor(
            feature_version=feature_version,
            print_feature_warning=print_warning
        )
        self.feature_dim = self.extractor.dim  # 2381 for v2

    def extract(self, file_path: str) -> Optional[np.ndarray]:
        """
        从 PE 文件提取 2381 维特征向量

        Args:
            file_path: PE 文件路径

        Returns:
            2381 维 numpy 数组，或 None（如果提取失败）
        """
        try:
            with open(file_path, 'rb') as f:
                bytez = f.read()

            # 使用官方提取器
            feature_vector = self.extractor.feature_vector(bytez)
            return feature_vector

        except Exception as e:
            print(f"特征提取失败: {file_path} - {str(e)}")
            return None

    def extract_raw_features(self, file_path: str) -> Optional[dict]:
        """
        提取原始特征（JSON 格式），用于分析和调试

        Args:
            file_path: PE 文件路径

        Returns:
            原始特征字典，包含各特征类型的详细信息
        """
        try:
            with open(file_path, 'rb') as f:
                bytez = f.read()

            raw_features = self.extractor.raw_features(bytez)
            return raw_features

        except Exception as e:
            print(f"原始特征提取失败: {file_path} - {str(e)}")
            return None

    def get_feature_groups(self, feature_vector: np.ndarray) -> dict:
        """
        将 2381 维特征向量拆分为特征组

        Args:
            feature_vector: 2381 维特征向量

        Returns:
            特征组字典:
                - histogram: 256维 (ByteHistogram)
                - byte_entropy: 256维 (ByteEntropyHistogram)
                - strings: 104维 (StringExtractor)
                - general: 1765维 (GeneralFileInfo+Header+Section+Imports+Exports+DataDirectories)
        """
        if len(feature_vector) != self.feature_dim:
            raise ValueError(f"特征维度不匹配: 期望 {self.feature_dim}, 实际 {len(feature_vector)}")

        return {
            'histogram': feature_vector[0:256],
            'byte_entropy': feature_vector[256:512],
            'strings': feature_vector[512:616],
            'general': feature_vector[616:2381]
        }


def extract_features_batch(file_paths: list, output_path: str = None) -> np.ndarray:
    """
    批量提取 PE 文件特征

    Args:
        file_paths: PE 文件路径列表
        output_path: 输出 numpy 文件路径（可选）

    Returns:
        特征矩阵 [N, 2381]
    """
    extractor = EMBERFeatureExtractor()
    features = []
    valid_paths = []

    for path in file_paths:
        feat = extractor.extract(path)
        if feat is not None:
            features.append(feat)
            valid_paths.append(path)

    feature_matrix = np.array(features, dtype=np.float32)

    if output_path:
        np.save(output_path, feature_matrix)
        print(f"特征已保存: {output_path} ({len(features)} 个样本)")

    return feature_matrix, valid_paths


if __name__ == "__main__":
    # 测试代码
    print("EMBER 官方特征提取器测试")

    if EMBER_AVAILABLE:
        extractor = EMBERFeatureExtractor()
        print(f"特征维度: {extractor.feature_dim}")

        # 测试特征分组
        test_vector = np.random.randn(2381).astype(np.float32)
        groups = extractor.get_feature_groups(test_vector)
        print("\n特征分组:")
        for name, vec in groups.items():
            print(f"  {name}: {len(vec)} 维")
    else:
        print("EMBER 模块未安装")