"""
二进制文件转图像工具
将PE文件转换为灰度图像用于多模态融合
"""

import numpy as np
import os
from pathlib import Path


def binary_to_grayscale_image(binary_data: bytes, img_size: int = 256) -> np.ndarray:
    """
    将二进制数据转换为灰度图像

    Args:
        binary_data: PE文件的二进制数据
        img_size: 输出图像大小（正方形）

    Returns:
        grayscale_image: 灰度图像数组 (img_size, img_size)
    """
    # 每个字节(0-255)对应一个灰度值
    data = np.frombuffer(binary_data, dtype=np.uint8)

    # 填充或裁剪到固定大小
    target_size = img_size * img_size
    if len(data) < target_size:
        # 填充0
        data = np.pad(data, (0, target_size - len(data)), mode='constant')
    else:
        # 裁剪到目标大小
        data = data[:target_size]

    # reshape为图像
    image = data.reshape(img_size, img_size)

    return image


def extract_image_features_simple(image: np.ndarray) -> np.ndarray:
    """
    简化的图像特征提取（不使用深度网络）
    提取统计特征和纹理特征

    Args:
        image: 灰度图像

    Returns:
        features: 图像特征向量（128维）
    """
    features = []

    # 1. 全局统计特征 (10维)
    features.append(image.mean())          # 均值
    features.append(image.std())           # 标准差
    features.append(image.max())           # 最大值
    features.append(image.min())           # 最小值
    features.append(np.median(image))      # 中位数

    # 计算直方图统计
    hist = np.histogram(image, bins=16, range=(0, 256))[0]
    hist_norm = hist / hist.sum()
    features.extend(hist_norm.tolist())    # 16维

    # 2. 分块统计特征（将图像分为4x4块）
    block_size = image.shape[0] // 4
    for i in range(4):
        for j in range(4):
            block = image[i*block_size:(i+1)*block_size,
                          j*block_size:(j+1)*block_size]
            features.append(block.mean())  # 每块均值
            features.append(block.std())   # 每块标准差
    # 32维

    # 3. 熵特征
    # 计算图像熵
    hist_full = np.histogram(image, bins=256, range=(0, 256))[0]
    hist_full = hist_full[hist_full > 0]
    entropy = -np.sum(hist_full / hist_full.sum() * np.log2(hist_full / hist_full.sum()))
    features.append(entropy)  # 1维

    # 4. 边缘特征（简化版）
    # 计算简单的梯度统计
    grad_x = np.abs(np.diff(image, axis=1))
    grad_y = np.abs(np.diff(image, axis=0))
    features.append(grad_x.mean())
    features.append(grad_x.std())
    features.append(grad_y.mean())
    features.append(grad_y.std())
    # 4维

    # 5. 空域特征
    # 中心区域统计
    center = image[image.shape[0]//4:3*image.shape[0]//4,
                   image.shape[1]//4:3*image.shape[1]//4]
    features.append(center.mean())
    features.append(center.std())
    # 2维

    # 6. 对角线特征
    diag_main = np.diag(image)
    diag_anti = np.diag(np.fliplr(image))
    features.append(diag_main.mean())
    features.append(diag_main.std())
    features.append(diag_anti.mean())
    features.append(diag_anti.std())
    # 4维

    # 7. 区域对比度
    quadrant_means = []
    for i in range(2):
        for j in range(2):
            q = image[i*image.shape[0]//2:(i+1)*image.shape[0]//2,
                      j*image.shape[1]//2:(j+1)*image.shape[1]//2]
            quadrant_means.append(q.mean())

    # 四个象限的差异
    for m in quadrant_means:
        features.append(m)
    features.append(max(quadrant_means) - min(quadrant_means))
    # 5维

    # 8. 高频特征（FFT简化）
    # 只取FFT的前几个分量
    fft = np.fft.fft2(image)
    fft_mag = np.abs(fft)
    features.append(fft_mag[0, 0])  # DC分量
    features.append(fft_mag[1, 1])
    features.append(fft_mag[2, 2])
    features.append(fft_mag[5, 5])
    # 4维

    # 总计: 10 + 32 + 1 + 4 + 2 + 4 + 5 + 4 = 62维
    # 填充到128维
    while len(features) < 128:
        features.append(0)

    return np.array(features[:128], dtype=np.float32)


def process_pe_to_multimodal(jsonl_path: str, output_path: str,
                              max_samples: int = 0) -> dict:
    """
    从JSONL文件处理PE数据，提取PE特征和图像特征

    Args:
        jsonl_path: JSONL文件路径
        output_path: 输出目录
        max_samples: 最大样本数

    Returns:
        处理统计信息
    """
    import json

    pe_features_list = []
    img_features_list = []
    labels_list = []

    count = 0

    with open(jsonl_path, 'r') as f:
        for line in f:
            if max_samples > 0 and count >= max_samples:
                break

            try:
                sample = json.loads(line.strip())

                # 提取PE特征（使用已有extract_full_pe_features）
                from extract_full_pe_features import extract_full_pe_features
                pe_feat = extract_full_pe_features(sample)
                pe_features_list.append(pe_feat)

                # 模拟图像特征（因为没有原始二进制文件）
                # 使用histogram和byteentropy作为图像特征的基础
                histogram = sample.get('histogram', [0]*256)
                byteentropy = sample.get('byteentropy', [0]*256)

                # 构造虚拟图像特征
                img_feat = np.zeros(128, dtype=np.float32)
                img_feat[:64] = np.array(histogram[:64], dtype=np.float32)
                img_feat[64:128] = np.array(byteentropy[:64], dtype=np.float32)
                img_features_list.append(img_feat)

                # 标签
                label = sample.get('label', sample.get('y', 0))
                labels_list.append(label)

                count += 1

            except Exception as e:
                continue

    return {
        'pe_features': np.array(pe_features_list, dtype=np.float32),
        'img_features': np.array(img_features_list, dtype=np.float32),
        'labels': np.array(labels_list, dtype=np.int64),
        'count': count
    }


def prepare_multimodal_dataset(data_dir: str, output_dir: str,
                                train_samples: int = 500000,
                                test_samples: int = 100000):
    """
    准备多模态数据集

    Args:
        data_dir: JSONL数据目录
        output_dir: 输出目录
        train_samples: 训练样本数
        test_samples: 测试样本数
    """
    os.makedirs(output_dir, exist_ok=True)

    import glob
    from tqdm import tqdm

    # 查找训练和测试文件
    train_files = glob.glob(os.path.join(data_dir, '*train*.jsonl'))
    test_files = glob.glob(os.path.join(data_dir, '*test*.jsonl'))

    print(f"找到 {len(train_files)} 个训练文件")
    print(f"找到 {len(test_files)} 个测试文件")

    # 处理训练数据
    print("\n处理训练数据...")
    all_pe_train = []
    all_img_train = []
    all_label_train = []
    train_count = 0

    for filepath in tqdm(train_files[:10]):  # 只处理前10个文件
        result = process_pe_to_multimodal(filepath, output_dir,
                                          max_samples=train_samples - train_count)
        if result['count'] > 0:
            all_pe_train.append(result['pe_features'])
            all_img_train.append(result['img_features'])
            all_label_train.append(result['labels'])
            train_count += result['count']

        if train_count >= train_samples:
            break

    if all_pe_train:
        pe_train = np.concatenate(all_pe_train, axis=0)
        img_train = np.concatenate(all_img_train, axis=0)
        label_train = np.concatenate(all_label_train, axis=0)

        print(f"训练数据: {len(pe_train)} 样本")
        print(f"  PE特征维度: {pe_train.shape[1]}")
        print(f"  图像特征维度: {img_train.shape[1]}")

        np.save(os.path.join(output_dir, 'train_pe_features.npy'), pe_train)
        np.save(os.path.join(output_dir, 'train_img_features.npy'), img_train)
        np.save(os.path.join(output_dir, 'train_labels.npy'), label_train)

    # 处理测试数据
    print("\n处理测试数据...")
    all_pe_test = []
    all_img_test = []
    all_label_test = []
    test_count = 0

    for filepath in tqdm(test_files[:5]):
        result = process_pe_to_multimodal(filepath, output_dir,
                                          max_samples=test_samples - test_count)
        if result['count'] > 0:
            all_pe_test.append(result['pe_features'])
            all_img_test.append(result['img_features'])
            all_label_test.append(result['labels'])
            test_count += result['count']

        if test_count >= test_samples:
            break

    if all_pe_test:
        pe_test = np.concatenate(all_pe_test, axis=0)
        img_test = np.concatenate(all_img_test, axis=0)
        label_test = np.concatenate(all_label_test, axis=0)

        print(f"测试数据: {len(pe_test)} 样本")

        np.save(os.path.join(output_dir, 'test_pe_features.npy'), pe_test)
        np.save(os.path.join(output_dir, 'test_img_features.npy'), img_test)
        np.save(os.path.join(output_dir, 'test_labels.npy'), label_test)

    print("\n" + "="*60)
    print("多模态数据集准备完成!")
    print(f"输出目录: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/ember_pe')
    parser.add_argument('--output_dir', default='data/ember_pe/multimodal')
    parser.add_argument('--train_samples', type=int, default=500000)
    parser.add_argument('--test_samples', type=int, default=100000)

    args = parser.parse_args()
    prepare_multimodal_dataset(args.data_dir, args.output_dir,
                                args.train_samples, args.test_samples)