"""
创新点 3: 轻量化集成优化训练脚本
实现特征筛选 + 权重剪枝 + 知识蒸馏的完整训练流程
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import json
import time
import copy

sys.path.insert(0, os.path.dirname(__file__))

from src.core.models import OptimizedMalwareDetector, EmberMLP
from src.core.heterogeneous_ensemble import HeterogeneousEnsembleDetector
from src.core.lightweight_optimization import (
    FeatureSelector,
    WeightPruner,
    KnowledgeDistiller,
    LightweightOptimizer,
    benchmark_inference
)


def load_ember_data(data_dir, sample_ratio=0.1):
    """加载EMBER数据集"""
    print(f"加载EMBER数据集 from {data_dir}")

    train_features = np.load(os.path.join(data_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    test_features = np.load(os.path.join(data_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))

    print(f"原始数据: train={len(train_features)}, test={len(test_features)}")

    if sample_ratio < 1.0:
        train_size = int(len(train_features) * sample_ratio)
        np.random.seed(42)
        indices = np.random.choice(len(train_features), train_size, replace=False)
        train_features = train_features[indices]
        train_labels = train_labels[indices]
        print(f"采样后: train={len(train_features)} ({sample_ratio*100:.0f}%)")

    return train_features, train_labels, test_features, test_labels


def create_lightweight_model(original_model, input_dim: int, num_classes: int = 2,
                              compression_ratio: float = 0.5) -> nn.Module:
    """
    创建轻量版模型

    Args:
        original_model: 原始模型
        input_dim: 输入维度
        num_classes: 类别数
        compression_ratio: 压缩比例

    Returns:
        轻量模型
    """
    # 计算目标参数量
    original_params = sum(p.numel() for p in original_model.parameters())
    target_params = int(original_params * compression_ratio)

    print(f"\n创建轻量模型:")
    print(f"  原始参数: {original_params:,}")
    print(f"  目标参数: {target_params:,}")

    # 创建更小的模型 (固定架构，保证足够的容量)
    # 使用多层但每层维度递减，保持表达能力
    hidden_dim1 = max(128, int(256 * compression_ratio))
    hidden_dim2 = max(64, hidden_dim1 // 2)

    lightweight_model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        nn.BatchNorm1d(hidden_dim1),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.BatchNorm1d(hidden_dim2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim2, num_classes)
    )

    lightweight_params = sum(p.numel() for p in lightweight_model.parameters())
    print(f"  实际参数: {lightweight_params:,} ({lightweight_params/original_params:.2%})")

    return lightweight_model


def train_with_feature_selection(model, train_features, train_labels,
                                  test_features, test_labels, config, device):
    """
    特征筛选训练流程
    """
    print("\n" + "=" * 60)
    print("特征筛选训练")
    print("=" * 60)

    # 特征筛选
    selector = FeatureSelector(method=config['method'])
    selected_indices = selector.fit(model, train_features, train_labels,
                                     top_k=config['top_k'], device=device)

    # 转换数据
    train_selected = selector.transform(train_features)
    test_selected = selector.transform(test_features)

    print(f"筛选后维度: {train_selected.shape[1]}")

    # 创建新模型 (适应新的输入维度)
    new_model = EmberMLP(input_dim=config['top_k'], num_classes=2)
    new_model = new_model.to(device)

    # 训练
    train_dataset = TensorDataset(torch.FloatTensor(train_selected), torch.LongTensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset = TensorDataset(torch.FloatTensor(test_selected), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(new_model.parameters(), lr=0.001)

    best_acc = 0

    for epoch in range(1, config['epochs'] + 1):
        new_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = new_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # 评估
        new_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                preds = torch.argmax(new_model(batch_x), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Acc={acc:.4f}")

    print(f"\n最佳准确率: {best_acc:.4f}")

    return new_model, selector, best_acc


def train_with_pruning(model, train_loader, test_loader, config, device):
    """
    权重剪枝训练流程
    """
    print("\n" + "=" * 60)
    print("权重剪枝训练")
    print("=" * 60)

    # 先训练原始模型
    print("\n训练原始模型...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, config['pretrain_epochs'] + 1):
        model.train()
        for batch_x, batch_y in tqdm(train_loader, desc=f"Pretrain Epoch {epoch}", leave=False):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 剪枝
    pruner = WeightPruner(prune_type=config['prune_type'], prune_ratio=config['prune_ratio'])
    pruned_model = pruner.prune(model)

    # 微调剪枝后的模型
    print("\n微调剪枝模型...")
    optimizer = optim.Adam(pruned_model.parameters(), lr=0.0001)

    best_acc = 0

    for epoch in range(1, config['finetune_epochs'] + 1):
        pruned_model.train()
        for batch_x, batch_y in tqdm(train_loader, desc=f"Finetune Epoch {epoch}", leave=False):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)

            optimizer.zero_grad()
            outputs = pruned_model(batch_x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # 评估
        pruned_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.long().to(device)

                outputs = pruned_model(batch_x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Acc={acc:.4f}")

    print(f"\n最佳准确率: {best_acc:.4f}")

    return pruned_model, pruner, best_acc


def train_with_distillation(teacher_model, train_features, train_labels,
                              test_features, test_labels, config, device):
    """
    知识蒸馏训练流程
    """
    print("\n" + "=" * 60)
    print("知识蒸馏训练")
    print("=" * 60)

    input_dim = train_features.shape[1]

    # 创建轻量学生模型
    student_model = create_lightweight_model(teacher_model, input_dim,
                                               num_classes=2,
                                               compression_ratio=config['compression_ratio'])

    # 设置蒸馏器
    distiller = KnowledgeDistiller(temperature=config['temperature'],
                                     alpha=config['alpha'])
    distiller.setup(teacher_model, student_model, device=device)

    # 训练
    train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    optimizer = optim.Adam(student_model.parameters(), lr=config['lr'])

    best_acc = 0

    for epoch in range(1, config['epochs'] + 1):
        student_model.train()

        epoch_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Distill Epoch {epoch}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            loss = distiller.train_step(batch_x, batch_y, optimizer)
            epoch_loss += loss

        # 评估
        student_model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            batch_size = 500
            for i in range(0, len(test_features), batch_size):
                batch = torch.FloatTensor(test_features[i:i+batch_size]).to(device)
                outputs = student_model(batch)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(test_labels[i:i+batch_size])

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        if acc > best_acc:
            best_acc = acc
            # 保存最佳模型
            torch.save({
                'model_state_dict': student_model.state_dict(),
                'accuracy': acc,
                'compression_ratio': config['compression_ratio']
            }, os.path.join(config['output_dir'], 'best_distilled_model.pth'))

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={epoch_loss/len(train_loader):.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    print(f"\n最佳准确率: {best_acc:.4f}")

    return student_model, distiller, best_acc


def load_step_result(output_dir, step_name):
    """加载步骤结果"""
    path = os.path.join(output_dir, f'result_{step_name}.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        print(f"  [续训] {step_name} 已完成，跳过 (Acc: {result.get('accuracy', 'N/A')})")
        return result
    return None


def save_step_result(output_dir, step_name, result):
    """保存步骤结果"""
    path = os.path.join(output_dir, f'result_{step_name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [存档] {step_name} 已保存")


def main():
    parser = argparse.ArgumentParser(description='轻量化优化训练')
    parser.add_argument('--data_dir', type=str, default='./data/ember_pe/full_features')
    parser.add_argument('--output_dir', type=str, default='./outputs_lightweight')
    parser.add_argument('--model_path', type=str, default='./outputs_ember/best_optimized_model.pth')
    parser.add_argument('--sample_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')

    # 特征筛选配置
    parser.add_argument('--feature_selection', type=bool, default=True)
    parser.add_argument('--fs_method', type=str, default='statistical', choices=['attention_based', 'statistical', 'mutual_info', 'combined'])
    parser.add_argument('--fs_top_k', type=int, default=2000)
    parser.add_argument('--fs_epochs', type=int, default=30)

    # 剪枝配置
    parser.add_argument('--pruning', type=bool, default=False)
    parser.add_argument('--prune_type', type=str, default='structured', choices=['structured', 'unstructured', 'global'])
    parser.add_argument('--prune_ratio', type=float, default=0.3)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--finetune_epochs', type=int, default=20)

    # 蒸馏配置
    parser.add_argument('--distillation', type=bool, default=True)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--compression_ratio', type=float, default=0.6)
    parser.add_argument('--distill_epochs', type=int, default=50)
    parser.add_argument('--distill_batch_size', type=int, default=256)
    parser.add_argument('--distill_lr', type=float, default=0.001)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_features, train_labels, test_features, test_labels = load_ember_data(
        args.data_dir, args.sample_ratio
    )

    # 加载原始模型
    print(f"\n加载原始模型: {args.model_path}")
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        original_model = OptimizedMalwareDetector(num_classes=2)
        original_model.load_state_dict(checkpoint['model_state_dict'])
        original_model = original_model.to(device)
        original_model.eval()
        print(f"原始模型准确率: {checkpoint.get('accuracy', 'N/A')}")
    else:
        print("模型文件不存在，创建新模型")
        original_model = OptimizedMalwareDetector(num_classes=2)
        original_model = original_model.to(device)

    original_params = sum(p.numel() for p in original_model.parameters())

    results = {
        'original': {
            'params': original_params
        }
    }

    # ============ 特征筛选 (断点续训) ============
    if args.feature_selection:
        fs_result = load_step_result(args.output_dir, 'feature_selection')
        if fs_result is not None:
            results['feature_selection'] = fs_result
        else:
            fs_config = {
                'method': args.fs_method,
                'top_k': args.fs_top_k,
                'epochs': args.fs_epochs
            }

            fs_model, selector, fs_acc = train_with_feature_selection(
                original_model, train_features, train_labels,
                test_features, test_labels, fs_config, device
            )

            fs_params = sum(p.numel() for p in fs_model.parameters())
            fs_selected = selector.transform(test_features[:500])
            fs_metrics = benchmark_inference(fs_model, fs_selected, device=device)

            fs_result = {
                'params': fs_params,
                'accuracy': fs_acc,
                'metrics': fs_metrics,
                'feature_dim': args.fs_top_k,
                'compression': fs_params / original_params
            }
            results['feature_selection'] = fs_result
            save_step_result(args.output_dir, 'feature_selection', fs_result)

            torch.save({
                'model_state_dict': fs_model.state_dict(),
                'selected_indices': selector.selected_indices,
                'accuracy': fs_acc
            }, os.path.join(args.output_dir, 'feature_selected_model.pth'))

            selector_info = selector.get_selected_feature_info()
            with open(os.path.join(args.output_dir, 'feature_selection_info.json'), 'w') as f:
                json.dump(selector_info, f, indent=2)

    # ============ 权重剪枝 (断点续训) ============
    if args.pruning:
        prune_result = load_step_result(args.output_dir, 'pruning')
        if prune_result is not None:
            results['pruning'] = prune_result
        else:
            prune_config = {
                'prune_type': args.prune_type,
                'prune_ratio': args.prune_ratio,
                'pretrain_epochs': args.pretrain_epochs,
                'finetune_epochs': args.finetune_epochs
            }

            train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

            test_dataset = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            pruned_model, pruner, prune_acc = train_with_pruning(
                copy.deepcopy(original_model), train_loader, test_loader,
                prune_config, device
            )

            prune_params = sum(p.numel() for p in pruned_model.parameters())
            prune_metrics = benchmark_inference(pruned_model, test_features[:500], device=device)

            prune_result = {
                'params': prune_params,
                'accuracy': prune_acc,
                'metrics': prune_metrics,
                'prune_info': pruner.get_pruning_info()
            }
            results['pruning'] = prune_result
            save_step_result(args.output_dir, 'pruning', prune_result)

            torch.save({
                'model_state_dict': pruned_model.state_dict(),
                'accuracy': prune_acc
            }, os.path.join(args.output_dir, 'pruned_model.pth'))

    # ============ 知识蒸馏 (断点续训) ============
    if args.distillation:
        distill_result = load_step_result(args.output_dir, 'distillation')
        if distill_result is not None:
            results['distillation'] = distill_result
        else:
            distill_config = {
                'temperature': args.temperature,
                'alpha': args.alpha,
                'compression_ratio': args.compression_ratio,
                'epochs': args.distill_epochs,
                'batch_size': args.distill_batch_size,
                'lr': args.distill_lr,
                'output_dir': args.output_dir
            }

            student_model, distiller, distill_acc = train_with_distillation(
                original_model, train_features, train_labels,
                test_features, test_labels, distill_config, device
            )

            student_params = sum(p.numel() for p in student_model.parameters())
            student_metrics = benchmark_inference(student_model, test_features[:500], device=device)

            distill_result = {
                'params': student_params,
                'accuracy': distill_acc,
                'metrics': student_metrics,
                'compression': student_params / original_params
            }
            results['distillation'] = distill_result
            save_step_result(args.output_dir, 'distillation', distill_result)

    # ============ 结果汇总 ============
    print("\n" + "=" * 60)
    print("轻量化优化结果汇总")
    print("=" * 60)

    print("\n| 方法 | 参数量 | 准确率 | 压缩比例 |")
    print("|------|--------|--------|----------|")
    print(f"| 原始模型 | {original_params:,} | - | 100% |")

    for method, data in results.items():
        if method != 'original':
            acc = data.get('accuracy', 'N/A')
            compression = data.get('compression', 1.0)
            print(f"| {method} | {data['params']:,} | {acc:.4f} | {compression:.2%} |")

    # 保存完整结果
    # 过滤掉不可序列化的 metrics 对象
    save_results = {}
    for k, v in results.items():
        save_entry = {}
        for sk, sv in v.items():
            if sk == 'metrics' and isinstance(sv, dict):
                # 只保留可序列化的标量值
                save_entry[sk] = {mk: mv for mk, mv in sv.items() if isinstance(mv, (int, float, str))}
            else:
                save_entry[sk] = sv
        save_results[k] = save_entry

    with open(os.path.join(args.output_dir, 'lightweight_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n结果已保存至: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()