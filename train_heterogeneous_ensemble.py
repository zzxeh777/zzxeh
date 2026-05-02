"""
异构集成模型训练脚本
训练 MLP + CNN1D + Transformer 多异构集成模型
对比单模型 vs 集成模型性能
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

# 添加src路径
sys.path.insert(0, os.path.dirname(__file__))

from src.core.heterogeneous_ensemble import (
    HeterogeneousEnsembleDetector,
    LightweightHeterogeneousEnsemble,
    EmberTransformer
)
from src.core.models import EmberMLP, EmberCNN1D


def load_ember_data(data_dir, sample_ratio=0.1):
    """加载EMBER数据集"""
    print(f"加载EMBER数据集 from {data_dir}")

    train_features = np.load(os.path.join(data_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    test_features = np.load(os.path.join(data_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))

    print(f"原始数据: train={len(train_features)}, test={len(test_features)}")

    # 清洗非法值（EMBER特征中可能存在NaN/Inf）
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)

    # 特征标准化（z-score），防止Transformer因输入值过大而溢出NaN
    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    train_std[train_std == 0] = 1.0  # 避免除零
    train_features = (train_features - train_mean) / train_std
    test_features = (test_features - train_mean) / train_std
    print(f"特征标准化完成: mean≈{train_features.mean():.4f}, std≈{train_features.std():.4f}")

    # 采样
    if sample_ratio < 1.0:
        train_size = int(len(train_features) * sample_ratio)
        np.random.seed(42)
        indices = np.random.choice(len(train_features), train_size, replace=False)
        train_features = train_features[indices]
        train_labels = train_labels[indices]
        print(f"采样后: train={len(train_features)} ({sample_ratio*100:.0f}%)")

    # 统计
    train_malicious = np.sum(train_labels == 1)
    train_benign = np.sum(train_labels == 0)
    print(f"训练集分布: 良性={train_benign}, 恶意={train_malicious}")

    return train_features, train_labels, test_features, test_labels


def train_single_model(model, train_loader, test_loader, device, epochs=30, model_name="Model"):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"训练 {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_accuracy = 0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            # 处理集成模型返回多个值的情况
            if isinstance(outputs, tuple):
                outputs = outputs[0] if outputs[0].dim() == 2 else outputs[1]

            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_acc = train_correct / train_total

        # 评估
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.long().to(device)

                outputs = model(batch_x)

                # 处理集成模型返回多个值的情况
                if isinstance(outputs, tuple):
                    probs = outputs[0]  # fused_probs
                    preds = torch.argmax(probs, dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)

                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        test_acc = test_correct / test_total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        all_probs = np.array(all_probs)
        if all_probs.shape[1] == 2:
            # 安全处理：若概率含NaN，用0.5填充并告警
            if np.isnan(all_probs).any():
                nan_count = np.isnan(all_probs).sum()
                print(f"    [警告] 检测到 {nan_count} 个 NaN 概率值，已用0.5填充")
                all_probs = np.nan_to_num(all_probs, nan=0.5)
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = 0

        history.append({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })

        print(f"  Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
              f"F1={f1:.4f}, AUC={auc:.4f}")

        scheduler.step(test_acc)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch

    print(f"\n{model_name} 最佳结果:")
    print(f"  最佳准确率: {best_accuracy:.4f} (Epoch {best_epoch})")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    return {
        'model_name': model_name,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'final_metrics': {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        },
        'history': history
    }


def train_heterogeneous_ensemble(model, train_loader, test_loader, device, epochs=30):
    """训练异构集成模型（记录各分支性能）"""
    print(f"\n{'='*60}")
    print(f"训练 HeterogeneousEnsembleDetector (异构集成)")
    print(f"{'='*60}")

    model = model.to(device)

    # 融合层返回的是 log-probabilities，使用 NLLLoss
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_accuracy = 0
    best_epoch = 0
    history = []
    branch_weights_history = []

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)

            optimizer.zero_grad()
            fused_probs, fused_logits, branch_probs_dict = model(batch_x)

            # 使用logits计算损失（更稳定）
            loss = criterion(fused_logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(fused_probs, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_acc = train_correct / train_total

        # 评估
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        # 各分支统计
        branch_correct = {'mlp': 0, 'cnn': 0, 'transformer': 0}
        fusion_weights_list = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.long().to(device)

                fused_probs, fused_logits, branch_probs_dict = model(batch_x)

                # 融合预测
                preds = torch.argmax(fused_probs, dim=1)
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)

                # 各分支预测
                for name, probs in branch_probs_dict.items():
                    branch_pred = torch.argmax(probs, dim=1)
                    branch_correct[name] += (branch_pred == batch_y).sum().item()

                # 获取融合权重
                weights = model.get_fusion_weights(batch_x)
                fusion_weights_list.append(weights.cpu().numpy())

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(fused_probs.cpu().numpy())

        test_acc = test_correct / test_total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        all_probs = np.array(all_probs)
        if all_probs.shape[1] == 2:
            # 安全处理：若概率含NaN，用0.5填充并告警
            if np.isnan(all_probs).any():
                nan_count = np.isnan(all_probs).sum()
                print(f"    [警告] 检测到 {nan_count} 个 NaN 概率值，已用0.5填充")
                all_probs = np.nan_to_num(all_probs, nan=0.5)
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = 0

        # 计算各分支准确率
        branch_acc = {
            name: correct / test_total for name, correct in branch_correct.items()
        }

        # 平均融合权重 (使用nanmean忽略偶尔的NaN)
        avg_weights = np.nanmean(np.concatenate(fusion_weights_list, axis=0), axis=0)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'branch_accuracy': branch_acc,
            'fusion_weights': avg_weights.tolist()
        })

        branch_weights_history.append(avg_weights)

        print(f"  Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        print(f"    分支准确率: MLP={branch_acc['mlp']:.4f}, "
              f"CNN={branch_acc['cnn']:.4f}, Transformer={branch_acc['transformer']:.4f}")
        print(f"    融合权重: MLP={avg_weights[0]:.3f}, "
              f"CNN={avg_weights[1]:.3f}, Transformer={avg_weights[2]:.3f}")

        scheduler.step(test_acc)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch

    print(f"\n异构集成模型最佳结果:")
    print(f"  最佳准确率: {best_accuracy:.4f} (Epoch {best_epoch})")

    return {
        'model_name': 'HeterogeneousEnsemble',
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'final_metrics': {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        },
        'branch_accuracy': branch_acc,
        'avg_fusion_weights': avg_weights.tolist(),
        'history': history
    }


def load_checkpoint(output_dir, model_name):
    """加载已有的模型checkpoint和结果"""
    ckpt_path = os.path.join(output_dir, f'checkpoint_{model_name}.pth')
    result_path = os.path.join(output_dir, f'result_{model_name}.json')
    if os.path.exists(ckpt_path) and os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        print(f"  [续训] {model_name} 已有checkpoint，跳过训练 (Acc: {result['best_accuracy']:.4f})")
        return result
    return None


def save_checkpoint(output_dir, model, model_name, result):
    """保存模型checkpoint和结果"""
    ckpt_path = os.path.join(output_dir, f'checkpoint_{model_name}.pth')
    result_path = os.path.join(output_dir, f'result_{model_name}.json')
    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [存档] {model_name} checkpoint已保存")


def main():
    parser = argparse.ArgumentParser(description='异构集成模型训练')
    parser.add_argument('--data_dir', type=str, default='./data/ember_pe/full_features',
                        help='EMBER数据目录')
    parser.add_argument('--output_dir', type=str, default='./outputs_heterogeneous',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='采样比例')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'lightweight'], help='模型类型')
    parser.add_argument('--fusion_type', type=str, default='probability_attention',
                        choices=['probability_attention', 'gating', 'meta_attention'],
                        help='融合策略')
    parser.add_argument('--compare_single', type=bool, default=True,
                        help='是否对比单模型性能')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_features, train_labels, test_features, test_labels = load_ember_data(
        args.data_dir, args.sample_ratio
    )

    # 创建DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(train_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_features),
        torch.LongTensor(test_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"DataLoader: train={len(train_loader)} batches, test={len(test_loader)} batches")

    # 存储所有结果
    all_results = []

    # ============ 训练单模型对比 (支持断点续训) ============
    if args.compare_single:
        print("\n" + "=" * 60)
        print("训练单模型作为对比基线")
        print("=" * 60)

        # MLP
        mlp_result = load_checkpoint(args.output_dir, 'EmberMLP')
        if mlp_result is None:
            mlp_model = EmberMLP(input_dim=2381, num_classes=2)
            mlp_result = train_single_model(mlp_model, train_loader, test_loader, device,
                                            epochs=args.epochs, model_name="EmberMLP")
            save_checkpoint(args.output_dir, mlp_model, 'EmberMLP', mlp_result)
        all_results.append(mlp_result)

        # CNN1D
        cnn_result = load_checkpoint(args.output_dir, 'EmberCNN1D')
        if cnn_result is None:
            cnn_model = EmberCNN1D(input_dim=2381, num_classes=2)
            cnn_result = train_single_model(cnn_model, train_loader, test_loader, device,
                                            epochs=args.epochs, model_name="EmberCNN1D")
            save_checkpoint(args.output_dir, cnn_model, 'EmberCNN1D', cnn_result)
        all_results.append(cnn_result)

        # Transformer (单独训练)
        trans_result = load_checkpoint(args.output_dir, 'EmberTransformer')
        if trans_result is None:
            trans_model = EmberTransformer(input_dim=2381, num_classes=2)
            trans_result = train_single_model(trans_model, train_loader, test_loader, device,
                                              epochs=args.epochs, model_name="EmberTransformer")
            save_checkpoint(args.output_dir, trans_model, 'EmberTransformer', trans_result)
        all_results.append(trans_result)

    # ============ 训练异构集成模型 (支持断点续训) ============
    ensemble_result = load_checkpoint(args.output_dir, 'HeterogeneousEnsemble')
    if ensemble_result is None:
        print("\n" + "=" * 60)
        print("训练异构集成模型")
        print("=" * 60)

        if args.model_type == 'standard':
            ensemble_model = HeterogeneousEnsembleDetector(
                input_dim=2381,
                num_classes=2,
                fusion_type=args.fusion_type
            )
        else:
            ensemble_model = LightweightHeterogeneousEnsemble(
                input_dim=2381,
                num_classes=2
            )

        ensemble_result = train_heterogeneous_ensemble(
            ensemble_model, train_loader, test_loader, device, epochs=args.epochs
        )
        save_checkpoint(args.output_dir, ensemble_model, 'HeterogeneousEnsemble', ensemble_result)

        # 额外保存完整模型文件
        model_file = os.path.join(args.output_dir, 'best_heterogeneous_ensemble.pth')
        torch.save({
            'model_state_dict': ensemble_model.state_dict(),
            'model_type': args.model_type,
            'fusion_type': args.fusion_type,
            'best_accuracy': ensemble_result['best_accuracy'],
            'branch_accuracy': ensemble_result.get('branch_accuracy', {}),
            'avg_fusion_weights': ensemble_result.get('avg_fusion_weights', [])
        }, model_file)
        print(f"模型已保存至: {model_file}")
    all_results.append(ensemble_result)

    # ============ 保存结果 ============
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    # 打印对比表
    print("\n| 模型 | 最佳准确率 | Precision | Recall | F1 | AUC |")
    print("|------|-----------|-----------|---------|-----|-----|")
    for result in all_results:
        metrics = result['final_metrics']
        print(f"| {result['model_name']} | {result['best_accuracy']:.4f} | "
              f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
              f"{metrics['f1']:.4f} | {metrics['auc']:.4f} |")

    # 保存详细结果
    results_file = os.path.join(args.output_dir, 'heterogeneous_ensemble_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {results_file}")

    # 保存融合权重变化历史
    if 'history' in ensemble_result:
        weights_history = [h.get('fusion_weights', []) for h in ensemble_result['history']]
        weights_file = os.path.join(args.output_dir, 'fusion_weights_history.json')
        with open(weights_file, 'w') as f:
            json.dump(weights_history, f, indent=2)
        print(f"融合权重历史已保存至: {weights_file}")

    # 计算集成提升
    if args.compare_single and len(all_results) >= 4:
        single_best = max(all_results[:3], key=lambda x: x['best_accuracy'])
        ensemble_acc = ensemble_result['best_accuracy']
        improvement = ensemble_acc - single_best['best_accuracy']

        print("\n" + "=" * 60)
        print(f"集成提升分析")
        print("=" * 60)
        print(f"最佳单模型: {single_best['model_name']} ({single_best['best_accuracy']:.4f})")
        print(f"集成模型: {ensemble_acc:.4f}")
        print(f"提升幅度: {improvement:.4f} ({improvement*100:.2f}%)")
        print("=" * 60)


if __name__ == '__main__':
    main()