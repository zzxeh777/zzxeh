"""
改进版轻量化训练：专注于知识蒸馏
- 更小的学生模型架构
- 更合理的超参数
- 数据归一化
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

sys.path.insert(0, os.path.dirname(__file__))
from src.core.models import OptimizedMalwareDetector


class SmallMLP(nn.Module):
    """小型MLP学生模型"""
    def __init__(self, input_dim=2381, num_classes=2):
        super(SmallMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(24, num_classes)
        )
    def forward(self, x):
        return self.model(x)


class MediumMLP(nn.Module):
    """中型MLP学生模型（保留更多表达能力）"""
    def __init__(self, input_dim=2381, num_classes=2):
        super(MediumMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, num_classes)
        )
    def forward(self, x):
        return self.model(x)


def load_ember_data(data_dir, sample_ratio=0.2):
    """加载EMBER数据集（带归一化）"""
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

    # z-score 归一化
    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    train_std[train_std == 0] = 1.0
    train_features = (train_features - train_mean) / train_std
    test_features = (test_features - train_mean) / train_std
    print(f"归一化后: mean={train_features.mean():.4f}, std={train_features.std():.4f}")

    return train_features, train_labels, test_features, test_labels


def distill(teacher, student, train_loader, test_features, test_labels,
            device, temperature=2.0, alpha=0.3, epochs=50, lr=0.001):
    """知识蒸馏训练"""
    print("\n" + "="*60)
    print("知识蒸馏训练")
    print("="*60)

    teacher.eval()
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_acc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 教师预测（无梯度）
            with torch.no_grad():
                teacher_out = teacher(batch_x)
                if isinstance(teacher_out, tuple):
                    teacher_out = teacher_out[0]

            # 学生预测
            student_out = student(batch_x)

            # 软标签损失 (KL散度)
            soft_loss = nn.functional.kl_div(
                nn.functional.log_softmax(student_out / temperature, dim=1),
                nn.functional.softmax(teacher_out / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            # 硬标签损失
            hard_loss = nn.functional.cross_entropy(student_out, batch_y)

            # 组合损失
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # 评估
        student.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            batch_size = 1000
            for i in range(0, len(test_features), batch_size):
                batch = torch.FloatTensor(test_features[i:i+batch_size]).to(device)
                out = student(batch)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # NaN保护
        if np.isnan(all_probs).any():
            all_probs = np.nan_to_num(all_probs, nan=0.5)

        acc = accuracy_score(test_labels, all_preds)
        f1 = f1_score(test_labels, all_preds, zero_division=0)
        auc = roc_auc_score(test_labels, all_probs[:, 1])

        scheduler.step(1 - acc)

        if acc > best_acc:
            best_acc = acc
            best_state = student.state_dict().copy()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={epoch_loss/len(train_loader):.4f}, Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # 加载最佳模型
    if best_state is not None:
        student.load_state_dict(best_state)

    print(f"\n最佳准确率: {best_acc:.4f}")
    return student, best_acc


def evaluate_model(model, features, labels, device, model_name="Model"):
    """评估模型"""
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(features), 1024):
            batch = torch.FloatTensor(features[i:i+1024]).to(device)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    if np.isnan(all_probs).any():
        all_probs = np.nan_to_num(all_probs, nan=0.5)

    return {
        'model_name': model_name,
        'accuracy': accuracy_score(labels, all_preds),
        'precision': precision_score(labels, all_preds, zero_division=0),
        'recall': recall_score(labels, all_preds, zero_division=0),
        'f1': f1_score(labels, all_preds, zero_division=0),
        'auc': roc_auc_score(labels, all_probs[:, 1])
    }


def benchmark(model, features, device):
    """推理性能测试"""
    model.eval()
    model = model.to(device)

    batch_size = 128
    times = []

    # 预热
    warmup = torch.FloatTensor(features[:batch_size]).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(warmup)

    # 测试
    with torch.no_grad():
        for i in range(50):
            batch = torch.FloatTensor(features[i*batch_size:(i+1)*batch_size]).to(device)
            start = time.perf_counter()
            _ = model(batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append(end - start)

    avg_time = np.mean(times) * 1000
    throughput = batch_size / np.mean(times)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024*1024)

    return {
        'avg_time_ms': avg_time,
        'throughput': throughput,
        'model_size_mb': param_size,
        'params': sum(p.numel() for p in model.parameters())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/ember_pe/full_features')
    parser.add_argument('--output_dir', type=str, default='./outputs_lightweight_improved')
    parser.add_argument('--model_path', type=str, default='./outputs_ember_fixed/best_optimized_model.pth')
    parser.add_argument('--sample_ratio', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--student_type', type=str, default='small', choices=['small', 'medium'])
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_features, train_labels, test_features, test_labels = load_ember_data(
        args.data_dir, args.sample_ratio
    )

    # 加载教师模型
    print(f"\n加载教师模型: {args.model_path}")
    teacher = OptimizedMalwareDetector(num_classes=2)
    checkpoint = torch.load(args.model_path, map_location=device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher = teacher.to(device)
    teacher.eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"教师模型参数: {teacher_params:,}")

    # 评估教师模型
    teacher_metrics = evaluate_model(teacher, test_features, test_labels, device, "Teacher")
    print(f"\n教师模型: Acc={teacher_metrics['accuracy']:.4f}, F1={teacher_metrics['f1']:.4f}, AUC={teacher_metrics['auc']:.4f}")

    # 创建学生模型
    if args.student_type == 'small':
        student = SmallMLP(input_dim=2381, num_classes=2)
    else:
        student = MediumMLP(input_dim=2381, num_classes=2)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"\n学生模型参数: {student_params:,}")
    print(f"压缩比例: {student_params/teacher_params:.2%}")

    # 蒸馏训练
    train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    student, best_acc = distill(
        teacher, student, train_loader, test_features, test_labels,
        device, temperature=args.temperature, alpha=args.alpha,
        epochs=args.epochs, lr=args.lr
    )

    # 评估学生模型
    student_metrics = evaluate_model(student, test_features, test_labels, device, "Student")
    perf = benchmark(student, test_features, device)

    print("\n" + "="*60)
    print("最终结果对比")
    print("="*60)
    print(f"{'模型':<15} {'参数量':<10} {'准确率':<8} {'F1':<8} {'AUC':<8}")
    print("-"*60)
    print(f"{'Teacher':<15} {teacher_params:<10,} {teacher_metrics['accuracy']:<8.4f} {teacher_metrics['f1']:<8.4f} {teacher_metrics['auc']:<8.4f}")
    print(f"{'Student':<15} {student_params:<10,} {student_metrics['accuracy']:<8.4f} {student_metrics['f1']:<8.4f} {student_metrics['auc']:<8.4f}")
    print("="*60)
    print(f"推理时间: {perf['avg_time_ms']:.2f} ms")
    print(f"吞吐量: {perf['throughput']:.0f} samples/sec")
    print(f"模型大小: {perf['model_size_mb']:.2f} MB")

    # 保存
    torch.save({
        'model_state_dict': student.state_dict(),
        'accuracy': student_metrics['accuracy'],
        'metrics': student_metrics,
        'performance': perf,
        'config': vars(args)
    }, os.path.join(args.output_dir, 'best_distilled_student.pth'))

    # 保存对比结果
    results = {
        'teacher': {**teacher_metrics, 'params': teacher_params},
        'student': {**student_metrics, 'params': student_params, **perf},
        'compression_ratio': student_params / teacher_params
    }
    with open(os.path.join(args.output_dir, 'distillation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] 结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
