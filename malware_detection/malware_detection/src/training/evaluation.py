"""
模型评估模块 - 论文版增强版
支持多种可视化：ROC曲线、PR曲线、混淆矩阵、注意力权重分析、特征贡献度分析
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互模式，适合服务器运行
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

class ModelEvaluator:
    def __init__(self, model, device: str = 'cuda', class_names: List[str] = None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names or ["良性文件", "恶意软件"]
        # 特征组名称，对应 models.py 中的分组
        self.group_names = ['字节直方图', '字节熵分布', '字符串特征', '元数据信息']
        # 存储评估数据用于后续可视化
        self.eval_data = {}

    def evaluate(self, data_loader, is_optimized: bool = True) -> Dict:
        """
        增强评估函数：收集完整数据用于多种可视化
        """
        all_preds = []
        all_labels = []
        all_probs = []
        all_weights = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)

                if is_optimized and hasattr(self.model, 'forward'):
                    try:
                        # 尝试获取双输出
                        outputs = self.model(inputs)
                        if isinstance(outputs, tuple):
                            logits, weights = outputs
                            all_weights.append(weights.cpu().numpy())
                        else:
                            logits = outputs
                    except Exception as e:
                        logits = self.model(inputs)
                else:
                    logits = self.model(inputs)

                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(probs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        # 存储评估数据
        self.eval_data = {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'weights': np.concatenate(all_weights, axis=0) if all_weights else None
        }

        # 基础指标计算
        metrics = self._calculate_basic_metrics(all_labels, all_preds, all_probs)

        # 创新点评估：特征组贡献度分析
        if all_weights:
            metrics['attention_analysis'] = self._analyze_attention(all_weights, all_labels)

        return metrics

    def _analyze_attention(self, weights_list: List[np.ndarray], labels: List[int]) -> Dict:
        """
        分析不同类别下的注意力分布差异
        """
        weights = np.concatenate(weights_list, axis=0) if isinstance(weights_list, list) else weights_list
        labels = np.array(labels)

        analysis = {
            'by_class': {},
            'overall': {},
            'statistics': {}
        }

        # 按类别分析
        for i, name in enumerate(self.class_names):
            class_weights = weights[labels == i]
            if len(class_weights) > 0:
                avg_w = np.mean(class_weights, axis=0)
                std_w = np.std(class_weights, axis=0)
                analysis['by_class'][name] = {
                    'mean': {self.group_names[j]: float(avg_w[j]) for j in range(min(len(avg_w), len(self.group_names)))},
                    'std': {self.group_names[j]: float(std_w[j]) for j in range(min(len(std_w), len(self.group_names)))}
                }

        # 总体分析
        overall_avg = np.mean(weights, axis=0)
        analysis['overall'] = {self.group_names[j]: float(overall_avg[j]) for j in range(min(len(overall_avg), len(self.group_names)))}

        # 统计显著性分析（良性vs恶意差异）
        if len(self.class_names) == 2:
            benign_w = weights[labels == 0]
            malicious_w = weights[labels == 1]
            if len(benign_w) > 0 and len(malicious_w) > 0:
                diff = np.mean(malicious_w, axis=0) - np.mean(benign_w, axis=0)
                analysis['statistics']['class_difference'] = {
                    self.group_names[j]: float(diff[j]) for j in range(min(len(diff), len(self.group_names)))
                }

        return analysis

    def _calculate_basic_metrics(self, y_true, y_pred, y_prob) -> Dict:
        y_prob = np.array(y_prob)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        # AUC计算
        if len(self.class_names) == 2 and y_prob.shape[1] >= 2:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            metrics['ap'] = average_precision_score(y_true, y_prob[:, 1])  # PR曲线的AP值

        return metrics

    # ==================== 可视化方法 ====================

    def plot_all_visualizations(self, save_dir: str, model_name: str = "Model"):
        """
        生成所有论文所需的可视化图表
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. ROC曲线
        self.plot_roc_curve(os.path.join(save_dir, f'{model_name}_roc_curve.png'))

        # 2. PR曲线
        self.plot_pr_curve(os.path.join(save_dir, f'{model_name}_pr_curve.png'))

        # 3. 混淆矩阵
        self.plot_confusion_matrix(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))

        # 4. 注意力权重热力图
        if self.eval_data.get('weights') is not None:
            self.plot_attention_heatmap(os.path.join(save_dir, f'{model_name}_attention_heatmap.png'))
            self.plot_attention_comparison(os.path.join(save_dir, f'{model_name}_attention_comparison.png'))

        # 5. 特征贡献度柱状图
        if self.eval_data.get('weights') is not None:
            self.plot_feature_contribution(os.path.join(save_dir, f'{model_name}_feature_contribution.png'))

        print(f"所有可视化图表已保存到: {save_dir}")

    def plot_roc_curve(self, save_path: str):
        """
        绘制ROC曲线 - 论文核心图表
        """
        if len(self.class_names) != 2:
            print("ROC曲线仅适用于二分类问题")
            return

        y_true = self.eval_data['labels']
        y_prob = self.eval_data['probabilities']

        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#2E86AB', linewidth=2, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='#A23B72', linestyle='--', linewidth=1.5, label='随机分类器')

        # 填充ROC曲线下方区域
        plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')

        plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        plt.title('ROC曲线分析', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"ROC曲线已保存: {save_path}")

    def plot_pr_curve(self, save_path: str):
        """
        绘制PR曲线 - 论文补充图表
        """
        if len(self.class_names) != 2:
            print("PR曲线仅适用于二分类问题")
            return

        y_true = self.eval_data['labels']
        y_prob = self.eval_data['probabilities']

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])
        ap = average_precision_score(y_true, y_prob[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='#F18F01', linewidth=2, label=f'PR曲线 (AP = {ap:.4f})')

        # 基线（正类比例）
        baseline = np.sum(y_true) / len(y_true)
        plt.plot([0, 1], [baseline, baseline], color='#A23B72', linestyle='--', linewidth=1.5,
                 label=f'基线 (正类比例 = {baseline:.2f})')

        plt.fill_between(recall, precision, alpha=0.2, color='#F18F01')

        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title('精确率-召回率曲线', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"PR曲线已保存: {save_path}")

    def plot_confusion_matrix(self, save_path: str):
        """
        绘制混淆矩阵 - 论文核心图表
        """
        y_true = self.eval_data['labels']
        y_pred = self.eval_data['predictions']

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    cbar_kws={'label': '样本数量'})

        # 添加百分比标注
        total = cm.sum()
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center', va='center', fontsize=10, color='gray')

        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('实际类别', fontsize=12)
        plt.title('混淆矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存: {save_path}")

    def plot_attention_heatmap(self, save_path: str):
        """
        绘制注意力权重热力图 - 论文创新点核心图表
        """
        if self.eval_data.get('weights') is None:
            print("无注意力权重数据")
            return

        weights = self.eval_data['weights']
        labels = self.eval_data['labels']

        # 计算每个类别的平均权重
        class_avg_weights = []
        for i in range(len(self.class_names)):
            class_weights = weights[labels == i]
            if len(class_weights) > 0:
                class_avg_weights.append(np.mean(class_weights, axis=0))

        if not class_avg_weights:
            return

        matrix = np.array(class_avg_weights)

        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=self.group_names[:matrix.shape[1]],
                    yticklabels=self.class_names,
                    annot_kws={'size': 12},
                    cbar_kws={'label': '注意力权重'})

        plt.xlabel('特征组', fontsize=12)
        plt.ylabel('样本类别', fontsize=12)
        plt.title('特征组注意力权重分布\n（场景适配分析）', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"注意力热力图已保存: {save_path}")

    def plot_attention_comparison(self, save_path: str):
        """
        绘制良性vs恶意的注意力权重对比图 - 论文创新点分析
        """
        if self.eval_data.get('weights') is None or len(self.class_names) != 2:
            return

        weights = self.eval_data['weights']
        labels = self.eval_data['labels']

        benign_avg = np.mean(weights[labels == 0], axis=0)
        malicious_avg = np.mean(weights[labels == 1], axis=0)

        groups = self.group_names[:len(benign_avg)]
        x = np.arange(len(groups))
        width = 0.35

        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(x - width/2, benign_avg, width, label='良性文件', color='#28965A', alpha=0.8)
        bars2 = plt.bar(x + width/2, malicious_avg, width, label='恶意软件', color='#A23B72', alpha=0.8)

        # 添加数值标注
        for bar in bars1:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)

        plt.xlabel('特征组', fontsize=12)
        plt.ylabel('平均注意力权重', fontsize=12)
        plt.title('良性文件与恶意软件的注意力权重对比\n（揭示模型决策依据差异）', fontsize=14, fontweight='bold')
        plt.xticks(x, groups, fontsize=11)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"注意力对比图已保存: {save_path}")

    def plot_feature_contribution(self, save_path: str):
        """
        绘制特征贡献度柱状图 - 论文可解释性分析
        """
        if self.eval_data.get('weights') is None:
            return

        weights = self.eval_data['weights']
        overall_avg = np.mean(weights, axis=0)

        groups = self.group_names[:len(overall_avg)]

        plt.figure(figsize=(10, 6))
        colors = ['#2E86AB', '#F18F01', '#A23B72', '#28965A'][:len(groups)]
        bars = plt.bar(groups, overall_avg, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # 添加数值标注
        for bar, val in zip(bars, overall_avg):
            height = bar.get_height()
            plt.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

        # 排序标注（贡献度排名）
        sorted_indices = np.argsort(overall_avg)[::-1]
        for rank, idx in enumerate(sorted_indices):
            bars[idx].set_hatch(['///', '...', '\\\\\\', '+++'][rank % 4])

        plt.xlabel('特征组', fontsize=12)
        plt.ylabel('平均贡献度权重', fontsize=12)
        plt.title('各特征组对检测决策的贡献度分析', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"特征贡献度图已保存: {save_path}")

    def generate_report(self, metrics: Dict) -> str:
        """
        生成论文用的评估报告文本
        """
        report = []
        report.append("=" * 60)
        report.append("模型评估报告")
        report.append("=" * 60)

        # 基础指标
        report.append("\n【基础性能指标】")
        report.append(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        report.append(f"精确率 (Precision): {metrics['precision']:.4f}")
        report.append(f"召回率 (Recall): {metrics['recall']:.4f}")
        report.append(f"F1分数 (F1-Score): {metrics['f1']:.4f}")
        if 'auc' in metrics:
            report.append(f"AUC值: {metrics['auc']:.4f}")
            report.append(f"AP值 (PR曲线): {metrics['ap']:.4f}")

        # 混淆矩阵分析
        report.append("\n【混淆矩阵分析】")
        cm = metrics['confusion_matrix']
        if len(cm) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            report.append(f"真阴性 (TN): {tn} - 正确识别的良性文件")
            report.append(f"假阳性 (FP): {fp} - 误报的良性文件")
            report.append(f"假阴性 (FN): {fn} - 漏报的恶意软件")
            report.append(f"真阳性 (TP): {tp} - 正确识别的恶意软件")
            report.append(f"误报率: {fp/(fp+tn)*100:.2f}%")
            report.append(f"漏报率: {fn/(fn+tp)*100:.2f}%")

        # 注意力分析
        if 'attention_analysis' in metrics:
            report.append("\n【特征组注意力分析】")
            analysis = metrics['attention_analysis']
            if 'overall' in analysis:
                report.append("总体特征贡献度:")
                for group, weight in analysis['overall'].items():
                    report.append(f"  - {group}: {weight:.4f}")

            if 'by_class' in analysis:
                report.append("\n按类别分析:")
                for class_name, data in analysis['by_class'].items():
                    report.append(f"  {class_name}:")
                    for group, weight in data['mean'].items():
                        report.append(f"    - {group}: {weight:.4f}")

        report.append("=" * 60)

        return "\n".join(report)


class MultiModelComparator:
    """
    多模型对比评估器 - 用于论文的对比实验
    """
    def __init__(self, models: Dict[str, torch.nn.Module], device: str = 'cuda'):
        self.models = models
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results = {}

    def compare_all(self, data_loader, is_optimized_map: Dict[str, bool] = None) -> Dict:
        """
        对比所有模型的性能
        """
        if is_optimized_map is None:
            is_optimized_map = {name: False for name in self.models}

        for name, model in self.models.items():
            evaluator = ModelEvaluator(model, self.device)
            self.results[name] = evaluator.evaluate(data_loader, is_optimized_map.get(name, False))

        return self.results

    def plot_comparison_bar(self, save_path: str, metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'auc']):
        """
        绘制多模型性能对比柱状图
        """
        model_names = list(self.results.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        plt.figure(figsize=(12, 7))

        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

        for i, (name, color) in enumerate(zip(model_names, colors)):
            values = [self.results[name].get(m, 0) for m in metrics]
            bars = plt.bar(x + i * width, values, width, label=name, color=color, alpha=0.8)

            # 数值标注
            for bar, val in zip(bars, values):
                plt.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9)

        plt.xlabel('评估指标', fontsize=12)
        plt.ylabel('指标值', fontsize=12)
        plt.title('多模型性能对比', fontsize=14, fontweight='bold')
        plt.xticks(x + width * (len(model_names) - 1) / 2, metrics, fontsize=11)
        plt.legend(loc='upper right', fontsize=10)
        plt.ylim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"模型对比图已保存: {save_path}")

    def plot_roc_comparison(self, save_path: str):
        """
        绘制多模型ROC曲线对比图
        """
        plt.figure(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))

        for (name, result), color in zip(self.results.items(), colors):
            if 'roc_data' in result:
                fpr, tpr = result['roc_data']
                auc = result.get('auc', 0)
                plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机分类器')

        plt.xlabel('假阳性率 (FPR)', fontsize=12)
        plt.ylabel('真阳性率 (TPR)', fontsize=12)
        plt.title('多模型ROC曲线对比', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def generate_comparison_table(self) -> str:
        """
        生成论文用的对比表格（Markdown格式）
        """
        lines = ["| 模型 | 准确率 | 精确率 | 召回率 | F1 | AUC |"]
        lines.append("|------|--------|--------|--------|-----|------|")

        for name, result in self.results.items():
            acc = result.get('accuracy', 0)
            prec = result.get('precision', 0)
            rec = result.get('recall', 0)
            f1 = result.get('f1', 0)
            auc = result.get('auc', 0)
            lines.append(f"| {name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} |")

        return "\n".join(lines)