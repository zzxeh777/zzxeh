"""
实验可视化模块
生成美观的论文/PPT级别图表
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
import numpy as np
import json
import os
from pathlib import Path

# 设置中文字体和美观样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# 美观配色方案
COLORS = {
    'primary': '#2E86AB',      # 主色蓝
    'secondary': '#A23B72',    # 辅色紫红
    'success': '#28A745',      # 成功绿
    'warning': '#FFC107',      # 警告黄
    'danger': '#DC3545',       # 危险红
    'info': '#17A2B8',         # 信息蓝
    'light': '#F8F9FA',        # 浅灰
    'dark': '#343A40',         # 深灰
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95C623', '#5C4D7D']
}


def plot_training_curves(history_path, output_path, title='训练过程'):
    """绘制训练曲线（Loss和Accuracy）"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss曲线
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], color=COLORS['primary'],
                 linewidth=2, marker='o', markersize=4, label='训练损失')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor(COLORS['light'])

    # Accuracy曲线
    axes[1].plot(epochs, history['val_acc'], color=COLORS['success'],
                 linewidth=2, marker='s', markersize=4, label='验证准确率')
    if 'test_acc' in history and isinstance(history['test_acc'], (int, float)):
        axes[1].axhline(y=history['test_acc'], color=COLORS['danger'],
                       linestyle='--', linewidth=2, label=f'测试准确率: {history["test_acc"]:.2f}%')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('准确率曲线', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor(COLORS['light'])

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_comparison_bar(data, labels, title, output_path, ylabel='准确率 (%)'):
    """绘制对比实验柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(labels))
    bars = ax.bar(x, data, color=COLORS['palette'][:len(labels)],
                  edgecolor='white', linewidth=2, width=0.6)

    # 添加数值标签
    for bar, val in zip(bars, data):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(data) * 1.15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor(COLORS['light'])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_ablation_experiment(results, output_path, title='消融实验结果'):
    """绘制消融实验对比图"""
    fig, ax = plt.subplots(figsize=(12, 7))

    baseline = results['baseline']
    ablations = results['ablations']

    # 数据准备
    labels = ['完整模型'] + [f'去除 {k}' for k in ablations.keys()]
    values = [baseline] + list(ablations.values())
    drops = [0] + [baseline - v for v in ablations.values()]

    x = range(len(labels))
    bars = ax.bar(x, values, color=[COLORS['primary']] + COLORS['palette'][1:len(labels)],
                  edgecolor='white', linewidth=2, width=0.7)

    # 添加下降标注
    for i, (bar, val, drop) in enumerate(zip(bars, values, drops)):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        if drop > 0:
            ax.annotate(f'↓{drop:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height/2),
                        ha='center', va='center',
                        fontsize=10, color=COLORS['danger'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, baseline * 1.15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor(COLORS['light'])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_feature_importance(importance_dict, output_path, title='特征重要性分析'):
    """绘制特征重要性图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    features = list(importance_dict.keys())
    values = list(importance_dict.values())

    # 水平柱状图
    y_pos = range(len(features))
    bars = ax.barh(y_pos, values, color=COLORS['palette'][:len(features)],
                   edgecolor='white', linewidth=2, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('重要性得分', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_facecolor(COLORS['light'])

    # 添加数值标签
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.annotate(f'{val:.2f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_confusion_matrix(cm, output_path, title='混淆矩阵'):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['良性', '恶意'],
                yticklabels=['良性', '恶意'],
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})

    ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
    ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_roc_curve(fpr, tpr, auc_score, output_path, title='ROC曲线'):
    """绘制ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2.5,
            label=f'ROC曲线 (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color=COLORS['secondary'], linestyle='--',
            linewidth=2, label='随机猜测')

    ax.fill_between(fpr, tpr, alpha=0.3, color=COLORS['primary'])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
    ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLORS['light'])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_model_comparison_table(data, output_path, title='模型性能对比'):
    """绘制模型对比表格图"""
    fig, ax = plt.subplots(figsize=(12, len(data) * 0.5 + 2))
    ax.axis('off')

    # 表格数据
    col_labels = ['模型', '测试准确率', '验证准确率', '特征维度', '训练时间']
    row_labels = [d['name'] for d in data]
    cell_data = [[d['test_acc'], d['val_acc'], d['dim'], d['time']] for d in data]

    # 创建表格
    table = ax.table(cellText=[[d['name']] + row for d, row in zip(data, cell_data)],
                      colLabels=col_labels,
                      loc='center',
                      cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 设置数据行样式
    for i in range(1, len(data) + 1):
        for j in range(len(col_labels)):
            if j == 0:
                table[(i, j)].set_facecolor(COLORS['light'])
            else:
                table[(i, j)].set_facecolor('white')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_multi_training_curves(histories, output_path, title='多模型训练对比'):
    """绘制多个模型的训练曲线对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = COLORS['palette']

    # Loss对比
    for i, (name, history) in enumerate(histories.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], color=colors[i],
                     linewidth=2, label=name, alpha=0.8)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练损失对比', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor(COLORS['light'])

    # Accuracy对比
    for i, (name, history) in enumerate(histories.items()):
        epochs = range(1, len(history['val_acc']) + 1)
        axes[1].plot(epochs, history['val_acc'], color=colors[i],
                     linewidth=2, label=name, alpha=0.8)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('验证准确率对比', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor(COLORS['light'])

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_attention_weights(weights, output_path, title='注意力权重分布'):
    """绘制注意力权重分布图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ['histogram', 'byte_entropy', 'strings', 'general']
    colors = COLORS['palette'][:4]

    bars = ax.bar(labels, weights, color=colors, edgecolor='white',
                  linewidth=2, width=0.6)

    for bar, val in zip(bars, weights):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_xlabel('特征组', fontsize=12)
    ax.set_ylabel('注意力权重', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(weights) * 1.2)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor(COLORS['light'])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def generate_all_visualizations(experiments_dir, output_dir):
    """生成所有实验可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("生成可视化图表")
    print("="*60)

    # 1. 训练曲线
    for exp_name in os.listdir(experiments_dir):
        exp_dir = os.path.join(experiments_dir, exp_name)
        history_path = os.path.join(exp_dir, 'training_history.json')
        if os.path.exists(history_path):
            plot_training_curves(history_path,
                                  os.path.join(output_dir, f'{exp_name}_training.png'),
                                  title=f'{exp_name} 训练过程')

    print("\n可视化图表生成完成!")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    # 测试
    generate_all_visualizations('./experiments', './docs/figures')