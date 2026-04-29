"""
配置文件 - 增强版 (支持特征组合优化与场景适配)
存储系统所有核心参数，确保全链路一致性
"""

import os
from pathlib import Path

# ================= 1. 基础路径配置 =================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = BASE_DIR / 'uploads'
LOGS_DIR = BASE_DIR / 'logs'

# 新增：专门存放带注意力机制模型的输出目录
OUTPUT_DIR_OPTIMIZED = BASE_DIR / 'outputs_optimized'

# 确保所有目录存在
for dir_path in [DATA_DIR, UPLOAD_DIR, LOGS_DIR, OUTPUT_DIR_OPTIMIZED]:
    dir_path.mkdir(exist_ok=True)

# ================= 2. 特征逻辑分组配置 (创新点核心) =================
# 这里的定义必须与 EMBER 官方 features.py 的特征顺序完全一致
# EMBER v2 2381维特征的官方分组标准 (ember_official/ember/features.py):
# - ByteHistogram: 256维 [0:256]
# - ByteEntropyHistogram: 256维 [256:512]
# - StringExtractor: 104维 [512:616] (1+1+1+96+1+1+1+1+1 = 104)
# - GeneralFileInfo: 10维
# - HeaderFileInfo: 62维
# - SectionInfo: 255维 (5+50+50+50+50+50)
# - ImportsInfo: 1280维 (256+1024)
# - ExportsInfo: 128维
# - DataDirectories: 30维 (15*2)
# Total: 256+256+104+10+62+255+1280+128+30 = 2381
FEATURE_GROUPS_CONFIG = {
    'group_names': ['字节直方图', '字节熵分布', '字符串特征', '通用元数据'],
    'group_dims': {
        'histogram': 256,       # ByteHistogram [0:256]
        'byte_entropy': 256,    # ByteEntropyHistogram [256:512]
        'strings': 104,         # StringExtractor [512:616]
        'general': 1765         # [616:2381] - GeneralFileInfo+HeaderFileInfo+SectionInfo+ImportsInfo+ExportsInfo+DataDirectories
    },
    'total_dim': 2381
}

# ================= 3. 模型配置 =================
MODEL_CONFIG = {
    'model_type': 'optimized', # 默认使用带注意力的模型
    'num_classes': 2,
    'embed_dim': 128,          # 注意力层的中间嵌入维度
    'dropout': 0.3,
    'learning_rate': 0.001,
    'weight_decay': 1e-5
}

# ================= 4. 训练与场景适配配置 =================
TRAINING_CONFIG = {
    'epochs': 20,
    'batch_size': 1024,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'patience': 5,             # 早停机制
    'save_history': True       # 是否保存注意力权重的演变历史
}

# ================= 5. 可视化与评估配置 =================
EVAL_CONFIG = {
    'plot_attention_heatmap': True,   # 是否生成场景适配热力图
    'plot_confusion_matrix': True,
    'plot_roc_curve': True,
    'class_names': ['Benign', 'Malicious']
}

# ================= 6. 路径映射表 =================
MODEL_PATHS = {
    'optimized': OUTPUT_DIR_OPTIMIZED / 'best_optimized_model.pth',
    'ember_base': BASE_DIR / 'outputs_ember' / 'best_model.pth'
}

def get_config(section: str):
    """获取指定部分的配置"""
    configs = {
        'data_groups': FEATURE_GROUPS_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'eval': EVAL_CONFIG,
        'paths': MODEL_PATHS
    }
    return configs.get(section, {})

if __name__ == "__main__":
    print("🔧 恶意软件检测系统配置 (组合优化版)")
    print(f"核心特征维度: {FEATURE_GROUPS_CONFIG['total_dim']}")
    print(f"注意力分组: {', '.join(FEATURE_GROUPS_CONFIG['group_names'])}")