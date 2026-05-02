import numpy as np
import torch
import sys
import os
sys.path.insert(0, 'src')

from core.stacking_ensemble import DLMLStackingEnsemble, MetaFeatureBuilder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 加载数据
print("加载数据...")
test_features = np.load('./data/ember_pe/full_features/test_features.npy')
test_labels = np.load('./data/ember_pe/full_features/test_labels.npy')

# 应用相同的 z-score 归一化
train_features = np.load('./data/ember_pe/full_features/train_features.npy')
train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
train_std[train_std == 0] = 1.0
test_features = (test_features - train_mean) / train_std

# 加载 meta 特征
print("加载 Meta 特征...")
test_meta = np.load('./outputs_stacking/test_meta.npy')
print(f"Test meta shape: {test_meta.shape}")

# 加载 XGBoost 元分类器
print("加载 XGBoost 元分类器...")
import joblib
xgb_model = joblib.load('./outputs_stacking/xgboost_meta.pkl')

# 预测
print("预测中...")
preds = xgb_model.predict(test_meta)
probs = xgb_model.predict_proba(test_meta)

# 计算指标
acc = accuracy_score(test_labels, preds)
prec = precision_score(test_labels, preds, zero_division=0)
rec = recall_score(test_labels, preds, zero_division=0)
f1 = f1_score(test_labels, preds, zero_division=0)
auc = roc_auc_score(test_labels, probs[:, 1])

print("\n" + "="*60)
print("Stacking Ensemble 最终结果")
print("="*60)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("="*60)

# 对比各 Base Learner
print("\n对比所有模型:")
print(f"  MLP:        Acc=0.9327, AUC=0.9789")
print(f"  CNN1D:      Acc=0.9196, AUC=0.9791")
print(f"  Transformer: Acc=0.9153, AUC=0.9770")
print(f"  Attention:  Acc=0.9381, AUC=0.9828")
print(f"  HeterogeneousEnsemble: Acc=0.9299, AUC=0.9847")
print(f"  StackingEnsemble:      Acc={acc:.4f}, AUC={auc:.4f}")

# 特征重要性
importance = xgb_model.feature_importances_
top_idx = np.argsort(importance)[-10:][::-1]
print("\nTop 10 重要特征:")
for i, idx in enumerate(top_idx):
    print(f"  特征 {idx}: {importance[idx]:.4f}")
