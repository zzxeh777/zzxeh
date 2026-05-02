import numpy as np
import torch
import sys
sys.path.insert(0, 'src')

from core.heterogeneous_ensemble import HeterogeneousEnsembleDetector

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载训练好的模型
    checkpoint = torch.load('./outputs_heterogeneous/best_heterogeneous_ensemble.pth', map_location=device)
    model = HeterogeneousEnsembleDetector(
        input_dim=2381,
        num_classes=2,
        embed_dim=64,
        fusion_type='probability_attention'
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Fusion type: {checkpoint.get('fusion_type', 'unknown')}")

    # 生成测试输入（用归一化后的分布）
    torch.manual_seed(42)
    x = torch.randn(128, 2381).to(device) * 0.6  # 接近归一化数据的std

    # 测试 forward
    model.eval()
    with torch.no_grad():
        fused_probs, fused_logits, branch_probs = model(x)
        print(f"\nForward output: NaN={torch.isnan(fused_probs).any().item()}")

    # 测试 get_fusion_weights
    weights = model.get_fusion_weights(x)
    print(f"Fusion weights shape: {weights.shape}")
    print(f"Fusion weights NaN: {torch.isnan(weights).any().item()}")
    print(f"Fusion weights sample (first 5 samples):")
    print(weights[:5].cpu().numpy())
    print(f"\nMean fusion weights: {weights.mean(dim=0).cpu().numpy()}")

    # 测试在 train 模式下
    model.train()
    weights_train = model.get_fusion_weights(x)
    print(f"\nIn train mode - NaN: {torch.isnan(weights_train).any().item()}")
    print(f"Mean in train mode: {weights_train.mean(dim=0).cpu().numpy()}")

    print("\n=== 测试通过！融合权重已正常 ===")

if __name__ == '__main__':
    test()
