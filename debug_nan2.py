import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from core.heterogeneous_ensemble import EmberTransformer

def test_conditions(model, x, name):
    print(f"\n--- {name} ---")
    # eval + no_grad
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(f"  eval + no_grad: NaN={torch.isnan(out).any().item()}")

    # eval + grad enabled
    model.eval()
    out = model(x)
    print(f"  eval + grad:    NaN={torch.isnan(out).any().item()}")

    # train + no_grad
    model.train()
    with torch.no_grad():
        out = model(x)
        print(f"  train + no_grad: NaN={torch.isnan(out).any().item()}")

    # train + grad enabled
    model.train()
    out = model(x)
    print(f"  train + grad:    NaN={torch.isnan(out).any().item()}")

    # Check intermediate layers in eval+no_grad
    model.eval()
    with torch.no_grad():
        batch_size = x.shape[0]
        pad_size = model.num_patches * model.patch_size - model.input_dim
        if pad_size > 0:
            x_pad = nn.functional.pad(x, (0, pad_size))
        else:
            x_pad = x
        x_view = x_pad.view(batch_size, model.num_patches, model.patch_size)
        x_embed = model.patch_embed(x_view)
        print(f"  patch_embed: NaN={torch.isnan(x_embed).any().item()}, "
              f"min={x_embed.min().item():.4f}, max={x_embed.max().item():.4f}")

        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        x_cat = torch.cat([cls_tokens, x_embed], dim=1)
        print(f"  after cat:   NaN={torch.isnan(x_cat).any().item()}")

        x_pos = x_cat + model.pos_embedding
        print(f"  after +pos:  NaN={torch.isnan(x_pos).any().item()}, "
              f"min={x_pos.min().item():.4f}, max={x_pos.max().item():.4f}")

        x_trans = model.transformer(x_pos)
        print(f"  transformer: NaN={torch.isnan(x_trans).any().item()}, "
              f"min={x_trans.min().item():.4f}, max={x_trans.max().item():.4f}")

        cls_out = x_trans[:, 0]
        logits = model.classifier(cls_out)
        print(f"  classifier:  NaN={torch.isnan(logits).any().item()}, "
              f"min={logits.min().item():.4f}, max={logits.max().item():.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")

    # Create dummy input with same scale as real data
    x = torch.randn(128, 2381).to(device)
    # Scale up to match real data magnitude
    x = x * 1e9

    model = EmberTransformer(input_dim=2381, num_classes=2).to(device)
    test_conditions(model, x, "Large-scale input (1e9)")

    # Try with normalized input
    x_norm = torch.randn(128, 2381).to(device)
    model2 = EmberTransformer(input_dim=2381, num_classes=2).to(device)
    test_conditions(model2, x_norm, "Normal-scale input (1.0)")

if __name__ == '__main__':
    main()
