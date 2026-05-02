import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, 'src')

from core.heterogeneous_ensemble import EmberTransformer

def load_ember_data(data_dir, sample_ratio=0.2):
    train_features = np.load(f'{data_dir}/train_features.npy')
    train_labels = np.load(f'{data_dir}/train_labels.npy')
    test_features = np.load(f'{data_dir}/test_features.npy')
    test_labels = np.load(f'{data_dir}/test_labels.npy')

    if sample_ratio < 1.0:
        train_size = int(len(train_features) * sample_ratio)
        np.random.seed(42)
        indices = np.random.choice(len(train_features), train_size, replace=False)
        train_features = train_features[indices]
        train_labels = train_labels[indices]

    # 清洗非法值
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)

    return train_features, train_labels, test_features, test_labels

def check_model_outputs(model, loader, name, device):
    model.eval()
    nan_batches = 0
    total_batches = 0
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):
            if i >= 5:
                break
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)
            has_nan = torch.isnan(outputs).any().item()
            has_inf = torch.isinf(outputs).any().item()
            total_batches += 1
            if has_nan:
                nan_batches += 1
            print(f"  {name} batch {i}: output shape={outputs.shape}, "
                  f"min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, "
                  f"mean={outputs.mean().item():.4f}, NaN={has_nan}, Inf={has_inf}")
    print(f"  {name}: NaN in {nan_batches}/{total_batches} batches")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_features, train_labels, test_features, test_labels = load_ember_data(
        './data/ember_pe/full_features', sample_ratio=0.2
    )

    print(f"Train features: shape={train_features.shape}, "
          f"NaN count={np.isnan(train_features).sum()}, "
          f"Inf count={np.isinf(train_features).sum()}, "
          f"min={train_features.min():.4f}, max={train_features.max():.4f}")
    print(f"Test features: shape={test_features.shape}, "
          f"NaN count={np.isnan(test_features).sum()}, "
          f"Inf count={np.isinf(test_features).sum()}, "
          f"min={test_features.min():.4f}, max={test_features.max():.4f}")

    train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    test_dataset = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = EmberTransformer(input_dim=2381, num_classes=2).to(device)

    print("\n=== Fresh model (eval mode) ===")
    check_model_outputs(model, train_loader, "Train", device)
    check_model_outputs(model, test_loader, "Test", device)

    # Train for 1 mini-batch to see if weights become NaN
    print("\n=== Training 1 batch ===")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_x, batch_y = next(iter(train_loader))
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.long().to(device)

    outputs = model(batch_x)
    print(f"  Train batch output: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, "
          f"NaN={torch.isnan(outputs).any().item()}")

    loss = criterion(outputs, batch_y)
    print(f"  Loss: {loss.item():.4f}, NaN in loss={torch.isnan(loss).item()}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Check gradients
    grad_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any().item())
    print(f"  Parameters with NaN gradients: {grad_nan}")

    optimizer.step()

    # Check weights after update
    weight_nan = sum(1 for p in model.parameters() if torch.isnan(p).any().item())
    print(f"  Parameters with NaN weights after update: {weight_nan}")

    print("\n=== After 1 batch update (eval mode) ===")
    check_model_outputs(model, train_loader, "Train", device)
    check_model_outputs(model, test_loader, "Test", device)

    # Train for 10 more batches
    print("\n=== Training 10 more batches ===")
    model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        if i >= 10:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.long().to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if torch.isnan(loss).item():
            print(f"  Batch {i}: NaN loss!")

    weight_nan = sum(1 for p in model.parameters() if torch.isnan(p).any().item())
    print(f"  Parameters with NaN weights after 11 batches: {weight_nan}")

    print("\n=== After 11 batches (eval mode) ===")
    check_model_outputs(model, train_loader, "Train", device)
    check_model_outputs(model, test_loader, "Test", device)

if __name__ == '__main__':
    main()
