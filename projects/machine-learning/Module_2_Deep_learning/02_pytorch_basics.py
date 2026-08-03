"""
Demo 02: PyTorch Basics with Financial Data
============================================
Module 2 - Introduction to Deep Learning
ML Engineer Training Curriculum

Covers: tensors, autograd, Dataset, DataLoader, and feature scaling
for stock market data.

Usage:
    python demos/02_pytorch_basics.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════
# PART 1: Tensors and Autograd
# ══════════════════════════════════════════════════════════════

def demo_tensors():
    """Demonstrate tensor operations relevant to finance."""
    print("=" * 60)
    print("PART 1: Tensors and Autograd")
    print("=" * 60)
    
    # Create tensors from financial data
    # Simulated daily returns for 5 stocks over 10 days
    np.random.seed(42)
    returns = np.random.randn(10, 5) * 0.02  # ~2% daily vol
    
    # NumPy array → PyTorch tensor
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    print(f"\nReturns tensor shape: {returns_tensor.shape}")
    print(f"Dtype: {returns_tensor.dtype}")
    print(f"Device: {returns_tensor.device}")
    
    # Basic operations (vectorized, just like NumPy)
    mean_returns = returns_tensor.mean(dim=0)  # Mean return per stock
    volatility = returns_tensor.std(dim=0)     # Volatility per stock
    sharpe = mean_returns / volatility * np.sqrt(252)
    
    print(f"\nMean daily returns: {mean_returns}")
    print(f"Daily volatility:   {volatility}")
    print(f"Annualized Sharpe:  {sharpe}")
    
    # ── Autograd: automatic differentiation ──
    # Simple example: minimize portfolio variance
    weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], requires_grad=True)
    
    # Covariance matrix
    cov_matrix = torch.tensor(np.cov(returns.T), dtype=torch.float32)
    
    # Portfolio variance = w^T * Sigma * w
    portfolio_var = weights @ cov_matrix @ weights
    
    # Backward pass: compute d(variance)/d(weights)
    portfolio_var.backward()
    
    print(f"\nPortfolio variance: {portfolio_var.item():.6f}")
    print(f"Gradient w.r.t. weights: {weights.grad}")
    print("(Gradient tells us how to change weights to reduce variance)")


# ══════════════════════════════════════════════════════════════
# PART 2: Custom Dataset for Financial Data
# ══════════════════════════════════════════════════════════════

class StockReturnDataset(Dataset):
    """
    PyTorch Dataset for stock return prediction.
    
    Each sample consists of:
    - features: technical indicators for a given stock on a given day
    - target: next-day log return
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            features: NumPy array of shape (n_samples, n_features)
            targets: NumPy array of shape (n_samples,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence-based prediction (LSTM input).
    
    Each sample is a sliding window of historical features.
    Input shape: (sequence_length, n_features)
    Target: scalar (next-day return)
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 60):
        """
        Args:
            features: (n_days, n_features) array of daily features
            targets: (n_days,) array of daily targets
            seq_length: number of historical days per sample
        """
        self.seq_length = seq_length
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.targets) - self.seq_length
    
    def __getitem__(self, idx):
        # Window of features: shape (seq_length, n_features)
        x = self.features[idx : idx + self.seq_length]
        # Target: the return on the day AFTER the window
        y = self.targets[idx + self.seq_length]
        return x, y


def demo_datasets():
    """Demonstrate Dataset and DataLoader with financial data."""
    print("\n" + "=" * 60)
    print("PART 2: Custom Datasets")
    print("=" * 60)
    
    # Simulate 1000 days of stock data with 10 features
    np.random.seed(42)
    n_days = 1000
    n_features = 10
    
    features = np.random.randn(n_days, n_features)
    targets = np.random.randn(n_days) * 0.02  # Daily returns
    
    # ── Cross-sectional Dataset (for feedforward networks) ──
    dataset = StockReturnDataset(features, targets)
    print(f"\nCross-sectional dataset: {len(dataset)} samples")
    
    sample_x, sample_y = dataset[0]
    print(f"Sample features shape: {sample_x.shape}")
    print(f"Sample target: {sample_y.item():.4f}")
    
    # DataLoader with batching and shuffling
    # NOTE: shuffle=True is OK for cross-sectional data (each row is independent)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    batch_x, batch_y = next(iter(loader))
    print(f"\nBatch features shape: {batch_x.shape}")
    print(f"Batch targets shape:  {batch_y.shape}")
    
    # ── Sequence Dataset (for LSTMs) ──
    seq_dataset = StockSequenceDataset(features, targets, seq_length=60)
    print(f"\nSequence dataset: {len(seq_dataset)} samples (from {n_days} days)")
    
    seq_x, seq_y = seq_dataset[0]
    print(f"Sequence input shape: {seq_x.shape}  (60 days x 10 features)")
    print(f"Sequence target: {seq_y.item():.4f}")
    
    # DataLoader for sequences
    # NOTE: shuffle=False for time-series to preserve order during evaluation
    seq_loader = DataLoader(seq_dataset, batch_size=32, shuffle=False)
    
    batch_seq_x, batch_seq_y = next(iter(seq_loader))
    print(f"\nSequence batch shape: {batch_seq_x.shape}  (batch x seq_len x features)")
    print(f"Sequence batch target: {batch_seq_y.shape}")


# ══════════════════════════════════════════════════════════════
# PART 3: Feature Scaling (Critical for Neural Networks)
# ══════════════════════════════════════════════════════════════

def demo_scaling():
    """Demonstrate proper feature scaling with temporal splits."""
    print("\n" + "=" * 60)
    print("PART 3: Feature Scaling with Temporal Splits")
    print("=" * 60)
    
    np.random.seed(42)
    n_days = 1000
    
    # Simulate features with very different scales (like real financial data)
    features = np.column_stack([
        np.random.randn(n_days) * 50000 + 75000,   # Income (~$75K)
        np.random.rand(n_days),                      # Utilization ratio (0-1)
        np.random.randn(n_days) * 20 + 50,           # RSI (0-100)
        np.random.randn(n_days) * 1000000 + 5000000, # Volume (~5M)
        np.random.randn(n_days) * 0.02,              # Daily return (~2% vol)
    ])
    
    feature_names = ["Price", "Utilization", "RSI", "Volume", "Return"]
    
    # Temporal split: 70/15/15
    train_end = int(n_days * 0.70)
    val_end = int(n_days * 0.85)
    
    X_train = features[:train_end]
    X_val = features[train_end:val_end]
    X_test = features[val_end:]
    
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # ── CORRECT: Fit scaler on training data ONLY ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
    X_val_scaled = scaler.transform(X_val)           # transform only
    X_test_scaled = scaler.transform(X_test)         # transform only
    
    print("\nBefore scaling (train):")
    for i, name in enumerate(feature_names):
        print(f"  {name:12s}: mean={X_train[:, i].mean():12.2f}, std={X_train[:, i].std():12.2f}")
    
    print("\nAfter scaling (train):")
    for i, name in enumerate(feature_names):
        print(f"  {name:12s}: mean={X_train_scaled[:, i].mean():8.4f}, std={X_train_scaled[:, i].std():8.4f}")
    
    print("\n⚠️  COMMON MISTAKE: fitting scaler on ALL data (leaks future info)")
    print("✓  CORRECT: fit on training data only, transform val/test with same params")
    
    # Convert to PyTorch tensors
    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    print(f"\nPyTorch tensors created:")
    print(f"  Train: {train_tensor.shape}, dtype={train_tensor.dtype}")
    print(f"  Val:   {val_tensor.shape}")
    print(f"  Test:  {test_tensor.shape}")


if __name__ == "__main__":
    demo_tensors()
    demo_datasets()
    demo_scaling()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Tensors are like NumPy arrays but support autograd and GPU")
    print("2. Dataset + DataLoader handle batching and iteration")
    print("3. For time-series: use StockSequenceDataset with shuffle=False")
    print("4. ALWAYS fit scaler on training data only (temporal split)")
