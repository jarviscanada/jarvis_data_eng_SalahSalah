"""
Demo 04: 1D CNN for Time-Series Financial Data
================================================
Module 2 - Introduction to Deep Learning
ML Engineer Training Curriculum

Demonstrates applying 1D convolutional layers to sliding windows
of stock return data to detect temporal patterns.

Usage:
    python demos/04_cnn_1d_timeseries.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════
# MODEL: 1D CNN for Financial Time Series
# ══════════════════════════════════════════════════════════════

class FinancialCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for stock return prediction.
    
    Input: (batch_size, sequence_length, n_features)
    Output: (batch_size,) - predicted next-day return
    
    Architecture:
        Conv1D(kernel=5) → BN → ReLU → MaxPool
        Conv1D(kernel=3) → BN → ReLU → MaxPool
        Conv1D(kernel=3) → BN → ReLU → GlobalAvgPool
        FC → Dropout → FC → Output
    """
    
    def __init__(self, n_features, seq_length=60, dropout_rate=0.3):
        super().__init__()
        
        # Convolutional layers
        # Conv1d expects (batch, channels, length)
        # We treat each feature as a channel
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # seq_length → seq_length/2
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # → seq_length/4
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        
        # Global average pooling: (batch, 16, time) → (batch, 16)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        # Input x: (batch, seq_length, n_features)
        # Conv1d expects: (batch, channels, length)
        x = x.permute(0, 2, 1)  # → (batch, n_features, seq_length)
        
        x = self.conv1(x)  # → (batch, 64, seq_length/2)
        x = self.conv2(x)  # → (batch, 32, seq_length/4)
        x = self.conv3(x)  # → (batch, 16, seq_length/4)
        
        x = self.global_pool(x)  # → (batch, 16, 1)
        x = x.squeeze(-1)       # → (batch, 16)
        
        x = self.fc(x)          # → (batch, 1)
        return x.squeeze(-1)    # → (batch,)


# ══════════════════════════════════════════════════════════════
# SLIDING WINDOW DATASET
# ══════════════════════════════════════════════════════════════

class SlidingWindowDataset(Dataset):
    """Create sliding windows from time-series data."""
    
    def __init__(self, features, targets, window_size=60):
        self.window_size = window_size
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.targets) - self.window_size
    
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]  # (window, features)
        y = self.targets[idx + self.window_size]          # scalar
        return x, y


# ══════════════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── Generate synthetic time-series data ──
    n_days = 2000
    n_features = 10
    window_size = 60
    
    # Features: simulated technical indicators
    features = np.random.randn(n_days, n_features)
    
    # Add a temporal pattern: 5-day momentum signal
    for t in range(5, n_days):
        features[t, 0] = np.mean(features[t-5:t, 0])  # 5-day moving average
    
    # Target with temporal dependency
    targets = np.zeros(n_days)
    for t in range(window_size, n_days):
        signal = 0.005 * np.mean(features[t-5:t, 0])  # Momentum signal
        noise = np.random.randn() * 0.02
        targets[t] = signal + noise
    
    print(f"Data: {n_days} days, {n_features} features, window={window_size}")
    
    # ── Temporal split ──
    train_end = int(n_days * 0.70)
    val_end = int(n_days * 0.85)
    
    # Scale features (fit on training portion only)
    scaler = StandardScaler()
    features[:train_end] = scaler.fit_transform(features[:train_end])
    features[train_end:val_end] = scaler.transform(features[train_end:val_end])
    features[val_end:] = scaler.transform(features[val_end:])
    
    # Create datasets
    train_dataset = SlidingWindowDataset(features[:train_end], targets[:train_end], window_size)
    val_dataset = SlidingWindowDataset(features[train_end:val_end], targets[train_end:val_end], window_size)
    test_dataset = SlidingWindowDataset(features[val_end:], targets[val_end:], window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # ── Build model ──
    model = FinancialCNN1D(
        n_features=n_features,
        seq_length=window_size,
        dropout_rate=0.3
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n1D CNN Model ({n_params:,} parameters):")
    print(model)
    
    # ── Training ──
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n" + "=" * 60)
    print("TRAINING 1D CNN")
    print("=" * 60)
    
    best_val_loss = float("inf")
    patience = 10
    wait = 0
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item()
        val_loss /= len(val_loader)
        
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            marker = " ★"
        else:
            wait += 1
        
        if epoch % 10 == 0 or marker or wait >= patience:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}{marker}")
        
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # ── Test ──
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            test_loss += criterion(preds, batch_y).item()
            correct += ((preds > 0) == (batch_y > 0)).sum().item()
            total += len(batch_y)
    
    test_loss /= len(test_loader)
    print(f"\nTest MSE: {test_loss:.6f}")
    print(f"Test Direction Accuracy: {correct/total:.4f}")
    
    # ── Inspect learned filters ──
    print("\n" + "=" * 60)
    print("INSPECTING LEARNED FILTERS")
    print("=" * 60)
    
    # First conv layer weights: (out_channels, in_channels, kernel_size)
    weights = model.conv1[0].weight.data.cpu().numpy()
    print(f"\nConv1 filter shape: {weights.shape}")
    print(f"  64 filters, each {n_features} channels wide, kernel size 5")
    print(f"  These filters detect 5-day patterns across {n_features} features")
    
    # Show top filter by weight magnitude
    filter_norms = np.linalg.norm(weights, axis=(1, 2))
    top_filter = np.argmax(filter_norms)
    print(f"\nStrongest filter (#{top_filter}):")
    print(f"  Weights norm: {filter_norms[top_filter]:.4f}")
    print(f"  This filter has learned to detect a specific 5-day pattern")


if __name__ == "__main__":
    main()
