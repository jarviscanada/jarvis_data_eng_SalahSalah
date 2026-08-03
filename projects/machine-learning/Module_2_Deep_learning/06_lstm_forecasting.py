"""
Demo 06: LSTM Forecasting with Walk-Forward Validation
=======================================================
Module 2 - Introduction to Deep Learning
ML Engineer Training Curriculum

Demonstrates:
- LSTM model for stock return prediction
- Sliding window dataset creation
- Walk-forward (expanding window) validation
- Gradient clipping and learning rate scheduling

Usage:
    python demos/06_lstm_forecasting.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy


# ══════════════════════════════════════════════════════════════
# LSTM MODEL
# ══════════════════════════════════════════════════════════════

class FinancialLSTM(nn.Module):
    """
    LSTM model for stock return prediction.
    
    Input: (batch_size, seq_length, n_features)
    Output: (batch_size,) - predicted next-day return
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,  # Input shape: (batch, seq, features)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        # x shape: (batch, seq_length, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use only the LAST time step's output for prediction
        # lstm_out[:, -1, :] shape: (batch, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        return self.fc(last_output).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    """Sliding window dataset for LSTM input."""
    
    def __init__(self, features, targets, seq_length=60):
        self.seq_length = seq_length
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y) - self.seq_length
    
    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_length],
            self.y[idx + self.seq_length],
        )


# ══════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    """Train one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        
        # ── GRADIENT CLIPPING (essential for LSTMs) ──
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, criterion, device):
    """Evaluate model and return loss + metrics."""
    model.eval()
    total_loss = 0
    preds_list, targets_list = [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += criterion(pred, y).item()
            preds_list.extend(pred.cpu().numpy())
            targets_list.extend(y.cpu().numpy())
    
    preds = np.array(preds_list)
    targets = np.array(targets_list)
    
    mse = total_loss / len(loader)
    direction_acc = np.mean((preds > 0) == (targets > 0))
    
    # Sharpe of long/short strategy
    strategy_returns = np.where(preds > 0, targets, -targets)
    sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
              if strategy_returns.std() > 0 else 0)
    
    return mse, direction_acc, sharpe


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════

def walk_forward_validation(
    features, targets, n_features, seq_length=60,
    n_folds=5, train_months=12, val_months=3,
    device=None, max_epochs=30, patience=5
):
    """
    Walk-forward (expanding window) validation for time-series.
    
    Simulates production deployment:
    - Train on historical data
    - Predict on the next unseen period
    - Expand training window and repeat
    
    This is the gold standard for financial model evaluation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_total = len(targets)
    # Approximate: 21 trading days per month
    train_size = train_months * 21
    val_size = val_months * 21
    step_size = val_size
    
    fold_results = []
    
    for fold in range(n_folds):
        fold_train_end = train_size + fold * step_size
        fold_val_end = fold_train_end + val_size
        
        if fold_val_end > n_total - seq_length:
            break
        
        print(f"\n--- Fold {fold+1} ---")
        print(f"  Train: days 0-{fold_train_end}, Val: days {fold_train_end}-{fold_val_end}")
        
        # Scale features
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(features[:fold_train_end])
        val_feats = scaler.transform(features[fold_train_end:fold_val_end])
        
        # Create datasets
        train_ds = SequenceDataset(train_feats, targets[:fold_train_end], seq_length)
        val_ds = SequenceDataset(val_feats, targets[fold_train_end:fold_val_end], seq_length)
        
        if len(train_ds) < 10 or len(val_ds) < 10:
            print("  Skipping (too few samples)")
            continue
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # Fresh model for each fold
        model = FinancialLSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=2,
            dropout=0.3
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )
        
        # Train with early stopping
        best_val_loss = float("inf")
        wait = 0
        
        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            
            if wait >= patience:
                break
        
        # Restore best and evaluate
        model.load_state_dict(best_state)
        val_mse, val_acc, val_sharpe = evaluate_model(model, val_loader, criterion, device)
        
        print(f"  Val MSE: {val_mse:.6f} | Dir Acc: {val_acc:.4f} | Sharpe: {val_sharpe:.4f}")
        
        fold_results.append({
            "fold": fold + 1,
            "mse": val_mse,
            "direction_acc": val_acc,
            "sharpe": val_sharpe,
        })
    
    return fold_results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate synthetic time-series data
    n_days = 1500
    n_features = 10
    seq_length = 60
    
    features = np.random.randn(n_days, n_features)
    
    # Add temporal structure
    for t in range(10, n_days):
        features[t, 0] = 0.5 * features[t-1, 0] + 0.5 * np.random.randn()
        features[t, 1] = np.mean(features[t-5:t, 0])
    
    # Target with autocorrelation
    targets = np.zeros(n_days)
    for t in range(1, n_days):
        targets[t] = 0.005 * features[t-1, 0] + 0.003 * features[t-1, 1] + np.random.randn() * 0.02
    
    print("=" * 60)
    print("LSTM WALK-FORWARD VALIDATION DEMO")
    print("=" * 60)
    print(f"Data: {n_days} days, {n_features} features")
    print(f"Sequence length: {seq_length}")
    
    # Run walk-forward validation
    results = walk_forward_validation(
        features, targets, n_features,
        seq_length=seq_length,
        n_folds=5,
        train_months=12,
        val_months=3,
        device=device,
        max_epochs=30,
        patience=5
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("=" * 60)
    
    if results:
        mses = [r["mse"] for r in results]
        accs = [r["direction_acc"] for r in results]
        sharpes = [r["sharpe"] for r in results]
        
        print(f"\nMSE:           {np.mean(mses):.6f} +/- {np.std(mses):.6f}")
        print(f"Direction Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"Sharpe Ratio:  {np.mean(sharpes):.4f} +/- {np.std(sharpes):.4f}")
        
        print("\nPer-fold results:")
        for r in results:
            print(f"  Fold {r['fold']}: MSE={r['mse']:.6f}, "
                  f"Acc={r['direction_acc']:.4f}, Sharpe={r['sharpe']:.4f}")


if __name__ == "__main__":
    main()
