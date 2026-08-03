"""
Demo 03: Training Loop with Early Stopping & TensorBoard
=========================================================
Module 2 - Introduction to Deep Learning
ML Engineer Training Curriculum

Demonstrates the complete PyTorch training pattern:
- Forward pass → Loss → Backward pass → Optimizer step
- Validation monitoring and early stopping
- TensorBoard logging
- Feedforward network for stock return prediction

Usage:
    python demos/03_training_loop.py
    tensorboard --logdir=runs/  # View training curves
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy
import os


# ══════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════════════

class StockPredictor(nn.Module):
    """
    Feedforward neural network for stock return prediction.
    
    Architecture:
        Input → [Linear → BatchNorm → ReLU → Dropout] × N → Output
    """
    
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        # Output layer: single neuron for regression
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════

class FinancialDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # ── The 5-step training pattern ──
        # Step 1: Forward pass
        predictions = model(batch_x)
        
        # Step 2: Compute loss
        loss = criterion(predictions, batch_y)
        
        # Step 3: Backward pass (compute gradients)
        loss.backward()
        
        # Step 4: Update weights
        optimizer.step()
        
        # Step 5: Zero gradients for next iteration
        optimizer.zero_grad()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns average loss and directional accuracy."""
    model.eval()
    total_loss = 0.0
    correct_direction = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
            
            # Directional accuracy: did we predict the sign correctly?
            correct_direction += ((predictions > 0) == (batch_y > 0)).sum().item()
            total_samples += len(batch_y)
    
    avg_loss = total_loss / len(loader)
    direction_acc = correct_direction / total_samples
    return avg_loss, direction_acc


def train_with_early_stopping(
    model, train_loader, val_loader, criterion, optimizer,
    device, max_epochs=100, patience=10, verbose=True
):
    """
    Full training loop with early stopping.
    
    Args:
        patience: Number of epochs without improvement before stopping
    
    Returns:
        dict with training history
    """
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    
    history = {"train_loss": [], "val_loss": [], "val_direction_acc": []}
    
    # Optional: TensorBoard logging
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter("runs/stock_predictor")
        use_tb = True
    except ImportError:
        use_tb = False
        if verbose:
            print("TensorBoard not available. Install: pip install tensorboard")
    
    for epoch in range(max_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_direction_acc"].append(val_acc)
        
        # TensorBoard logging
        if use_tb:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalar("DirectionAccuracy/val", val_acc, epoch)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            marker = " ★"  # New best
        else:
            epochs_without_improvement += 1
            marker = ""
        
        if verbose and (epoch % 10 == 0 or marker or epoch == max_epochs - 1):
            print(
                f"Epoch {epoch+1:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Dir Acc: {val_acc:.4f}{marker}"
            )
        
        # Stop if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Restored best model (val_loss={best_val_loss:.6f})")
    
    if use_tb:
        writer.close()
    
    return history


# ══════════════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════════════

def main():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ── Generate synthetic financial data ──
    n_days = 2000
    n_features = 15
    
    # Simulate features (like technical indicators)
    X = np.random.randn(n_days, n_features)
    
    # Target: next-day return with weak signal + noise
    # Real financial data has ~0.5-2% signal-to-noise ratio
    signal = 0.01 * (X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3])
    noise = np.random.randn(n_days) * 0.02
    y = signal + noise
    
    print(f"\nData: {n_days} days, {n_features} features")
    print(f"Target stats: mean={y.mean():.6f}, std={y.std():.6f}")
    print(f"Signal-to-noise ratio: {signal.std()/noise.std():.4f}")
    
    # ── Temporal split ──
    train_end = int(n_days * 0.70)
    val_end = int(n_days * 0.85)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # ── Scale features ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # ── Create DataLoaders ──
    train_dataset = FinancialDataset(X_train, y_train)
    val_dataset = FinancialDataset(X_val, y_val)
    test_dataset = FinancialDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # ── Build model ──
    model = StockPredictor(
        input_dim=n_features,
        hidden_dims=(128, 64, 32),
        dropout_rate=0.3
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {n_params:,}")
    
    # ── Training setup ──
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # ── Train! ──
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = train_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer,
        device, max_epochs=100, patience=10
    )
    
    # ── Test evaluation ──
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test MSE:  {test_loss:.6f}")
    print(f"Test Direction Accuracy: {test_acc:.4f}")
    
    # Baseline: always predict mean
    baseline_mse = np.mean((y_test - y_train.mean()) ** 2)
    print(f"\nBaseline MSE (predict mean): {baseline_mse:.6f}")
    print(f"Improvement over baseline: {(1 - test_loss/baseline_mse)*100:.1f}%")
    
    # ── Save model ──
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/stock_predictor.pt")
    print("\nModel saved to models/stock_predictor.pt")
    
    # ── Compute Sharpe ratio of a simple strategy ──
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Strategy: go long when predicted return > 0, else stay flat
    strategy_returns = np.where(all_preds > 0, all_targets, 0)
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    print(f"\nStrategy Sharpe Ratio: {sharpe:.4f}")
    print(f"(Positive Sharpe = model has some predictive power)")


if __name__ == "__main__":
    main()
