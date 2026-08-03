"""
Demo 05: Candlestick Chart Image Generator
============================================
Module 2 - Introduction to Deep Learning
ML Engineer Training Curriculum

Generates 224x224 candlestick chart images from OHLCV data
for use with 2D CNNs and transfer learning (ResNet).

Usage:
    python demos/05_chart_image_generator.py
"""

import numpy as np
import os

# Use non-interactive backend for server environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


def generate_candlestick_image(
    dates, opens, highs, lows, closes, volumes=None,
    output_path=None, img_size=(224, 224), dpi=72
):
    """
    Render a candlestick chart as an image.
    
    Args:
        dates: Array of date indices (ints or datetime)
        opens, highs, lows, closes: Price arrays
        volumes: Optional volume array (adds volume subplot)
        output_path: If provided, save to this path
        img_size: Image dimensions in pixels (width, height)
        dpi: Resolution
    
    Returns:
        NumPy array of shape (height, width, 3) if output_path is None
    """
    fig_w = img_size[0] / dpi
    fig_h = img_size[1] / dpi
    
    if volumes is not None:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(fig_w, fig_h), dpi=dpi,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
        ax2 = None
    
    n = len(opens)
    x = np.arange(n)
    
    # Width of candle body (relative to spacing)
    body_width = 0.6
    
    for i in range(n):
        color = "#26A69A" if closes[i] >= opens[i] else "#EF5350"  # Green / Red
        
        # Wick (high-low line)
        ax1.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=0.8)
        
        # Body (open-close rectangle)
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        if body_height < 0.001:
            body_height = 0.001  # Minimum visible body
        
        rect = Rectangle(
            (x[i] - body_width / 2, body_bottom),
            body_width, body_height,
            facecolor=color, edgecolor=color, linewidth=0.5
        )
        ax1.add_patch(rect)
    
    ax1.set_xlim(-1, n)
    ax1.set_ylim(lows.min() * 0.995, highs.max() * 1.005)
    
    # Clean up axes for CNN input (minimal decoration)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    
    # Volume subplot
    if ax2 is not None and volumes is not None:
        colors = ["#26A69A" if c >= o else "#EF5350" for o, c in zip(opens, closes)]
        ax2.bar(x, volumes, color=colors, width=body_width, alpha=0.7)
        ax2.set_xlim(-1, n)
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_visible(False)
    
    fig.tight_layout(pad=0.1)
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return output_path
    else:
        # Return as numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img


def generate_chart_dataset(
    prices_df, window_size=30, target_horizon=5,
    output_dir="data/chart_images", img_size=(224, 224)
):
    """
    Generate labeled chart images from OHLCV data.
    
    For each trading day, render the previous `window_size` days
    as a candlestick chart, and label with the next `target_horizon`
    day return direction.
    
    Args:
        prices_df: DataFrame with columns [Open, High, Low, Close, Volume]
        window_size: Number of trading days per chart
        target_horizon: Days ahead for the target label
        output_dir: Directory to save images
        img_size: Image dimensions
    
    Returns:
        labels_df: DataFrame with [filename, label, return]
    """
    os.makedirs(os.path.join(output_dir, "up"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "down"), exist_ok=True)
    
    labels = []
    n = len(prices_df)
    
    for i in range(window_size, n - target_horizon):
        # Extract window
        window = prices_df.iloc[i - window_size : i]
        
        opens = window["Open"].values
        highs = window["High"].values
        lows = window["Low"].values
        closes = window["Close"].values
        volumes = window["Volume"].values if "Volume" in window.columns else None
        
        # Compute target: future return direction
        current_close = prices_df.iloc[i]["Close"]
        future_close = prices_df.iloc[i + target_horizon]["Close"]
        future_return = np.log(future_close / current_close)
        label = 1 if future_return > 0 else 0
        label_name = "up" if label == 1 else "down"
        
        # Generate image
        filename = f"{label_name}/chart_{i:06d}.png"
        filepath = os.path.join(output_dir, filename)
        
        generate_candlestick_image(
            dates=np.arange(window_size),
            opens=opens, highs=highs, lows=lows, closes=closes,
            volumes=volumes,
            output_path=filepath,
            img_size=img_size
        )
        
        labels.append({
            "filename": filename,
            "label": label,
            "return": round(future_return, 6),
            "date_index": i,
        })
    
    import pandas as pd
    labels_df = pd.DataFrame(labels)
    labels_path = os.path.join(output_dir, "labels.csv")
    labels_df.to_csv(labels_path, index=False)
    
    return labels_df


def main():
    """Demo: generate sample chart images from synthetic data."""
    print("=" * 60)
    print("CHART IMAGE GENERATOR DEMO")
    print("=" * 60)
    
    # Generate synthetic OHLCV data (replace with real data in practice)
    np.random.seed(42)
    n_days = 500
    
    # Random walk for close prices
    returns = np.random.randn(n_days) * 0.015
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n_days) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_days) * 0.01))
    open_price = close * (1 + np.random.randn(n_days) * 0.005)
    volume = np.random.lognormal(mean=15, sigma=0.5, size=n_days)
    
    import pandas as pd
    prices = pd.DataFrame({
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    
    print(f"\nSynthetic data: {n_days} days")
    print(f"Price range: ${close.min():.2f} - ${close.max():.2f}")
    
    # Generate a few sample images
    output_dir = "data/sample_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Single chart
    window = prices.iloc[100:130]
    img_path = os.path.join(output_dir, "sample_chart.png")
    generate_candlestick_image(
        dates=np.arange(30),
        opens=window["Open"].values,
        highs=window["High"].values,
        lows=window["Low"].values,
        closes=window["Close"].values,
        volumes=window["Volume"].values,
        output_path=img_path,
    )
    print(f"\nSample chart saved to: {img_path}")
    
    # Generate small labeled dataset (first 200 days for demo speed)
    print("\nGenerating labeled chart dataset (first 200 days)...")
    small_prices = prices.iloc[:200]
    labels = generate_chart_dataset(
        small_prices,
        window_size=30,
        target_horizon=5,
        output_dir="data/chart_images_demo",
        img_size=(224, 224),
    )
    
    print(f"\nGenerated {len(labels)} chart images")
    print(f"  Up:   {(labels['label'] == 1).sum()} ({(labels['label'] == 1).mean()*100:.1f}%)")
    print(f"  Down: {(labels['label'] == 0).sum()} ({(labels['label'] == 0).mean()*100:.1f}%)")
    print(f"\nImages saved to: data/chart_images_demo/")
    print(f"Labels saved to: data/chart_images_demo/labels.csv")
    
    print("\n" + "=" * 60)
    print("USAGE IN CNN TRAINING")
    print("=" * 60)
    print("""
To use these images with a PyTorch CNN / transfer learning model:

    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    dataset = datasets.ImageFolder(
        root='data/chart_images_demo/',
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """)


if __name__ == "__main__":
    main()
