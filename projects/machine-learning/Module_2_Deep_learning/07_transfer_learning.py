"""
Demo 07: Transfer Learning with ResNet on Chart Images
=======================================================
Module 2 - Introduction to Deep Learning
ML Engineer Training Curriculum

Demonstrates:
- Loading pre-trained ResNet-18 from torchvision
- Replacing the classification head
- Fine-tuning vs feature extraction approaches
- Grad-CAM for visual explainability

Usage:
    python demos/07_transfer_learning.py

Prerequisites:
    Run demos/05_chart_image_generator.py first to create chart images
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# ══════════════════════════════════════════════════════════════
# TRANSFER LEARNING MODEL
# ══════════════════════════════════════════════════════════════

def create_transfer_model(num_classes=2, freeze_features=True):
    """
    Create a ResNet-18 model for chart image classification.
    
    Args:
        num_classes: Number of output classes (2 for up/down)
        freeze_features: If True, freeze all layers except the final FC
                        (feature extraction mode). If False, fine-tune
                        the last 2 residual blocks.
    
    Returns:
        model: Modified ResNet-18
    """
    # Load pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if freeze_features:
        # ── FEATURE EXTRACTION MODE ──
        # Freeze ALL parameters
        for param in model.parameters():
            param.requires_grad = False
    else:
        # ── FINE-TUNING MODE ──
        # Freeze early layers, unfreeze later ones
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer3 and layer4 (the last 2 residual blocks)
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    # Replace the final fully connected layer
    # Original: nn.Linear(512, 1000) for ImageNet
    # New: nn.Linear(512, num_classes) for our task
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    return model


# ══════════════════════════════════════════════════════════════
# DATASET FOR CHART IMAGES
# ══════════════════════════════════════════════════════════════

class ChartImageDataset(Dataset):
    """
    Dataset for loading candlestick chart images.
    
    Expected directory structure:
        root/
        ├── up/
        │   ├── chart_000100.png
        │   └── ...
        └── down/
            ├── chart_000200.png
            └── ...
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {"down": 0, "up": 1}
        
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in sorted(os.listdir(class_dir)):
                if filename.endswith((".png", ".jpg")):
                    filepath = os.path.join(class_dir, filename)
                    self.samples.append((filepath, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ══════════════════════════════════════════════════════════════
# IMAGE TRANSFORMS
# ══════════════════════════════════════════════════════════════

def get_transforms():
    """
    Define image transforms for training and evaluation.
    
    Note: We use ImageNet normalization because we're using
    a model pre-trained on ImageNet.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # NOTE: Do NOT use RandomHorizontalFlip for charts!
        # Time flows left-to-right; flipping changes the meaning.
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],   # ImageNet stds
        ),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    return train_transform, eval_transform


# ══════════════════════════════════════════════════════════════
# GRAD-CAM VISUALIZATION
# ══════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights which regions of the chart image the model
    focuses on when making its prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Only forward hook here
        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        # Save activations
        self.activations = output.detach()

        # Save gradients from this output tensor during backward
        def _save_gradients(grad):
            self.gradients = grad.detach()

        output.register_hook(_save_gradients)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        if self.gradients is None:
            raise RuntimeError("Grad-CAM gradients were not captured.")

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()

# ══════════════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING DEMO")
    print("=" * 60)
    
    # ── 1. Create Feature Extraction Model ──
    print("\n--- Feature Extraction Mode ---")
    print("(Freeze all pre-trained layers, only train the new FC head)")
    fe_model = create_transfer_model(num_classes=2, freeze_features=True)
    
    # ── 2. Create Fine-Tuning Model ──
    print("\n--- Fine-Tuning Mode ---")
    print("(Unfreeze last 2 residual blocks + new FC head)")
    ft_model = create_transfer_model(num_classes=2, freeze_features=False)
    
    # ── 3. Show transforms ──
    train_transform, eval_transform = get_transforms()
    print("\n--- Image Transforms ---")
    print("Train:", train_transform)
    print("\nEval:", eval_transform)
    
    # ── 4. Demo with synthetic data ──
    print("\n" + "=" * 60)
    print("TRAINING DEMO (synthetic data)")
    print("=" * 60)
    
    # Create synthetic chart images for demo
    demo_dir = "data/chart_demo"
    os.makedirs(os.path.join(demo_dir, "up"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "down"), exist_ok=True)
    
    # Generate small random images (replace with real charts in practice)
    for i in range(20):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        label = "up" if i % 2 == 0 else "down"
        Image.fromarray(img).save(os.path.join(demo_dir, label, f"chart_{i:04d}.png"))
    
    dataset = ChartImageDataset(demo_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset: {len(dataset)} images")
    
    # Quick training demo (2 epochs on synthetic data)
    model = create_transfer_model(num_classes=2, freeze_features=False).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize parameters that require gradients
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    
    model.train()
    for epoch in range(2):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)
        
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={acc:.4f}")
    
    # ── 5. Grad-CAM Demo ──
    print("\n--- Grad-CAM Visualization ---")
    print("(Shows which chart regions the model focuses on)")
    
    grad_cam = GradCAM(model, model.layer4[-1])
    
    # Get one sample
    sample_image, sample_label = dataset[0]
    sample_input = sample_image.unsqueeze(0).to(device)
    
    cam = grad_cam.generate(sample_input)
    print(f"Grad-CAM heatmap shape: {cam.shape}")
    print(f"Predicted class: {'up' if model(sample_input).argmax().item() == 1 else 'down'}")
    print(f"True class: {'up' if sample_label == 1 else 'down'}")
    print("(In practice, overlay this heatmap on the chart image)")
    
    # Cleanup demo data
    import shutil
    shutil.rmtree(demo_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW SUMMARY")
    print("=" * 60)
    print("""
1. Generate chart images: demos/05_chart_image_generator.py
2. Load pre-trained ResNet-18 from torchvision
3. Replace final FC layer: nn.Linear(512, 2)
4. Choose approach:
   - Feature extraction: freeze all, train FC only (fast, less data)
   - Fine-tuning: unfreeze last blocks + FC (slower, better results)
5. Train with CrossEntropyLoss and Adam (lr=1e-4)
6. Use Grad-CAM to visualize which chart patterns drive predictions
7. Compare with LSTM and feedforward baselines
""")


if __name__ == "__main__":
    main()
