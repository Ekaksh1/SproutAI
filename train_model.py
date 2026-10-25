"""
SPROUT AI - Model Training Pipeline
Performs genuine machine learning training using PyTorch.
Supports MobileNetV2 transfer learning and SproutCustomCNN.
Can be run via CLI or invoked programmatically with progress callbacks.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

from model_utils import create_model, save_model, get_device, CLASSES

DEFAULT_DATASET_DIR = "dataset"
DEFAULT_MODELS_DIR = "models"
MODEL_FILENAME = "sprout_model.pth"
METADATA_FILENAME = "model_metadata.json"

# Supported image extensions
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def check_dataset_status(dataset_dir: str = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    """
    Validates folder structure and counts images in healthy and stressed directories.
    """
    healthy_dir = os.path.join(dataset_dir, "healthy")
    stressed_dir = os.path.join(dataset_dir, "stressed")

    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(stressed_dir, exist_ok=True)

    def count_valid_images(folder: str) -> List[str]:
        if not os.path.exists(folder):
            return []
        return [
            f for f in os.listdir(folder)
            if f.lower().endswith(VALID_EXTENSIONS) and not f.startswith(".")
        ]

    healthy_files = count_valid_images(healthy_dir)
    stressed_files = count_valid_images(stressed_dir)

    total = len(healthy_files) + len(stressed_files)
    is_ready = len(healthy_files) >= 2 and len(stressed_files) >= 2

    return {
        "dataset_dir": dataset_dir,
        "healthy_count": len(healthy_files),
        "stressed_count": len(stressed_files),
        "total_count": total,
        "is_ready": is_ready,
        "healthy_samples": healthy_files[:5],
        "stressed_samples": stressed_files[:5]
    }


def get_data_transforms():
    """Generates train and validation PyTorch transforms with data augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class TransformSubset(torch.utils.data.Dataset):
    """Applies a specific transform to a PyTorch dataset subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def train_sprout_model(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    output_dir: str = DEFAULT_MODELS_DIR,
    arch: str = "mobilenet_v2",
    pretrained: bool = True,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Main training function. Executes genuine backpropagation training,
    computes validation metrics, and writes saved weights to disk.
    """
    status = check_dataset_status(dataset_dir)
    if not status["is_ready"]:
        raise ValueError(
            f"Insufficient training images. Found {status['healthy_count']} healthy and "
            f"{status['stressed_count']} stressed images. At least 2 images per category are required. "
            f"Please place labeled photos in '{os.path.join(dataset_dir, 'healthy')}' and "
            f"'{os.path.join(dataset_dir, 'stressed')}'."
        )

    device = get_device()
    print(f"[*] SPROUT AI Training Engine active on device: {device}")

    # Load dataset with ImageFolder (raw images without transforms initially)
    raw_dataset = datasets.ImageFolder(dataset_dir)
    class_to_idx = raw_dataset.class_to_idx
    print(f"[*] Found classes: {class_to_idx}")

    # Validate classes match expected
    if "healthy" not in class_to_idx or "stressed" not in class_to_idx:
        raise ValueError(
            f"Directory structure in '{dataset_dir}' must contain 'healthy/' and 'stressed/' subfolders."
        )

    # Train / Val Split
    total_samples = len(raw_dataset)
    val_size = max(1, int(total_samples * val_split))
    train_size = total_samples - val_size

    # Fix seed for reproducible train/val partitioning
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(raw_dataset, [train_size, val_size], generator=generator)

    train_tf, val_tf = get_data_transforms()
    train_dataset = TransformSubset(train_subset, transform=train_tf)
    val_dataset = TransformSubset(val_subset, transform=val_tf)

    # Effective batch size
    eff_batch_size = min(batch_size, train_size)
    train_loader = DataLoader(train_dataset, batch_size=eff_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=min(eff_batch_size, val_size), shuffle=False)

    # Instantiate Model
    model = create_model(arch=arch, pretrained=pretrained)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Filter only parameters that require grad
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 1

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)

        lr_scheduler.step(epoch_val_loss)

        # Track history
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        # Check best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_summary = {
            "epoch": epoch,
            "total_epochs": epochs,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "is_best": (epoch == best_epoch)
        }

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}%"
        )

        if progress_callback:
            progress_callback(epoch_summary)

    duration = time.time() - start_time

    # Load best weights into model before saving
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save artifact
    model_path = os.path.join(output_dir, MODEL_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)

    metadata = {
        "project": "SPROUT AI",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": arch,
        "pretrained_weights": pretrained,
        "classes": CLASSES,
        "class_to_idx": class_to_idx,
        "training_samples": train_size,
        "validation_samples": val_size,
        "total_samples": total_samples,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "final_train_accuracy": history["train_acc"][-1],
        "final_val_accuracy": history["val_acc"][-1],
        "best_val_loss": round(best_val_loss, 4),
        "training_duration_seconds": round(duration, 2),
        "history": history
    }

    save_model(model, model_path, metadata_path, metadata)
    print(f"[*] Training complete in {duration:.1f}s. Model saved to '{model_path}'.")

    return {
        "success": True,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata": metadata
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPROUT AI - Model Trainer")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR, help="Path to dataset directory")
    parser.add_argument("--output", type=str, default=DEFAULT_MODELS_DIR, help="Path to save models")
    parser.add_argument("--arch", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "custom_cnn"])
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    try:
        train_sprout_model(
            dataset_dir=args.dataset,
            output_dir=args.output,
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    except Exception as e:
        print(f"[!] Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
