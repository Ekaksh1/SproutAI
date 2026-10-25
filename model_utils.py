"""
SPROUT AI - Model Architecture and Utilities
Provides neural network architectures (MobileNetV2 Transfer Learning and SproutCustomCNN),
device configuration, and weight management.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

CLASSES = ["healthy", "stressed"]
CLASS_LABELS = {
    0: "Healthy-looking plant",
    1: "Visibly stressed plant"
}

def get_device() -> torch.device:
    """Detect available hardware accelerator, fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SproutCustomCNN(nn.Module):
    """
    Lightweight 4-block CNN designed for fast CPU training from scratch
    without requiring external pretrained weights. Perfect for offline college labs.
    """
    def __init__(self, num_classes: int = 2):
        super(SproutCustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))            # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_mobilenet_v2(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Builds a MobileNetV2 architecture with a custom classification head.
    """
    try:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            model = mobilenet_v2(weights=weights)
            # Freeze early feature layers for transfer learning efficiency
            for param in model.features[:10].parameters():
                param.requires_grad = False
        else:
            model = mobilenet_v2(weights=None)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        return model
    except Exception as e:
        # Fallback to custom CNN if torchvision pretrained fails
        print(f"Notice: Pretrained MobileNetV2 unavailable ({e}). Initializing SproutCustomCNN.")
        return SproutCustomCNN(num_classes=num_classes)


def create_model(arch: str = "mobilenet_v2", pretrained: bool = True) -> nn.Module:
    """Factory function to instantiate the selected model architecture."""
    if arch.lower() == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=2, pretrained=pretrained)
    elif arch.lower() in ["custom_cnn", "cnn"]:
        return SproutCustomCNN(num_classes=2)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose 'mobilenet_v2' or 'custom_cnn'.")


def save_model(
    model: nn.Module,
    model_path: str,
    metadata_path: str,
    metadata: Dict[str, Any]
) -> None:
    """Saves model weights and corresponding JSON metadata."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    torch.save(model.state_dict(), model_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_model(
    model_path: str,
    metadata_path: str,
    device: Optional[torch.device] = None
) -> Tuple[Optional[nn.Module], Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads saved model weights and metadata.
    Returns (model, metadata, error_message).
    """
    if device is None:
        device = get_device()

    if not os.path.exists(model_path):
        return None, None, f"Model file not found at '{model_path}'."
    if not os.path.exists(metadata_path):
        return None, None, f"Metadata file not found at '{metadata_path}'."

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        arch = metadata.get("architecture", "mobilenet_v2")
        model = create_model(arch=arch, pretrained=False)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, metadata, None
    except Exception as e:
        return None, None, f"Failed to load model weights: {str(e)}"
