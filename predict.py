"""
SPROUT AI - Inference and Prediction Pipeline
Handles image validation, OpenCV quality checks, PyTorch inference,
calibrated probability estimation, and confidence thresholding.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
from typing import Tuple, Dict, Any, Optional, Union
import torchvision.transforms as transforms

from model_utils import load_model, get_device, CLASSES, CLASS_LABELS
from recommendations import get_recommendation

DEFAULT_MODEL_PATH = os.path.join("models", "sprout_model.pth")
DEFAULT_METADATA_PATH = os.path.join("models", "model_metadata.json")

# Standard normalization for pretrained vision backbones
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def assess_image_quality(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Evaluates image lighting, contrast, and blurriness using OpenCV.
    Provides proactive feedback before neural inference.
    """
    warnings = []
    # Convert PIL to OpenCV BGR
    img_np = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Blur detection using Laplacian variance
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = lap_var < 35.0
    if is_blurry:
        warnings.append(f"Image appears blurry (sharpness score {lap_var:.1f}/100). Focus closely on leaves.")

    # 2. Lighting / exposure check
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 35:
        warnings.append("Image is severely underexposed (too dark). Move plant into natural daylight.")
    elif mean_brightness > 230:
        warnings.append("Image is overexposed (glare detected). Avoid intense direct flash.")

    return {
        "sharpness_score": round(lap_var, 2),
        "mean_brightness": round(mean_brightness, 2),
        "is_blurry": is_blurry,
        "warnings": warnings
    }


def preprocess_image(image_input: Union[str, bytes, Image.Image]) -> Tuple[Optional[Image.Image], Optional[torch.Tensor], Optional[str]]:
    """
    Loads, corrects orientation, and preprocesses input image into a PyTorch tensor.
    Returns (pil_image, tensor, error_msg).
    """
    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                return None, None, f"Image path '{image_input}' not found."
            pil_img = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            import io
            pil_img = Image.open(io.BytesIO(image_input))
        else:
            return None, None, "Unsupported image input type."

        # Auto-orient based on camera EXIF tags
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")

        # Transform for model
        tensor = INFERENCE_TRANSFORMS(pil_img).unsqueeze(0)  # Shape [1, 3, 224, 224]
        return pil_img, tensor, None
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"


class SproutPredictor:
    """
    Inference manager with model caching and confidence thresholding.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, metadata_path: str = DEFAULT_METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.device = get_device()
        self.model = None
        self.metadata = None
        self.load_error = None
        self.reload_model()

    def reload_model(self) -> None:
        """Loads or reloads the trained model from disk."""
        model, meta, err = load_model(self.model_path, self.metadata_path, self.device)
        self.model = model
        self.metadata = meta
        self.load_error = err

    def is_model_ready(self) -> bool:
        """Check if a trained model is loaded and available for inference."""
        return self.model is not None and self.load_error is None

    def predict(
        self,
        image_input: Union[str, bytes, Image.Image],
        confidence_threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Executes genuine AI inference on the input image.
        Applies confidence thresholding to categorize as Healthy, Stressed, or Uncertain.
        """
        pil_img, tensor, err = preprocess_image(image_input)
        if err:
            return {
                "success": False,
                "error": err,
                "predicted_class": "uncertain",
                "confidence": 0.0,
                "recommendation": get_recommendation("uncertain", 0.0, confidence_threshold)
            }

        # Quality assessment
        quality = assess_image_quality(pil_img)

        # Check model availability
        if not self.is_model_ready():
            return {
                "success": False,
                "error": self.load_error or "Model weights not found. Please train or load a model first.",
                "model_missing": True,
                "predicted_class": "uncertain",
                "confidence": 0.0,
                "image_quality": quality,
                "recommendation": get_recommendation("uncertain", 0.0, confidence_threshold)
            }

        try:
            tensor = tensor.to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            p_healthy = float(probs[0])
            p_stressed = float(probs[1])

            # Determine dominant class
            raw_class_idx = int(np.argmax(probs))
            raw_class = CLASSES[raw_class_idx]
            top_confidence = float(probs[raw_class_idx])

            # Apply confidence threshold
            if top_confidence < confidence_threshold:
                effective_class = "uncertain"
                explanation = (
                    f"Model confidence ({top_confidence:.1%}) is below the {confidence_threshold:.0%} reliability threshold. "
                    f"Probabilities are divided: Healthy={p_healthy:.1%}, Stressed={p_stressed:.1%}. "
                    "Classified as Uncertain to prevent misleading diagnosis."
                )
            else:
                effective_class = raw_class
                if effective_class == "healthy":
                    explanation = (
                        f"Classified as Healthy with {top_confidence:.1%} confidence. "
                        "Leaf coloring and structural patterns are consistent with healthy foliage."
                    )
                else:
                    explanation = (
                        f"Classified as Stressed with {top_confidence:.1%} confidence. "
                        "Visible signatures of stress (chlorosis, browning, or wilting) were identified."
                    )

            # Get tailored care recommendation
            rec = get_recommendation(effective_class, top_confidence, confidence_threshold)

            return {
                "success": True,
                "predicted_class": effective_class,
                "raw_class": raw_class,
                "confidence": top_confidence,
                "confidence_threshold": confidence_threshold,
                "probabilities": {
                    "healthy": round(p_healthy, 4),
                    "stressed": round(p_stressed, 4)
                },
                "explanation": explanation,
                "quality_assessment": quality,
                "recommendation": rec,
                "model_metadata": self.metadata
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference execution failed: {str(e)}",
                "predicted_class": "uncertain",
                "confidence": 0.0,
                "recommendation": get_recommendation("uncertain", 0.0, confidence_threshold)
            }
