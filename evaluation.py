"""
SPROUT AI - Testing and Evaluation Engine
Evaluates trained models on unseen test images, computes genuine scikit-learn
metrics (Accuracy, Precision, Recall, F1-Score), generates confusion matrices,
and exports results to CSV.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from predict import SproutPredictor


def evaluate_test_dataset(
    test_samples: List[Dict[str, Any]],
    predictor: SproutPredictor,
    confidence_threshold: float = 0.70
) -> Dict[str, Any]:
    """
    Evaluates a collection of unseen test images against ground-truth labels.
    Each item in test_samples must have:
      - 'image': PIL Image or file path
      - 'true_label': 'healthy' or 'stressed'
      - 'filename': optional display name
    """
    if not test_samples:
        return {"success": False, "error": "No test samples provided."}

    if not predictor.is_model_ready():
        return {"success": False, "error": "Predictor has no loaded model weights."}

    results = []
    y_true = []
    y_pred = []
    y_raw_pred = []

    for idx, sample in enumerate(test_samples):
        img_input = sample["image"]
        true_label = str(sample["true_label"]).lower().strip()
        filename = sample.get("filename", f"sample_{idx+1}.jpg")

        pred_res = predictor.predict(img_input, confidence_threshold=confidence_threshold)
        predicted_class = pred_res.get("predicted_class", "uncertain")
        raw_class = pred_res.get("raw_class", "uncertain")
        conf = pred_res.get("confidence", 0.0)
        probs = pred_res.get("probabilities", {"healthy": 0.0, "stressed": 0.0})

        is_correct = (predicted_class == true_label)

        results.append({
            "filename": filename,
            "true_label": true_label,
            "predicted_class": predicted_class,
            "raw_class": raw_class,
            "confidence": round(conf, 4),
            "p_healthy": probs.get("healthy", 0.0),
            "p_stressed": probs.get("stressed", 0.0),
            "is_correct": is_correct,
            "is_uncertain": (predicted_class == "uncertain"),
            "explanation": pred_res.get("explanation", "")
        })

        y_true.append(true_label)
        y_pred.append(predicted_class)
        y_raw_pred.append(raw_class)

    df_results = pd.DataFrame(results)

    # Filter out 'uncertain' for standard binary metrics or report them transparently
    confident_mask = df_results["predicted_class"] != "uncertain"
    confident_count = int(confident_mask.sum())
    uncertain_count = int((~confident_mask).sum())
    total_count = len(df_results)

    labels = ["healthy", "stressed"]

    if confident_count > 0:
        y_true_conf = df_results.loc[confident_mask, "true_label"]
        y_pred_conf = df_results.loc[confident_mask, "predicted_class"]

        acc = float(accuracy_score(y_true_conf, y_pred_conf))
        prec = float(precision_score(y_true_conf, y_pred_conf, labels=labels, average="weighted", zero_division=0))
        rec = float(recall_score(y_true_conf, y_pred_conf, labels=labels, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true_conf, y_pred_conf, labels=labels, average="weighted", zero_division=0))

        # Binary confusion matrix on confident predictions
        cm = confusion_matrix(y_true_conf, y_pred_conf, labels=labels)
    else:
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
        cm = np.zeros((2, 2), dtype=int)

    # Raw metrics (without threshold rejection) for comparison
    raw_acc = float(accuracy_score(y_true, y_raw_pred))
    raw_cm = confusion_matrix(y_true, y_raw_pred, labels=labels)

    metrics = {
        "total_test_samples": total_count,
        "confident_predictions": confident_count,
        "uncertain_predictions": uncertain_count,
        "coverage_rate": round(confident_count / max(1, total_count), 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "raw_accuracy": round(raw_acc, 4),
        "confusion_matrix": cm.tolist(),
        "raw_confusion_matrix": raw_cm.tolist(),
        "labels": labels
    }

    return {
        "success": True,
        "metrics": metrics,
        "results_table": df_results
    }


def generate_confusion_matrix_plot(cm_array: List[List[int]], labels: List[str] = ["Healthy", "Stressed"], title: str = "SPROUT AI - Test Set Confusion Matrix") -> plt.Figure:
    """
    Renders a styled Seaborn confusion matrix figure for Streamlit display.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=120)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(title, fontsize=12, pad=12, weight="bold", color="#1E3F20")
    ax.set_xlabel("Predicted Condition", fontsize=10, weight="bold", labelpad=8)
    ax.set_ylabel("Ground Truth (Actual)", fontsize=10, weight="bold", labelpad=8)
    plt.tight_layout()
    return fig
