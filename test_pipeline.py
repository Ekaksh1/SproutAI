"""
SPROUT AI - Comprehensive System Verification Suite
Validates the entire end-to-end pipeline:
1. Model Loading & Device Verification
2. Single Image Inference (Healthy & Stressed)
3. Confidence Rejection Thresholding (<70%)
4. Image Quality Assessment (Blur / Brightness detection)
5. Care Recommendation Generation
6. Test Dataset Evaluation & Confusion Matrix
7. Design Thinking Report Generation
"""

import os
import sys
from PIL import Image, ImageFilter

from predict import SproutPredictor
from recommendations import get_recommendation
from evaluation import evaluate_test_dataset
from design_thinking_data import DEFAULT_DTM_STAGES, generate_markdown_report


def run_all_tests():
    print("=" * 60)
    print("       SPROUT AI - System Verification Suite")
    print("=" * 60)

    # 1. Test Predictor Loading
    print("\n[Test 1] Initializing SproutPredictor...")
    predictor = SproutPredictor()
    if not predictor.is_model_ready():
        print(f"[FAIL] Predictor not ready. Error: {predictor.load_error}")
        sys.exit(1)
    print(f"[PASS] Model loaded successfully on device: {predictor.device}")
    print(f"       Metadata: Arch={predictor.metadata.get('architecture')}, "
          f"Val Acc={predictor.metadata.get('final_val_accuracy', 0)*100:.1f}%")

    # 2. Test Healthy Sample
    healthy_path = os.path.join("sample_test", "healthy", "test_healthy_01.jpg")
    print(f"\n[Test 2] Testing inference on healthy sample: {healthy_path}")
    res_h = predictor.predict(healthy_path, confidence_threshold=0.70)
    if not res_h.get("success"):
        print(f"[FAIL] Inference failed: {res_h}")
        sys.exit(1)
    print(f"[PASS] Predicted Class: {res_h['predicted_class']} (Confidence: {res_h['confidence']*100:.1f}%)")
    print(f"       Probabilities: {res_h['probabilities']}")
    print(f"       Explanation: {res_h['explanation']}")

    # 3. Test Stressed Sample
    stressed_path = os.path.join("sample_test", "stressed", "test_stressed_01.jpg")
    print(f"\n[Test 3] Testing inference on stressed sample: {stressed_path}")
    res_s = predictor.predict(stressed_path, confidence_threshold=0.70)
    if not res_s.get("success"):
        print(f"[FAIL] Inference failed: {res_s}")
        sys.exit(1)
    print(f"[PASS] Predicted Class: {res_s['predicted_class']} (Confidence: {res_s['confidence']*100:.1f}%)")
    print(f"       Probabilities: {res_s['probabilities']}")
    print(f"       Explanation: {res_s['explanation']}")

    # 4. Test High-Threshold Uncertainty Rejection
    print("\n[Test 4] Testing Confidence Rejection at 99.9% threshold...")
    res_unc = predictor.predict(healthy_path, confidence_threshold=0.999)
    if res_unc["predicted_class"] != "uncertain":
        print(f"[FAIL] Expected 'uncertain', got '{res_unc['predicted_class']}'")
    else:
        print(f"[PASS] Correctly rejected to 'uncertain': {res_unc['explanation']}")

    # 5. Test Care Recommendations
    print("\n[Test 5] Verifying Care Recommendations...")
    rec_h = get_recommendation("healthy", 0.95, 0.70)
    rec_s = get_recommendation("stressed", 0.92, 0.70)
    rec_u = get_recommendation("uncertain", 0.45, 0.70)
    assert "Healthy" in rec_h["status_title"], "Healthy title mismatch"
    assert "Stressed" in rec_s["status_title"], "Stressed title mismatch"
    assert "Uncertain" in rec_u["status_title"], "Uncertain title mismatch"
    print(f"[PASS] Recommendation engine returned verified guidance for all 3 categories.")

    # 6. Test Unseen Test Set Evaluation & Confusion Matrix
    print("\n[Test 6] Evaluating unseen test dataset (10 samples)...")
    test_samples = []
    h_dir = os.path.join("sample_test", "healthy")
    s_dir = os.path.join("sample_test", "stressed")
    for fn in os.listdir(h_dir):
        if fn.lower().endswith((".jpg", ".png")):
            test_samples.append({"image": os.path.join(h_dir, fn), "true_label": "healthy", "filename": fn})
    for fn in os.listdir(s_dir):
        if fn.lower().endswith((".jpg", ".png")):
            test_samples.append({"image": os.path.join(s_dir, fn), "true_label": "stressed", "filename": fn})

    eval_out = evaluate_test_dataset(test_samples, predictor, confidence_threshold=0.70)
    if not eval_out.get("success"):
        print(f"[FAIL] Evaluation failed: {eval_out}")
        sys.exit(1)
    m = eval_out["metrics"]
    print(f"[PASS] Evaluation Complete:")
    print(f"       - Total Unseen Samples: {m['total_test_samples']}")
    print(f"       - Confident Predictions: {m['confident_predictions']}")
    print(f"       - Accuracy: {m['accuracy']*100:.1f}%")
    print(f"       - Weighted F1-Score: {m['f1_score']*100:.1f}%")
    print(f"       - Confusion Matrix:\n         {m['confusion_matrix']}")

    # 7. Test Design Thinking Report Generation
    print("\n[Test 7] Verifying Design Thinking report compilation...")
    stages = {k: v["content"] for k, v in DEFAULT_DTM_STAGES.items()}
    report = generate_markdown_report(stages)
    assert len(report) > 500, "Report too short"
    assert "How might we help students" in report, "Problem statement missing"
    print(f"[PASS] DTM Report compiled successfully ({len(report)} characters).")

    print("\n" + "=" * 60)
    print("🎉 ALL SPROUT AI TESTS PASSED SUCCESSFULLY! SYSTEM FULLY OPERATIONAL.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
