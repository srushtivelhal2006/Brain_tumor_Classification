"""
Comprehensive System & Environment Verification Suite.
Validates dependencies, model architecture, weights loading, inference pipeline, and Grad-CAM generation.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

print("=" * 64)
print("[*] CLINICAL BRAIN TUMOR CLASSIFICATION VERIFIER")
print("=" * 64)

try:
    print("[1/5] Checking TensorFlow & Core Dependencies...")
    import tensorflow as tf
    from PIL import Image
    import matplotlib
    print(f"      - TensorFlow Version : {tf.__version__}")
    print(f"      - Pillow Version     : {Image.__version__}")
    print(f"      - Matplotlib Version : {matplotlib.__version__}")

    print("[2/5] Importing Project Modules...")
    from src.config import CLASS_NAMES, DEFAULT_MODEL_PATH, SAMPLES_DIR
    from src.model import build_custom_cnn, load_trained_model
    from src.predictor import ClinicalBrainTumorPredictor
    from src.explainability import generate_gradcam_heatmap
    print(f"      - Target Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

    print("[3/5] Verifying CNN Architecture Construction...")
    test_model = build_custom_cnn()
    param_count = test_model.count_params()
    print(f"      - Architecture Compiled.")
    print(f"      - Parameter Count: {param_count:,} (~{param_count/1e6:.2f}M)")
    assert param_count == 8711204, f"Unexpected parameter count: {param_count}"
    print("      - Parameter count precisely matches 8.71M architecture specs!")

    print("[4/5] Testing End-to-End Inference & Grad-CAM Pipeline...")
    predictor = ClinicalBrainTumorPredictor(model_path=str(DEFAULT_MODEL_PATH))
    mock_input = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    results = predictor.predict(mock_input, include_gradcam=True)

    print(f"      - Predicted Class    : {results['predicted_class']}")
    print(f"      - Confidence         : {results['confidence_percent']}")
    print(f"      - Probabilities      : {results['probabilities']}")
    print(f"      - Model Weight State : {results['model_status']}")

    assert len(results["probabilities"]) == 4, f"Expected 4 classes, got {len(results['probabilities'])}"
    for cls in CLASS_NAMES:
        assert cls in results["probabilities"], f"Class '{cls}' missing from probabilities"
    assert results["gradcam_overlay"] is not None, "Grad-CAM overlay failed to generate"
    print("      - Inference and Grad-CAM attention heatmap successfully verified!")

    print("[5/5] Checking Clinical MRI Sample Images...")
    samples = ["normal_sample.png", "glioma_sample.png", "meningioma_sample.png", "pituitary_sample.png"]
    for sample in samples:
        sample_path = SAMPLES_DIR / sample
        exists = sample_path.exists()
        print(f"      - Checking '{sample_path.name}': {'Found [OK]' if exists else 'Missing [FAIL]'}")
        assert exists, f"Error: Synthetic sample scan '{sample_path}' not found."

    print("\n" + "=" * 64)
    print("[+] ALL CHECKS PASSED: System & Pipeline are 100% operational!")
    print("=" * 64)

except Exception as e:
    print(f"\n[-] VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

sys.exit(0)
