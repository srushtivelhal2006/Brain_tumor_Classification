import os
import sys
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("🧠 CLINICAL BRAIN TUMOR CLASSIFICATION VERIFIER")
print("="*60)

try:
    print("[1/5] Importing TensorFlow and project dependencies...")
    import tensorflow as tf
    print(f"      - TensorFlow version: {tf.__version__}")
    
    print("[2/5] Importing App Components...")
    import app
    print("      - Successfully imported app.py core modules.")
    
    print("[3/5] Verifying CNN Architecture Construction...")
    test_model = app.build_custom_cnn()
    test_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    param_count = test_model.count_params()
    print(f"      - Custom CNN compiled successfully.")
    print(f"      - Model Parameters Count: {param_count:,} (~{param_count/1e6:.1f} Million)")
    
    # Assert size matching 8.7M specs
    assert param_count > 8000000 and param_count < 9500000, f"Error: Unexpected parameter count: {param_count}"
    print("      - Parameters size matched custom specs!")
    
    print("[4/5] Testing Image Preprocessing & Inference...")
    # Generate random mock image of size 200x200x3 (similar to raw upload)
    mock_input = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    results, status = app.predict_brain_tumor(mock_input)
    
    print(f"      - Inference Results Dictionary: {results}")
    print(f"      - Prediction Findings Status: '{status}'")
    
    assert len(results) == 4, f"Error: Expected 4 classes in prediction dictionary, got {len(results)}"
    for cls in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]:
        assert cls in results, f"Error: Class '{cls}' not found in predictions"
        
    print("      - Image processing pipeline and predictions validated successfully!")

    print("[5/5] Checking Synthetic MRI Sample Images...")
    samples = ["normal_sample.png", "glioma_sample.png", "meningioma_sample.png", "pituitary_sample.png"]
    for sample in samples:
        path = os.path.join("samples", sample)
        exists = os.path.exists(path)
        print(f"      - Checking '{path}': {'Found ✅' if exists else 'Missing ❌'}")
        assert exists, f"Error: Synthetic sample image '{path}' was not generated"
        
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED: Gradio Backend is fully operational!")
    print("="*60)

except Exception as e:
    print(f"\n❌ VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
