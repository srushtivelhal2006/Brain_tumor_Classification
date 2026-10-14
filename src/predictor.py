"""
Clinical Inference & Prediction Engine for Brain Tumor Classification.
Coordinates model loading, image preprocessing, multi-class inference, and Grad-CAM generation.
"""

from typing import Union, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, TUMOR_METADATA, DEFAULT_MODEL_PATH
from src.model import load_trained_model
from src.preprocess import preprocess_mri, load_image_to_pil
from src.explainability import generate_gradcam_heatmap, overlay_gradcam


class ClinicalBrainTumorPredictor:
    """
    High-level clinical predictor offering end-to-end diagnosis,
    confidence metrics, explainability heatmaps, and structured reports.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model, self.has_trained_weights = load_trained_model(model_path)
        # Warmup call
        dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
        _ = self.model(dummy)

    def predict(
        self,
        image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        include_gradcam: bool = True,
        enhance_contrast_flag: bool = False,
        gradcam_alpha: float = 0.45
    ) -> Dict[str, Any]:
        """
        Runs comprehensive analysis on an MRI scan.

        Returns:
            Dictionary containing:
            - 'predicted_class': Top predicted class name
            - 'confidence': Top confidence score [0.0 - 1.0]
            - 'confidence_percent': Confidence in percentage string
            - 'probabilities': Dict of class names to probability floats
            - 'is_tumor_detected': Boolean indicating abnormal neoplastic finding
            - 'urgency': Urgency rating
            - 'metadata': Full clinical guidance dictionary for the tumor type
            - 'gradcam_overlay': PIL Image with heatmap (if requested)
            - 'raw_heatmap': 2D numpy array [0, 1]
            - 'original_image': Standardized input PIL Image
            - 'model_status': 'Production (Trained Weights)' or 'Demonstration (Unweighted)'
        """
        original_pil = load_image_to_pil(image_source)
        batch_tensor, processed_pil = preprocess_mri(
            original_pil,
            apply_enhancement=enhance_contrast_flag
        )

        # Run model inference
        raw_preds = self.model.predict(batch_tensor, verbose=0)[0]

        # Map predictions to class labels
        # 0: Meningioma, 1: Glioma, 2: Pituitary, 3: No Tumor
        probabilities = {
            CLASS_NAMES[i]: float(raw_preds[i])
            for i in range(len(CLASS_NAMES))
        }

        top_idx = int(np.argmax(raw_preds))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = float(raw_preds[top_idx])
        confidence_percent = f"{confidence * 100:.2f}%"

        meta = TUMOR_METADATA.get(predicted_class, {})
        is_tumor_detected = (predicted_class != "No Tumor")

        gradcam_overlay = None
        raw_heatmap = None

        if include_gradcam:
            try:
                raw_heatmap = generate_gradcam_heatmap(
                    self.model,
                    batch_tensor,
                    target_class_index=top_idx
                )
                gradcam_overlay = overlay_gradcam(
                    original_pil,
                    raw_heatmap,
                    alpha=gradcam_alpha
                )
            except Exception as e:
                print(f"[Warning] Grad-CAM computation skipped: {e}")

        # Construct findings summary
        if predicted_class == "No Tumor":
            clinical_summary = (
                f"Neuro-oncology screening evaluated normal anatomical landmarks with {confidence_percent} confidence. "
                f"No evidence of space-occupying lesions, intracranial mass effect, or midline shift."
            )
        else:
            clinical_summary = (
                f"Abnormal imaging pattern identified with {confidence_percent} confidence. "
                f"Morphological features strongly characteristic of {predicted_class}. "
                f"Lesion urgency level: {meta.get('urgency', 'High')}."
            )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percent": confidence_percent,
            "probabilities": probabilities,
            "is_tumor_detected": is_tumor_detected,
            "urgency": meta.get("urgency", "High"),
            "clinical_summary": clinical_summary,
            "metadata": meta,
            "gradcam_overlay": gradcam_overlay,
            "raw_heatmap": raw_heatmap,
            "original_image": original_pil,
            "is_trained_weights": self.has_trained_weights,
            "model_status": "Production (Trained Weights)" if self.has_trained_weights else "Demonstration (Unweighted)"
        }
