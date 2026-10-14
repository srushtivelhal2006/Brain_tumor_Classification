"""
Unit tests for preprocessing and clinical prediction pipeline.
"""

import unittest
import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, NUM_CLASSES
from src.preprocess import load_image_to_pil, preprocess_mri, enhance_contrast
from src.predictor import ClinicalBrainTumorPredictor


class TestPredictionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = ClinicalBrainTumorPredictor()

    def test_image_loader_variants(self):
        """Verifies loader handles numpy array, PIL, and grayscale formats."""
        gray_arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        pil_from_gray = load_image_to_pil(gray_arr)
        self.assertEqual(pil_from_gray.mode, "RGB")
        self.assertEqual(pil_from_gray.size, (100, 100))

        rgba_arr = np.random.randint(0, 255, (80, 80, 4), dtype=np.uint8)
        pil_from_rgba = load_image_to_pil(rgba_arr)
        self.assertEqual(pil_from_rgba.mode, "RGB")

    def test_mri_preprocessing(self):
        """Verifies tensor shape, normalization to [0, 1], and resizing."""
        dummy_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        tensor, processed_pil = preprocess_mri(dummy_img, target_size=(128, 128))

        self.assertEqual(tensor.shape, (1, 128, 128, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(tensor >= 0.0) and np.all(tensor <= 1.0))
        self.assertEqual(processed_pil.size, (128, 128))

    def test_clinical_predictor_output_structure(self):
        """Tests that prediction dictionary has all expected fields and valid probabilities."""
        dummy_img = Image.new("RGB", (128, 128), color=(50, 50, 50))
        result = self.predictor.predict(dummy_img, include_gradcam=True)

        self.assertIn("predicted_class", result)
        self.assertIn(result["predicted_class"], CLASS_NAMES)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertEqual(len(result["probabilities"]), NUM_CLASSES)

        # Check probabilities sum to 1.0
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # Check metadata
        self.assertIn("metadata", result)
        self.assertIn("urgency", result)
        self.assertIn("clinical_summary", result)
        self.assertIsNotNone(result["gradcam_overlay"])


if __name__ == "__main__":
    unittest.main()
