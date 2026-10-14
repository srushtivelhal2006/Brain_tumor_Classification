"""
Unit tests for Grad-CAM explainability and heatmap visualization.
"""

import unittest
import numpy as np
from PIL import Image
import tensorflow as tf

from src.model import build_custom_cnn
from src.explainability import generate_gradcam_heatmap, overlay_gradcam


class TestExplainability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = build_custom_cnn()
        # Initialize graph
        dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
        _ = cls.model(dummy)

    def test_gradcam_heatmap_dimensions_and_range(self):
        """Verifies Grad-CAM computes 2D heatmap bounded in [0.0, 1.0]."""
        dummy_tensor = np.random.uniform(0.0, 1.0, (1, 128, 128, 3)).astype(np.float32)
        heatmap = generate_gradcam_heatmap(
            self.model,
            dummy_tensor,
            target_class_index=0,
            target_layer_name="conv2d_5"
        )

        self.assertEqual(heatmap.ndim, 2)
        self.assertEqual(heatmap.shape, (32, 32))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_overlay_gradcam(self):
        """Verifies that heatmap overlays cleanly onto original image."""
        orig_img = Image.new("RGB", (256, 256), color=(40, 40, 40))
        heatmap = np.random.uniform(0.0, 1.0, (32, 32)).astype(np.float32)
        overlay = overlay_gradcam(orig_img, heatmap, alpha=0.5)

        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, (256, 256))
        self.assertEqual(overlay.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
