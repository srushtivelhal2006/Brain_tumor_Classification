"""
Unit tests for neural network architecture and weight loading.
"""

import unittest
import numpy as np
import tensorflow as tf

from src.config import INPUT_SHAPE, NUM_CLASSES, DEFAULT_MODEL_PATH
from src.model import build_custom_cnn, load_trained_model


class TestModelArchitecture(unittest.TestCase):

    def test_cnn_architecture_shapes_and_params(self):
        """Validates that custom CNN matches exact 6-conv specs and layer dimensions."""
        model = build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
        self.assertIsNotNone(model)

        # Check total parameter count
        param_count = model.count_params()
        self.assertEqual(param_count, 8711204, f"Expected 8,711,204 params, got {param_count}")

        # Check output tensor shape
        dummy_input = np.zeros((2, *INPUT_SHAPE), dtype=np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, NUM_CLASSES))

        # Check softmax properties (sums to ~1.0)
        sums = np.sum(output.numpy(), axis=-1)
        np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-5)

    def test_load_trained_model(self):
        """Tests that trained model weights load correctly from disk if present."""
        model, is_loaded = load_trained_model(str(DEFAULT_MODEL_PATH))
        self.assertIsNotNone(model)
        self.assertTrue(is_loaded, "Trained weights file brain_tumor_model.h5 failed to load")
        self.assertEqual(model.output_shape[-1], NUM_CLASSES)


if __name__ == "__main__":
    unittest.main()
