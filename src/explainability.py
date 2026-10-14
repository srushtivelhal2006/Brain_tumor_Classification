"""
Explainable AI (XAI) Module: Gradient-Weighted Class Activation Mapping (Grad-CAM).
Provides visual localization of neuro-oncological lesions to enhance clinical trust.
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    target_class_index: Optional[int] = None,
    target_layer_name: str = "conv2d_5"
) -> np.ndarray:
    """
    Computes a 2D Grad-CAM heatmap highlighting regions the CNN prioritized
    when predicting target_class_index.

    Args:
        model: Trained tf.keras.Model.
        input_tensor: Normalized image batch tensor of shape (1, 128, 128, 3).
        target_class_index: Index of class to explain (if None, uses argmax prediction).
        target_layer_name: Name of final convolutional layer (default: 'conv2d_5').

    Returns:
        2D numpy array with values in [0.0, 1.0].
    """
    # Find layer and index
    conv_layer = None
    conv_idx = -1
    for idx, layer in enumerate(model.layers):
        if layer.name == target_layer_name:
            conv_layer = layer
            conv_idx = idx
            break

    if conv_layer is None:
        # Fallback to the last Conv2D layer in the model
        for idx in reversed(range(len(model.layers))):
            if isinstance(model.layers[idx], tf.keras.layers.Conv2D):
                conv_layer = model.layers[idx]
                conv_idx = idx
                break

    if conv_layer is None:
        raise ValueError("No suitable Conv2D layer found in the model for Grad-CAM.")

    tf_inputs = tf.constant(input_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = tf_inputs
        conv_output = None
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == conv_idx:
                conv_output = x
                tape.watch(conv_output)
        predictions = x

        if target_class_index is None:
            target_class_index = int(tf.argmax(predictions[0]))

        loss = predictions[:, target_class_index]

    # Compute gradient of the score with respect to feature map activations
    grads = tape.gradient(loss, conv_output)

    # Global average pooling over spatial dimensions to get channel importance
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by gradient importance
    conv_features = conv_output[0]
    heatmap = conv_features @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (we only care about features that have positive impact on target class)
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize between 0.0 and 1.0
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Image.Image:
    """
    Superimposes the Grad-CAM heatmap over the original patient MRI scan.

    Args:
        original_image: PIL Image of the original scan.
        heatmap: 2D numpy array [0, 1] from generate_gradcam_heatmap.
        alpha: Blending weight for the heatmap overlay (0.0 = original, 1.0 = heatmap).
        colormap_name: Matplotlib colormap (e.g. 'jet', 'turbo', 'inferno').

    Returns:
        PIL Image with superimposed lesion localization attention overlay.
    """
    orig_w, orig_h = original_image.size

    # Resize heatmap to match original image dimensions using bicubic interpolation
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = heatmap_pil.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
    heatmap_arr = np.array(heatmap_resized, dtype=np.float32) / 255.0

    # Colorize heatmap with colormap
    try:
        import matplotlib
        colormap = matplotlib.colormaps[colormap_name]
    except Exception:
        colormap = cm.get_cmap(colormap_name)
    colored_heatmap = colormap(heatmap_arr)[:, :, :3]  # Strip alpha, retain RGB [0, 1]
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

    # Blend original and colored heatmap
    orig_arr = np.array(original_image.convert("RGB"), dtype=np.float32)
    blended_arr = (1.0 - alpha) * orig_arr + alpha * colored_heatmap_uint8
    blended_arr = np.clip(blended_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_arr)
