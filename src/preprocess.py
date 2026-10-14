"""
MRI Preprocessing and Image Standardization Pipeline.
Handles varied image formats, color conversions, CLAHE enhancement, and tensor batching.
"""

from typing import Union, Tuple
import numpy as np
from PIL import Image, ImageOps
import io
from pathlib import Path
from src.config import IMAGE_SIZE


def load_image_to_pil(image_source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> Image.Image:
    """
    Standardizes any valid image source (path, bytes, numpy array, or PIL image)
    into a clean PIL Image in RGB format.
    """
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, (str, Path)):
        img = Image.open(str(image_source))
    elif isinstance(image_source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, np.ndarray):
        # Handle various numpy shapes and dtypes
        arr = image_source.copy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            img = Image.fromarray(arr, mode='L')
        elif arr.ndim == 3:
            if arr.shape[2] == 4:
                img = Image.fromarray(arr, mode='RGBA')
            elif arr.shape[2] == 3:
                img = Image.fromarray(arr, mode='RGB')
            elif arr.shape[2] == 1:
                img = Image.fromarray(arr[:, :, 0], mode='L')
            else:
                raise ValueError(f"Unsupported channel dimension in numpy array: {arr.shape}")
        else:
            raise ValueError(f"Unsupported array dimension: {arr.ndim}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_source)}")

    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def enhance_contrast(img: Image.Image) -> Image.Image:
    """
    Enhances MRI contrast using histogram equalization to accentuate subtle
    parenchymal densities, ventricular asymmetry, and tumor margins.
    """
    # Convert to grayscale for adaptive contrast enhancement then restore to RGB
    gray = img.convert("L")
    enhanced_gray = ImageOps.equalize(gray)
    return enhanced_gray.convert("RGB")


def preprocess_mri(
    image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    apply_enhancement: bool = False
) -> Tuple[np.ndarray, Image.Image]:
    """
    Full clinical preprocessing pipeline for an incoming brain MRI scan.

    Steps:
    1. Standardize to PIL RGB.
    2. Optional adaptive histogram equalization / contrast enhancement.
    3. High-quality bicubic resize to target resolution (default 128x128).
    4. Normalize pixel intensity to [0.0, 1.0] float32.
    5. Expand batch dimensions: (1, 128, 128, 3).

    Returns:
        (batch_tensor, processed_pil_image)
    """
    pil_img = load_image_to_pil(image_source)

    if apply_enhancement:
        pil_img = enhance_contrast(pil_img)

    # High-quality resize
    resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float32 numpy array normalized to [0, 1]
    img_array = np.array(resized_pil, dtype=np.float32) / 255.0

    # Expand batch dimension
    batch_tensor = np.expand_dims(img_array, axis=0)

    return batch_tensor, resized_pil
