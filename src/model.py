"""
Neural Network Architecture Definition & Weights Loading Module.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
from src.config import DEFAULT_MODEL_PATH, INPUT_SHAPE, NUM_CLASSES


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """
    Constructs the exact 6-convolutional layer deep CNN architecture
    matching the trained brain_tumor_model.h5 weights file (8,711,206 parameters).

    Architecture:
    - Conv Block 1: 2x Conv2D(32), BatchNorm, MaxPool2D(2,2), Dropout(0.25)
    - Conv Block 2: 2x Conv2D(64), BatchNorm, MaxPool2D(2,2), Dropout(0.25)
    - Conv Block 3: 2x Conv2D(128), BatchNorm, MaxPool2D(2,2), Dropout(0.25)
    - Classifier: Flatten, Dense(256), BatchNorm, Dropout(0.50), Dense(128), Dropout(0.30), Dense(4, Softmax)
    """
    model = tf.keras.models.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='conv2d'),
        tf.keras.layers.BatchNormalization(name='batch_normalization'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d'),
        tf.keras.layers.Dropout(0.25, name='dropout'),

        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
        tf.keras.layers.BatchNormalization(name='batch_normalization_1'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_1'),
        tf.keras.layers.Dropout(0.25, name='dropout_1'),

        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_4'),
        tf.keras.layers.BatchNormalization(name='batch_normalization_2'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_5'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_2'),
        tf.keras.layers.Dropout(0.25, name='dropout_2'),

        # Fully Connected / Classification Head
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(256, activation='relu', name='dense'),
        tf.keras.layers.BatchNormalization(name='batch_normalization_3'),
        tf.keras.layers.Dropout(0.50, name='dropout_3'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.30, name='dropout_4'),

        # Output Layer
        tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_2')
    ], name="brain_tumor_custom_cnn")

    return model


def load_trained_model(model_path: Optional[str] = None) -> Tuple[tf.keras.Model, bool]:
    """
    Loads pre-trained model weights from disk.
    If the specified file is absent, initializes the matching architecture in demo/unweighted mode.

    Returns:
        (model, is_loaded_from_weights)
    """
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if target_path.exists():
        try:
            print(f"[Model Loader] Loading trained weights from: {target_path}")
            model = tf.keras.models.load_model(str(target_path))
            return model, True
        except Exception as err:
            print(f"[Model Loader Warning] Failed loading via load_model: {err}. Building architecture and trying load_weights...")
            try:
                model = build_custom_cnn()
                model.load_weights(str(target_path))
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                return model, True
            except Exception as w_err:
                print(f"[Model Loader Error] Could not load weights: {w_err}. Falling back to unweighted architecture.")

    print(f"[Model Loader Notice] Weights file '{target_path}' not found or incompatible. Initializing blank architecture.")
    model = build_custom_cnn()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, False
