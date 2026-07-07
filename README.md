---
title: Clinical Brain Tumor Classifier
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
---

# Clinical Brain Tumor Classification Dashboard

A Deep Learning computer vision system designed for clinical assistant use, capable of identifying brain tumor categories from T1-weighted MRI images with high confidence breakdowns.

## System Features
- **4-Class Prediction**: Glioma, Meningioma, Pituitary, or No Tumor.
- **Trained Model Detection**: Automatically loads `brain_tumor_model.h5` with full 8.7M parameters if present.
- **Visual Clinical Cards**: Detailed medical reference guides for each tumor classification directly in the interface.
- **Robust Image Preprocessing**: Standardized OpenCV resize to 128x128x3, pixel normalization, and RGB conversion matching Google Colab training.

## Local Installation & Launch

1. Clone or download this project folder.
2. Place your trained `brain_tumor_model.h5` file inside this folder.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   python app.py
   ```
5. Open `http://127.0.0.1:7860` in your web browser.
