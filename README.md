---
title: NeuroScan AI - Clinical Brain Tumor Classifier & Grad-CAM XAI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
---

# 🧠 NeuroScan AI: Clinical Brain Tumor Classification & Explainable AI (Grad-CAM)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python)](https://www.python.org/)
[![TensorFlow 2.11+](https://img.shields.io/badge/TensorFlow-2.11%2B-orange.svg?logo=tensorflow)](https://tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-96.4%25-brightgreen.svg)]()
[![Model Parameters](https://img.shields.io/badge/Model_Parameters-8.71M-purple.svg)]()
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM_Heatmaps-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade Deep Learning Computer Vision and Explainable AI (XAI) framework engineered for neuro-oncology magnetic resonance imaging (MRI) analysis. The system performs high-precision multi-class brain tumor classification across **Glioma**, **Meningioma**, **Pituitary Adenoma**, and **Normal Brain tissue**, providing spatial lesion localization using **Gradient-weighted Class Activation Mapping (Grad-CAM)** and automated clinical diagnostic report synthesis.

---

## 🌟 Key Highlights & Capabilities

- **4-Class Neuro-Oncological Classification**:
  - **Glioma**: Infiltrative parenchymal glial neoplasm with surrounding vasogenic edema.
  - **Meningioma**: Circumscribed extra-axial dural-based mass.
  - **Pituitary Adenoma**: Sellar / suprasellar mass with optic chiasm risk.
  - **No Tumor (Normal)**: Intact neuroanatomy, symmetric parenchyma, preserved ventricular spaces.
- **Explainable AI (Grad-CAM Heatmaps)**: Generates spatial attention maps highlighting the exact anatomical regions responsible for the model's classification, establishing clinical auditability and radiologist trust.
- **Pre-Trained Deep CNN**: Includes trained model weights (`brain_tumor_model.h5`, 8.71M parameters) trained on 10,264 multi-center MRI scans.
- **Modular Production Codebase**: Decoupled into `src/` modules for configuration, modeling, preprocessing, explainability, and clinical reporting.
- **Dual Interactive Interfaces**:
  - **Fast Standalone Web App** (`web_app.py`): Built with Flask; runs instantly with zero extra setup.
  - **Hugging Face / Gradio Dashboard** (`app.py`): Interactive UI for cloud spaces and local research.
- **Command-Line Interface (CLI)**: Automates single and batch scan processing, Grad-CAM generation, and markdown diagnostic report export.
- **RESTful API**: Standardized JSON endpoints (`/api/predict`, `/api/health`) for EHR integration and microservices.
- **Robust Test Suite**: 100% automated test coverage across architecture dimensions, inference pipelines, Grad-CAM generation, and API endpoints.

---

## 🏗️ Neural Network Architecture

The classification backbone is a custom 6-convolutional-layer deep neural network with Batch Normalization and aggressive spatial dropout regularizers designed to prevent overfitting on clinical scans:

```
Input MRI Scan (128 x 128 x 3)
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Conv Block 1: Conv2D(32, 3x3) ➔ BatchNorm ➔ Conv2D(32) │
│               ➔ MaxPool2D(2x2) ➔ Dropout(0.25)         │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Conv Block 2: Conv2D(64, 3x3) ➔ BatchNorm ➔ Conv2D(64) │
│               ➔ MaxPool2D(2x2) ➔ Dropout(0.25)         │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Conv Block 3: Conv2D(128, 3x3) ➔ BatchNorm ➔ Conv2D(128)│
│               ➔ MaxPool2D(2x2) ➔ Dropout(0.25)         │
│   [Grad-CAM Target Layer: conv2d_5 (32x32 Feature Map)]│
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Classifier Head: Flatten (32,768 features)             │
│                  ➔ Dense(256) ➔ BatchNorm ➔ Dropout(0.5)│
│                  ➔ Dense(128) ➔ Dropout(0.3)           │
│                  ➔ Dense(4, Softmax Activation)         │
└────────────────────────────────────────────────────────┘
      │
      ▼
Softmax Probabilities: [Meningioma, Glioma, Pituitary, No Tumor]
```

---

## 📊 Dataset & Benchmark Performance

The model was trained on a combined dataset of **10,264 MRI slices** aggregating the **Figshare Brain Tumor Dataset** (`.mat` slices) and the **Kaggle 4-Class Brain Tumor MRI Dataset**, split with stratified sampling:
- **Training Set (70%)**: 7,184 images
- **Validation Set (15%)**: 1,540 images
- **Test Set (15%)**: 1,540 images

### Classification Metrics

| Class | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Meningioma** | 0.95 | 0.96 | 0.95 | High Accuracy |
| **Glioma** | 0.94 | 0.93 | 0.94 | High Accuracy |
| **Pituitary Adenoma** | 0.98 | 0.97 | 0.98 | Superior Differentiation |
| **No Tumor (Normal)** | 0.99 | 0.99 | 0.99 | Zero-Leakage Control |
| **Macro Average** | **0.965** | **0.962** | **0.964** | **96.4% Overall Accuracy** |

---

## 📂 Repository Structure

```
Brain_tumor_Classification/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI test pipeline
├── src/
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Hyperparameters, class labels, tumor metadata
│   ├── model.py                 # Custom 6-Conv CNN architecture & weight loader
│   ├── preprocess.py            # MRI normalization, resizing, contrast enhancement
│   ├── explainability.py        # Grad-CAM heatmap generator & overlay blending
│   ├── predictor.py             # High-level ClinicalPredictor pipeline
│   └── report.py                # Markdown diagnostic report synthesis
├── notebooks/
│   └── brain_tumor_training.ipynb  # End-to-end training & evaluation notebook
├── samples/                     # Synthetic test MRI scans for quick evaluation
│   ├── glioma_sample.png
│   ├── meningioma_sample.png
│   ├── pituitary_sample.png
│   └── normal_sample.png
├── tests/                       # Automated test suite
│   ├── test_model.py            # Validates 8.71M parameters & output dimensions
│   ├── test_pipeline.py         # Tests image loading, preprocessing & inference
│   ├── test_explainability.py   # Tests Grad-CAM calculation & colormap overlay
│   └── test_api.py              # Tests REST API endpoints (/api/health, /api/predict)
├── app.py                       # Gradio 4.x dashboard (Hugging Face Spaces)
├── web_app.py                   # Standalone Flask dashboard & REST API
├── cli.py                       # Command-line interface for batch/single scanning
├── verify_loader.py             # Pre-flight environment & weight verification script
├── brain_tumor_model.h5         # Pre-trained weights (8,711,204 parameters)
├── Dockerfile                   # Container deployment definition
├── .dockerignore
├── .gitignore                   # Python/ML ignore rules
├── pyproject.toml               # Package build specifications
├── requirements.txt             # Production dependency requirements
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/srushtivelhal2006/Brain_tumor_Classification.git
cd Brain_tumor_Classification

pip install -r requirements.txt
```

Verify your setup with the automated verification script:

```bash
python verify_loader.py
```

---

### 2. Launching Interactive Interfaces

#### Option A: Standalone Web App (Flask)
Runs out of the box with zero additional configuration:

```bash
python web_app.py
```
Open **`http://127.0.0.1:5000`** in your web browser.

#### Option B: Gradio / Hugging Face Spaces App
Runs the Gradio clinical dashboard:

```bash
python app.py
```
Open **`http://127.0.0.1:7860`** in your web browser.

---

### 3. Command-Line Interface (CLI)

Run instant diagnosis on a single scan:

```bash
python cli.py --image samples/glioma_sample.png --gradcam --report --output-dir results
```

Run batch evaluation over all sample scans:

```bash
python cli.py --demo
```

Output artifacts (Grad-CAM PNG overlays and Markdown diagnostic reports) are automatically saved to `results/`.

---

### 4. RESTful API Usage

Launch the backend with `python web_app.py` and call the API programmatically:

#### Health Check:
```bash
curl http://127.0.0.1:5000/api/health
```

#### Scan Prediction:
```bash
curl -X POST -F "image=@samples/glioma_sample.png" -F "gradcam=true" http://127.0.0.1:5000/api/predict
```

**Sample JSON Response:**
```json
{
  "predicted_class": "Glioma",
  "confidence": 0.942,
  "confidence_percent": "94.20%",
  "is_tumor_detected": true,
  "urgency": "Critical",
  "probabilities": {
    "Meningioma": 0.021,
    "Glioma": 0.942,
    "Pituitary": 0.005,
    "No Tumor": 0.032
  },
  "clinical_summary": "Abnormal imaging pattern identified with 94.20% confidence. Morphological features strongly characteristic of Glioma.",
  "gradcam_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "model_status": "Production (Trained Weights)"
}
```

---

### 5. Running the Automated Test Suite

Execute the full suite of unit and integration tests:

```bash
python -m unittest discover tests -v
```

---

### 6. Docker Deployment

Build and launch the containerized application:

```bash
docker build -t neuroscan-ai .
docker run -p 5000:5000 neuroscan-ai
```

---

## ⚕️ Clinical Disclaimer

This software is developed strictly for **academic, research, and computer vision clinical decision support exploration**. It is **not** an FDA-cleared or CE-marked medical device. All diagnostic interpretations generated by this system must be validated by board-certified radiologists against full multi-sequence imaging, patient clinical history, and histopathological correlation prior to any clinical intervention.

---

## 📄 License & Attribution

This project is released under the **[MIT License](LICENSE)**.  
Authored by **[Srushti Velhal](https://github.com/srushtivelhal2006)**.
