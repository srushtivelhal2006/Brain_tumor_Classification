"""
Standalone Clinical Brain Tumor Classification & Grad-CAM Web Application.
Built with Flask (pre-installed, lightweight, production-ready, no heavy dependencies required).

Usage:
    python web_app.py
    Open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import io
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CLASS_NAMES, TUMOR_METADATA, DATASET_METRICS, DEFAULT_MODEL_PATH, SAMPLES_DIR
from src.predictor import ClinicalBrainTumorPredictor
from src.report import generate_markdown_report

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload

# Initialize inference predictor
print("[Flask App] Initializing Clinical Brain Tumor Predictor...")
predictor = ClinicalBrainTumorPredictor(model_path=str(DEFAULT_MODEL_PATH))
print("[Flask App] Predictor ready!")


def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Brain Tumor Classifier & Grad-CAM XAI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.75);
            --border-color: rgba(148, 163, 184, 0.15);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-indigo: #6366f1;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(circle at 20% 15%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 80% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            border-bottom: 1px solid var(--border-color);
            background: rgba(10, 14, 23, 0.8);
            backdrop-filter: blur(12px);
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-icon {
            font-size: 2rem;
            background: rgba(59, 130, 246, 0.15);
            padding: 8px 12px;
            border-radius: 12px;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .brand-title {
            font-size: 1.35rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-subtitle {
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.1);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
        }

        main {
            flex: 1;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            padding: 30px 24px;
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 28px;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            margin-bottom: 24px;
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-desc {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-bottom: 18px;
        }

        /* Upload Area */
        .dropzone {
            border: 2px dashed rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: rgba(15, 23, 42, 0.4);
            position: relative;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.05);
        }

        .dropzone input[type="file"] {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }

        .dropzone-icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        /* Options */
        .options-group {
            display: flex;
            gap: 16px;
            margin: 18px 0;
        }

        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            cursor: pointer;
        }

        .btn-analyze {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-indigo) 100%);
            color: #ffffff;
            border: none;
            padding: 14px 24px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35);
        }

        .btn-analyze:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(37, 99, 235, 0.5);
        }

        .btn-analyze:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Samples Section */
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 12px;
        }

        .sample-item {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }

        .sample-item:hover {
            border-color: var(--accent-blue);
            transform: translateY(-2px);
        }

        .sample-item img {
            width: 100%;
            height: 80px;
            object-fit: contain;
            border-radius: 6px;
            background: #000;
        }

        .sample-label {
            font-size: 0.75rem;
            margin-top: 6px;
            color: #cbd5e1;
            font-weight: 600;
        }

        /* Results Display */
        .result-view {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .image-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        @media (max-width: 640px) {
            .image-comparison {
                grid-template-columns: 1fr;
            }
        }

        .image-box {
            background: #000000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            text-align: center;
        }

        .image-box img {
            width: 100%;
            height: 240px;
            object-fit: contain;
            display: block;
        }

        .image-caption {
            padding: 8px 12px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            background: rgba(15, 23, 42, 0.9);
            font-weight: 500;
        }

        /* Diagnostic Banner */
        .finding-banner {
            padding: 16px 20px;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.8);
            border-left: 5px solid var(--accent-blue);
        }

        .finding-title {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 4px;
        }

        .finding-meta {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .finding-desc {
            font-size: 0.85rem;
            color: #cbd5e1;
            line-height: 1.5;
        }

        /* Probability Bars */
        .prob-item {
            margin-bottom: 12px;
        }

        .prob-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .prob-track {
            background: rgba(30, 41, 59, 0.8);
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
        }

        .prob-bar {
            height: 100%;
            border-radius: 5px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        /* Medical Reference Guide */
        .reference-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 14px;
        }

        .ref-card {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 10px;
            padding: 12px;
            border-left: 4px solid var(--border-color);
        }

        .ref-title {
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .ref-desc {
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        footer {
            border-top: 1px solid var(--border-color);
            padding: 20px 40px;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-secondary);
            background: rgba(10, 14, 23, 0.9);
        }
    </style>
</head>
<body>
    <header>
        <div class="header-brand">
            <div class="brand-icon">🧠</div>
            <div>
                <h1 class="brand-title">NeuroScan AI · Clinical Tumor Classifier</h1>
                <p class="brand-subtitle">Deep Learning Diagnostic Decision Support with Grad-CAM Explainability</p>
            </div>
        </div>
        <div>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span>Model: 8.71M Params (Loaded)</span>
            </div>
        </div>
    </header>

    <main>
        <!-- Left Section: Upload & Controls -->
        <div>
            <div class="card">
                <h2 class="card-title">📥 Patient MRI Scan</h2>
                <p class="card-desc">Upload a cranial axial or coronal MRI slice in PNG, JPEG, or DICOM-exported format.</p>

                <div class="dropzone" id="dropzone">
                    <input type="file" id="fileInput" accept="image/*">
                    <div class="dropzone-icon">🔬</div>
                    <div style="font-weight: 600; margin-bottom: 4px;">Click or Drag & Drop MRI Image</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem;">Supports PNG, JPG, JPEG, TIFF (Max 32MB)</div>
                </div>

                <div class="options-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="gradcamToggle" checked>
                        <span>Grad-CAM Attention Overlay</span>
                    </label>
                    <label class="checkbox-label">
                        <input type="checkbox" id="enhanceToggle">
                        <span>Contrast Equalization (CLAHE)</span>
                    </label>
                </div>

                <button class="btn-analyze" id="analyzeBtn" onclick="runAnalysis()">🔍 Analyze MRI Scan</button>
            </div>

            <div class="card">
                <h2 class="card-title">📂 Pre-Loaded Clinical Cases</h2>
                <p class="card-desc">Click any synthetic MRI scan to load and evaluate instantaneously:</p>
                
                <div class="sample-grid">
                    <div class="sample-item" onclick="loadSample('glioma_sample.png', 'Glioma')">
                        <img src="/sample/glioma_sample.png" alt="Glioma">
                        <div class="sample-label">Glioma</div>
                    </div>
                    <div class="sample-item" onclick="loadSample('meningioma_sample.png', 'Meningioma')">
                        <img src="/sample/meningioma_sample.png" alt="Meningioma">
                        <div class="sample-label">Meningioma</div>
                    </div>
                    <div class="sample-item" onclick="loadSample('pituitary_sample.png', 'Pituitary')">
                        <img src="/sample/pituitary_sample.png" alt="Pituitary">
                        <div class="sample-label">Pituitary</div>
                    </div>
                    <div class="sample-item" onclick="loadSample('normal_sample.png', 'Normal')">
                        <img src="/sample/normal_sample.png" alt="Normal">
                        <div class="sample-label">No Tumor</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2 class="card-title">📚 Neuro-Oncology Reference Guide</h2>
                <div class="reference-grid">
                    <div class="ref-card" style="border-left-color: var(--accent-amber);">
                        <div class="ref-title" style="color: var(--accent-amber);">Glioma</div>
                        <div class="ref-desc">Intra-axial glial origin. Infiltrative margins, heterogeneous signal, edema, necrosis.</div>
                    </div>
                    <div class="ref-card" style="border-left-color: var(--accent-red);">
                        <div class="ref-title" style="color: var(--accent-red);">Meningioma</div>
                        <div class="ref-desc">Extra-axial dural attachment. Sharp margins, intense contrast enhancement, dural tail.</div>
                    </div>
                    <div class="ref-card" style="border-left-color: var(--accent-purple);">
                        <div class="ref-title" style="color: var(--accent-purple);">Pituitary</div>
                        <div class="ref-desc">Sellar base lesion. Well-delineated mass, optic chiasm compression, endocrine signs.</div>
                    </div>
                    <div class="ref-card" style="border-left-color: var(--accent-green);">
                        <div class="ref-title" style="color: var(--accent-green);">No Tumor</div>
                        <div class="ref-desc">Normal neuroanatomy. Symmetric parenchyma, normal ventricular spaces.</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Section: Results, Heatmap & Probabilities -->
        <div class="result-view">
            <div class="card">
                <h2 class="card-title">🎯 Diagnostic Findings</h2>
                <div id="findingsContainer">
                    <div style="text-align: center; padding: 40px 20px; color: var(--text-secondary);">
                        <div style="font-size: 2.2rem; margin-bottom: 10px;">🩺</div>
                        <p>Upload a patient MRI scan or select a sample case above to generate an immediate diagnostic breakdown.</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2 class="card-title">🖼️ Anatomical & Grad-CAM Visualizations</h2>
                <div class="image-comparison">
                    <div class="image-box">
                        <img id="originalScanImg" src="/sample/normal_sample.png" alt="Original Scan">
                        <div class="image-caption">Standardized Patient MRI</div>
                    </div>
                    <div class="image-box">
                        <img id="gradcamOverlayImg" src="/sample/normal_sample.png" alt="Grad-CAM Overlay">
                        <div class="image-caption">Grad-CAM Lesion Heatmap</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2 class="card-title">📊 Multi-Class Probability Distribution</h2>
                <div id="probBars">
                    <div class="prob-item">
                        <div class="prob-header">
                            <span>Meningioma</span>
                            <span id="probVal0">--%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-bar" id="probBar0" style="width: 0%; background: var(--accent-red);"></div>
                        </div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-header">
                            <span>Glioma</span>
                            <span id="probVal1">--%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-bar" id="probBar1" style="width: 0%; background: var(--accent-amber);"></div>
                        </div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-header">
                            <span>Pituitary</span>
                            <span id="probVal2">--%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-bar" id="probBar2" style="width: 0%; background: var(--accent-purple);"></div>
                        </div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-header">
                            <span>No Tumor (Healthy)</span>
                            <span id="probVal3">--%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-bar" id="probBar3" style="width: 0%; background: var(--accent-green);"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer>
        Dataset: 10,264 Combined MRI Scans (Figshare + Kaggle) | Architecture: Custom Deep CNN (8.71M Params) | Test Accuracy: 96.4% | Developed for Clinical Decision Support
    </footer>

    <script>
        let currentFile = null;
        let selectedSampleName = null;

        const fileInput = document.getElementById('fileInput');
        const dropzone = document.getElementById('dropzone');

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                currentFile = e.target.files[0];
                selectedSampleName = null;
                const reader = new FileReader();
                reader.onload = (re) => {
                    document.getElementById('originalScanImg').src = re.target.result;
                    document.getElementById('gradcamOverlayImg').src = re.target.result;
                };
                reader.readAsDataURL(currentFile);
            }
        });

        function loadSample(sampleName, label) {
            selectedSampleName = sampleName;
            currentFile = null;
            const url = '/sample/' + sampleName;
            document.getElementById('originalScanImg').src = url;
            document.getElementById('gradcamOverlayImg').src = url;
            runAnalysis();
        }

        async function runAnalysis() {
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.innerText = "⏳ Processing MRI...";

            const formData = new FormData();
            if (currentFile) {
                formData.append('image', currentFile);
            } else if (selectedSampleName) {
                formData.append('sample', selectedSampleName);
            } else {
                alert("Please select a sample scan or upload an MRI image.");
                btn.disabled = false;
                btn.innerText = "🔍 Analyze MRI Scan";
                return;
            }

            formData.append('gradcam', document.getElementById('gradcamToggle').checked);
            formData.append('enhance', document.getElementById('enhanceToggle').checked);

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "Analysis failed");
                }

                // Update Findings Banner
                const meta = data.metadata || {};
                const badgeColor = meta.badge_color || '#3b82f6';
                const statusIcon = data.is_tumor_detected ? '⚠️' : '✅';

                document.getElementById('findingsContainer').innerHTML = `
                    <div class="finding-banner" style="border-left-color: ${badgeColor};">
                        <div class="finding-title" style="color: ${badgeColor};">
                            ${statusIcon} ${data.predicted_class} (${data.confidence_percent})
                        </div>
                        <div class="finding-meta">
                            <strong>Status:</strong> ${data.is_tumor_detected ? 'Neoplasm Identified' : 'Physiological Parenchyma'} | 
                            <strong>Clinical Urgency:</strong> ${data.urgency}
                        </div>
                        <p class="finding-desc">${data.clinical_summary}</p>
                        <div style="margin-top: 10px; font-size: 0.8rem; color: #94a3b8;">
                            <strong>Recommended Step:</strong> ${meta.recommendation || 'Consult attending physician.'}
                        </div>
                    </div>
                `;

                // Update Images
                if (data.gradcam_base64) {
                    document.getElementById('gradcamOverlayImg').src = "data:image/png;base64," + data.gradcam_base64;
                }
                if (data.original_base64) {
                    document.getElementById('originalScanImg').src = "data:image/png;base64," + data.original_base64;
                }

                // Update Probability Bars
                const probs = data.probabilities;
                updateBar(0, probs['Meningioma']);
                updateBar(1, probs['Glioma']);
                updateBar(2, probs['Pituitary']);
                updateBar(3, probs['No Tumor']);

            } catch (err) {
                alert("Error during scan analysis: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerText = "🔍 Analyze MRI Scan";
            }
        }

        function updateBar(idx, val) {
            const pct = (val * 100).toFixed(2);
            document.getElementById('probVal' + idx).innerText = pct + '%';
            document.getElementById('probBar' + idx).style.width = pct + '%';
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/sample/<filename>")
def serve_sample(filename):
    sample_path = SAMPLES_DIR / filename
    if sample_path.exists():
        return send_file(str(sample_path), mimetype="image/png")
    return jsonify({"error": "Sample file not found"}), 404


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor.has_trained_weights,
        "classes": CLASS_NAMES,
        "metrics": DATASET_METRICS
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for brain tumor classification.
    Accepts multipart form-data with 'image' file or 'sample' filename.
    """
    try:
        include_gradcam = request.form.get("gradcam", "true").lower() == "true"
        enhance = request.form.get("enhance", "false").lower() == "true"

        img_input = None
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            img_input = file.read()
        elif "sample" in request.form:
            sample_file = SAMPLES_DIR / request.form["sample"]
            if sample_file.exists():
                img_input = str(sample_file)

        if img_input is None:
            return jsonify({"error": "No image uploaded or sample selected"}), 400

        result = predictor.predict(
            img_input,
            include_gradcam=include_gradcam,
            enhance_contrast_flag=enhance
        )

        gradcam_b64 = None
        if result["gradcam_overlay"] is not None:
            gradcam_b64 = pil_to_base64(result["gradcam_overlay"])

        original_b64 = None
        if result["original_image"] is not None:
            original_b64 = pil_to_base64(result["original_image"])

        return jsonify({
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "confidence_percent": result["confidence_percent"],
            "probabilities": result["probabilities"],
            "is_tumor_detected": result["is_tumor_detected"],
            "urgency": result["urgency"],
            "clinical_summary": result["clinical_summary"],
            "metadata": result["metadata"],
            "gradcam_base64": gradcam_b64,
            "original_base64": original_b64,
            "model_status": result["model_status"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================================")
    print(f"🚀 NeuroScan Clinical Tumor Web Dashboard running at:")
    print(f"   http://127.0.0.1:{port}")
    print(f"==================================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
