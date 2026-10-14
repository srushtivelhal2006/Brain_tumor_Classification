"""
Clinical Brain Tumor Classification & Explainable AI (Grad-CAM) Dashboard.
Compatible with Gradio 4.x / Hugging Face Spaces.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CLASS_NAMES, TUMOR_METADATA, DATASET_METRICS, DEFAULT_MODEL_PATH
from src.predictor import ClinicalBrainTumorPredictor
from src.report import generate_markdown_report

# Initialize predictor
predictor = ClinicalBrainTumorPredictor(model_path=str(DEFAULT_MODEL_PATH))


def analyze_mri_scan(input_image, enable_gradcam, enable_enhancement, gradcam_blend):
    """
    Callback function executed when an MRI scan is submitted for analysis.
    """
    if input_image is None:
        return (
            None,
            None,
            "⚠️ **No Scan Uploaded**: Please select or upload a valid cranial MRI image.",
            ""
        )

    try:
        results = predictor.predict(
            input_image,
            include_gradcam=enable_gradcam,
            enhance_contrast_flag=enable_enhancement,
            gradcam_alpha=gradcam_blend
        )

        pred_class = results["predicted_class"]
        conf_pct = results["confidence_percent"]
        meta = results["metadata"]
        urgency = results["urgency"]
        probs = results["probabilities"]

        # Build clean clinical status banner
        if results["is_tumor_detected"]:
            banner = (
                f"### ⚠️ **Diagnostic Finding: {pred_class.upper()}**\n\n"
                f"**Confidence Level:** `{conf_pct}` | **Clinical Urgency:** `{urgency}`\n\n"
                f"{results['clinical_summary']}\n\n"
                f"**Recommended Action:** {meta.get('recommendation', 'Clinical review required.')}"
            )
        else:
            banner = (
                f"### ✅ **Diagnostic Finding: NORMAL (NO TUMOR)**\n\n"
                f"**Confidence Level:** `{conf_pct}` | **Status:** `Physiological Landmarks Intact`\n\n"
                f"{results['clinical_summary']}"
            )

        # Generate complete Markdown Clinical Report
        report_md = generate_markdown_report(results, patient_id="PT-WEB-DEMO")

        # Visual Grad-CAM or processed output
        visual_output = results["gradcam_overlay"] if enable_gradcam and results["gradcam_overlay"] else results["original_image"]

        return (
            visual_output,
            probs,
            banner,
            report_md
        )

    except Exception as exc:
        import traceback
        err_msg = f"❌ **Analysis Error**: {str(exc)}\n\n```python\n{traceback.format_exc()}\n```"
        return None, None, err_msg, ""


# -----------------------------------------------------------------------------
# Gradio Interface Definition
# -----------------------------------------------------------------------------
def build_gradio_app():
    import gradio as gr

    custom_css = """
    body, .gradio-container {
        background-color: #030712 !important;
        background: radial-gradient(circle at 50% 0%, #1e293b 0%, #030712 100%) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: #f3f4f6 !important;
    }
    .clinical-card {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        backdrop-filter: blur(8px) !important;
        margin-bottom: 16px !important;
    }
    .predict-btn {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .predict-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(37, 99, 235, 0.6) !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Default(), css=custom_css, title="Clinical Brain Tumor Classifier") as demo:
        gr.HTML(
            "<div style='text-align: center; padding: 24px 0 10px 0;'>"
            "  <h1 style='font-size: 2.4rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;'>🧠 Clinical Brain Tumor Classifier & Grad-CAM XAI</h1>"
            "  <p style='color: #94a3b8; font-size: 1.05rem; max-width: 800px; margin: 0 auto;'>Deep Learning Multi-Class Computer Vision for Neuro-Oncology MRI Diagnostics with Explainable AI Lesion Localization</p>"
            "</div>"
        )

        with gr.Row():
            # Left Column: Upload & Controls
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["clinical-card"]):
                    gr.Markdown("### 📥 Patient MRI Scan Upload")
                    input_mri = gr.Image(
                        label="Upload Cranial MRI Slice",
                        type="pil",
                        sources=["upload", "clipboard"]
                    )

                    with gr.Accordion("⚙️ Diagnostic & Visualization Settings", open=False):
                        gradcam_toggle = gr.Checkbox(
                            label="Generate Grad-CAM Attention Heatmap",
                            value=True,
                            info="Visualizes spatial lesion localization learned by the CNN."
                        )
                        contrast_toggle = gr.Checkbox(
                            label="Apply Contrast Enhancement (Equalization)",
                            value=False,
                            info="Enhances soft tissue boundaries and subtle lesions."
                        )
                        alpha_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.45,
                            step=0.05,
                            label="Heatmap Transparency (Alpha)"
                        )

                    analyze_btn = gr.Button("🔍 Analyze MRI Scan", elem_classes=["predict-btn"])

                with gr.Group(elem_classes=["clinical-card"]):
                    gr.Markdown("### 📂 Built-in Clinical Test Scans")
                    gr.Examples(
                        examples=[
                            ["samples/glioma_sample.png", True, False, 0.45],
                            ["samples/meningioma_sample.png", True, False, 0.45],
                            ["samples/pituitary_sample.png", True, False, 0.45],
                            ["samples/normal_sample.png", True, False, 0.45]
                        ],
                        inputs=[input_mri, gradcam_toggle, contrast_toggle, alpha_slider],
                        label="Click to Evaluate Test Scan"
                    )

            # Right Column: Visual Heatmap & Findings
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["clinical-card"]):
                    gr.Markdown("### 🎯 Diagnostic & Explainability Visualization")
                    output_gradcam = gr.Image(
                        label="Grad-CAM Lesion Localization Overlay",
                        type="pil"
                    )
                    status_findings = gr.Markdown("✨ **Ready**: Upload an MRI scan and click **Analyze MRI Scan** to inspect.")

                with gr.Group(elem_classes=["clinical-card"]):
                    gr.Markdown("### 📊 Multi-Class Probability Breakdown")
                    output_probs = gr.Label(
                        num_top_classes=4,
                        label="Predicted Class Confidence"
                    )

        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes=["clinical-card"]):
                    gr.Markdown("### 📋 Generated Clinical Diagnostic Report")
                    report_output = gr.Markdown("The comprehensive medical finding summary will generate here following scan analysis.")

        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes=["clinical-card"]):
                    gr.Markdown("### 📚 Medical Reference Guide")
                    gr.HTML(
                        "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;'>"
                        "  <div style='background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 8px; border-left: 4px solid #F59E0B;'>"
                        "    <h4 style='color: #F59E0B; margin: 0 0 6px 0;'>Glioma</h4>"
                        "    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>Intra-axial glial mass. Infiltrative margins, heterogeneous signal, surrounding edema, central necrosis.</p>"
                        "  </div>"
                        "  <div style='background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 8px; border-left: 4px solid #EF4444;'>"
                        "    <h4 style='color: #EF4444; margin: 0 0 6px 0;'>Meningioma</h4>"
                        "    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>Extra-axial dural-based lesion. Sharp circumscribed margins, intense post-contrast enhancement, dural tail.</p>"
                        "  </div>"
                        "  <div style='background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 8px; border-left: 4px solid #8B5CF6;'>"
                        "    <h4 style='color: #8B5CF6; margin: 0 0 6px 0;'>Pituitary Adenoma</h4>"
                        "    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>Sellar/suprasellar mass. Well-delineated, potential optic chiasm elevation ('snowman' sign), endocrine effects.</p>"
                        "  </div>"
                        "  <div style='background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 8px; border-left: 4px solid #10B981;'>"
                        "    <h4 style='color: #10B981; margin: 0 0 6px 0;'>No Tumor (Normal)</h4>"
                        "    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>Symmetric hemispheres, preserved ventricles, gray-white differentiation without focal mass effect.</p>"
                        "  </div>"
                        "</div>"
                    )

        # Trigger on button click
        analyze_btn.click(
            fn=analyze_mri_scan,
            inputs=[input_mri, gradcam_toggle, contrast_toggle, alpha_slider],
            outputs=[output_gradcam, output_probs, status_findings, report_output]
        )

    return demo


if __name__ == "__main__":
    try:
        import gradio as gr
        app = build_gradio_app()
        app.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except ImportError:
        print("[Notice] Gradio is not installed in the active environment.")
        print("To launch the Gradio interface, run: pip install gradio>=4.0.0")
        print("Or run the standalone Flask web app: python web_app.py")
        sys.exit(1)
