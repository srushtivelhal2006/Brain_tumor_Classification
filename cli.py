"""
Command-Line Interface (CLI) for Brain Tumor Classification & Grad-CAM Analysis.

Usage examples:
    python cli.py --image samples/glioma_sample.png
    python cli.py --image samples/meningioma_sample.png --gradcam --output-dir results
    python cli.py --demo
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import SAMPLES_DIR
from src.predictor import ClinicalBrainTumorPredictor
from src.report import generate_markdown_report


def print_banner():
    print("=" * 68)
    print("[*] CLINICAL BRAIN TUMOR CLASSIFIER & GRAD-CAM EXPLAINABILITY CLI")
    print("=" * 68)


def evaluate_single_image(predictor: ClinicalBrainTumorPredictor, image_path: Path, args):
    if not image_path.exists():
        print(f"[Error] File not found: {image_path}")
        return

    print(f"\n[Scanning] Analyzing MRI Scan: {image_path.name}")
    result = predictor.predict(
        image_path,
        include_gradcam=args.gradcam,
        enhance_contrast_flag=args.enhance,
        gradcam_alpha=args.alpha
    )

    pred_class = result["predicted_class"]
    conf = result["confidence_percent"]
    urgency = result["urgency"]

    print("-" * 68)
    print(f"  Classification Finding : {pred_class.upper()}")
    print(f"  Diagnostic Confidence  : {conf}")
    print(f"  Clinical Urgency       : {urgency}")
    print(f"  Neoplasm Detected      : {'YES [ALERT]' if result['is_tumor_detected'] else 'NO (Normal) [OK]'}")
    print("-" * 68)
    print("  Class Probability Distribution:")
    for cls_name, prob in result["probabilities"].items():
        bar = "#" * int(prob * 25)
        print(f"    - {cls_name:<12}: {prob*100:6.2f}% {bar}")
    print("-" * 68)

    # Save output artifacts if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        if result["gradcam_overlay"] is not None:
            gradcam_path = out_dir / f"{stem}_gradcam.png"
            result["gradcam_overlay"].save(gradcam_path)
            print(f"  [Artifact] Saved Grad-CAM overlay -> {gradcam_path}")

        if args.report:
            report_content = generate_markdown_report(result, patient_id=f"PT-{stem.upper()}")
            report_path = out_dir / f"{stem}_diagnostic_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"  [Artifact] Saved Clinical Report   -> {report_path}")


def run_demo(predictor: ClinicalBrainTumorPredictor, args):
    print("\n[Demo Mode] Running batch evaluation on repository sample scans...")
    sample_files = sorted([f for f in SAMPLES_DIR.glob("*.png")])
    if not sample_files:
        print("[Error] No sample PNG files found in samples/ directory.")
        return

    for sample in sample_files:
        evaluate_single_image(predictor, sample, args)


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Brain Tumor Classification & Explainable AI (Grad-CAM) CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", "-i", type=str, help="Path to input brain MRI scan (PNG, JPG, DICOM-exported)")
    parser.add_argument("--demo", action="store_true", help="Run batch evaluation on all sample scans")
    parser.add_argument("--gradcam", "-g", action="store_true", default=True, help="Generate Grad-CAM lesion heatmap overlay")
    parser.add_argument("--no-gradcam", action="store_false", dest="gradcam", help="Disable Grad-CAM heatmap generation")
    parser.add_argument("--enhance", "-e", action="store_true", help="Apply adaptive contrast equalization prior to inference")
    parser.add_argument("--alpha", type=float, default=0.45, help="Grad-CAM overlay blending weight (0.0 to 1.0)")
    parser.add_argument("--output-dir", "-o", type=str, default="results", help="Directory to save generated outputs and reports")
    parser.add_argument("--report", "-r", action="store_true", default=True, help="Generate clinical diagnostic markdown report")
    parser.add_argument("--model-path", "-m", type=str, default=None, help="Custom path to model weights .h5 file")

    args = parser.parse_args()
    print_banner()

    predictor = ClinicalBrainTumorPredictor(model_path=args.model_path)

    if args.demo or (not args.image and not sys.stdin.isatty()):
        run_demo(predictor, args)
    elif args.image:
        evaluate_single_image(predictor, Path(args.image), args)
    else:
        print("\nNo image specified. Use --image <path> or --demo to evaluate test scans.")
        parser.print_help()


if __name__ == "__main__":
    main()
