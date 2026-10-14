"""
Clinical Diagnostic Report Generator.
Formats patient findings, probability distribution, and recommendations for clinician review.
"""

from datetime import datetime
from typing import Dict, Any


def generate_markdown_report(prediction_result: Dict[str, Any], patient_id: str = "PT-SCAN-9021") -> str:
    """
    Generates a structured, professional Markdown diagnostic summary report.
    """
    cls = prediction_result["predicted_class"]
    conf = prediction_result["confidence_percent"]
    probs = prediction_result["probabilities"]
    meta = prediction_result.get("metadata", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# 🧠 CLINICAL NEURO-ONCOLOGY AI DIAGNOSTIC REPORT

**Patient Reference ID:** `{patient_id}`  
**Analysis Timestamp:** {timestamp}  
**Model Architecture:** Custom 6-Conv Layer Deep CNN (8.71M parameters)  
**Execution Environment:** {prediction_result.get('model_status', 'Clinical Production')}  

---

## 📋 PRIMARY DIAGNOSTIC FINDINGS

| Metric | Result |
| :--- | :--- |
| **Classified Finding** | **{meta.get('full_name', cls)}** |
| **Diagnostic Confidence** | **{conf}** |
| **Pathologic Status** | {'⚠️ Neoplastic Mass Identified' if prediction_result['is_tumor_detected'] else '✅ No Neoplasm Detected'} |
| **Clinical Urgency** | **{prediction_result.get('urgency', 'Routine')}** |

---

## 📊 DIFFERENTIAL PROBABILITY BREAKDOWN

"""
    for class_name, prob in probs.items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        md += f"- **{class_name:<12}**: `{bar}` **{prob * 100:6.2f}%**\n"

    md += f"""
---

## 🔬 PATHOLOGICAL CORRELATION & MRI FEATURES

- **Anatomical Origin:** {meta.get('origin', 'N/A')}
- **Typical MRI Features:** {meta.get('mri_features', 'N/A')}
- **Clinical Presentation:** {meta.get('clinical_presentation', 'N/A')}
- **Epidemiology / Prevalence:** {meta.get('prevalence', 'N/A')}

---

## 🩺 RECOMMENDED NEXT CLINICAL STEPS

> {meta.get('recommendation', 'Consult attending physician.')}

---
*Notice: This system is an AI diagnostic assistance tool intended to augment radiologist review. All outputs must be clinically validated against full multi-sequence imaging and patient history.*
"""
    return md
