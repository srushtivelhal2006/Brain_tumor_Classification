"""
Configuration and Medical Knowledge Base for Brain Tumor Classification.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "brain_tumor_model.h5"
SAMPLES_DIR = BASE_DIR / "samples"

# Model Parameters
IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 4

# Target Classes in Exact Order of Trained Softmax Output
# Index 0: Meningioma
# Index 1: Glioma
# Index 2: Pituitary
# Index 3: No Tumor
CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary", "No Tumor"]

CLASS_LABELS = {
    0: "Meningioma",
    1: "Glioma",
    2: "Pituitary",
    3: "No Tumor"
}

# Clinical Knowledge Base & Metadata
TUMOR_METADATA = {
    "Meningioma": {
        "full_name": "Meningioma (Extra-Axial Neoplasm)",
        "origin": "Arachnoid 'cap' cells in the meningeal layers surrounding the brain and spinal cord.",
        "prevalence": "Most common primary non-glial intracranial tumor (~37% of primary brain neoplasms).",
        "mri_features": "Extra-axial, dural-based mass with sharp circumscribed margins, frequent intense homogeneous post-contrast enhancement, and characteristic 'dural tail' sign.",
        "clinical_presentation": "Headache, localized focal neurological deficits, visual field disturbances, seizure activity depending on cranial nerve and parenchymal compression.",
        "urgency": "High",
        "badge_color": "#EF4444",  # Red
        "icon": "🛡️",
        "recommendation": "Contrast-enhanced high-resolution cranial MRI, neurosurgical oncology evaluation for surgical resection or stereotactic radiosurgery consideration."
    },
    "Glioma": {
        "full_name": "Glioma (Intra-Axial Parenchymal Neoplasm)",
        "origin": "Supportive glial cells (astrocytes, oligodendrocytes, or ependymal cells) within the brain parenchyma.",
        "prevalence": "Accounts for ~80% of all malignant primary central nervous system tumors.",
        "mri_features": "Intra-axial lesion, typically ill-defined or infiltrating margins, heterogeneous signal intensity, central necrosis, surrounding vasogenic edema with mass effect on ventricles.",
        "clinical_presentation": "Progressive morning headaches, seizure onset, behavioral/cognitive changes, hemiparesis, intracranial hypertension.",
        "urgency": "Critical",
        "badge_color": "#F59E0B",  # Amber/Orange
        "icon": "⚠️",
        "recommendation": "Urgent neurosurgical multidisciplinary consultation (MDT), advanced MRI spectroscopy / perfusion imaging, surgical debulking and histopathological/molecular profiling (IDH, 1p/19q, MGMT)."
    },
    "Pituitary": {
        "full_name": "Pituitary Adenoma / Sellar Region Tumor",
        "origin": "Anterior pituitary epithelial cells within the sella turcica at the base of the skull.",
        "prevalence": "Accounts for ~10-15% of all intracranial neoplasms; overwhelmingly benign.",
        "mri_features": "Sellar or suprasellar well-delineated mass lesion; can expand the sella turcica with superior extension causing characteristic compression of the optic chiasm ('snowman' or figure-8 sign).",
        "clinical_presentation": "Bitemporal hemianopsia (optic chiasm compression), hypopituitarism or hypersecretory endocrine syndromes (prolactinoma, Cushing's, Acromegaly), persistent dull headaches.",
        "urgency": "Moderate to High",
        "badge_color": "#8B5CF6",  # Purple
        "icon": "🔬",
        "recommendation": "Comprehensive serum pituitary hormone evaluation (Prolactin, IGF-1, ACTH, Cortisol, TSH, LH/FSH), dedicated sellar thin-slice contrast MRI, visual field perimetry test."
    },
    "No Tumor": {
        "full_name": "Normal Brain Neuroanatomy (No Neoplasm Detected)",
        "origin": "Physiological brain parenchyma, ventricular system, and cranial vault without focal lesion.",
        "prevalence": "Normal control / non-neoplastic scan.",
        "mri_features": "Symmetric cerebral and cerebellar hemispheres, midline cerebral falx, normal ventricular morphology, preservation of gray-white matter differentiation without focal space-occupying lesion.",
        "clinical_presentation": "No imaging evidence of mass effect, parenchymal distortion, midline shift, or pathologic focal contrast uptake.",
        "urgency": "Routine",
        "badge_color": "#10B981",  # Green
        "icon": "✅",
        "recommendation": "Correlate with non-structural clinical causes for any present symptoms (e.g. vascular, migraine, metabolic, functional), routine clinical follow-up as clinically indicated."
    }
}

# Project Dataset Statistics
DATASET_METRICS = {
    "total_dataset_size": 10264,
    "train_samples": 7184,
    "val_samples": 1540,
    "test_samples": 1540,
    "image_resolution": "128x128x3",
    "architecture": "Custom 6-Conv Layer Deep CNN + Batch Normalization",
    "total_parameters": 8711206,
    "trained_epochs": 30,
    "reported_test_accuracy": "96.4%"
}
