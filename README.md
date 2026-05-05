# Developing Predictive Models for Early ASD Detection Based on fMRI Scans

[![IEEE](https://img.shields.io/badge/Published-IEEE%20Transactions-blue)](Docs/paper.tex)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](requirements.txt)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.96%25-brightgreen)](streamlit_model_tester/asd_model_metrics.csv)
[![Dataset](https://img.shields.io/badge/Dataset-ABIDE--I%20(N%3D1%2C112)-orange)](http://fcon_1000.projects.nitrc.org/indi/abide/)

> **Final Year Project** | Department of Computer Science & Engineering  
> Madanapalle Institute of Technology & Science (MITS), Andhra Pradesh — 2026  
> Authors: Enduluri Reddy Rohith, Jalla Reddy Manoj, MachireddyGari Monika, Jonna Mounika  
> Supervisor: Dr. R. Nidhya, SMIEEE

---

## Overview

This repository contains the full source code, trained models, evaluation plots, and documentation for a multimodal machine learning framework for **Autism Spectrum Disorder (ASD) classification** using the ABIDE-I neuroimaging dataset.

### The Core Idea

Standard approaches to ASD classification using fMRI treat **head motion artefacts as noise** to be removed. Our work flips this assumption:

> **fMRI scan-quality metrics (Framewise Displacement, DVARS, Signal-to-Noise Ratio) encode the same motor and sensory dysregulation that characterises ASD. They are diagnostic signal disguised as noise.**

By combining these quality metrics with phenotypic data and feeding them into an optimized XGBoost classifier with Mutual Information feature selection, we achieve:

| Metric | Value |
|---|---|
| **Test Accuracy** | **95.96%** |
| **Precision** | 99.01% |
| **Recall** | 92.59% |
| **Specificity** | 98.36% |
| **F1-Score** | 95.69% |
| **AUC-ROC** | 0.9705 |

This surpasses the best prior art on ABIDE-I (74.1% — Transformer-based models) by **+22 percentage points**.

---

## Quick Start

### 1. Clone & Set Up Environment

```bash
git clone <repository-url>
cd FYP_ASD

python -m venv .venv
.venv\Scripts\Activate.ps1         # Windows
# source .venv/bin/activate        # Linux/Mac

pip install -r requirements.txt
```

### 2. Launch the Web App (Instant Demo)

```bash
cd streamlit_model_tester
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

### 3. Retrain the Model from Scratch

```bash
# Ensure Phenotypic_V1_0b_preprocessed1.csv is present
python train_model_reproducible.py
# → Outputs: improved_asd_model_FIXED.pkl  (95.96% accuracy)
```

---

## Repository Structure

```
FYP_ASD/
├── Docs/
│   ├── paper.tex                      # IEEE research paper (LaTeX)
│   ├── references.bib                 # 40+ citations
│   ├── final_report.pdf               # Compiled final report
│   ├── workspace_execution_guide.md   # ← Full project guide (READ THIS)
│   ├── ABIDE_LEGEND_V1.02.pdf         # ABIDE-I data dictionary
│   └── ABIDEII_Data_Legend.pdf        # ABIDE-II data dictionary
│
├── streamlit_model_tester/
│   ├── app.py                         # Streamlit web application
│   ├── asd_model.pkl                  # Trained model artifact
│   ├── asd_model_features_all.csv     # Processed features (all 1,112 subjects)
│   ├── requirements.txt               # App dependencies
│   └── README.md                      # App-specific guide
│
├── plots/                             # All evaluation figures (for paper.tex)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   ├── system_architecture.png
│   └── ...
│
│   ── TRAINING SCRIPTS ──
├── train_model_reproducible.py        # ★ Primary training script (95.96%)
├── improved_asd_model_FIXED.py        # Advanced training with validation
│
│   ── DATA PREPARATION ──
├── generate_full_dataset.py           # Feature extraction for Streamlit app
├── clean_duplicates.py                # Remove duplicate SUB_IDs
│
│   ── DATA ACQUISITION ──
├── download_abide_preproc.py          # Download ABIDE from FCP-INDI S3
├── abide_s3_utils.py                  # S3 client utilities
├── abide_streaming_analysis.py        # Memory-efficient S3 streaming analysis
├── abide_loader.py                    # ABIDE data loader (generator)
├── map_rawout_to_organized_s3.py      # Reorganize S3 download structure
│
│   ── EVALUATION ──
├── model_visualizations.py            # Generate all evaluation plots
│
│   ── RAW DATA ──
├── Phenotypic_V1_0b.csv               # Raw ABIDE-I phenotypic file
├── Phenotypic_V1_0b_preprocessed1.csv # ★ Processed dataset (N=1,112)
├── ABIDEII_Composite_Phenotypic.csv   # ABIDE-II phenotypic data
│
│   ── ARTIFACTS ──
├── improved_asd_model_FIXED.pkl       # Trained model (joblib serialized)
├── requirements.txt                   # Root environment dependencies
└── .gitignore
```

---

## The Pipeline at a Glance

```
ABIDE-I Dataset (17 sites, N=1,112)
          │
          ▼
  [1] C-PAC Preprocessing
      • Slice-timing correction
      • Motion realignment → FD computation
      • MNI-152 normalisation
      • Nuisance regression, bandpass filter
          │
          ▼
  [2] Feature Engineering (85 features)
      • Phenotypic: demographics, IQ, ADOS, ADI-R, SRS, VINELAND, WISC-IV
      • Quality Metrics: FD, DVARS, SNR, EFC, FBER, FWHMQuality
          │
          ▼
  [3] Mutual Information Feature Selection
      • k-NN MI estimator (k=5)
      • Top K=40 features retained
      • RobustScaler normalization
          │
          ▼
  [4] XGBoost Classification
      • 80/20 stratified train/test split
      • 10-fold cross-validation
      • Grid search: γ ∈ {0, 0.1, 0.5, 1.0}, λ ∈ {0.1, 1, 5, 10}
      • Final: 300 trees, depth=5, lr=0.05
          │
          ▼
  95.96% Accuracy | AUC-ROC 0.9705
```

---

## Key Results

### vs. State of the Art on ABIDE-I

| Method | Accuracy |
|---|---|
| Nielsen et al. (2013) — GLM/FC | 60.0% |
| Abraham et al. (2017) — SVM/FC | 67.0% |
| Heinsfeld et al. (2018) — Autoencoders | 70.1% |
| BrainGNN (2021) | 72.3% |
| ASD-DiagNet (2019) | 73.2% |
| Transformer/FC (2023) | 74.1% |
| **This Work — XGBoost + Quality Metrics** | **95.96%** |

### Feature Importance Finding

Quality metrics **FD, DVARS, and tSNR** rank in the **top 20 most important features** alongside ADOS total score and FIQ — empirically validating the paper's core hypothesis.

---

## Documentation

| Document | Description |
|---|---|
| 📖 [Full Project Guide](Docs/workspace_execution_guide.md) | Complete beginner-friendly explanation of every file, concept, and execution step |
| 📄 [Research Paper](Docs/paper.tex) | Full IEEE-format LaTeX manuscript |
| 📊 [Visualization Report](VISUALIZATION_REPORT.md) | Interpretation of all generated evaluation plots |
| ⚠️ [Problem Statement Analysis](PROBLEM_STATEMENT_DETAILED_COMPARISON.md) | Gap analysis between "early detection" scope and current model capability |
| 🖥️ [Streamlit App Guide](streamlit_model_tester/README.md) | How to run and use the web application |

---

## Dataset

This project uses the **ABIDE-I** (Autism Brain Imaging Data Exchange, Phase I) dataset:

- **Access:** [http://fcon_1000.projects.nitrc.org/indi/abide/](http://fcon_1000.projects.nitrc.org/indi/abide/)
- **Preprocessed version:** [http://preprocessed-connectomes-project.org/abide/](http://preprocessed-connectomes-project.org/abide/)
- **Citation:** Di Martino et al. (2014), *Molecular Psychiatry*, 19(6):659–667. DOI: 10.1038/mp.2013.78

Data is used in accordance with the ABIDE data sharing agreement for non-commercial research purposes.

---

## Dependencies

```
pandas, numpy, scikit-learn, xgboost, joblib,
streamlit, matplotlib, seaborn, nilearn (optional, for fMRI loading)
```

Install all with: `pip install -r requirements.txt`

---

## Limitations & Ethical Considerations

1. **Not a clinical tool** — This is a research prototype. Do not use for clinical decision-making without proper validation.
2. **Post-diagnosis features** — ADOS/ADI-R scores used as features are available only *after* formal evaluation; the model is a decision-support tool, not a pre-assessment screen.
3. **Age range** — Trained on subjects 6.5–64 years. Not validated for infants or toddlers.
4. **Multi-site bias** — Test subjects may share sites with training subjects; leave-one-site-out validation remains as future work.
5. **Future work** — ADOS-free variant (quality metrics + demographics only), ComBat harmonisation, SHAP explainability.

---

*© 2026 Enduluri Reddy Rohith et al., MITS. For research use only.*
