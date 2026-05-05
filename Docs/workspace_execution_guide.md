# ASD Detection Project — Complete Guide & Workspace Reference

**Paper Title:** *Developing Predictive Models for Early ASD Detection Based on fMRI Scans*  
**Affiliation:** Department of Computer Science & Engineering, MITS, Madanapalle, Andhra Pradesh  
**Published to:** IEEE Transactions on Neural Systems and Rehabilitation Engineering (2026)

---

## Table of Contents

1. [What is ASD? (For Absolute Beginners)](#1-what-is-asd-for-absolute-beginners)
2. [The Research Problem](#2-the-research-problem)
3. [Our Novel Solution — Key Idea](#3-our-novel-solution--key-idea)
4. [Technical Background (Plain English)](#4-technical-background-plain-english)
5. [The Dataset: ABIDE-I](#5-the-dataset-abide-i)
6. [The Four-Stage Pipeline](#6-the-four-stage-pipeline)
7. [Results Achieved](#7-results-achieved)
8. [Complete File-by-File Reference](#8-complete-file-by-file-reference)
9. [Execution Order (Step-by-Step)](#9-execution-order-step-by-step)
10. [How to Set Up & Run the Project](#10-how-to-set-up--run-the-project)
11. [Key References from the Paper](#11-key-references-from-the-paper)

---

## 1. What is ASD? (For Absolute Beginners)

**Autism Spectrum Disorder (ASD)** is a lifelong neurodevelopmental condition. "Spectrum" means it affects people very differently — from mild to severe. The two defining features are:

1. **Social communication difficulties** — trouble understanding or engaging in typical social interactions, conversations, and non-verbal cues.
2. **Restricted/repetitive behaviours** — repetitive movements, insistence on routines, or highly focused interests.

**How common is it?**  
The U.S. CDC (2023) estimates **1 in 36 children** aged 8 years has ASD — a number that has risen dramatically over the past two decades. ASD is four times more prevalent in boys than in girls.

**Why does early diagnosis matter?**  
Research has conclusively shown that interventions started **before 36 months of age** (Applied Behaviour Analysis, speech-language therapy) produce dramatically better life outcomes. However, a formal ASD diagnosis is typically not confirmed until age 3–5 years, causing a critical intervention window to be missed.

**Why is diagnosis hard today?**  
Current gold-standard tools — the **ADOS-2** (Autism Diagnostic Observation Schedule) and **ADI-R** (Autism Diagnostic Interview–Revised) — require:
- Trained specialist clinicians
- Several hours of structured observation
- Expensive and resource-intensive evaluations
- Subjectivity — results can vary between raters

This project uses Machine Learning to help automate and support the diagnostic process using brain scan data.

---

## 2. The Research Problem

### Why fMRI?

**Functional Magnetic Resonance Imaging (fMRI)** measures brain activity by detecting changes in blood oxygen levels (called the **BOLD signal** — Blood Oxygen Level Dependent). When a brain region is active, it needs more oxygen, and fMRI captures that.

In **resting-state fMRI (rs-fMRI)**, the patient simply lies still in the scanner. The brain's spontaneous, low-frequency activity patterns (< 0.1 Hz) reveal which brain regions are functionally "connected" — working together. This is called **Functional Connectivity (FC)**.

Multiple studies have found that **people with ASD show abnormal FC** — specifically:
- **Under-connectivity** in long-range networks (regions far apart in the brain don't communicate well)
- **Hyper-connectivity** in local networks (nearby regions are over-coupled)

### The Big Challenge: Head Motion

When a patient moves their head even **sub-millimeter amounts** during an fMRI scan, it creates **systematic noise** in the BOLD signal — called **motion artefacts**. This is a major problem because:

- People with ASD have **higher rates of motor dysregulation** — they move more in the scanner
- The standard fix is to **exclude high-motion participants**, making datasets smaller and introducing bias
- Excluding these people may actually discard the most diagnostically informative subjects

### The Gap in Existing Research

Previous ML models on the ABIDE dataset achieved only **60–74% accuracy** using functional connectivity features. They treated motion statistics purely as noise to be removed. **No prior work had exploited these quality metrics as diagnostic features themselves.**

---

## 3. Our Novel Solution — Key Idea

> **Core Hypothesis:** The same motor and sensory symptoms that define ASD also cause measurable, systematic patterns in fMRI scan-quality metrics. These patterns carry diagnostic information — if we use them as features instead of discarding them.

In other words: **"The noise IS the signal."**

We propose a **multimodal machine learning framework** that:
1. Combines **phenotypic data** (demographics, clinical scales, IQ scores) with
2. **fMRI scan-quality metrics** (how clean/noisy the scan is), then
3. Selects the **40 most informative features** using information theory (Mutual Information), and
4. Classifies subjects using **XGBoost** — a state-of-the-art gradient boosting algorithm

This is novel because no prior work on ABIDE explicitly used scan quality indices **as features for classification**.

---

## 4. Technical Background (Plain English)

### 4.1 Framewise Displacement (FD)

FD measures how much a patient's head moved **between consecutive brain volumes**. It combines all six motion parameters (3 translations + 3 rotations):

```
FD_t = |Δx| + |Δy| + |Δz| + r(|Δα| + |Δβ| + |Δγ|)
```

Where `r = 50mm` (converts rotation to mm). **Higher mean FD = more head motion.** The ASD group had significantly higher mean FD (0.21mm) vs TDC controls (0.17mm).

### 4.2 DVARS

DVARS (Temporal Derivative of Root Mean Square Variance) measures how much the **BOLD signal changes from one brain volume to the next**, across all voxels:

```
DVARS_t = sqrt( (1/V) × Σ [I(v,t) - I(v,t-1)]² )
```

High DVARS = large instantaneous signal changes = scanning artefacts or subject motion.

### 4.3 Signal-to-Noise Ratio (SNR / tSNR)

Temporal SNR measures the **quality of the signal** — how much real brain signal there is vs random noise:

```
tSNR = (1/V) × Σ (mean_signal_at_voxel / std_signal_at_voxel)
```

Higher tSNR = cleaner scan. People with ASD, who move more, tend to have lower tSNR.

### 4.4 Mutual Information (MI) Feature Selection

With ~85 raw features, we need to select the most informative ones. **Mutual Information** measures how much knowing a feature value reduces our uncertainty about the diagnosis:

```
I(X;Y) = ∫∫ p(x,y) × log[ p(x,y) / (p(x)×p(y)) ] dx dy
```

We use the k-nearest-neighbour estimator (k=5) to compute MI for continuous features, rank all features, and keep the **top K=40**.

### 4.5 XGBoost (Extreme Gradient Boosting)

XGBoost builds an ensemble of decision trees. Each new tree corrects the errors of all previous trees. The final prediction is the sum of all trees:

```
ŷ(t) = Σ f_k(x_i)  for k=1 to t
```

At each step, it minimises a penalised loss (binary cross-entropy + regularisation penalty Ω), using a second-order Taylor expansion for efficiency. Key hyperparameters tuned via grid search:
- **300 trees**, max depth 5, learning rate η=0.05
- Column/row subsampling = 0.8 (prevents overfitting)
- γ=0.1 (minimum split gain), λ=1.0 (L2 regularisation)

---

## 5. The Dataset: ABIDE-I

| Property | Value |
|---|---|
| **Full Name** | Autism Brain Imaging Data Exchange — Phase I |
| **Type** | Multi-site, retrospectively aggregated rs-fMRI data |
| **Sites** | 17 international laboratories |
| **Total Subjects Used** | N = 1,112 |
| **ASD Cases** | 539 |
| **Typically Developing Controls (TDC)** | 573 |
| **Age Range** | 6.5 – 64 years (mean ~17.5 years) |
| **Sex** | ~87% male (consistent with ASD epidemiology) |
| **Preprocessing Pipeline** | C-PAC (Configurable Pipeline for Analysis of Connectomes) v1.8 |

### Demographics Summary

| Characteristic | ASD (n=539) | TDC (n=573) | p-value |
|---|---|---|---|
| Age (mean ± SD, years) | 17.5 ± 7.8 | 17.0 ± 7.4 | > 0.05 |
| Sex (M/F) | 470/69 | 483/90 | > 0.05 |
| Full-Scale IQ (mean ± SD) | 106.3 ± 16.5 | 111.6 ± 12.4 | < 0.01 |
| Mean FD (mm, mean ± SD) | 0.21 ± 0.11 | 0.17 ± 0.09 | < 0.01 |

The significantly higher mean FD in ASD provides **preliminary empirical validation** of the quality-metrics hypothesis.

### C-PAC Preprocessing Steps

The raw BOLD volumes go through 6 preprocessing steps before feature extraction:

1. **Slice-timing correction** — corrects for the fact that different brain slices are acquired at slightly different times
2. **Motion realignment** — registers every brain volume to a reference image using rigid-body (6 DoF) registration
3. **Spatial normalisation** — warps each brain to the standard MNI-152 template (3mm isotropic)
4. **Nuisance signal regression** — removes white-matter and CSF (cerebrospinal fluid) signals that are not neural in origin
5. **Temporal band-pass filtering** — retains only low-frequency oscillations (0.01–0.10 Hz)
6. **Spatial smoothing** — applies a 6mm FWHM Gaussian kernel to improve signal consistency

---

## 6. The Four-Stage Pipeline

```
RAW ABIDE DATA
      │
      ▼
[Stage 1] DATA CURATION & PREPROCESSING
  • Load Phenotypic_V1_0b_preprocessed1.csv
  • Filter valid ASD/TDC labels (DX_GROUP ∈ {1,2})
  • Remove duplicate subjects (by SUB_ID)
  • Impute missing values (<5%) with column-wise median
      │
      ▼
[Stage 2] FEATURE ENGINEERING (85 raw features)
  ├── Phenotypic Features
  │     ├── Demographics: age, sex, handedness
  │     ├── Cognitive: FIQ, VIQ, PIQ
  │     ├── Clinical Scales: ADOS total, ADI-R subscores,
  │     │                    SRS, SCQ, AQ, VINELAND, WISC-IV
  │     └── Site metadata: acquisition site ID
  └── fMRI Quality Metrics
        ├── func_mean_fd  (mean Framewise Displacement)
        ├── func_dvars    (DVARS)
        ├── anat_snr      (Signal-to-Noise Ratio)
        ├── func_fber, func_efc, func_outlier
        └── anat_cnr, anat_efc, anat_fber, anat_fwhm
      │
      ▼
[Stage 3] MUTUAL INFORMATION FEATURE SELECTION
  • Compute MI between each feature and ASD/TDC label
  • Rank 85 features by MI score (descending)
  • Retain Top K=40 most informative features
  • Apply RobustScaler (robust to outliers)
      │
      ▼
[Stage 4] XGBOOST CLASSIFICATION
  • Train/Test split: 80% train (889) / 20% test (223)
  • Stratified 10-fold cross-validation on training set
  • Grid search hyperparameter tuning (γ, λ, depth, lr)
  • Final model: 300 trees, depth=5, lr=0.05
      │
      ▼
RESULTS & EVALUATION
  • Accuracy: 95.96%
  • Precision: 99.01%
  • Recall: 92.59%
  • Specificity: 98.36%
  • F1-Score: 95.69%
  • AUC-ROC: 0.9705
```

---

## 7. Results Achieved

### Performance on Independent Test Set (n=223)

| Metric | Value | What it means |
|---|---|---|
| **Accuracy** | **95.96%** | Out of 223 test patients, 214 were correctly classified |
| **Precision (PPV)** | **99.01%** | When the model says "ASD", it's correct 99% of the time |
| **Recall (Sensitivity)** | **92.59%** | The model catches 92.6% of all actual ASD cases |
| **Specificity (TNR)** | **98.36%** | The model correctly clears 98.4% of healthy controls |
| **F1-Score** | **95.69%** | Balanced harmonic mean of Precision and Recall |
| **AUC-ROC** | **0.9705** | Near-perfect discrimination (1.0 = perfect) |

### Comparison with State-of-the-Art (ABIDE-I Benchmark)

| Method | Approach | Accuracy |
|---|---|---|
| Nielsen et al. (2013) | GLM on FC features | 60.0% |
| Abraham et al. (2017) | SVM + atlas connectivity | 67.0% |
| Heinsfeld et al. (2018) | Stacked autoencoders | 70.1% |
| Li et al. (2021) | BrainGNN | 72.3% |
| Eslami et al. (2019) | ASD-DiagNet (hybrid) | 73.2% |
| Hu et al. (2023) | Transformer on FC windows | 74.1% |
| **This Work** | **XGBoost + Quality Metrics** | **95.96%** |

Our model surpasses the best prior art by **+22 percentage points**.

### Why Feature Importance Validates Our Hypothesis

The XGBoost gain-based feature importance ranking shows that **fMRI quality metrics (FD, DVARS, tSNR) rank prominently alongside ADOS total score and FIQ** in the top 20 most important features. This empirically confirms that scan quality indices encode genuine diagnostic information — not just noise.

---

## 8. Complete File-by-File Reference

### 8.1 Documentation & Configuration

| File | Purpose |
|---|---|
| `Docs/paper.tex` | The full IEEE-format research paper in LaTeX source |
| `Docs/references.bib` | 40+ bibliographic references cited in the paper |
| `Docs/final_report.pdf` | Final compiled PDF report for submission |
| `Docs/workspace_execution_guide.md` | This file — the comprehensive project guide |
| `PROBLEM_STATEMENT_DETAILED_COMPARISON.md` | Analysis of the alignment/gap between "early detection" and the current model's actual capability |
| `VISUALIZATION_REPORT.md` | Interpretation guide for all generated evaluation plots |
| `requirements.txt` | Python dependencies for the root project environment |
| `.gitignore` | Version control exclusion rules (`.venv/`, `__pycache__/`, etc.) |
| `.venv/` | Local Python virtual environment (not committed to Git) |
| `ABIDE_LEGEND_V1.02.pdf` | Official ABIDE-I data dictionary (explains all column names) |
| `ABIDEII_Data_Legend.pdf` | Official ABIDE-II data dictionary |

---

### 8.2 Raw Data Files

| File | Purpose |
|---|---|
| `Phenotypic_V1_0b.csv` | **Raw** ABIDE-I phenotypic file (downloaded from FCP-INDI). Contains demographics, clinical scores, and site info for all subjects. |
| `Phenotypic_V1_0b_preprocessed1.csv` | **Processed** phenotypic file — the primary dataset used for all training. Contains the merged quality metrics and cleaned features for 1,112 subjects. |
| `ABIDEII_Composite_Phenotypic.csv` | ABIDE-II phenotypic data (for future multi-cohort validation). |
| `abideII_5-10_download_checklist.csv` | Download manifest listing ABIDE-II subjects in age range 5–10 years. |
| `abide_download/` | Directory where raw ABIDE fMRI files are downloaded. |
| `abide1_fmri_data/` | Organised local storage for ABIDE-I imaging data. |
| `ABIDEII_MRI_Quality_Metrics/` | Quality metric outputs for ABIDE-II subjects. |

---

### 8.3 Data Acquisition Scripts

| File | When to Run | Purpose |
|---|---|---|
| `download_abide_preproc.py` | **Step 1** — One-time data fetch | Downloads preprocessed ABIDE derivatives from FCP-INDI AWS S3. Supports filtering by age, sex, site, and diagnosis. Usage: `python download_abide_preproc.py -d func_preproc -p cpac -s nofilt_noglobal -o ./abide_download` |
| `abide_s3_utils.py` | Called internally | Helper library for S3 connectivity: `S3ABIDEClient` (fetches phenotypic data and NIfTI images), `ABIDEDataFilter` (applies age/sex/site filters). |
| `map_rawout_to_organized_s3.py` | After download | Reorganises the S3 raw directory structure into a clean, browsable local hierarchy. |

---

### 8.4 Data Preparation & Feature Engineering Scripts

| File | When to Run | Purpose |
|---|---|---|
| `clean_duplicates.py` | **Step 2** — After download | Detects and removes duplicate `SUB_ID` entries in the phenotypic CSV. ABIDE aggregates from 17 sites, and some subjects can appear twice. |
| `generate_full_dataset.py` | **Step 3** — Feature extraction | Loads `Phenotypic_V1_0b_preprocessed1.csv`, extracts 85 raw features (demographics + clinical scales + fMRI quality metrics), fills missing values with column-wise median, and applies the trained model's scaler/selector pipeline to produce the `asd_model_features_all.csv` used by the Streamlit app. |
| `abide_loader.py` | Utility / Notebooks | A Python generator that streams ABIDE subject data (phenotypic + NIfTI image) directly from S3 without requiring a full 180GB local download. Useful for Jupyter notebook exploration. |
| `abide_streaming_analysis.py` | Optional analysis | Full production-ready streaming pipeline. Demonstrates filtering and feature extraction across multiple sites (NYU, UCLA, Stanford, Caltech) directly from S3, saving lightweight CSV summaries. |

---

### 8.5 Model Training Scripts

| File | When to Run | Purpose |
|---|---|---|
| `train_model_reproducible.py` | **Step 4 (Primary)** | **The canonical training script.** Reproduces the exact 95.96% accuracy result from the paper. Pipeline: load data → select 40 features (SelectKBest + MI) → RobustScaler → XGBoost (300 trees, depth=5, lr=0.05) → 10-fold CV → serialize `.pkl`. |
| `improved_asd_model_FIXED.py` | **Step 4 (Advanced)** | An extended training script with additional validation: checks for subject leakage between train/test, validates feature-to-sample ratio, trains an ensemble (RF + GB + LR + XGBoost) to compare, and explicitly removes post-diagnostic assessment columns to test for data leakage. |

---

### 8.6 Serialized Model Artifacts

| File | Description |
|---|---|
| `improved_asd_model.pkl` | Initial model artifact (may have pre-fix issues). |
| `improved_asd_model_FIXED.pkl` | **Current production model.** Serialized `joblib` archive containing: `model` (XGBoostClassifier), `selector` (SelectKBest), `scaler` (RobustScaler), `feature_names`, `selected_features`, and `metrics` dictionary. |
| `streamlit_model_tester/asd_model.pkl` | Model artifact packaged for the Streamlit deployment. |

---

### 8.7 Evaluation & Visualization

| File | When to Run | Purpose |
|---|---|---|
| `model_visualizations.py` | **Step 5** — After training | Generates all publication-quality plots referenced in the paper's Section 4. Saves them to `plots/`. |

**Output plots in `plots/` directory (referenced in paper.tex):**

| Plot File | Figure in Paper | Description |
|---|---|---|
| `confusion_matrix.png` | Fig. 3 | Count + normalized confusion matrix on test set |
| `roc_curve.png` | Fig. 4a | ROC curve (AUC = 0.9705) |
| `precision_recall_curve.png` | Fig. 4b | PR curve (AP = 0.97) |
| `feature_importance.png` | Fig. 5 | Top 20 XGBoost gain-importance features |
| `calibration_curve.png` | Discussion | Model probability calibration |
| `classification_report.png` | Discussion | Precision/Recall/F1 per class |
| `prediction_distribution.png` | Discussion | Distribution of predicted probabilities |
| `threshold_analysis.png` | Discussion | Sensitivity/Specificity vs. threshold |
| `performance_summary.png` | Summary | All metrics at a glance |
| `system_architecture.png` | Fig. 1 | Full pipeline architecture diagram |
| `model_arc.png` | Fig. 2 | XGBoost ensemble structure |

---

### 8.8 Deployment (Streamlit Web Application)

| File | Purpose |
|---|---|
| `streamlit_model_tester/app.py` | **The interactive web application.** A Streamlit dashboard with three tabs: (1) *Test on Sample* — select any of the 1,112 subjects and see the model's prediction vs. the true label; (2) *Test on Custom Data* — upload a CSV of new subjects and get batch predictions; (3) *Model Performance* — view all metrics, CV results, and dataset distribution. |
| `streamlit_model_tester/asd_model.pkl` | Model artifact used by the app |
| `streamlit_model_tester/asd_model_features.csv` | Pre-processed features for quick app demos |
| `streamlit_model_tester/asd_model_features_all.csv` | Full 1,112-subject processed feature dataset |
| `streamlit_model_tester/asd_model_metrics.csv` | Model performance metrics CSV |
| `streamlit_model_tester/requirements.txt` | App-specific Python dependencies |
| `streamlit_model_tester/README.md` | App-specific setup and usage guide |
| `create_test_data.py` | Generates sample test CSVs (5, 10, 20, 50 samples) for testing the app upload feature |

---

## 9. Execution Order (Step-by-Step)

If you are rebuilding the project from scratch, follow this exact sequence:

```
STEP 1: Download the data
─────────────────────────
python download_abide_preproc.py \
  -d func_preproc \
  -p cpac \
  -s nofilt_noglobal \
  -o ./abide_download \
  -a   # both ASD and TDC

STEP 2: Sanitize the phenotypic data
─────────────────────────────────────
python clean_duplicates.py
  → Ensures unique SUB_IDs in Phenotypic_V1_0b_preprocessed1.csv

STEP 3: Generate the full feature dataset
──────────────────────────────────────────
python generate_full_dataset.py
  → Reads Phenotypic_V1_0b_preprocessed1.csv
  → Extracts 85 features + median imputation
  → Outputs: streamlit_model_tester/asd_model_features_all.csv

STEP 4: Train the model (canonical)
────────────────────────────────────
python train_model_reproducible.py
  → MI feature selection (top 40)
  → XGBoost training (300 trees)
  → 10-fold cross-validation
  → Outputs: improved_asd_model_FIXED.pkl
             improved_model_features_FIXED.csv
             improved_model_metrics_FIXED.csv

STEP 5: Generate evaluation plots
──────────────────────────────────
python model_visualizations.py
  → Outputs all .png files to plots/ directory

STEP 6: Launch the web application
────────────────────────────────────
cd streamlit_model_tester
streamlit run app.py
  → Opens at http://localhost:8501
```

---

## 10. How to Set Up & Run the Project

### Prerequisites

- Python 3.8 or higher
- pip package manager
- ~500 MB disk space for the phenotypic data and models
- (Optional) AWS CLI for full S3 download (~180 GB for imaging files)

### Installation

```bash
# 1. Navigate to the project root
cd FYP_ASD

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Core Dependencies (from requirements.txt)

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation and numerical computing |
| `scikit-learn` | SelectKBest, RobustScaler, cross-validation, metrics |
| `xgboost` | The primary classifier |
| `joblib` | Model serialization (save/load .pkl files) |
| `streamlit` | Interactive web application framework |
| `matplotlib`, `seaborn` | Plotting and visualization |

### Run the Streamlit App (Quickest Demo)

```bash
cd streamlit_model_tester
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 11. Key References from the Paper

The following are the most important citations from `references.bib`:

| Citation | Significance |
|---|---|
| Di Martino et al. (2014) — *Molecular Psychiatry* | Original ABIDE dataset paper |
| Maenner et al. (2023) — *MMWR* | CDC ASD prevalence: 1 in 36 children |
| Power et al. (2012) — *NeuroImage* | Defined Framewise Displacement (FD) and DVARS |
| Chen & Guestrin (2016) — *KDD* | XGBoost algorithm paper |
| Kraskov et al. (2004) — *Physical Review E* | k-NN Mutual Information estimator |
| Dawson et al. (2010) — *Pediatrics* | Early intervention (Early Start Denver Model) efficacy |
| Nielsen et al. (2013) — *Frontiers* | First multi-site ABIDE classification baseline (60%) |
| Heinsfeld et al. (2018) — *NeuroImage: Clinical* | Deep learning on ABIDE baseline (70.1%) |
| Eslami et al. (2019) — *Frontiers* | ASD-DiagNet hybrid baseline (73.2%) |
| Fournier et al. (2010) — *JADD* | Motor coordination difficulties in ASD (neurobiological rationale) |
| Lundberg & Lee (2017) — *NeurIPS* | SHAP explainability (future work direction) |
| Craddock et al. (2013) — *Frontiers* | C-PAC preprocessing pipeline |

---

*Last Updated: May 2026 | Prepared by: Enduluri Reddy Rohith et al., MITS*
