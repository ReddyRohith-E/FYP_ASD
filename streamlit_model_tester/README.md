# ASD Detection — Streamlit Web Application

> Part of the Final Year Project: *"Developing Predictive Models for Early ASD Detection Based on fMRI Scans"*  
> Department of CSE, Madanapalle Institute of Technology & Science (MITS)

This folder contains the interactive web application for testing and demonstrating the trained **XGBoost-based ASD detection model** that achieved **95.96% accuracy** on the ABIDE-I dataset.

---

## What Does This App Do?

The app allows anyone — clinicians, researchers, or students — to interact with the trained model without writing any code. It provides three modes of use:

| Tab | What you can do |
|---|---|
| 🎯 **Test on Sample** | Pick any of the 1,112 ABIDE subjects, see the model's prediction, and compare it to the true clinical diagnosis |
| 📊 **Test on Custom Data** | Upload your own CSV file with subject features and get batch predictions + downloadable results |
| 📈 **Model Performance** | View full performance metrics (Accuracy, Precision, Recall, F1, AUC-ROC), cross-validation results, and dataset composition |

---

## 📁 Contents of This Folder

| File / Folder | Description |
|---|---|
| `app.py` | Main Streamlit application source code |
| `requirements.txt` | Python dependencies needed to run the app |
| `asd_model.pkl` | **Trained model artifact** — contains the XGBoost model, SelectKBest feature selector, and RobustScaler (packaged with `joblib`) |
| `asd_model_features.csv` | Subset of pre-processed ABIDE features for quick in-app demos |
| `asd_model_features_all.csv` | Full 1,112-subject processed feature dataset |
| `asd_model_metrics.csv` | Model performance metrics in CSV format |
| `test_asd_5_samples.csv` | 5 ASD sample rows for testing the CSV upload feature |
| `test_tdc_5_samples.csv` | 5 TDC sample rows for testing the CSV upload feature |
| `test_data_20_samples.csv` | 20 mixed samples for batch testing |
| `test_data_50_samples.csv` | 50 mixed samples for batch testing |
| `test_mixed_10_samples.csv` | 10 mixed ASD/TDC samples |

---

## 🚀 How to Run

### Option 1: Use the parent project's virtual environment (Recommended)

```powershell
# From the root FYP_ASD directory
cd streamlit_model_tester
..\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Create a fresh virtual environment

```bash
cd streamlit_model_tester

# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will open at **`http://localhost:8501`** in your browser.

---

## 🧠 About the Model

The model loaded by this app (`asd_model.pkl`) is the result of the full training pipeline described in `paper.tex`. Here is a summary:

| Property | Value |
|---|---|
| **Algorithm** | XGBoost (Extreme Gradient Boosting) |
| **Dataset** | ABIDE-I (17 international sites) |
| **Total Subjects** | 1,112 (539 ASD + 573 TDC) |
| **Training Set** | 889 subjects (80%) |
| **Test Set** | 223 subjects (20%) |
| **Feature Selection** | Top 40 features via Mutual Information (SelectKBest) |
| **Scaling** | RobustScaler (robust to fMRI data outliers) |
| **Validation** | Stratified 10-fold cross-validation |
| **Test Accuracy** | **95.96%** |
| **AUC-ROC** | **0.9705** |

### Key Features Used by the Model

The model's 40 selected features come from two categories:

1. **Clinical Assessment Scores** (post-diagnosis):
   - ADOS total score, ADOS Gotham severity
   - ADI-R social, verbal, and RRB subscores
   - SRS (Social Responsiveness Scale) subscores
   - VINELAND adaptive behaviour scores
   - WISC-IV cognitive subscores

2. **fMRI Scan Quality Metrics** (the novel contribution):
   - `func_mean_fd` — Mean Framewise Displacement (head motion)
   - `func_num_fd` / `func_perc_fd` — Number/percentage of high-motion frames
   - `func_quality` — Overall functional scan quality metric

> **Note:** The quality metrics appear alongside clinical scales in the top feature importance rankings, validating the paper's core hypothesis that *motion artefacts encode diagnostically useful information*.

---

## 📊 How to Use the CSV Upload Feature

To test on your own data, prepare a CSV file with the same 40 feature columns as the training data. A template can be found in any of the `test_*.csv` files included in this folder.

### Required column order

The app expects exactly the features selected during training. The easiest way to prepare data is to:

1. Download a fresh ABIDE phenotypic record (same columns as `Phenotypic_V1_0b_preprocessed1.csv`)
2. Run `generate_full_dataset.py` from the parent directory
3. Export the resulting pre-processed rows and upload to the app

---

## ⚠️ Important Limitations

- This model is a **research prototype**, not a clinical diagnostic tool.
- The ADOS and ADI-R scores used as features are **only available after** a clinical evaluation has already been conducted — making this model best suited as a **decision-support** or **confirmation** tool, not a standalone screen.
- The model was trained on subjects aged **6.5–64 years** from 17 research sites. Performance may differ on populations outside this range.
- For genuinely early (pre-diagnosis) detection in infants, a different dataset (e.g., IBIS) and different features (early developmental markers) would be required.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `streamlit: command not found` | Run `python -m streamlit run app.py` |
| `FileNotFoundError: asd_model.pkl` | Ensure `asd_model.pkl` is in this folder (run `train_model_reproducible.py` from parent dir if missing) |
| CSV upload fails | Check that your CSV has the same column names as the `test_*.csv` example files |
| App opens but shows errors | Ensure `.venv` is activated and all packages are installed |

---

## 📝 License & Attribution

This application is part of the Final Year Project submitted to the Department of Computer Science & Engineering, MITS, Madanapalle, 2026.  
Data sourced from the publicly available [ABIDE Preprocessed](http://preprocessed-connectomes-project.org/abide/) repository under their data-sharing agreement.
