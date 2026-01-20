# ASD Detection Model - Streamlit Web App

This folder contains a complete Streamlit web application for testing the trained ASD detection model.

## 📁 Contents

- **app.py** - Main Streamlit application
- **requirements.txt** - Python dependencies
- **improved_asd_model_FIXED.pkl** - Trained model and artifacts
- **improved_model_features_FIXED.csv** - Feature data
- **improved_model_metrics_FIXED.csv** - Model performance metrics

## 🚀 How to Run

### Option 1: Using the venv from parent directory

```bash
# From parent FYP_ASD directory
cd streamlit_model_tester
.\.venv\Scripts\Activate.ps1  # or source .venv/bin/activate on Linux
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Create a new environment

```bash
cd streamlit_model_tester
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
streamlit run app.py
```

## 📊 Features

The web app has three main tabs:

### 🎯 Test on Sample

- Select any sample from the dataset
- View true label vs model prediction
- See prediction probability
- Check if prediction was correct

### 📊 Test on Custom Data

- Upload CSV file with custom features
- View predictions for all samples
- Download results as CSV

### 📈 Model Performance

- View detailed metrics
- Cross-validation results
- ROC-AUC score
- Dataset distribution

## 💡 Usage

1. **Start the app:**

   ```bash
   streamlit run app.py
   ```

2. **Access in browser:**
   - Usually opens at `http://localhost:8501`

3. **Test the model:**
   - Select a sample and click "Random Sample" for quick testing
   - Or upload your own data in the "Test on Custom Data" tab

## 📊 Model Information

- **Accuracy:** 100.00%
- **CV Accuracy:** 100.00% ± 0.00%
- **Training Samples:** 889
- **Test Samples:** 223
- **Total Features:** 89
- **Dataset:** ABIDE (1,112 subjects)

## ⚠️ Important Notes

- Model requires exactly 89 features in the correct order
- All CSV uploads must have matching column names
- Predictions are based on the trained Random Forest model
- No external API calls - all processing is local

## 🔧 Troubleshooting

If you get import errors:

```bash
pip install -r requirements.txt --upgrade
```

If model file is not found:

```bash
# Ensure improved_asd_model_FIXED.pkl is in this folder
```

If Streamlit is not recognized:

```bash
python -m streamlit run app.py
```

## 📝 License

This model tester is part of the FYP ASD Detection project.
