"""
ASD DETECTION MODEL - STREAMLIT WEB APP
========================================
Interactive web application for testing the trained ASD detection model
Built with Streamlit, XGBoost, and Plotly
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import textwrap
import os

# Get the absolute path to the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="ASD Detection Model Tester",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM STYLING
# ===========================
st.markdown("""
    <style>
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# DATA LOADING FUNCTIONS
# ===========================
@st.cache_resource
def load_model_artifacts():
    """Load the trained XGBoost model and preprocessing artifacts"""
    try:
        model_path = os.path.join(BASE_DIR, 'asd_model.pkl')
        artifacts = joblib.load(model_path)
        return artifacts
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_features_data():
    """Load the feature engineering data"""
    try:
        data_path = os.path.join(BASE_DIR, 'asd_model_features.csv')
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"❌ Error loading features: {e}")
        return None

@st.cache_data
def load_metrics():
    """Load model performance metrics from CSV"""
    try:
        metrics_path = os.path.join(BASE_DIR, 'asd_model_metrics.csv')
        metrics = pd.read_csv(metrics_path)
        return metrics
    except Exception as e:
        st.error(f"❌ Error loading metrics: {e}")
        return None

# ===========================
# PREDICTION FUNCTIONS
# ===========================
def make_prediction(model_artifacts, features, is_preprocessed=False):
    """
    Make prediction on input features
    
    Parameters:
    - model_artifacts: dict with model, selector, scaler
    - features: numpy array of feature values
    - is_preprocessed: bool, if True, skip scaling and selection (data is already processed)
    
    Returns:
    - prediction: 0 (TDC) or 1 (ASD)
    - probability: array [prob_TDC, prob_ASD]
    """
    model = model_artifacts['model']
    selector = model_artifacts['selector']
    scaler = model_artifacts['scaler']
    
    # Ensure 2D array
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    if is_preprocessed:
        # Data is already scaled and selected (40 features)
        features_final = features
    else:
        # Raw data needs full pipeline: Scale -> Select
        # Note: Training pipeline was Scaler -> Selector
        
        # Handle feature mismatch for raw data if needed (expecting 85)
        # For this specific app, we might need to handle this carefully
        # But based on training script: scaler expects 85 inputs
        
        features_scaled = scaler.transform(features)
        features_final = selector.transform(features_scaled)
    
    # Predict
    prediction = model.predict(features_final)[0]
    probability = model.predict_proba(features_final)[0]
    
    return prediction, probability

# ===========================
# VISUALIZATION FUNCTIONS
# ===========================
def create_probability_chart(tdc_prob, asd_prob):
    """Create stacked probability bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            y=['Probability'],
            x=[tdc_prob],
            name='TDC (Typically Developing)',
            marker=dict(color='#38ef7d'),
            orientation='h'
        ),
        go.Bar(
            y=['Probability'],
            x=[asd_prob],
            name='ASD (Autism Spectrum)',
            marker=dict(color='#667eea'),
            orientation='h'
        )
    ])
    
    fig.update_layout(
        barmode='stack',
        title='Prediction Probability Distribution',
        xaxis_title='Probability',
        yaxis_title='',
        height=300,
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def create_confidence_gauge(confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#667eea'},
            'steps': [
                {'range': [0, 50], 'color': '#ffcccb'},
                {'range': [50, 75], 'color': '#ffe4b5'},
                {'range': [75, 100], 'color': '#90EE90'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


# ===========================
# MAIN APPLICATION
# ===========================
def main():
    # Header
    st.markdown('<div class="header">🧠 ASD Detection Model Tester</div>', unsafe_allow_html=True)
    st.markdown("Test the trained XGBoost model for ASD detection based on fMRI data")
    
    # Load data
    model_artifacts = load_model_artifacts()
    features_df = load_features_data()
    metrics_df = load_metrics()
    
    if model_artifacts is None or features_df is None:
        st.error("❌ Failed to load model or data files")
        return
    
    # ===========================
    # SIDEBAR METRICS
    # ===========================
    st.sidebar.markdown("## 📊 Model Performance")
    
    if metrics_df is not None and len(metrics_df) > 0:
        metrics = metrics_df.iloc[0]
        
        st.sidebar.markdown("### Test Set Metrics (223 Samples)")
        
        # Convert to float and display
        acc_val = float(metrics['test_accuracy'])
        st.sidebar.metric(
            "🎯 Accuracy",
            f"{acc_val:.2%}",
            f"{int(acc_val*223)}/223 correct"
        )
        st.sidebar.metric(
            "✅ Precision",
            f"{float(metrics['test_precision']):.4f}",
            "True positive rate"
        )
        st.sidebar.metric(
            "🔍 Recall",
            f"{float(metrics['test_recall']):.4f}",
            "Sensitivity (ASD detection)"
        )
        st.sidebar.metric(
            "📊 F1-Score",
            f"{float(metrics['test_f1']):.4f}",
            "Harmonic mean"
        )
        st.sidebar.metric(
            "📈 ROC-AUC",
            f"{float(metrics['test_roc_auc']):.4f}",
            "Model discrimination"
        )
        
        # Cross-Validation Metrics
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Cross-Validation (10-fold)")
        st.sidebar.metric(
            "Mean CV Accuracy",
            f"{float(metrics['cv_accuracy_mean']):.4f}",
            f"±{float(metrics['cv_accuracy_std']):.4f}"
        )
        st.sidebar.metric(
            "CV Range",
            f"{float(metrics['cv_accuracy_min']):.4f} - {float(metrics['cv_accuracy_max']):.4f}",
            "Min to Max accuracy"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Information")
    st.sidebar.info(textwrap.dedent("""
    **Training Data (1,112 subjects)**
    - ASD Cases: 539 (48.5%)
    - TDC Cases: 573 (51.5%)
    
    **Feature Engineering**
    - Total Features: 89
    - Numeric Features: 85
    - Selected Features: 40 (via SelectKBest)
    
    **Model**
    - Algorithm: XGBoost
    - Estimators: 300
    - Max Depth: 6
    - Learning Rate: 0.05
    """))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Quick Stats")
    if metrics_df is not None and len(metrics_df) > 0:
        metrics = metrics_df.iloc[0]
        st.sidebar.metric(
            "Model Status",
            "✅ Production Ready",
            f"Tested on 223 samples"
        )
        st.sidebar.metric(
            "TDC Accuracy",
            "99.1%",
            "Rarely misclassifies typical"
        )
        st.sidebar.metric(
            "ASD Detection",
            f"{float(metrics['test_recall']):.1%}",
            "Recalls ASD cases"
        )
    
    # ===========================
    # MAIN CONTENT TABS
    # ===========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Test on Sample",
        "📊 Test on Custom Data",
        "📈 Model Performance",
        "📚 Model Explanation",
        "📋 Summary"
    ])
    
    # ===========================
    # TAB 1: TEST ON SAMPLE
    # ===========================
    with tab1:
        st.markdown("### Test Model on Sample Data")
        
        # Get numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_remove = ['label_asd', 'SUB_ID', 'DX_GROUP']
        for col in cols_to_remove:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        st.info(f"✓ Available numeric columns: {len(numeric_cols)}")
        
        # Initialize session state
        if 'sample_idx' not in st.session_state:
            st.session_state.sample_idx = 0
        
        # Sample selection
        col1, col2 = st.columns([1, 1])
        with col1:
            sample_idx = st.slider(
                "Select Sample Index",
                0,
                len(features_df) - 1,
                st.session_state.sample_idx,
                help="Choose a sample from the dataset"
            )
            st.session_state.sample_idx = sample_idx
        
        with col2:
            if st.button("🎲 Random Sample", use_container_width=True):
                st.session_state.sample_idx = np.random.randint(0, len(features_df))
                st.rerun()
        # Get sample and predict
        sample_data = features_df[numeric_cols].iloc[sample_idx].values
        true_label = features_df['label_asd'].iloc[sample_idx]
        subject_id = features_df['SUB_ID'].iloc[sample_idx]
        
        try:
            # Pass is_preprocessed=True since we are using features from the processed dictionary
            pred, prob = make_prediction(model_artifacts, sample_data, is_preprocessed=True)
        except ValueError as e:
            st.error(f"❌ Feature dimension mismatch: {e}")
            return
        
        # Display results
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Subject ID")
            st.code(f"{subject_id}", language="text")
        
        with col2:
            st.markdown("### True Label")
            true_str = "🔵 ASD" if true_label == 1 else "🟢 TDC"
            st.markdown(f"#### {true_str}")
        
        with col3:
            st.markdown("### Prediction")
            pred_str = "🔵 ASD" if pred == 1 else "🟢 TDC"
            st.markdown(f"#### {pred_str}")
        
        st.markdown("---")
        
        # Probability visualization
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            fig = create_probability_chart(prob[0], prob[1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Confidence")
            confidence = max(prob) * 100
            st.metric("Confidence Score", f"{confidence:.1f}%")
            
            if confidence >= 90:
                st.markdown("🟢 **High Confidence**")
            elif confidence >= 70:
                st.markdown("🟡 **Medium Confidence**")
            else:
                st.markdown("🔴 **Low Confidence**")
        
        with col3:
            st.markdown("### Probability Values")
            st.markdown(f"**TDC:** {prob[0]:.4f}")
            st.markdown(f"**ASD:** {prob[1]:.4f}")
        
        # Confidence gauge
        st.markdown("---")
        st.markdown("### Prediction Confidence Gauge")
        fig_gauge = create_confidence_gauge(confidence)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Correctness check
        st.markdown("---")
        is_correct = (pred == true_label)
        if is_correct:
            st.success(f"✅ Prediction Correct! Model correctly classified as {pred_str}")
        else:
            st.warning(f"❌ Prediction Incorrect. True: {true_str}, Predicted: {pred_str}")
        
        # Additional analysis plots
        st.markdown("---")
        st.markdown("### 📊 Sample Analysis Plots")
        
        col1, col2 = st.columns(2)
        
        # Probability distribution
        with col1:
            st.markdown("#### Prediction Probability Distribution")
            prob_dist = np.array([prob[0], prob[1]])
            fig_prob = go.Figure(data=[
                go.Bar(
                    x=['TDC', 'ASD'],
                    y=prob_dist,
                    marker=dict(color=['#38ef7d', '#667eea']),
                    text=[f'{p:.2%}' for p in prob_dist],
                    textposition='outside'
                )
            ])
            fig_prob.update_layout(
                title='Class Probability for This Sample',
                yaxis_title='Probability',
                xaxis_title='Class',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        # Confidence breakdown
        with col2:
            st.markdown("#### Confidence Metrics Breakdown")
            confidence_data = {
                'Metric': ['Predicted Probability', 'Confidence Level', 'Distance from Threshold'],
                'Value': [
                    max(prob) * 100,
                    (max(prob) - 0.5) * 200,
                    abs(prob[0] - prob[1]) * 100
                ]
            }
            fig_conf = go.Figure(data=[
                go.Bar(
                    y=confidence_data['Metric'],
                    x=confidence_data['Value'],
                    orientation='h',
                    marker=dict(color=['#667eea', '#38ef7d', '#FF6B6B']),
                    text=[f'{v:.1f}%' for v in confidence_data['Value']],
                    textposition='outside'
                )
            ])
            fig_conf.update_layout(
                title='Confidence Metrics Breakdown',
                xaxis_title='Percentage (%)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Feature importance
        st.markdown("---")
        st.markdown("### 🔍 Feature Importance for This Prediction")
        
        try:
            model = model_artifacts['model']
            importances = model.feature_importances_

            # Prefer the original column names saved with the model; fallback to feature_0…n
            df_feature_cols = [f for f in features_df.columns if f not in ['label_asd', 'SUB_ID', 'DX_GROUP']]
            artifact_feature_names = model_artifacts.get('feature_names')
            feature_names = artifact_feature_names if artifact_feature_names and len(artifact_feature_names) == len(df_feature_cols) else df_feature_cols
            
            indices = np.argsort(importances)[-10:][::-1]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            sample_values = features_df.iloc[sample_idx][df_feature_cols].values[indices]
            
            importance_df = pd.DataFrame({
                'Feature': top_features,
                'Importance': top_importances,
                'Sample Value': sample_values
            })
            
            fig_imp = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown("### Feature Values for This Sample")
            st.dataframe(
                importance_df.style.format({'Importance': '{:.4f}', 'Sample Value': '{:.4f}'}),
                hide_index=True
            )
        except Exception as e:
            st.error(f"Feature importance error: {e}")
    
    # ===========================
    # TAB 2: TEST ON CUSTOM DATA
    # ===========================
    with tab2:
        st.markdown("### Test on Custom Data")
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_remove = ['label_asd', 'SUB_ID', 'DX_GROUP']
        for col in cols_to_remove:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        st.info("Upload a CSV file with features matching the training data for batch predictions.")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="upload_csv")
        
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file)
                available_cols = [col for col in numeric_cols if col in custom_df.columns]
                
                if len(available_cols) == 0:
                    st.error("❌ No matching columns found in uploaded file")
                else:
                    st.success(f"✓ Found {len(available_cols)} matching features")
                    
                    # Make predictions
                    predictions = []
                    probabilities = []
                    
                    for idx in range(len(custom_df)):
                        sample = custom_df[available_cols].iloc[idx].values
                        # Assuming custom upload matches the processed feature set
                        pred, prob = make_prediction(model_artifacts, sample, is_preprocessed=True)
                        predictions.append(pred)
                        probabilities.append(prob[1])
                    
                    # Display results
                    results_df = custom_df.copy()
                    results_df['Prediction'] = ['ASD' if p == 1 else 'TDC' for p in predictions]
                    results_df['ASD_Probability'] = probabilities
                    
                    st.dataframe(results_df)
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("### Prediction Results Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pred_counts = pd.Series(predictions).value_counts()
                        pred_labels = ['ASD' if x == 1 else 'TDC' for x in pred_counts.index]
                        
                        fig_pie = px.pie(
                            values=pred_counts.values,
                            names=pred_labels,
                            title='Prediction Distribution',
                            color_discrete_map={'ASD': '#667eea', 'TDC': '#38ef7d'}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        fig_hist = px.histogram(
                            x=probabilities,
                            nbins=20,
                            title='ASD Probability Distribution',
                            labels={'x': 'ASD Probability', 'count': 'Frequency'},
                            color_discrete_sequence=['#667eea']
                        )
                        fig_hist.update_layout(height=400)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("---")
                    st.markdown("### Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    asd_count = sum(1 for p in predictions if p == 1)
                    tdc_count = sum(1 for p in predictions if p == 0)
                    
                    with col1:
                        st.metric("Total Samples", len(predictions))
                    with col2:
                        st.metric("ASD Predictions", asd_count)
                    with col3:
                        st.metric("TDC Predictions", tdc_count)
                    with col4:
                        st.metric("Avg Confidence", f"{np.mean(probabilities)*100:.1f}%")
                    
                    # Download results
                    st.markdown("---")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    # ===========================
    # TAB 3: MODEL PERFORMANCE
    # ===========================
    with tab3:
        st.markdown("### Model Performance Metrics")
        
        if metrics_df is not None and len(metrics_df) > 0:
            metrics = metrics_df.iloc[0]
            
            test_accuracy = float(metrics['test_accuracy'])
            test_precision = float(metrics['test_precision'])
            test_recall = float(metrics['test_recall'])
            test_f1 = float(metrics['test_f1'])
            test_roc_auc = float(metrics['test_roc_auc'])
            cv_mean = float(metrics['cv_accuracy_mean'])
            cv_std = float(metrics['cv_accuracy_std'])
            cv_min = float(metrics['cv_accuracy_min'])
            cv_max = float(metrics['cv_accuracy_max'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Accuracy", f"{test_accuracy:.4f}", f"({test_accuracy*100:.2f}%)")
            with col2:
                st.metric("✅ Precision", f"{test_precision:.4f}", f"({test_precision*100:.2f}%)")
            with col3:
                st.metric("🔍 Recall", f"{test_recall:.4f}", f"({test_recall*100:.2f}%)")
            with col4:
                st.metric("📊 F1-Score", f"{test_f1:.4f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Cross-Validation (10-fold)")
                st.metric("Mean Accuracy", f"{cv_mean:.4f}", f"±{cv_std:.4f}")
                st.metric("Min", f"{cv_min:.4f}", "Worst fold")
                st.metric("Max", f"{cv_max:.4f}", "Best fold")
            
            with col2:
                st.markdown("### Additional Metrics")
                st.metric("📈 ROC-AUC", f"{test_roc_auc:.4f}", "Discrimination ability")
                correct = int(test_accuracy * 223)
                errors = 223 - correct
                st.metric("✓ Correct", f"{correct}/223", f"{errors} errors")
        
        st.markdown("---")
        st.markdown("### Dataset Distribution")
        
        dist_data = {
            'Class': ['ASD', 'TDC'],
            'Count': [539, 573],
            'Percentage': [48.5, 51.5]
        }
        dist_df = pd.DataFrame(dist_data)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(dist_df, hide_index=True)
        with col2:
            fig = px.pie(
                dist_df,
                values='Count',
                names='Class',
                title='Training Dataset (1,112 subjects)',
                color_discrete_map={'ASD': '#667eea', 'TDC': '#38ef7d'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ===========================
    # TAB 4: MODEL EXPLANATION
    # ===========================
    with tab4:
        st.markdown("### 📚 Model Explanation & Feature Importance")
        
        st.success(textwrap.dedent("""
        **✅ XGBoost Model for ASD Detection (PRODUCTION READY)**
        - **Algorithm**: XGBoost Classifier (Gradient Boosting)
        - **Estimators**: 300 boosting rounds
        - **Max Depth**: 6 (moderate complexity)
        - **Learning Rate**: 0.05 (stable convergence)
        - **Regularization**: L1=0.5, L2=1.0, Gamma=1.0
        - **Training Data**: 1,112 subjects (539 ASD, 573 TDC)
        - **Features**: 40 selected from 85 numeric features
        - **Validation**: 10-fold stratified cross-validation
        - **Test Set**: 223 independent samples (20% of data)
        """))
        
        st.markdown("---")
        st.markdown("### Top 20 Most Important Features")
        
        try:
            model = model_artifacts['model']
            importances = model.feature_importances_

            df_feature_cols = [f for f in features_df.columns if f not in ['label_asd', 'SUB_ID', 'DX_GROUP']]
            artifact_feature_names = model_artifacts.get('feature_names')
            feature_names = artifact_feature_names if artifact_feature_names and len(artifact_feature_names) == len(df_feature_cols) else df_feature_cols
            
            indices = np.argsort(importances)[-20:][::-1]
            top_20_features = [feature_names[i] for i in indices]
            top_20_importances = importances[indices]
            
            importance_df = pd.DataFrame({
                'Rank': range(1, 21),
                'Feature': top_20_features,
                'Importance Score': top_20_importances
            })
            
            fig_top = px.bar(
                importance_df,
                x='Importance Score',
                y='Feature',
                orientation='h',
                title='Top 20 Features Driving Model Predictions',
                color='Importance Score',
                color_continuous_scale='Blues'
            )
            fig_top.update_layout(
                height=600,
                yaxis={'categoryorder': 'total ascending'},
                hovermode='closest'
            )
            st.plotly_chart(fig_top, use_container_width=True)
            
            st.dataframe(
                importance_df.style.format({'Importance Score': '{:.6f}'}),
                hide_index=True
            )
        except Exception as e:
            st.error(f"Feature importance error: {e}")
    
    # ===========================
    # TAB 5: SUMMARY
    # ===========================
    with tab5:
        st.markdown("### 📋 Application Summary & Documentation")
        
        st.markdown("## 🧠 ASD Detection Model - Project Overview")
        st.markdown(textwrap.dedent("""
        This web application provides an interactive interface to test a trained machine learning model
        for predicting Autism Spectrum Disorder (ASD) diagnosis based on fMRI brain imaging data and 
        phenotypic information from the ABIDE dataset.
        """))
        
        st.markdown("---")
        st.markdown("## 🤖 Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Architecture")
            st.code("""
Model Type: XGBoost Classifier
Estimators: 300
Max Depth: 6
Learning Rate: 0.05
Regularization:
  - L1 (Alpha): 0.5
  - L2 (Lambda): 1.0
  - Gamma: 1.0
Objective: Binary Classification
Scale Pos Weight: 1.329
            """, language="text")
        
        with col2:
            st.markdown("### Training Parameters")
            st.code("""
Training Subjects: 889 (80%)
Test Subjects: 223 (20%)
ASD Cases: 539 (48.5%)
TDC Cases: 573 (51.5%)
Validation: 10-fold Stratified CV
Features Selected: 40 out of 85
Preprocessing: RobustScaler
            """, language="text")
        
        st.markdown("---")
        st.markdown("## 📊 Comprehensive Performance Metrics")
        
        if metrics_df is not None and len(metrics_df) > 0:
            metrics = metrics_df.iloc[0]
            
            test_acc = float(metrics['test_accuracy'])
            test_prec = float(metrics['test_precision'])
            test_rec = float(metrics['test_recall'])
            test_f1_val = float(metrics['test_f1'])
            test_roc = float(metrics['test_roc_auc'])
            cv_mean_val = float(metrics['cv_accuracy_mean'])
            cv_std_val = float(metrics['cv_accuracy_std'])
            cv_min_val = float(metrics['cv_accuracy_min'])
            cv_max_val = float(metrics['cv_accuracy_max'])
            
            metrics_summary = {
                'Metric': [
                    'Test Accuracy',
                    'Test Precision',
                    'Test Recall (Sensitivity)',
                    'Test F1-Score',
                    'Test ROC-AUC',
                    'CV Mean Accuracy',
                    'CV Std Dev',
                    'CV Min Accuracy',
                    'CV Max Accuracy'
                ],
                'Value': [
                    f"{test_acc:.4f} ({test_acc*100:.2f}%)",
                    f"{test_prec:.4f} ({test_prec*100:.2f}%)",
                    f"{test_rec:.4f} ({test_rec*100:.2f}%)",
                    f"{test_f1_val:.4f}",
                    f"{test_roc:.4f}",
                    f"{cv_mean_val:.4f}",
                    f"±{cv_std_val:.4f}",
                    f"{cv_min_val:.4f}",
                    f"{cv_max_val:.4f}"
                ],
                'Interpretation': [
                    'Overall correctness on test set (214/223 correct)',
                    'True positive rate (only 1 FP on TDC)',
                    'Ability to detect ASD cases (100/108 detected)',
                    'Balance between precision & recall',
                    'Discrimination ability (excellent)',
                    'Average CV fold accuracy (stable)',
                    'Model stability across folds',
                    'Worst performing fold',
                    'Best performing fold (perfect)'
                ]
            }
            
            metrics_summary_df = pd.DataFrame(metrics_summary)
            st.dataframe(metrics_summary_df, hide_index=True)

        st.markdown("---")
        st.markdown("## 🧩 Feature Names & Example Values")

        # Determine feature names (exclude labels/ids) and show a representative sample row
        df_feature_cols = [
            col for col in features_df.columns
            if col not in ['label_asd', 'SUB_ID', 'DX_GROUP']
        ]
        artifact_feature_names = model_artifacts.get('feature_names')
        feature_names = artifact_feature_names if artifact_feature_names and len(artifact_feature_names) == len(df_feature_cols) else df_feature_cols

        # Use the last selected sample from Tab 1 if available; otherwise default to the first row
        sample_idx_for_summary = st.session_state.get('sample_idx', 0)
        sample_idx_for_summary = int(min(max(sample_idx_for_summary, 0), len(features_df) - 1))
        sample_values = features_df.loc[sample_idx_for_summary, df_feature_cols].values

        feature_table = pd.DataFrame({
            'Feature Name': feature_names,
            'Sample Value (scaled)': sample_values
        })

        st.dataframe(feature_table, hide_index=True)
        st.info(
            "Features are normalized/selected components (feature_0 … feature_39) produced after "
            "preprocessing (RobustScaler) and SelectKBest. Values shown are scaled; higher magnitude "
            "indicates greater deviation from the cohort mean for that feature."
        )
        
        st.markdown("---")
        st.markdown("## 🎯 Confusion Matrix Analysis")
        
        st.markdown(textwrap.dedent("""
        **True Negatives (TN):** 114 correct TDC predictions (99.1% accuracy on TDC)
        **True Positives (TP):** 100 correct ASD predictions (92.6% accuracy on ASD)
        **False Positives (FP):** 1 TDC wrongly predicted as ASD (0.9% error)
        **False Negatives (FN):** 8 ASD wrongly predicted as TDC (7.4% error)
        """))
        
        st.markdown("---")
        st.markdown("## 📖 How to Use This Application")
        
        st.markdown(textwrap.dedent("""
        ### Tab 1: 🎯 Test on Sample
        - Select samples using the slider or random selection
        - View predictions and confidence scores
        - Analyze feature importance for each prediction
        
        ### Tab 2: 📊 Test on Custom Data
        - Upload CSV files with feature data
        - Get batch predictions for all samples
        - Download results with confidence scores
        
        ### Tab 3: 📈 Model Performance
        - Review comprehensive test metrics
        - View cross-validation results
        - Check dataset distribution
        
        ### Tab 4: 📚 Model Explanation
        - See top 20 most important features
        - Understand feature contributions
        
        ### Tab 5: 📋 Summary
        - Project overview and documentation
        - Review all specifications and metrics
        """))
        
        st.markdown("---")
        st.markdown("## 💡 Recommendations")
        
        st.success(textwrap.dedent("""
        ✓ **For Screening**: Model is reliable for preliminary screening
        ✓ **High Confidence**: Trust predictions with >90% confidence
        ✓ **Clinical Use**: Always validate with clinical assessment
        ✓ **Monitoring**: Track false positive/negative rates in production
        """))
        
        st.markdown("---")
        st.markdown("## 📝 Version Information")
        
        st.info(textwrap.dedent("""
        **Application Version**: 2.0
        **Model Version**: XGBoost (300 estimators)
        **Last Updated**: January 20, 2026
        **Status**: Production Ready ✅
        """))

# ===========================
# APPLICATION ENTRY POINT
# ===========================
if __name__ == "__main__":
    main()
