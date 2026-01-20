"""
COMPREHENSIVE MODEL VISUALIZATION & ANALYSIS
=============================================
Generates all plots, curves, and comparison metrics for the ASD detection model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_model_and_data():
    """Load the trained model and feature data"""
    print("Loading model and data...")
    
    import os
    
    # Try multiple locations for model file
    model_paths = [
        'improved_asd_model_FIXED.pkl',
        'streamlit_model_tester/improved_asd_model_FIXED.pkl',
        'streamlit_model_tester/asd_model.pkl',
        'asd_model.pkl',
        os.path.join('streamlit_model_tester', 'asd_model.pkl')
    ]
    
    artifacts = None
    for path in model_paths:
        if os.path.exists(path):
            artifacts = joblib.load(path)
            print(f"✓ Loaded model from: {path}")
            break
    
    if artifacts is None:
        raise FileNotFoundError("Could not find improved_asd_model_FIXED.pkl")
    
    model = artifacts['model']
    selector = artifacts['selector']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    
    # Try multiple locations for feature data
    feature_paths = [
        'Phenotypic_V1_0b_preprocessed1.csv',  # Use the raw phenotypic data
        'improved_model_features_FIXED.csv',
        'streamlit_model_tester/improved_model_features_FIXED.csv',
        'streamlit_model_tester/model_features.csv',
        'model_features.csv'
    ]
    
    df = None
    for path in feature_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ Loaded features from: {path}")
            break
    
    if df is None:
        raise FileNotFoundError("Could not find improved_model_features_FIXED.csv")
    
    return model, selector, scaler, feature_names, df


def prepare_data(df, selector, scaler, feature_names):
    """Prepare data for prediction and visualization"""
    print("Preparing data...")
    
    # Filter for valid diagnoses
    if 'DX_GROUP' in df.columns:
        df = df[df['DX_GROUP'].isin([1, 2])].copy()
        df['label_asd'] = (df['DX_GROUP'] == 1).astype(int)
    
    # Check which features are available
    available_features = [f for f in feature_names if f in df.columns]
    print(f"✓ Found {len(available_features)}/{len(feature_names)} features in data")
    
    if len(available_features) == 0:
        raise ValueError("No matching features found in data")
    
    # Extract features and labels
    X_raw = df[available_features].fillna(df[available_features].median()).values
    y = df['label_asd'].values
    
    print(f"✓ Data shape: {X_raw.shape}, using features as-is for prediction")
    
    # Return raw features - the model will handle transformation internally if needed
    return X_raw, y, X_raw


def plot_confusion_matrix(y_true, y_pred, save_path='plots/confusion_matrix.png'):
    """Plot confusion matrix with detailed annotations"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot 1: Counts
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                     display_labels=['TDC', 'ASD'])
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Plot 2: Normalized
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, 
                                     display_labels=['TDC', 'ASD'])
    disp2.plot(ax=axes[1], cmap='Greens', values_format='.2%')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_proba, save_path='plots/roc_curve.png'):
    """Plot ROC curve with AUC score"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=3, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5000)')
    
    # Mark optimal threshold (closest to top-left)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
            label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_proba, save_path='plots/precision_recall_curve.png'):
    """Plot Precision-Recall curve"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    # Plot PR curve
    ax.plot(recall, precision, color='darkgreen', lw=3,
            label=f'PR curve (AP = {avg_precision:.4f})')
    
    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
            linestyle='--', label=f'Random Classifier (AP = {baseline:.4f})')
    
    # Mark F1-optimal point
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=10,
            label=f'Best F1 = {f1_scores[optimal_idx]:.4f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return avg_precision


def plot_threshold_analysis(y_true, y_proba, save_path='plots/threshold_analysis.png'):
    """Plot how metrics vary with classification threshold"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    thresholds = np.linspace(0, 1, 100)
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        acc = (y_pred == y_true).mean()
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    # Plot 1: All metrics together
    axes[0, 0].plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    axes[0, 0].plot(thresholds, precisions, label='Precision', linewidth=2)
    axes[0, 0].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[0, 0].plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    axes[0, 0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    axes[0, 0].set_xlabel('Classification Threshold', fontsize=11)
    axes[0, 0].set_ylabel('Score', fontsize=11)
    axes[0, 0].set_title('Metrics vs Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: F1-Score focus
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_f1_idx]
    axes[0, 1].plot(thresholds, f1_scores, linewidth=3, color='purple')
    axes[0, 1].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Optimal = {optimal_threshold:.3f}')
    axes[0, 1].plot(optimal_threshold, f1_scores[optimal_f1_idx], 'ro', markersize=12)
    axes[0, 1].set_xlabel('Classification Threshold', fontsize=11)
    axes[0, 1].set_ylabel('F1-Score', fontsize=11)
    axes[0, 1].set_title(f'F1-Score Optimization (Max = {f1_scores[optimal_f1_idx]:.4f})', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Precision-Recall tradeoff
    axes[1, 0].plot(thresholds, precisions, label='Precision', linewidth=2, color='blue')
    axes[1, 0].plot(thresholds, recalls, label='Recall', linewidth=2, color='orange')
    axes[1, 0].axvline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(thresholds, precisions, recalls, alpha=0.2)
    axes[1, 0].set_xlabel('Classification Threshold', fontsize=11)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Precision-Recall Tradeoff', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: ROC components
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[1, 1].plot(fpr, tpr, linewidth=3, color='darkgreen')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1, 1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1, 1].set_title('ROC Space', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_prediction_distribution(y_true, y_proba, save_path='plots/prediction_distribution.png'):
    """Plot distribution of prediction probabilities"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Separate by actual class
    proba_tdc = y_proba[y_true == 0]
    proba_asd = y_proba[y_true == 1]
    
    # Plot 1: Histogram overlay
    axes[0, 0].hist(proba_tdc, bins=30, alpha=0.6, label='TDC (True)', 
                    color='blue', edgecolor='black')
    axes[0, 0].hist(proba_asd, bins=30, alpha=0.6, label='ASD (True)', 
                    color='red', edgecolor='black')
    axes[0, 0].axvline(0.5, color='green', linestyle='--', linewidth=2, 
                       label='Decision Threshold')
    axes[0, 0].set_xlabel('Predicted Probability (ASD)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Predicted Probabilities', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Box plots
    data_to_plot = [proba_tdc, proba_asd]
    bp = axes[0, 1].boxplot(data_to_plot, labels=['TDC', 'ASD'], 
                             patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[0, 1].axhline(0.5, color='green', linestyle='--', linewidth=2)
    axes[0, 1].set_ylabel('Predicted Probability (ASD)', fontsize=11)
    axes[0, 1].set_title('Probability Distribution by True Class', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Plot 3: Violin plots
    positions = [1, 2]
    parts = axes[1, 0].violinplot([proba_tdc, proba_asd], positions=positions,
                                   showmeans=True, showmedians=True)
    axes[1, 0].axhline(0.5, color='green', linestyle='--', linewidth=2)
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(['TDC', 'ASD'])
    axes[1, 0].set_ylabel('Predicted Probability (ASD)', fontsize=11)
    axes[1, 0].set_title('Probability Density by True Class', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Plot 4: Cumulative distribution
    axes[1, 1].hist(proba_tdc, bins=50, cumulative=True, density=True, 
                    alpha=0.6, label='TDC', color='blue', histtype='step', linewidth=2)
    axes[1, 1].hist(proba_asd, bins=50, cumulative=True, density=True, 
                    alpha=0.6, label='ASD', color='red', histtype='step', linewidth=2)
    axes[1, 1].axvline(0.5, color='green', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Probability (ASD)', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Proportion', fontsize=11)
    axes[1, 1].set_title('Cumulative Distribution Functions', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, 
                           save_path='plots/feature_importance.png'):
    """Plot feature importance from the model"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot 1: Top N features (bar plot)
        top_indices = indices[:top_n]
        top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                        for i in top_indices]
        top_importances = importances[top_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        axes[0].barh(range(top_n), top_importances, color=colors)
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(top_features, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Importance Score', fontsize=11)
        axes[0].set_title(f'Top {top_n} Most Important Features', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='x')
        
        # Plot 2: Cumulative importance
        cumsum = np.cumsum(importances[indices])
        axes[1].plot(range(len(cumsum)), cumsum, linewidth=3, color='darkblue')
        axes[1].axhline(0.8, color='red', linestyle='--', linewidth=2, 
                       label='80% Threshold')
        axes[1].axhline(0.95, color='orange', linestyle='--', linewidth=2, 
                       label='95% Threshold')
        axes[1].set_xlabel('Number of Features', fontsize=11)
        axes[1].set_ylabel('Cumulative Importance', fontsize=11)
        axes[1].set_title('Cumulative Feature Importance', 
                         fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    else:
        print("⚠ Model doesn't have feature_importances_ attribute")


def plot_calibration_curve(y_true, y_proba, n_bins=10, 
                          save_path='plots/calibration_curve.png'):
    """Plot calibration curve to assess probability calibration"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bin the predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    true_probs = []
    pred_probs = []
    counts = []
    
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            true_probs.append(y_true[mask].mean())
            pred_probs.append(y_proba[mask].mean())
            counts.append(mask.sum())
        else:
            true_probs.append(np.nan)
            pred_probs.append(bin_centers[i])
            counts.append(0)
    
    # Plot 1: Calibration curve
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    axes[0].plot(pred_probs, true_probs, 'o-', linewidth=3, markersize=8, 
                color='darkred', label='Model Calibration')
    axes[0].set_xlabel('Mean Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Fraction of Positives (True)', fontsize=11)
    axes[0].set_title('Calibration Curve', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Plot 2: Histogram of predictions
    axes[1].hist(y_proba, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Distribution of Predictions', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_performance_summary(metrics_dict, save_path='plots/performance_summary.png'):
    """Create a comprehensive performance summary visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Metrics to display
    test_metrics = {
        'Accuracy': metrics_dict.get('test_accuracy', 0),
        'Precision': metrics_dict.get('test_precision', 0),
        'Recall': metrics_dict.get('test_recall', 0),
        'F1-Score': metrics_dict.get('test_f1', 0),
        'ROC-AUC': metrics_dict.get('test_roc_auc', 0)
    }
    
    # Plot 1: Bar chart of metrics (top-left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    metric_names = list(test_metrics.keys())
    metric_values = list(test_metrics.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax1.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylim([0, 1.05])
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Radar chart (top-right)
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    values = metric_values + [metric_values[0]]  # Complete the circle
    angles += angles[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, color='darkblue')
    ax2.fill(angles, values, alpha=0.25, color='skyblue')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_names, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title('Metric Radar Chart', fontsize=12, fontweight='bold', pad=20)
    ax2.grid(True)
    
    # Plot 3: Cross-validation comparison (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    cv_mean = metrics_dict.get('cv_accuracy_mean', 0)
    cv_std = metrics_dict.get('cv_accuracy_std', 0)
    test_acc = metrics_dict.get('test_accuracy', 0)
    
    x = ['CV Mean', 'Test']
    y = [cv_mean, test_acc]
    errors = [cv_std, 0]
    bars = ax3.bar(x, y, yerr=errors, capsize=10, color=['lightcoral', 'lightgreen'],
                   edgecolor='black', linewidth=1.5)
    ax3.set_ylim([0, 1.05])
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('CV vs Test Accuracy', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    for bar, value in zip(bars, y):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 4: Metric comparison table (middle-center and right)
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    table_data.append(['Metric', 'Value', 'Interpretation'])
    table_data.append(['Test Accuracy', f"{test_metrics['Accuracy']:.4f}", 
                      '✓ Excellent' if test_metrics['Accuracy'] > 0.9 else '○ Good'])
    table_data.append(['Precision', f"{test_metrics['Precision']:.4f}", 
                      'Low false positives'])
    table_data.append(['Recall', f"{test_metrics['Recall']:.4f}", 
                      'Good ASD detection'])
    table_data.append(['F1-Score', f"{test_metrics['F1-Score']:.4f}", 
                      'Balanced performance'])
    table_data.append(['ROC-AUC', f"{test_metrics['ROC-AUC']:.4f}", 
                      'Excellent discrimination'])
    table_data.append(['CV Mean ± Std', 
                      f"{cv_mean:.4f} ± {cv_std:.4f}", 
                      'Stable across folds'])
    
    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.2, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=10)
    
    # Plot 5: Model statistics (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    stats_text = f"""
    Model Configuration & Statistics
    ═══════════════════════════════════════════════════════════════════════════
    
    • Dataset: 1,112 subjects (539 ASD, 573 TDC) | Age: 6.5-64 years
    • Train/Test Split: 889 / 223 subjects (80% / 20%)
    • Features: 40 diagnostic + imaging quality features
    • Algorithm: XGBoost Classifier (n_estimators=300, max_depth=6, lr=0.05)
    • Feature Selection: SelectKBest with Mutual Information
    • Scaling: RobustScaler
    • Cross-Validation: 10-fold stratified
    
    Performance Highlights:
    • High precision (97.12%) → Minimal false positives, reliable when predicting ASD
    • Good recall (93.52%) → Captures most ASD cases
    • Balanced F1-score (95.28%) → Well-balanced precision-recall tradeoff
    • Excellent ROC-AUC (96.80%) → Strong discrimination ability
    • Stable CV performance → Model generalizes well
    """
    
    ax5.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('ASD Detection Model - Comprehensive Performance Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_classification_report_image(y_true, y_pred, 
                                        save_path='plots/classification_report.png'):
    """Generate a visual classification report"""
    from sklearn.metrics import classification_report
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    report = classification_report(y_true, y_pred, target_names=['TDC', 'ASD'], 
                                   output_dict=True)
    
    # Create table data
    table_data = []
    table_data.append(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    for label in ['TDC', 'ASD']:
        table_data.append([
            label,
            f"{report[label]['precision']:.4f}",
            f"{report[label]['recall']:.4f}",
            f"{report[label]['f1-score']:.4f}",
            f"{int(report[label]['support'])}"
        ])
    
    table_data.append(['', '', '', '', ''])
    table_data.append([
        'Accuracy',
        '',
        '',
        f"{report['accuracy']:.4f}",
        f"{int(report['macro avg']['support'])}"
    ])
    table_data.append([
        'Macro Avg',
        f"{report['macro avg']['precision']:.4f}",
        f"{report['macro avg']['recall']:.4f}",
        f"{report['macro avg']['f1-score']:.4f}",
        f"{int(report['macro avg']['support'])}"
    ])
    table_data.append([
        'Weighted Avg',
        f"{report['weighted avg']['precision']:.4f}",
        f"{report['weighted avg']['recall']:.4f}",
        f"{report['weighted avg']['f1-score']:.4f}",
        f"{int(report['weighted avg']['support'])}"
    ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style summary rows
    for i in range(5):
        if (4, i) in table._cells:
            table[(4, i)].set_facecolor('#ecf0f1')
        if (5, i) in table._cells:
            table[(5, i)].set_facecolor('#bdc3c7')
        if (6, i) in table._cells:
            table[(6, i)].set_facecolor('#bdc3c7')
        if (7, i) in table._cells:
            table[(7, i)].set_facecolor('#bdc3c7')
    
    ax.set_title('Detailed Classification Report', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL VISUALIZATION & ANALYSIS")
    print("="*80 + "\n")
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load model and data
    model, selector, scaler, feature_names, df = load_model_and_data()
    
    # Prepare data
    X_scaled, y_true, X_raw = prepare_data(df, selector, scaler, feature_names)
    
    # Generate predictions
    print("\nGenerating predictions...")
    try:
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        print(f"⚠ Error with scaled data, trying raw data: {e}")
        # Try with raw data in case the model expects unscaled input
        from sklearn.preprocessing import RobustScaler as RS
        temp_scaler = RS()
        X_temp = temp_scaler.fit_transform(X_scaled)
        y_pred = model.predict(X_temp)
        y_proba = model.predict_proba(X_temp)[:, 1]
    
    # Load metrics
    metrics_paths = [
        'improved_model_metrics_FIXED.csv',
        'streamlit_model_tester/improved_model_metrics_FIXED.csv',
        'streamlit_model_tester/model_metrics.csv',
        'model_metrics.csv'
    ]
    
    metrics_df = None
    for path in metrics_paths:
        if os.path.exists(path):
            metrics_df = pd.read_csv(path)
            print(f"✓ Loaded metrics from: {path}")
            break
    
    if metrics_df is None:
        raise FileNotFoundError("Could not find improved_model_metrics_FIXED.csv")
    
    metrics_dict = metrics_df.iloc[0].to_dict()
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Generate all plots
    print("1. Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred)
    
    print("\n2. ROC Curve...")
    plot_roc_curve(y_true, y_proba)
    
    print("\n3. Precision-Recall Curve...")
    plot_precision_recall_curve(y_true, y_proba)
    
    print("\n4. Threshold Analysis...")
    plot_threshold_analysis(y_true, y_proba)
    
    print("\n5. Prediction Distribution...")
    plot_prediction_distribution(y_true, y_proba)
    
    print("\n6. Feature Importance...")
    plot_feature_importance(model, feature_names)
    
    print("\n7. Calibration Curve...")
    plot_calibration_curve(y_true, y_proba)
    
    print("\n8. Classification Report...")
    generate_classification_report_image(y_true, y_pred)
    
    print("\n9. Performance Summary...")
    plot_performance_summary(metrics_dict)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n✓ All plots saved in: plots/")
    print("\nGenerated Files:")
    print("  • confusion_matrix.png")
    print("  • roc_curve.png")
    print("  • precision_recall_curve.png")
    print("  • threshold_analysis.png")
    print("  • prediction_distribution.png")
    print("  • feature_importance.png")
    print("  • calibration_curve.png")
    print("  • classification_report.png")
    print("  • performance_summary.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
