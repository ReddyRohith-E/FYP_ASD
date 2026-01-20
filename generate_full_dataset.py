"""
Generate preprocessed features for ALL 1,112 subjects from ABIDE dataset
Uses the same preprocessing pipeline as the trained model
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_all_data():
    """Load all data and preprocess using the model's pipeline"""
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load phenotypic data
    df = pd.read_csv('Phenotypic_V1_0b_preprocessed1.csv')
    print(f"✓ Loaded {len(df)} subjects from raw ABIDE dataset")
    
    # Filter for valid diagnosis
    df = df[df['DX_GROUP'].isin([1, 2])].copy()
    print(f"✓ Valid ASD/TDC labels: {len(df)} subjects")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['SUB_ID'], keep='first')
    print(f"✓ After deduplication: {len(df)} subjects")
    
    # Create target
    df['label_asd'] = (df['DX_GROUP'] == 1).astype(int)
    
    print(f"\n  ASD cases: {df['label_asd'].sum()}")
    print(f"  TDC cases: {(df['label_asd'] == 0).sum()}")
    
    return df


def extract_features(df):
    """Extract the 85 raw features (model expects all raw features before SelectKBest)"""
    
    # These are the features the model's scaler was trained on
    feature_list = [
        'Unnamed: 0.1', 'Unnamed: 0', 'X', 'subject', 'DSM_IV_TR', 'AGE_AT_SCAN', 'SEX',
        'HANDEDNESS_SCORES', 'FIQ', 'VIQ', 'PIQ', 'ADI_R_SOCIAL_TOTAL_A',
        'ADI_R_VERBAL_TOTAL_BV', 'ADI_RRB_TOTAL_C', 'ADI_R_ONSET_TOTAL_D',
        'ADI_R_RSRCH_RELIABLE', 'ADOS_MODULE', 'ADOS_TOTAL', 'ADOS_COMM',
        'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV', 'ADOS_RSRCH_RELIABLE',
        'ADOS_GOTHAM_SOCAFFECT', 'ADOS_GOTHAM_RRB', 'ADOS_GOTHAM_TOTAL',
        'ADOS_GOTHAM_SEVERITY', 'SRS_VERSION', 'SRS_RAW_TOTAL', 'SRS_AWARENESS',
        'SRS_COGNITION', 'SRS_COMMUNICATION', 'SRS_MOTIVATION', 'SRS_MANNERISMS',
        'SCQ_TOTAL', 'AQ_TOTAL', 'OFF_STIMULANTS_AT_SCAN',
        'VINELAND_RECEPTIVE_V_SCALED', 'VINELAND_EXPRESSIVE_V_SCALED',
        'VINELAND_WRITTEN_V_SCALED', 'VINELAND_COMMUNICATION_STANDARD',
        'VINELAND_PERSONAL_V_SCALED', 'VINELAND_DOMESTIC_V_SCALED',
        'VINELAND_COMMUNITY_V_SCALED', 'VINELAND_DAILYLVNG_STANDARD',
        'VINELAND_INTERPERSONAL_V_SCALED', 'VINELAND_PLAY_V_SCALED',
        'VINELAND_COPING_V_SCALED', 'VINELAND_SOCIAL_STANDARD',
        'VINELAND_SUM_SCORES', 'VINELAND_ABC_STANDARD', 'VINELAND_INFORMANT',
        'WISC_IV_VCI', 'WISC_IV_PRI', 'WISC_IV_WMI', 'WISC_IV_PSI',
        'WISC_IV_SIM_SCALED', 'WISC_IV_VOCAB_SCALED', 'WISC_IV_INFO_SCALED',
        'WISC_IV_BLK_DSN_SCALED', 'WISC_IV_PIC_CON_SCALED', 'WISC_IV_MATRIX_SCALED',
        'WISC_IV_DIGIT_SPAN_SCALED', 'WISC_IV_LET_NUM_SCALED',
        'WISC_IV_CODING_SCALED', 'WISC_IV_SYM_SCALED', 'EYE_STATUS_AT_SCAN',
        'AGE_AT_MPRAGE', 'BMI', 'anat_cnr', 'anat_efc', 'anat_fber', 'anat_fwhm',
        'anat_qi1', 'anat_snr', 'func_efc', 'func_fber', 'func_fwhm', 'func_dvars',
        'func_outlier', 'func_quality', 'func_mean_fd', 'func_num_fd', 'func_perc_fd',
        'func_gsr', 'SUB_IN_SMP'
    ]
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION")
    print("="*80)
    
    # Keep only existing columns
    available_features = [f for f in feature_list if f in df.columns]
    print(f"✓ Using {len(available_features)} raw features (85 expected)")
    
    # Extract features
    X = df[available_features].copy()
    
    # Fill missing values with median
    X = X.fillna(X.median())
    
    print(f"✓ Filled missing values with median")
    print(f"✓ Shape: {X.shape}")
    
    return X, df[['label_asd', 'SUB_ID']], available_features


def apply_model_preprocessing(X_full):
    """Apply the model's preprocessing pipeline"""
    
    print("\n" + "="*80)
    print("APPLYING MODEL PREPROCESSING")
    print("="*80)
    
    # Load the trained model artifacts
    try:
        artifacts = joblib.load('streamlit_model_tester/asd_model.pkl')
        print("✓ Loaded model from streamlit_model_tester/asd_model.pkl")
    except:
        print("✗ Could not load model - file not found")
        return None
    
    scaler = artifacts['scaler']
    selector = artifacts['selector']
    
    # Apply scaling
    X_scaled = scaler.transform(X_full)
    print(f"✓ Applied RobustScaler")
    
    # Apply feature selection
    X_selected = selector.transform(X_scaled)
    print(f"✓ Applied SelectKBest (40 features)")
    print(f"✓ Final shape: {X_selected.shape}")
    
    return X_selected


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("GENERATING FULL DATASET FOR STREAMLIT APP (ALL 1,112 SAMPLES)")
    print("="*80)
    
    # Load and filter data
    df = load_and_preprocess_all_data()
    
    # Extract raw features
    X_raw, y_info, feature_names = extract_features(df)
    
    # Apply preprocessing
    X_processed = apply_model_preprocessing(X_raw)
    
    if X_processed is None:
        print("\n✗ Failed to generate full dataset")
        return
    
    # Create output dataframe
    output_data = {}
    
    # Add processed features
    for i in range(X_processed.shape[1]):
        output_data[f'feature_{i}'] = X_processed[:, i]
    
    # Add label and ID
    output_data['label_asd'] = y_info['label_asd'].values
    output_data['SUB_ID'] = y_info['SUB_ID'].values
    
    output_df = pd.DataFrame(output_data)
    
    # Save to CSV
    output_path = 'streamlit_model_tester/asd_model_features_all.csv'
    output_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"✓ Saved {len(output_df)} preprocessed samples")
    print(f"✓ Features: {X_processed.shape[1]}")
    print(f"✓ Output file: {output_path}")
    print(f"✓ File size: {output_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"\nASD: {output_df['label_asd'].sum()}")
    print(f"TDC: {(output_df['label_asd'] == 0).sum()}")
    
    return output_df


if __name__ == "__main__":
    main()
