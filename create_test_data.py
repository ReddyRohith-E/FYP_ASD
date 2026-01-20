"""
Create test CSV files for Streamlit app testing
"""
import pandas as pd
import numpy as np
import os

# Change to streamlit_model_tester directory
os.chdir('streamlit_model_tester')

# Load the full dataset  
df = pd.read_csv('improved_model_features_FIXED.csv')

# Get true labels for validation
true_labels = df['label_asd'] if 'label_asd' in df.columns else None

# Remove non-feature columns
feature_cols = [col for col in df.columns if col not in ['Unnamed: 0.1', 'SUB_ID', 'label_asd', 'DX_GROUP']]

print(f"Total samples in dataset: {len(df)}")
print(f"Features available: {len(feature_cols)}")
print(f"ASD samples: {sum(df['label_asd'] == 1)}")
print(f"TDC samples: {sum(df['label_asd'] == 0)}")
print("\n" + "="*60)

# 1. Create 20 random samples
np.random.seed(42)
test_indices = np.random.choice(len(df), size=20, replace=False)
test_data = df.iloc[test_indices][feature_cols].copy()
test_data.to_csv('test_data_20_samples.csv', index=False)
print(f'✓ Created test_data_20_samples.csv with {len(test_data)} samples')

# 2. Create 5 ASD samples
if true_labels is not None:
    asd_indices = df[df['label_asd'] == 1].index
    asd_samples = np.random.choice(asd_indices, size=5, replace=False)
    test_asd = df.iloc[asd_samples][feature_cols].copy()
    test_asd.to_csv('test_asd_5_samples.csv', index=False)
    print(f'✓ Created test_asd_5_samples.csv with 5 ASD samples')

# 3. Create 5 TDC samples
if true_labels is not None:
    tdc_indices = df[df['label_asd'] == 0].index
    tdc_samples = np.random.choice(tdc_indices, size=5, replace=False)
    test_tdc = df.iloc[tdc_samples][feature_cols].copy()
    test_tdc.to_csv('test_tdc_5_samples.csv', index=False)
    print(f'✓ Created test_tdc_5_samples.csv with 5 TDC samples')

# 4. Create mixed 10 samples (5 ASD + 5 TDC)
if true_labels is not None:
    mixed_indices = np.concatenate([asd_samples[:5], tdc_samples[:5]])
    test_mixed = df.iloc[mixed_indices][feature_cols].copy()
    test_mixed.to_csv('test_mixed_10_samples.csv', index=False)
    print(f'✓ Created test_mixed_10_samples.csv with 10 mixed samples (5 ASD + 5 TDC)')

# 5. Create 50 samples for larger batch testing
test_indices_50 = np.random.choice(len(df), size=50, replace=False)
test_data_50 = df.iloc[test_indices_50][feature_cols].copy()
test_data_50.to_csv('test_data_50_samples.csv', index=False)
print(f'✓ Created test_data_50_samples.csv with 50 samples')

print("\n" + "="*60)
print(f'\n📊 All test files created in streamlit_model_tester/')
print(f'   Features per sample: {len(feature_cols)}')
print(f'\n📝 Files created:')
print(f'   1. test_data_20_samples.csv - 20 random samples')
print(f'   2. test_asd_5_samples.csv - 5 ASD samples')
print(f'   3. test_tdc_5_samples.csv - 5 TDC samples')
print(f'   4. test_mixed_10_samples.csv - 5 ASD + 5 TDC')
print(f'   5. test_data_50_samples.csv - 50 random samples')
print(f'\n🚀 Upload these files in Tab 2: "Test on Custom Data" in the Streamlit app')
