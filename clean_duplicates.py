"""
Data Cleaning and Deduplication
Remove duplicate subjects and create clean datasets for trustworthy validation
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("CLEANING DUPLICATE SUBJECTS")
print("="*80)

# Load data
ultra_df = pd.read_csv('ultra_features.csv')
infant_df = pd.read_csv('infant_asd_features.csv')

print(f"\nOriginal Ultra Model:")
print(f"  Total rows: {len(ultra_df)}")
print(f"  Unique subjects: {ultra_df['subject_id'].nunique()}")

print(f"\nOriginal Infant Model:")
print(f"  Total rows: {len(infant_df)}")
print(f"  Unique subjects: {infant_df['subject_id'].nunique()}")

# Remove duplicates, keeping first occurrence
ultra_df_clean = ultra_df.drop_duplicates(subset=['subject_id'], keep='first')
infant_df_clean = infant_df.drop_duplicates(subset=['subject_id'], keep='first')

print(f"\n{'─'*80}")
print(f"After Deduplication:")
print(f"{'─'*80}")

print(f"\nCleaned Ultra Model:")
print(f"  Total rows: {len(ultra_df_clean)}")
print(f"  Unique subjects: {ultra_df_clean['subject_id'].nunique()}")
print(f"  Removed: {len(ultra_df) - len(ultra_df_clean)} duplicates")
print(f"  ASD/TDC: {(ultra_df_clean['label_asd']==1).sum()}/{(ultra_df_clean['label_asd']==0).sum()}")

print(f"\nCleaned Infant Model:")
print(f"  Total rows: {len(infant_df_clean)}")
print(f"  Unique subjects: {infant_df_clean['subject_id'].nunique()}")
print(f"  Removed: {len(infant_df) - len(infant_df_clean)} duplicates")
print(f"  ASD/TDC: {(infant_df_clean['label_asd']==1).sum()}/{(infant_df_clean['label_asd']==0).sum()}")

# Save cleaned data
ultra_df_clean.to_csv('ultra_features_CLEAN.csv', index=False)
infant_df_clean.to_csv('infant_asd_features_CLEAN.csv', index=False)

print(f"\n{'─'*80}")
print("✓ Saved cleaned datasets:")
print("  - ultra_features_CLEAN.csv")
print("  - infant_asd_features_CLEAN.csv")
print(f"{'─'*80}\n")

# Sample check
print("\nSample of cleaned data (Ultra Model):")
print(ultra_df_clean[['subject_id', 'label_asd', 'age', 'FIQ']].head(10))

print("\nSample of cleaned data (Infant Model):")
print(infant_df_clean[['subject_id', 'label_asd', 'age', 'FIQ']].head(10))
