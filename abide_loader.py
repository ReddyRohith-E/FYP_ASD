
"""
ABIDE Dataset Loader for ASD Detection
======================================

This module provides a simple interface to load ABIDE data directly from S3
without downloading the entire dataset. It is designed for use in Jupyter notebooks.
"""

import sys
import os
import numpy as np
import pandas as pd
# Add current directory to path to ensure we can import sibling modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from abide_s3_utils import S3ABIDEClient, ABIDEDataFilter
except ImportError as e:
    print(f"Error importing helper modules: {e}")
    print("Ensure 'abide_s3_utils.py' is in the same directory.")
    raise

def load_abide_dataset(
    max_subjects=None,
    age_range=None,
    diagnosis_group=None,
    site=None,
    apply_mask=False
):
    """
    Generator that yields subject data and features.
    
    Parameters:
    -----------
    max_subjects : int, optional
        Maximum number of subjects to load.
    age_range : tuple, optional
        (min_age, max_age)
    diagnosis_group : str, optional
        'ASD' or 'TDC'. If None, returns both.
    site : str, optional
        Name of the site (e.g., 'NYU').
    apply_mask : bool
        If True, applies a brain mask to the data (requires nilearn).
        
    Yields:
    -------
    tuple
        (subject_id, phenotypic_dict, nifti_img)
    """
    
    client = S3ABIDEClient(use_anonymous=True)
    pheno_df = client.get_phenotypic_data()
    
    if pheno_df is None:
        raise RuntimeError("Could not load phenotypic data.")

    # Clean obvious bad/missing file IDs to avoid S3 404s
    if 'FILE_ID' in pheno_df.columns:
        pheno_df = pheno_df.dropna(subset=['FILE_ID']).copy()
        pheno_df['FILE_ID'] = pheno_df['FILE_ID'].astype(str).str.strip()
        pheno_df = pheno_df[pheno_df['FILE_ID'].str.lower() != 'no_filename']
        
    filter_tool = ABIDEDataFilter(pheno_df)
    
    # Map friendly diagnosis names to ABIDE codes
    dx_map = {'ASD': 1, 'TDC': 2}
    
    # Build filter arguments
    filter_args = {}
    if age_range:
        filter_args['age_range'] = age_range
    if diagnosis_group:
        dx_code = dx_map.get(diagnosis_group)
        if dx_code:
            filter_args['diagnosis'] = dx_code
    if site:
        filter_args['site'] = site
        
    # Apply filters
    filtered_df = filter_tool.apply_filters(**filter_args)
    
    subject_ids = filtered_df['FILE_ID'].tolist()
    
    if max_subjects:
        subject_ids = subject_ids[:max_subjects]
        
    print(f"Generator prepared for {len(subject_ids)} subjects.")
    
    for subject_id in subject_ids:
        try:
            pheno_data, nifti_img = client.get_subject_data(subject_id)
            if nifti_img is not None:
                yield subject_id, pheno_data, nifti_img
            else:
                print(f"Skipping {subject_id}: Image data missing.")
        except Exception as e:
            print(f"Error yielding subject {subject_id}: {e}")

def simple_test():
    print("Running simple test...")
    # Load just 1 subject to verify
    loader = load_abide_dataset(max_subjects=1, age_range=(5, 10), diagnosis_group='ASD')
    for sid, pheno, img in loader:
        print(f"Successfully loaded: {sid}")
        print(f"Age: {pheno['AGE_AT_SCAN']}")
        print(f"Image shape: {img.shape}")
        
if __name__ == "__main__":
    simple_test()
