"""
ABIDE S3 Utilities - Stream fMRI data directly from AWS S3 without local storage

This module provides tools to access ABIDE preprocessed fMRI data directly from S3
without downloading all 180GB+ of data to your local machine.
"""

import boto3
import nibabel as nib
import io
import numpy as np
import pandas as pd
from io import BytesIO
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.config import Config
import os

# S3 Configuration
ABIDE_BUCKET = 'fcp-indi'
ABIDE_PREFIX = 'data/Projects/ABIDE_Initiative/'
ABIDE_REGION = 'us-east-1'

# Cache for phenotypic data
_phenotypic_cache = None


class S3ABIDEClient:
    """Client for accessing ABIDE data from AWS S3

    By default uses anonymous (unsigned) access which works with the public
    FCP-INDI bucket. Set use_anonymous=False if you want to use credentials.
    """
    
    def __init__(self, use_anonymous: bool = True):
        if use_anonymous:
            cfg = Config(signature_version=UNSIGNED)
            self.s3_client = boto3.client('s3', region_name=ABIDE_REGION, config=cfg)
            self.s3_resource = boto3.resource('s3', region_name=ABIDE_REGION, config=cfg)
        else:
            self.s3_client = boto3.client('s3', region_name=ABIDE_REGION)
            self.s3_resource = boto3.resource('s3', region_name=ABIDE_REGION)
        
    def get_phenotypic_data(self):
        """Load ABIDE phenotypic data.

        Attempts S3 anonymous read first, then falls back to local CSV files
        in the workspace if S3 access is unavailable.
        """
        global _phenotypic_cache
        
        if _phenotypic_cache is not None:
            return _phenotypic_cache
            
        try:
            print("Loading phenotypic data from S3...")
            obj = self.s3_client.get_object(
                Bucket=ABIDE_BUCKET,
                Key=ABIDE_PREFIX + 'Phenotypic_V1_0b_preprocessed1.csv'
            )
            df = pd.read_csv(BytesIO(obj['Body'].read()))
            _phenotypic_cache = df
            print(f"Loaded {len(df)} subjects")
            return df
        except Exception as e:
            print(f"Error loading phenotypic data from S3 (will try local): {e}")
            # Fallback to local CSVs if available
            local_candidates = [
                'Phenotypic_V1_0b_preprocessed1.csv',
                'Phenotypic_V1_0b.csv',
                os.path.join('ABIDEII_MRI_Quality_Metrics', 'functional_qap.csv')  # alternate QC metrics
            ]
            for path in local_candidates:
                if os.path.exists(path):
                    try:
                        print(f"Loading phenotypic data from local file: {path}")
                        df = pd.read_csv(path)
                        _phenotypic_cache = df
                        print(f"Loaded {len(df)} subjects from local CSV")
                        return df
                    except Exception as le:
                        print(f"Failed to read local CSV {path}: {le}")
            print("No phenotypic CSV available from S3 or local.")
            return None
    
    def list_available_files(self, pipeline='cpac', strategy='nofilt_noglobal', 
                            derivative='func_preproc'):
        """List all available files in S3 for a given pipeline/strategy"""
        try:
            prefix = f'{ABIDE_PREFIX}Outputs/{pipeline}/{strategy}/{derivative}/'
            response = self.s3_client.list_objects_v2(
                Bucket=ABIDE_BUCKET,
                Prefix=prefix,
                MaxKeys=10000
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.nii.gz'):
                        filename = obj['Key'].split('/')[-1]
                        size_mb = obj['Size'] / (1024 * 1024)
                        files.append({
                            'filename': filename,
                            'size_mb': size_mb,
                            's3_path': obj['Key']
                        })
            
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def load_nifti_from_s3(self, subject_id, pipeline='cpac', strategy='nofilt_noglobal',
                          derivative='func_preproc'):
        """Load NIfTI file directly from S3 without saving to disk"""
        try:
            # Construct S3 path
            s3_path = f'{ABIDE_PREFIX}Outputs/{pipeline}/{strategy}/{derivative}/{subject_id}_{derivative}.nii.gz'
            
            print(f"Downloading {subject_id} from S3...")
            # Get object from S3
            response = self.s3_client.get_object(
                Bucket=ABIDE_BUCKET,
                Key=s3_path
            )
            
            # Load directly into memory
            nifti_data = nib.load(BytesIO(response['Body'].read()))
            print(f"Successfully loaded {subject_id}")
            return nifti_data
            
        except ClientError as e:
            print(f"Error downloading {subject_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error loading {subject_id} from S3: {e}")
            return None
    
    def get_subject_data(self, subject_id):
        """Get both phenotypic data and NIfTI image for a subject"""
        phenotypic = self.get_phenotypic_data()
        
        if phenotypic is None:
            return None, None
        
        # Find subject in phenotypic data
        subject_data = phenotypic[phenotypic['FILE_ID'].str.contains(subject_id, na=False)]
        
        if subject_data.empty:
            print(f"Subject {subject_id} not found in phenotypic data")
            return None, None
        
        # Load NIfTI
        nifti_img = self.load_nifti_from_s3(subject_id)
        
        return subject_data.iloc[0].to_dict(), nifti_img
    
    def get_subjects_by_diagnosis(self, diagnosis='1', pipeline='cpac'):
        """Get list of subjects by diagnosis group"""
        phenotypic = self.get_phenotypic_data()
        
        if phenotypic is None:
            return []
        
        # diagnosis: 1 = ASD, 2 = TDC
        subjects = phenotypic[phenotypic['DX_GROUP'] == int(diagnosis)]
        file_ids = subjects['FILE_ID'].tolist()
        
        return file_ids[:10]  # Return first 10 for testing
    
    def batch_load_subjects(self, subject_ids, max_subjects=5):
        """Load multiple subjects' data in batch"""
        results = []
        for i, subject_id in enumerate(subject_ids[:max_subjects]):
            print(f"[{i+1}/{min(max_subjects, len(subject_ids))}] Loading {subject_id}...")
            phenotypic_data, nifti_img = self.get_subject_data(subject_id)
            
            if nifti_img is not None:
                results.append({
                    'subject_id': subject_id,
                    'phenotypic': phenotypic_data,
                    'nifti': nifti_img
                })
        
        return results


class ABIDEDataFilter:
    """Filter ABIDE subjects by various criteria"""
    
    def __init__(self, phenotypic_df):
        self.df = phenotypic_df
    
    def by_age(self, min_age=None, max_age=None):
        """Filter subjects by age"""
        df = self.df.copy()
        if min_age:
            df = df[df['AGE_AT_SCAN'] >= min_age]
        if max_age:
            df = df[df['AGE_AT_SCAN'] <= max_age]
        return df
    
    def by_diagnosis(self, diagnosis=1):
        """Filter by diagnosis (1=ASD, 2=TDC)"""
        return self.df[self.df['DX_GROUP'] == diagnosis]
    
    def by_site(self, site_name):
        """Filter by scanning site"""
        return self.df[self.df['SITE_ID'] == site_name]
    
    def by_sex(self, sex='M'):
        """Filter by sex (M/F)"""
        return self.df[self.df['SEX'] == sex]
    
    def by_motion(self, max_mean_fd=0.5):
        """Filter by head motion (mean frame displacement)"""
        return self.df[self.df['func_mean_fd'] <= max_mean_fd]
    
    def apply_filters(self, **kwargs):
        """Apply multiple filters at once"""
        if self.df is None:
            print("Phenotypic DataFrame is not loaded; returning empty result.")
            return pd.DataFrame()

        df = self.df.copy()
        
        for key, value in kwargs.items():
            if key == 'age_range' and value:
                min_age, max_age = value
                df = df[(df['AGE_AT_SCAN'] >= min_age) & (df['AGE_AT_SCAN'] <= max_age)]
            elif key == 'diagnosis' and value:
                df = df[df['DX_GROUP'] == value]
            elif key == 'site' and value:
                df = df[df['SITE_ID'] == value]
            elif key == 'sex' and value:
                df = df[df['SEX'] == value]
            elif key == 'max_motion' and value:
                df = df[df['func_mean_fd'] <= value]
        
        return df


def quick_load_sample(num_subjects=5):
    """Quick function to load a few sample subjects for testing"""
    client = S3ABIDEClient(use_anonymous=True)
    
    # Get phenotypic data
    phenotypic = client.get_phenotypic_data()
    
    if phenotypic is None:
        return []
    
    # Get first N healthy controls for speed
    asd_subjects = phenotypic[phenotypic['DX_GROUP'] == 1]['FILE_ID'].head(num_subjects).tolist()
    
    print(f"\nLoading {num_subjects} sample ASD subjects...")
    results = client.batch_load_subjects(asd_subjects, max_subjects=num_subjects)
    
    return results


if __name__ == "__main__":
    # Example usage
    client = S3ABIDEClient(use_anonymous=True)
    
    # Load phenotypic data
    pheno = client.get_phenotypic_data()
    if pheno is not None:
        print(f"\nAvailable columns: {pheno.columns.tolist()[:10]}...")
    else:
        print("\nPhenotypic data unavailable.")
    
    # Get some file info
    files = client.list_available_files()
    print(f"\nTotal available files: {len(files)}")
    print(f"Sample files: {files[:3]}")
