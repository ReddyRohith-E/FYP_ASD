"""
Example: Loading ABIDE Data from S3 Without Local Download

This script demonstrates how to access ABIDE fMRI data directly from S3
without downloading the entire 180GB+ dataset locally.
"""

from abide_s3_utils import S3ABIDEClient, ABIDEDataFilter, quick_load_sample
import nibabel as nib
import numpy as np

# ============================================================================
# Method 1: Quick Load - Get a few sample subjects
# ============================================================================

print("=" * 70)
print("METHOD 1: Quick Load Sample Subjects")
print("=" * 70)

# Load 5 sample ASD subjects without local storage
samples = quick_load_sample(num_subjects=5)

for result in samples:
    subject_id = result['subject_id']
    nifti = result['nifti']
    pheno = result['phenotypic']
    
    print(f"\nSubject: {subject_id}")
    print(f"  Age: {pheno.get('AGE_AT_SCAN', 'N/A')}")
    print(f"  Diagnosis: {'ASD' if pheno.get('DX_GROUP') == 1 else 'TDC'}")
    print(f"  Image shape: {nifti.shape}")
    print(f"  Mean FD: {pheno.get('func_mean_fd', 'N/A')}")


# ============================================================================
# Method 2: Filter and Load Specific Subjects
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 2: Filter by Criteria")
print("=" * 70)

client = S3ABIDEClient(use_anonymous=True)
pheno_df = client.get_phenotypic_data()

if pheno_df is None:
    print("Phenotypic data could not be loaded from S3; attempting local fallback handled in utils.")
    filtered = np.array([])
else:
    filter_obj = ABIDEDataFilter(pheno_df)

# Filter: Children (age 5-10) with ASD from NYU site
    filtered = filter_obj.apply_filters(
        age_range=(5, 10),
        diagnosis=1,  # ASD
        site='NYU',
        max_motion=0.5
    )

    print(f"Found {len(filtered)} subjects matching criteria")
    print(f"Columns available: {list(filtered.columns[:10])}")

    # Load first 3 matching subjects
    matching_subjects = filtered['FILE_ID'].head(3).tolist()
    results = client.batch_load_subjects(matching_subjects, max_subjects=3)

    for result in results:
        print(f"\n{result['subject_id']}: {result['nifti'].shape}")


# ============================================================================
# Method 3: Access Specific Subject
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 3: Load Specific Subject by ID")
print("=" * 70)

# Load a specific subject directly
subject_id = "Pitt_0050004"  # Example subject ID

phenotypic_data, nifti_img = client.get_subject_data(subject_id)

if nifti_img is not None:
    print(f"\nSubject: {subject_id}")
    print(f"Image shape: {nifti_img.shape}")
    print(f"Data type: {nifti_img.get_data_dtype()}")
    print(f"Affine matrix:\n{nifti_img.affine}")
    print(f"Age: {phenotypic_data.get('AGE_AT_SCAN')}")
    print(f"Sex: {phenotypic_data.get('SEX')}")


# ============================================================================
# Method 4: Process Data Without Saving
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 4: Process Directly in Memory")
print("=" * 70)

# Load data and process without saving to disk
if pheno_df is not None:
    subject_ids = pheno_df[pheno_df['DX_GROUP'] == 1]['FILE_ID'].head(3).tolist()

    for subject_id in subject_ids:
        _, nifti_img = client.get_subject_data(subject_id)
        
        if nifti_img is not None:
            # Get data array
            data = nifti_img.get_fdata()
            
            # Example calculations
            mean_intensity = np.mean(data)
            max_intensity = np.max(data)
            num_voxels = data.shape[0] * data.shape[1] * data.shape[2]
            
            print(f"\n{subject_id}:")
            print(f"  Shape: {data.shape}")
            print(f"  Mean intensity: {mean_intensity:.2f}")
            print(f"  Max intensity: {max_intensity:.2f}")
            print(f"  Total voxels: {num_voxels}")


# ============================================================================
# Method 5: Get Statistics Without Loading Full Data
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 5: Phenotypic Statistics")
print("=" * 70)

# Work with phenotypic data only (no NIfTI downloads)
print(f"Total subjects: {len(pheno_df)}")
print(f"ASD subjects: {len(pheno_df[pheno_df['DX_GROUP'] == 1])}")
print(f"TDC subjects: {len(pheno_df[pheno_df['DX_GROUP'] == 2])}")

age_stats = pheno_df['AGE_AT_SCAN'].describe()
print(f"\nAge statistics:")
print(age_stats)

# By site
print(f"\nSubjects by site:")
print(pheno_df['SITE_ID'].value_counts())


print("\n" + "=" * 70)
print("NOTE: This approach streams data on-demand from S3.")
print("No 180GB download needed - only pay for what you use!")
print("=" * 70)
