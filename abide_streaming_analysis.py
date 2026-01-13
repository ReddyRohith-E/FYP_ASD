"""
Production Ready: ABIDE Analysis Pipeline with S3 Streaming

This is a complete, production-ready example that demonstrates:
1. Efficient phenotypic filtering
2. Smart batch loading from S3
3. Feature extraction without local storage
4. Saving results to CSV (lightweight)
"""

import sys
sys.path.insert(0, '.')

from abide_s3_utils import S3ABIDEClient, ABIDEDataFilter
import numpy as np
import pandas as pd
from datetime import datetime

class ABIDEAnalyzer:
    """Streaming analysis of ABIDE data from S3"""
    
    def __init__(self):
        self.client = S3ABIDEClient()
        self.pheno_df = self.client.get_phenotypic_data()
        self.filter_obj = ABIDEDataFilter(self.pheno_df)
        self.results = []
    
    def extract_features(self, nifti_img):
        """Extract basic features from NIfTI image"""
        data = nifti_img.get_fdata()
        
        return {
            'shape': data.shape,
            'mean_intensity': float(np.mean(data)),
            'std_intensity': float(np.std(data)),
            'min_intensity': float(np.min(data)),
            'max_intensity': float(np.max(data)),
            'num_voxels': int(data.size),
            'voxel_volume_mm3': float(np.prod(nifti_img.header.get_zooms()))
        }
    
    def analyze_subset(self, **filter_args):
        """
        Analyze a subset of ABIDE data based on filter criteria
        
        Parameters:
        -----------
        age_range : tuple (min, max)
            Age range in years
        diagnosis : int
            1 = ASD, 2 = TDC
        site : str
            Scanning site name
        sex : str
            'M' or 'F'
        max_motion : float
            Maximum mean frame displacement
        max_subjects : int
            Max subjects to analyze (default 50)
        
        Example:
        --------
        analyzer = ABIDEAnalyzer()
        analyzer.analyze_subset(
            age_range=(6, 12),
            diagnosis=1,  # ASD
            site='NYU',
            sex='M',
            max_motion=0.5,
            max_subjects=20
        )
        """
        
        # Extract max_subjects before filtering
        max_subjects = filter_args.pop('max_subjects', 50)
        
        # Apply filters to get matching subjects
        filtered_df = self.filter_obj.apply_filters(**filter_args)
        subject_ids = filtered_df['FILE_ID'].head(max_subjects).tolist()
        
        print(f"\n{'='*70}")
        print(f"Analyzing {len(subject_ids)} subjects")
        print(f"Filter criteria: {filter_args}")
        print(f"{'='*70}\n")
        
        # Process subjects one by one (memory efficient)
        for i, subject_id in enumerate(subject_ids, 1):
            try:
                print(f"[{i}/{len(subject_ids)}] Processing {subject_id}...", end=' ')
                
                # Stream from S3
                pheno_data, nifti_img = self.client.get_subject_data(subject_id)
                
                if nifti_img is None:
                    print("FAILED (download)")
                    continue
                
                # Extract features
                features = self.extract_features(nifti_img)
                
                # Combine with phenotypic data
                result = {
                    'subject_id': subject_id,
                    'age': pheno_data.get('AGE_AT_SCAN'),
                    'sex': pheno_data.get('SEX'),
                    'diagnosis': 'ASD' if pheno_data.get('DX_GROUP') == 1 else 'TDC',
                    'site': pheno_data.get('SITE_ID'),
                    'mean_fd': pheno_data.get('func_mean_fd'),
                    **features
                }
                
                self.results.append(result)
                print("OK")
                
            except Exception as e:
                print(f"ERROR ({str(e)[:30]})")
                continue
        
        print(f"\n{'='*70}")
        print(f"Successfully analyzed {len(self.results)} subjects")
        print(f"{'='*70}\n")
    
    def save_results(self, filename=None):
        """Save results to CSV"""
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"abide_analysis_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} results to {filename}")
        return df
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}\n")
        
        print(f"Total subjects: {len(df)}")
        print(f"\nBy Diagnosis:")
        print(df['diagnosis'].value_counts())
        print(f"\nBy Sex:")
        print(df['sex'].value_counts())
        print(f"\nBy Site:")
        print(df['site'].value_counts())
        
        print(f"\nAge statistics:")
        print(df['age'].describe())
        
        print(f"\nMean motion (func_mean_fd):")
        print(df['mean_fd'].describe())
        
        print(f"\nIntensity statistics:")
        print(df['mean_intensity'].describe())
        
        print(f"\n{'='*70}\n")


# ============================================================================
# EXAMPLE 1: Analyze Children with ASD
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Children with ASD (ages 6-12)")
print("="*70)

analyzer1 = ABIDEAnalyzer()
analyzer1.analyze_subset(
    age_range=(6, 12),
    diagnosis=1,  # ASD
    max_motion=0.5,
    max_subjects=10  # Keep small for testing
)
analyzer1.print_summary()
results1 = analyzer1.save_results('asd_children.csv')


# ============================================================================
# EXAMPLE 2: Analyze Typically Developing Controls
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Typically Developing Controls (ages 18-25)")
print("="*70)

analyzer2 = ABIDEAnalyzer()
analyzer2.analyze_subset(
    age_range=(18, 25),
    diagnosis=2,  # TDC
    max_motion=0.5,
    max_subjects=10
)
analyzer2.print_summary()
results2 = analyzer2.save_results('tdc_adults.csv')


# ============================================================================
# EXAMPLE 3: Site-specific Analysis
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: NYU Site Analysis")
print("="*70)

analyzer3 = ABIDEAnalyzer()
analyzer3.analyze_subset(
    site='NYU',
    max_subjects=10
)
analyzer3.print_summary()
results3 = analyzer3.save_results('nyu_site.csv')


# ============================================================================
# EXAMPLE 4: Multi-site Comparison (ASD only)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Multi-site ASD Comparison")
print("="*70)

all_results = []
for site in ['NYU', 'UCLA_1', 'Stanford', 'Caltech']:
    print(f"\nProcessing {site}...")
    analyzer = ABIDEAnalyzer()
    analyzer.analyze_subset(
        site=site,
        diagnosis=1,
        max_subjects=5  # Small for speed
    )
    all_results.extend(analyzer.results)

# Save combined results
combined_df = pd.DataFrame(all_results)
combined_df.to_csv('multisite_asd_comparison.csv', index=False)

print(f"\nCombined {len(combined_df)} subjects from multiple sites")
print("\nSubjects per site:")
print(combined_df['site'].value_counts())


print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nKey points:")
print("✓ All data streamed from S3 - no 180GB download")
print("✓ Results saved as lightweight CSV files")
print("✓ Can analyze 100+ subjects on a laptop")
print("✓ Run on AWS EC2 for production scale (1000+ subjects)")
print("="*70 + "\n")
