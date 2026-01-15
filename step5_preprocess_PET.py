"""

import argparse
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PET Imaging Preprocessing (Methods 2.2) - No Premature Standardization'
    )
    parser.add_argument('--amyloid_file', type=str,
                        default='./ADNI_Raw_Data/PET/Amyloid_PET.csv',
                        help='Path to Amyloid PET data')
    parser.add_argument('--tau_file', type=str,
                        default='./ADNI_Raw_Data/PET/Tau_PET.csv',
                        help='Path to Tau PET data')
    parser.add_argument('--output_file', type=str,
                        default='./processed_data/PET_imaging_features_raw.csv',
                        help='Output file path (log-transformed, but not standardized)')
    parser.add_argument('--output_dir', type=str,
                        default='./processed_data',
                        help='Output directory')
    parser.add_argument('--log_transform', action='store_true', default=True,
                        help='Apply log transformation for normality')
    return parser.parse_args()


def read_csv_robust(filepath):
    """Read CSV with multiple encoding attempts."""
    encodings = ['utf-8-sig', 'gbk', 'latin-1', 'utf-8']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except Exception:
            continue
    raise ValueError(f"Could not read file: {filepath}")


def extract_baseline(df):
    """Extract baseline visit records."""
    if 'VISCODE' not in df.columns:
        return df
    
    baseline_codes = ['bl', 'BL', 'sc', 'SC', 'v01', 'V01']
    df_bl = df[df['VISCODE'].isin(baseline_codes)].copy()
    
    if len(df_bl) == 0:
        id_col = 'PTID' if 'PTID' in df.columns else 'RID'
        df_bl = df.sort_values('VISCODE').groupby(id_col).first().reset_index()
    
    return df_bl


def extract_pet_features(df, roi_keywords, pet_type='Amyloid'):
    """Extract PET SUVR features for specified ROIs."""
    features = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Look for SUVR columns matching ROI keywords
        if 'suvr' in col_lower and any(roi in col_lower for roi in roi_keywords):
            features.append(col)
    
    # If not enough features found, get all SUVR columns
    if len(features) < 5:
        suvr_cols = [col for col in df.columns if 'suvr' in col.lower()]
        features = suvr_cols[:15]
    
    return features


def preprocess_pet(amyloid_file, tau_file, output_file, output_dir, log_transform=True):
    """
    Main preprocessing function for PET imaging data.
    
    Methods 2.2 Implementation:
    - Load Amyloid and Tau PET data
    - Extract baseline visits
    - Select regional SUVR features
    - Apply log transformation (monotonic, no leakage)
    - OUTPUT WITHOUT Z-SCORE STANDARDIZATION (deferred to Step 7)
    
    Args:
        amyloid_file: Path to Amyloid PET CSV
        tau_file: Path to Tau PET CSV
        output_file: Output file path
        output_dir: Output directory
        log_transform: Whether to apply log transformation
    """
    print("=" * 70)
    print("Step 5: PET Imaging Preprocessing (Methods 2.2)")
    print("CORRECTED: No premature standardization (deferred to Step 7)")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/6] Loading PET imaging data...")
    amyloid_pet = read_csv_robust(amyloid_file)
    tau_pet = read_csv_robust(tau_file)
    print(f"  Amyloid PET: {len(amyloid_pet)} records")
    print(f"  Tau PET: {len(tau_pet)} records")
    
    # Extract baseline data
    print("\n[2/6] Extracting baseline visits...")
    amyloid_baseline = extract_baseline(amyloid_pet)
    tau_baseline = extract_baseline(tau_pet)
    print(f"  Amyloid baseline: {len(amyloid_baseline)} records")
    print(f"  Tau baseline: {len(tau_baseline)} records")
    
    # Extract PET SUVR features (Methods 2.2)
    print("\n[3/6] Extracting PET SUVR features...")
    
    # Amyloid PET ROIs
    amyloid_roi_keywords = ['frontal', 'temporal', 'parietal', 'cingulate', 
                           'precuneus', 'global', 'composite', 'cortical']
    amyloid_features = extract_pet_features(amyloid_baseline, amyloid_roi_keywords, 'Amyloid')
    print(f"  Amyloid features: {len(amyloid_features)}")
    
    # Tau PET ROIs (Methods 2.2: entorhinal, hippocampus, temporal)
    tau_roi_keywords = ['entorhinal', 'hippocampus', 'temporal', 'fusiform', 
                       'parahippocampal', 'global', 'composite', 'braak']
    tau_features = extract_pet_features(tau_baseline, tau_roi_keywords, 'Tau')
    print(f"  Tau features: {len(tau_features)}")
    
    # Extract ID and features
    print("\n[4/6] Merging Amyloid and Tau PET data...")
    
    amyloid_id_col = 'PTID' if 'PTID' in amyloid_baseline.columns else 'RID'
    amyloid_data = amyloid_baseline[[amyloid_id_col] + amyloid_features].copy()
    amyloid_data.rename(columns={amyloid_id_col: 'ID'}, inplace=True)
    # Add prefix to distinguish features
    amyloid_data.columns = ['ID'] + [f'Amyloid_{col}' for col in amyloid_features]
    
    tau_id_col = 'PTID' if 'PTID' in tau_baseline.columns else 'RID'
    tau_data = tau_baseline[[tau_id_col] + tau_features].copy()
    tau_data.rename(columns={tau_id_col: 'ID'}, inplace=True)
    tau_data.columns = ['ID'] + [f'Tau_{col}' for col in tau_features]
    
    # Merge Amyloid and Tau PET data
    pet_merged = pd.merge(amyloid_data, tau_data, on='ID', how='inner')
    print(f"  Merged records: {len(pet_merged)}")
    
    all_features = [col for col in pet_merged.columns if col != 'ID']
    print(f"  Total PET features: {len(all_features)}")
    
    # Report missing values (will be handled in Step 7)
    print("\n[5/6] Reporting missing values (will be imputed in Step 7)...")
    for col in all_features[:10]:  # Show first 10
        n_missing = pet_merged[col].isnull().sum()
        if n_missing > 0:
            pct_missing = 100 * n_missing / len(pet_merged)
            print(f"  {col}: {n_missing} missing ({pct_missing:.1f}%)")
    
    # Log transformation (optional, monotonic - no leakage)
    if log_transform:
        print("\n[6/6] Applying log transformation...")
        print("  NOTE: Log transformation is monotonic, does not cause data leakage")
        
        features_matrix = pet_merged[all_features].copy()
        
        # Handle negative values for log transformation
        for col in all_features:
            min_val = features_matrix[col].min()
            if pd.notna(min_val) and min_val <= 0:
                offset = -min_val + 0.01
                features_matrix[col] = features_matrix[col] + offset
        
        # Log transformation
        log_transformed = np.log2(features_matrix + 1)
        
        # Update features
        pet_merged[all_features] = log_transformed
        print("  Log2 transformation applied")
    else:
        print("\n[6/6] Skipping log transformation (disabled)")
    
    # Summary statistics (after log transform, but before standardization)
    print("\n  PET Feature Summary (log-transformed, NOT standardized):")
    for col in all_features[:5]:
        valid_data = pet_merged[col].dropna()
        if len(valid_data) > 0:
            print(f"    {col}: median={valid_data.median():.3f}, IQR=[{valid_data.quantile(0.25):.3f}, {valid_data.quantile(0.75):.3f}]")
    
    # Save output (without Z-score standardization)
    output_path = output_file if os.path.isabs(output_file) else os.path.join(
        output_dir, os.path.basename(output_file)
    )
    pet_merged.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  Total subjects: {len(pet_merged)}")
    print(f"  Total features: {len(all_features)}")
    
    print("\n" + "-" * 70)
    print("NOTE: Output contains log-transformed values (NOT standardized)")
    print("Z-score standardization will be applied in Step 7 after train/test split")
    print("This ensures Methods 2.3 compliance: 'parameters derived exclusively from training set'")
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Step 5: PET Preprocessing Complete")
    print("=" * 70)
    
    return pet_merged


def main():
    """Main entry point."""
    args = parse_args()
    preprocess_pet(
        amyloid_file=args.amyloid_file,
        tau_file=args.tau_file,
        output_file=args.output_file,
        output_dir=args.output_dir,
        log_transform=args.log_transform
    )


if __name__ == "__main__":
    main()
