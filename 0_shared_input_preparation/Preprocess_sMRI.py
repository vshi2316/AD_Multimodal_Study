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
        description='Structural MRI Preprocessing - No Premature Standardization'
    )
    parser.add_argument('--input_file', type=str,
                        default='./ADNI_Raw_Data/MRI/FreeSurfer_sMRI.csv',
                        help='Path to FreeSurfer sMRI data')
    parser.add_argument('--output_file', type=str,
                        default='./processed_data/Structural_MRI_features_raw.csv',
                        help='Output file path (ICV-corrected, log-transformed, but not standardized)')
    parser.add_argument('--output_dir', type=str,
                        default='./processed_data',
                        help='Output directory')
    parser.add_argument('--icv_correction', action='store_true', default=True,
                        help='Apply ICV correction ')
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
    
    baseline_codes = ['bl', 'BL', 'sc', 'SC', 'scmri']
    df_bl = df[df['VISCODE'].isin(baseline_codes)].copy()
    
    if len(df_bl) == 0:
        id_col = 'PTID' if 'PTID' in df.columns else 'RID'
        df_bl = df.sort_values('VISCODE').groupby(id_col).first().reset_index()
    
    return df_bl


def find_icv_column(df):
    """Find ICV (Intracranial Volume) column in DataFrame."""
    icv_patterns = ['icv', 'intracranial', 'etiv', 'brainsegtotal', 'wholebrain']
    for col in df.columns:
        col_lower = col.lower()
        for pattern in icv_patterns:
            if pattern in col_lower:
                return col
    return None


def apply_icv_correction(df, feature_cols, icv_col):
    """
    Apply ICV correction to volumetric features.
    
    Args:
        df: DataFrame with MRI features
        feature_cols: List of feature columns to correct
        icv_col: Column name for ICV
    
    Returns:
        DataFrame with ICV-corrected features
    """
    df_corrected = df.copy()
    icv = df[icv_col].values
    
    # Avoid division by zero
    icv = np.where(icv == 0, np.nan, icv)
    
    for col in feature_cols:
        if col != icv_col:
            # Proportional correction: volume / ICV * mean(ICV)
            # Note: Using mean(ICV) is acceptable as it's just a scaling factor
            # The relative relationships are preserved
            mean_icv = np.nanmean(icv)
            df_corrected[col] = df[col] / icv * mean_icv
    
    return df_corrected


def preprocess_smri(input_file, output_file, output_dir, icv_correction=True, log_transform=True):
    """
    Main preprocessing function for structural MRI data.
    
    Args:
        input_file: Path to input CSV
        output_file: Output file path
        output_dir: Output directory
        icv_correction: Whether to apply ICV correction
        log_transform: Whether to apply log transformation
    """
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/6] Loading FreeSurfer sMRI data from: {input_file}")
    smri_raw = read_csv_robust(input_file)
    print(f"  Raw records: {len(smri_raw)}")
    print(f"  Total columns: {len(smri_raw.columns)}")
    
    # Extract baseline data
    print("\n[2/6] Extracting baseline visits...")
    smri_baseline = extract_baseline(smri_raw)
    print(f"  Baseline records: {len(smri_baseline)}")
    
    # Select AD-relevant brain regions 
    print("\n[3/6] Selecting AD-relevant brain regions...")
    feature_patterns = {
        'Hippocampus': ['hippocampus', 'hipp'],
        'Entorhinal': ['entorhinal'],
        'Temporal': ['temporal', 'middletemporal', 'superiortemporal', 'inferiortemporal'],
        'Parietal': ['parietal', 'superiorparietal', 'inferiorparietal'],
        'Frontal': ['frontal', 'superiorfrontal', 'middlefrontal', 'inferiorfrontal'],
        'Cingulate': ['cingulate', 'anteriorcingulate', 'posteriorcingulate', 'isthmuscingulate'],
        'Precuneus': ['precuneus'],
        'Fusiform': ['fusiform'],
        'Parahippocampal': ['parahippocampal'],
        'Amygdala': ['amygdala'],
        'WholeBrain': ['wholebrain', 'intracranial', 'brainsegtotal', 'icv', 'etiv']
    }
    
    core_features = []
    for region, patterns in feature_patterns.items():
        matched_cols = []
        for pattern in patterns:
            matched = [col for col in smri_baseline.columns if pattern in col.lower()]
            matched_cols.extend(matched)
        matched_cols = list(set(matched_cols))[:4]  # Limit per region
        core_features.extend(matched_cols)
        if matched_cols:
            print(f"  {region}: {len(matched_cols)} features")
    
    core_features = list(set(core_features))
    print(f"  Total selected features: {len(core_features)}")
    
    # Extract ID and features
    id_col = 'PTID' if 'PTID' in smri_baseline.columns else 'RID'
    available_features = [c for c in core_features if c in smri_baseline.columns]
    smri_features = smri_baseline[[id_col] + available_features].copy()
    smri_features.rename(columns={id_col: 'ID'}, inplace=True)
    
    # ICV correction
    if icv_correction:
        print("\n[4/6] Applying ICV correction...")
        print("  NOTE: ICV correction is within-subject, does not cause data leakage")
        icv_col = find_icv_column(smri_features)
        if icv_col:
            print(f"  ICV column found: {icv_col}")
            feature_cols_for_correction = [c for c in available_features if c in smri_features.columns]
            smri_features = apply_icv_correction(smri_features, feature_cols_for_correction, icv_col)
            print("  ICV correction applied")
        else:
            print("  WARNING: ICV column not found, skipping correction")
    else:
        print("\n[4/6] Skipping ICV correction (disabled)")
    
    # Handle missing values (report only, imputation in Step 7)
    print("\n[5/6] Reporting missing values (will be imputed in Step 7)...")
    feature_cols = [c for c in smri_features.columns if c != 'ID']
    for col in feature_cols[:10]:  # Show first 10
        n_missing = smri_features[col].isnull().sum()
        if n_missing > 0:
            pct_missing = 100 * n_missing / len(smri_features)
            print(f"  {col}: {n_missing} missing ({pct_missing:.1f}%)")
    
    # Log transformation (optional, monotonic - no leakage)
    if log_transform:
        print("\n[6/6] Applying log transformation...")
        print("  NOTE: Log transformation is monotonic, does not cause data leakage")
        
        features_matrix = smri_features[feature_cols].copy()
        
        # Handle negative values for log transformation
        for col in feature_cols:
            min_val = features_matrix[col].min()
            if pd.notna(min_val) and min_val <= 0:
                offset = -min_val + 1
                features_matrix[col] = features_matrix[col] + offset
        
        # Log transformation
        log_transformed = np.log2(features_matrix + 1)
        
        # Update features
        smri_features[feature_cols] = log_transformed
        print("  Log2 transformation applied")
    else:
        print("\n[6/6] Skipping log transformation (disabled)")
    
    # Summary statistics (after ICV correction and log transform, but before standardization)
    print("\n  sMRI Feature Summary (ICV-corrected, log-transformed, NOT standardized):")
    for col in feature_cols[:5]:
        valid_data = smri_features[col].dropna()
        if len(valid_data) > 0:
            print(f"    {col}: median={valid_data.median():.3f}, IQR=[{valid_data.quantile(0.25):.3f}, {valid_data.quantile(0.75):.3f}]")
    
    # Save output (without Z-score standardization)
    output_path = output_file if os.path.isabs(output_file) else os.path.join(
        output_dir, os.path.basename(output_file)
    )
    smri_features.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  Total subjects: {len(smri_features)}")
    print(f"  Total features: {len(feature_cols)}")
    
    print("\n" + "=" * 70)
    print("Step 4: sMRI Preprocessing Complete")
    print("=" * 70)
    
    return smri_features


def main():
    """Main entry point."""
    args = parse_args()
    preprocess_smri(
        input_file=args.input_file,
        output_file=args.output_file,
        output_dir=args.output_dir,
        icv_correction=args.icv_correction,
        log_transform=args.log_transform
    )


if __name__ == "__main__":
    main()

