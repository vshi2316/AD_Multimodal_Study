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
        description='CSF Biomarker Preprocessing'
    )
    parser.add_argument('--roche_file', type=str,
                        default='./ADNI_Raw_Data/CSF/CSF_Roche_Elecsys.csv',
                        help='Path to Roche Elecsys CSF data')
    parser.add_argument('--alzbio3_file', type=str,
                        default='./ADNI_Raw_Data/CSF/CSF_Alzbio3.csv',
                        help='Path to AlzBio3 CSF data')
    parser.add_argument('--output_file', type=str,
                        default='./processed_data/CSF_biomarkers_raw.csv',
                        help='Output file path (raw values, no standardization)')
    parser.add_argument('--output_dir', type=str,
                        default='./processed_data',
                        help='Output directory')
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


def harmonize_platforms(roche_data, alzbio3_data):
    """
    Harmonize CSF measurements across Roche Elecsys and AlzBio3 platforms.
    
    Args:
        roche_data: DataFrame with Roche Elecsys measurements
        alzbio3_data: DataFrame with AlzBio3 measurements
    
    Returns:
        DataFrame with harmonized CSF biomarkers
    """
    # Extract core biomarkers from Roche platform
    roche_cols = ['PTID', 'RID', 'ABETA40', 'ABETA42', 'TAU', 'PTAU', 'BATCH']
    roche_available = [c for c in roche_cols if c in roche_data.columns]
    roche_core = roche_data[roche_available].copy()
    
    # Calculate Aβ42/Aβ40 ratio (primary amyloid marker per
    if 'ABETA42' in roche_core.columns and 'ABETA40' in roche_core.columns:
        roche_core['ABETA42_ABETA40_RATIO'] = roche_core['ABETA42'] / roche_core['ABETA40']
    
    # Rename columns for consistency
    rename_map = {'TAU': 'TAU_TOTAL', 'PTAU': 'PTAU181'}
    roche_core.rename(columns=rename_map, inplace=True)
    
    # Extract core biomarkers from AlzBio3 platform
    alzbio3_cols = ['RID', 'ABETA', 'TAU', 'PTAU', 'BATCH']
    alzbio3_available = [c for c in alzbio3_cols if c in alzbio3_data.columns]
    alzbio3_core = alzbio3_data[alzbio3_available].copy()
    
    alzbio3_rename = {'ABETA': 'ABETA42', 'TAU': 'TAU_TOTAL', 'PTAU': 'PTAU181'}
    alzbio3_core.rename(columns=alzbio3_rename, inplace=True)
    
    # Merge platforms using outer join
    csf_merged = pd.merge(roche_core, alzbio3_core, on='RID', how='outer', 
                          suffixes=('_ROCHE', '_ALZBIO3'))
    
    # Harmonize measurements: prefer Roche, fill with AlzBio3
    for biomarker in ['TAU_TOTAL', 'PTAU181']:
        roche_col = f'{biomarker}_ROCHE'
        alzbio_col = f'{biomarker}_ALZBIO3'
        
        if roche_col in csf_merged.columns and alzbio_col in csf_merged.columns:
            csf_merged[biomarker] = csf_merged[roche_col].fillna(csf_merged[alzbio_col])
        elif roche_col in csf_merged.columns:
            csf_merged[biomarker] = csf_merged[roche_col]
        elif alzbio_col in csf_merged.columns:
            csf_merged[biomarker] = csf_merged[alzbio_col]
    
    return csf_merged


def preprocess_csf(roche_file, alzbio3_file, output_file, output_dir):
    """
    Main preprocessing function for CSF biomarker data.
    Args:
        roche_file: Path to Roche Elecsys CSV
        alzbio3_file: Path to AlzBio3 CSV
        output_file: Output file path
        output_dir: Output directory
    """
    print("=" * 70)
    print("Step 2: CSF Biomarker Preprocessing 
    print("CORRECTED: No premature standardization (deferred to Step 7)")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/4] Loading CSF data...")
    roche = read_csv_robust(roche_file)
    alzbio3 = read_csv_robust(alzbio3_file)
    print(f"  Roche Elecsys: {len(roche)} records")
    print(f"  AlzBio3: {len(alzbio3)} records")
    
    # Harmonize platforms
    print("\n[2/4] Harmonizing platforms 
    csf_merged = harmonize_platforms(roche, alzbio3)
    print(f"  Merged records: {len(csf_merged)}")
    
    # Select final features ( Aβ42/Aβ40 ratio, p-tau181, total tau)
    print("\n[3/4] Selecting CSF biomarkers...")
    feature_cols = ['PTAU181', 'ABETA42_ABETA40_RATIO', 'TAU_TOTAL']
    
    # Ensure ID column exists
    id_col = 'PTID' if 'PTID' in csf_merged.columns else 'RID'
    csf_final = csf_merged[[id_col] + [c for c in feature_cols if c in csf_merged.columns]].copy()
    csf_final.rename(columns={id_col: 'ID'}, inplace=True)
    
    # Handle infinite values only (keep NaN for later MICE imputation)
    print("\n[4/4] Handling infinite values (keeping NaN for MICE in Step 7)...")
    csf_final = csf_final.replace([np.inf, -np.inf], np.nan)
    
    available_features = [c for c in feature_cols if c in csf_final.columns]
    
    # Report missing values (will be handled by MICE in Step 7)
    for feat in available_features:
        n_missing = csf_final[feat].isnull().sum()
        pct_missing = 100 * n_missing / len(csf_final)
        print(f"  {feat}: {n_missing} missing ({pct_missing:.1f}%) - will be imputed in Step 7")
    
    # Summary statistics (RAW values)
    print("\n  CSF Biomarker Summary (RAW values, before standardization):")
    for feat in available_features:
        valid_data = csf_final[feat].dropna()
        print(f"    {feat}: median={valid_data.median():.2f}, IQR=[{valid_data.quantile(0.25):.2f}, {valid_data.quantile(0.75):.2f}]")
    
    # Save output (RAW values)
    output_path = output_file if os.path.isabs(output_file) else os.path.join(
        output_dir, os.path.basename(output_file)
    )
    csf_final.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  Total subjects: {len(csf_final)}")
    
    print("\n" + "=" * 70)
    print("Step 2: CSF Preprocessing Complete")
    print("=" * 70)
    
    return csf_final


def main():
    """Main entry point."""
    args = parse_args()
    preprocess_csf(
        roche_file=args.roche_file,
        alzbio3_file=args.alzbio3_file,
        output_file=args.output_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

