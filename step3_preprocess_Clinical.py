"""

import argparse
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Clinical Assessment Preprocessing '
    )
    parser.add_argument('--data_dir', type=str,
                        default='./ADNI_Raw_Data/Clinical',
                        help='Directory containing clinical data files')
    parser.add_argument('--output_file', type=str,
                        default='./processed_data/Clinical_cognitive_scores_raw.csv',
                        help='Output file path (raw values)')
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


def calculate_age_at_baseline(demographics_df, baseline_date_col='EXAMDATE'):
    """
    Calculate age at baseline visit (not using fixed year).
    """
    if 'PTDOBYY' not in demographics_df.columns:
        return None
    
    birth_year = demographics_df['PTDOBYY']
    
    # Try to use actual baseline date
    if baseline_date_col in demographics_df.columns:
        try:
            exam_dates = pd.to_datetime(demographics_df[baseline_date_col], errors='coerce')
            exam_year = exam_dates.dt.year
            age = exam_year - birth_year
            return age
        except Exception:
            pass
    
    # Fallback: use USERDATE or current year
    if 'USERDATE' in demographics_df.columns:
        try:
            user_dates = pd.to_datetime(demographics_df['USERDATE'], errors='coerce')
            user_year = user_dates.dt.year
            age = user_year - birth_year
            return age
        except Exception:
            pass
    
    # Last resort: use median year from data
    return datetime.now().year - birth_year


def preprocess_clinical(data_dir, output_file, output_dir):
    """
    Main preprocessing function for clinical assessment data.
    
    Args:
        data_dir: Directory containing clinical data files
        output_file: Output file path
        output_dir: Output directory
    """
    print("=" * 70)
    print("Step 3: Clinical Assessment Preprocessing ")
    print("CORRECTED: No premature MICE/standardization (deferred to Step 7)")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define expected files
    file_mapping = {
        'diagnostic': 'Diagnostic_Summary.csv',
        'adas': 'ADAS_Cognitive_Behavior.csv',
        'cdr': 'Clinical_Dementia_Rating.csv',
        'faq': 'Functional_Activities_Questionnaire.csv',
        'demographics': 'Subject_Demographics.csv',
        'gds': 'Geriatric_Depression_Scale.csv',
        'mmse': 'MMSE.csv'
    }
    
    # Load data files
    print(f"\n[1/5] Loading clinical data files from: {data_dir}")
    data_dict = {}
    for key, filename in file_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data_dict[key] = read_csv_robust(filepath)
            print(f"  {key}: {len(data_dict[key])} records")
        else:
            print(f"  {key}: FILE NOT FOUND ({filename})")
    
    # Extract baseline visits
    print("\n[2/5] Extracting baseline visits...")
    baseline_dict = {key: extract_baseline(df) for key, df in data_dict.items()}
    for key, df in baseline_dict.items():
        print(f"  {key}: {len(df)} baseline records")
    
    # Extract features from each dataset
    print("\n[3/5] Extracting clinical features...")
    
    # Diagnosis
    if 'diagnostic' in baseline_dict:
        diag_df = baseline_dict['diagnostic']
        diag_features = diag_df[['PTID', 'DIAGNOSIS']].copy() if 'DIAGNOSIS' in diag_df.columns else diag_df[['PTID']].copy()
        if 'DIAGNOSIS' in diag_features.columns:
            diagnosis_map = {'CN': 1, 'MCI': 2, 'AD': 3, 'Dementia': 3}
            diag_features['DIAGNOSIS'] = diag_features['DIAGNOSIS'].map(diagnosis_map)
    else:
        diag_features = pd.DataFrame()
    
    # ADAS-Cog13 
    if 'adas' in baseline_dict:
        adas_df = baseline_dict['adas']
        adas_col = 'TOTAL13' if 'TOTAL13' in adas_df.columns else 'TOTSCORE'
        adas_features = adas_df[['PTID', adas_col]].copy()
        adas_features.rename(columns={adas_col: 'ADAS13'}, inplace=True)
    else:
        adas_features = pd.DataFrame()
    
    # CDR-SB 
    if 'cdr' in baseline_dict:
        cdr_df = baseline_dict['cdr']
        cdr_features = cdr_df[['PTID', 'CDRSB']].copy() if 'CDRSB' in cdr_df.columns else pd.DataFrame()
    else:
        cdr_features = pd.DataFrame()
    
    # FAQ 
    if 'faq' in baseline_dict:
        faq_df = baseline_dict['faq']
        faq_features = faq_df[['PTID', 'FAQTOTAL']].copy() if 'FAQTOTAL' in faq_df.columns else pd.DataFrame()
    else:
        faq_features = pd.DataFrame()
    
    # MMSE 
    if 'mmse' in baseline_dict:
        mmse_df = baseline_dict['mmse']
        mmse_col = 'MMSCORE' if 'MMSCORE' in mmse_df.columns else 'MMSE'
        mmse_features = mmse_df[['PTID', mmse_col]].copy()
        mmse_features.rename(columns={mmse_col: 'MMSE'}, inplace=True)
    else:
        mmse_features = pd.DataFrame()
    
    # Demographics (Age, Sex, Education)
    if 'demographics' in baseline_dict:
        demo_df = baseline_dict['demographics']
        demo_features = demo_df[['PTID']].copy()
        
        # Age calculation (from baseline date, not fixed year)
        demo_features['AGE'] = calculate_age_at_baseline(demo_df)
        
        # Sex (0=Male, 1=Female)
        if 'PTGENDER' in demo_df.columns:
            demo_features['SEX'] = demo_df['PTGENDER'].map({1: 0, 2: 1})
        
        # Education years
        if 'PTEDUCAT' in demo_df.columns:
            demo_features['EDUCATION'] = demo_df['PTEDUCAT']
    else:
        demo_features = pd.DataFrame()
    
    # GDS (Geriatric Depression Scale)
    if 'gds' in baseline_dict:
        gds_df = baseline_dict['gds']
        gds_features = gds_df[['PTID', 'GDTOTAL']].copy() if 'GDTOTAL' in gds_df.columns else pd.DataFrame()
        gds_features.rename(columns={'GDTOTAL': 'GDS'}, inplace=True)
    else:
        gds_features = pd.DataFrame()
    
    # Merge all clinical features
    print("\n[4/5] Merging clinical features...")
    feature_dfs = [diag_features, adas_features, cdr_features, faq_features, 
                   mmse_features, demo_features, gds_features]
    feature_dfs = [df for df in feature_dfs if len(df) > 0]
    
    if len(feature_dfs) == 0:
        raise ValueError("No clinical features could be extracted")
    
    clinical_merged = feature_dfs[0]
    for df in feature_dfs[1:]:
        clinical_merged = pd.merge(clinical_merged, df, on='PTID', how='outer')
    
    print(f"  Merged records: {len(clinical_merged)}")
    print(f"  Features: {list(clinical_merged.columns)}")
    
    # Report missing values (will be handled by MICE in Step 7)
    print("\n[5/5] Reporting missing values (will be imputed by MICE in Step 7)...")
    
    for col in clinical_merged.columns:
        if col == 'PTID':
            continue
        n_missing = clinical_merged[col].isnull().sum()
        pct_missing = 100 * n_missing / len(clinical_merged)
        if n_missing > 0:
            print(f"  {col}: {n_missing} missing ({pct_missing:.1f}%)")
    
    # Prepare final output (RAW values)
    clinical_final = clinical_merged.rename(columns={'PTID': 'ID'})
    final_cols = ['ID'] + [col for col in clinical_final.columns if col != 'ID']
    clinical_final = clinical_final[final_cols]
    
    # Summary statistics (RAW values)
    print("\n  Clinical Feature Summary (RAW values, before standardization):")
    continuous_cols = [c for c in clinical_final.columns if c not in ['ID', 'DIAGNOSIS', 'SEX']]
    for col in continuous_cols[:5]:
        valid_data = clinical_final[col].dropna()
        if len(valid_data) > 0:
            print(f"    {col}: median={valid_data.median():.1f}, IQR=[{valid_data.quantile(0.25):.1f}, {valid_data.quantile(0.75):.1f}]")
    
    # Save output (RAW values)
    output_path = output_file if os.path.isabs(output_file) else os.path.join(
        output_dir, os.path.basename(output_file)
    )
    clinical_final.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  Total subjects: {len(clinical_final)}")
    
    print("\n" + "=" * 70)
    print("Step 3: Clinical Preprocessing Complete")
    print("=" * 70)
    
    return clinical_final


def main():
    """Main entry point."""
    args = parse_args()
    preprocess_clinical(
        data_dir=args.data_dir,
        output_file=args.output_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

