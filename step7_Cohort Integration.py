"""

import argparse
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cohort Integration with Frozen Pipeline (Methods 2.3 & 2.6)'
    )
    parser.add_argument('--data_dir', type=str,
                        default='./processed_data',
                        help='Directory containing preprocessed data files (raw values)')
    parser.add_argument('--output_dir', type=str,
                        default='./processed_data',
                        help='Output directory')
    parser.add_argument('--test_size', type=int, default=196,
                        help='Number of MCI patients for independent test set (Methods 2.6: 196)')
    parser.add_argument('--mice_iterations', type=int, default=15,
                        help='MICE imputation iterations (Methods 2.6: 15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--discovery_output', type=str,
                        default='ADNI_Discovery_Scaled.csv',
                        help='Output filename for discovery (training) cohort')
    parser.add_argument('--test_output', type=str,
                        default='ADNI_Test_Scaled.csv',
                        help='Output filename for independent test set')
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


def integrate_and_process_strictly(data_dir, output_dir, test_size=196, 
                                    mice_iterations=15, seed=42,
                                    discovery_output='ADNI_Discovery_Scaled.csv',
                                    test_output='ADNI_Test_Scaled.csv'):
    """
    Main function implementing the Frozen Pipeline Strategy.
    
    Methods 2.3 & 2.6 Strict Compliance:
    - All preprocessing parameters fitted on training set ONLY
    - Test set transformed using training-fitted parameters
    - No data leakage from test to training
    
    Args:
        data_dir: Directory containing raw preprocessed files
        output_dir: Output directory
        test_size: Number of samples for independent test set
        mice_iterations: MICE iterations (Methods 2.6: 15)
        seed: Random seed
        discovery_output: Filename for discovery cohort
        test_output: Filename for test set
    """
    print("=" * 70)
    print("Step 7: Cohort Integration with Frozen Pipeline Strategy")
    print("Methods 2.3 & 2.6 STRICT COMPLIANCE")
    print("=" * 70)
    
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define expected input files (RAW values from Steps 1-6)
    file_mapping = {
        'APOE': 'APOE_genetics.csv',
        'CSF': 'CSF_biomarkers_raw.csv',
        'Clinical': 'Clinical_cognitive_scores_raw.csv',
        'MRI': 'Structural_MRI_features_raw.csv',
        'PET': 'PET_imaging_features_raw.csv',
        'Outcome': 'AD_conversion_outcomes.csv'
    }
    
    # =========================================================================
    # PHASE 1: Load all RAW data from previous steps
    # =========================================================================
    print(f"\n[1/7] Loading RAW modality files from: {data_dir}")
    print("  NOTE: These files should contain RAW values (no standardization)")
    
    data_dict = {}
    for key, filename in file_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data_dict[key] = read_csv_robust(filepath)
            print(f"  {key}: {len(data_dict[key])} subjects, {len(data_dict[key].columns)} columns")
        else:
            # Try alternative filename without _raw suffix
            alt_filepath = os.path.join(data_dir, filename.replace('_raw', ''))
            if os.path.exists(alt_filepath):
                data_dict[key] = read_csv_robust(alt_filepath)
                print(f"  {key}: {len(data_dict[key])} subjects (from {alt_filepath})")
            else:
                print(f"  {key}: FILE NOT FOUND ({filename})")
    
    if len(data_dict) == 0:
        raise ValueError("No data files found!")
    
    # =========================================================================
    # PHASE 2: Merge all modalities
    # =========================================================================
    print("\n[2/7] Merging all modalities (inner join for complete data)...")
    
    first_key = list(data_dict.keys())[0]
    integrated = data_dict[first_key].copy()
    
    for key, df in data_dict.items():
        if key == first_key:
            continue
        if 'ID' not in df.columns:
            print(f"  WARNING: {key} missing ID column, skipping")
            continue
        
        before = len(integrated)
        integrated = pd.merge(integrated, df, on='ID', how='inner')
        after = len(integrated)
        print(f"  After merging {key}: {after} subjects (lost {before - after})")
    
    print(f"\n  Integrated dataset: {len(integrated)} subjects, {len(integrated.columns)} features")
    
    # =========================================================================
    # PHASE 3: Split into Discovery (Training) and Test sets
    # =========================================================================
    print(f"\n[3/7] Splitting into Discovery and Test sets...")
    print(f"  Methods 2.6: {test_size} MCI patients for independent test set")
    
    # Identify MCI patients for test set (based on outcome)
    if 'AD_Conversion' in integrated.columns:
        # Stratified sampling to maintain conversion rate
        converters = integrated[integrated['AD_Conversion'] == 1]
        non_converters = integrated[integrated['AD_Conversion'] == 0]
        
        # Calculate proportions
        total_n = len(integrated)
        conv_rate = len(converters) / total_n
        
        # Sample test set maintaining conversion rate
        n_test_converters = int(test_size * conv_rate)
        n_test_non_converters = test_size - n_test_converters
        
        test_converters = converters.sample(n=min(n_test_converters, len(converters)), 
                                            random_state=seed)
        test_non_converters = non_converters.sample(n=min(n_test_non_converters, len(non_converters)), 
                                                     random_state=seed)
        
        test_data = pd.concat([test_converters, test_non_converters])
        test_ids = test_data['ID'].tolist()
        
        discovery_data = integrated[~integrated['ID'].isin(test_ids)].copy()
        test_data = integrated[integrated['ID'].isin(test_ids)].copy()
    else:
        # Random split if no outcome available
        test_data = integrated.sample(n=min(test_size, len(integrated)), random_state=seed)
        test_ids = test_data['ID'].tolist()
        discovery_data = integrated[~integrated['ID'].isin(test_ids)].copy()
    
    print(f"  Discovery (Training): {len(discovery_data)} subjects")
    print(f"  Test (Independent): {len(test_data)} subjects")
    
    if 'AD_Conversion' in integrated.columns:
        disc_conv_rate = discovery_data['AD_Conversion'].mean() * 100
        test_conv_rate = test_data['AD_Conversion'].mean() * 100
        print(f"  Discovery conversion rate: {disc_conv_rate:.1f}%")
        print(f"  Test conversion rate: {test_conv_rate:.1f}%")
    
    # =========================================================================
    # PHASE 4: Identify feature columns for processing
    # =========================================================================
    print("\n[4/7] Identifying feature columns...")
    
    # Exclude metadata and outcome columns
    exclude_cols = ['ID', 'AD_Conversion', 'Time_to_Event', 'Censored', 
                    'Followup_Months', 'DIAGNOSIS', 'SEX', 'APOE4_STATUS', 'APOE4_DOSAGE']
    
    all_cols = integrated.columns.tolist()
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    
    # Identify numeric columns only
    numeric_cols = []
    for col in feature_cols:
        if integrated[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            numeric_cols.append(col)
    
    print(f"  Total columns: {len(all_cols)}")
    print(f"  Numeric feature columns for processing: {len(numeric_cols)}")
    
    # =========================================================================
    # PHASE 5: MICE Imputation (Fit on Discovery ONLY)
    # =========================================================================
    print(f"\n[5/7] MICE Imputation ({mice_iterations} iterations)")
    print("  CRITICAL: Fitting imputer on DISCOVERY set ONLY (Methods 2.6)")
    
    # Report missing values before imputation
    n_missing_disc = discovery_data[numeric_cols].isnull().sum().sum()
    n_missing_test = test_data[numeric_cols].isnull().sum().sum()
    print(f"  Missing values - Discovery: {n_missing_disc}, Test: {n_missing_test}")
    
    # Fit MICE imputer on Discovery set ONLY
    imputer = IterativeImputer(
        max_iter=mice_iterations, 
        random_state=seed, 
        initial_strategy='median',
        verbose=0
    )
    
    print("  Fitting MICE imputer on Discovery set...")
    imputer.fit(discovery_data[numeric_cols])
    
    # Transform Discovery set
    print("  Transforming Discovery set...")
    discovery_imputed = imputer.transform(discovery_data[numeric_cols])
    discovery_data[numeric_cols] = discovery_imputed
    
    # Transform Test set using Discovery-fitted imputer (NO REFITTING!)
    print("  Transforming Test set (using Discovery-fitted imputer)...")
    test_imputed = imputer.transform(test_data[numeric_cols])
    test_data[numeric_cols] = test_imputed
    
    # Verify no missing values remain
    n_missing_disc_after = discovery_data[numeric_cols].isnull().sum().sum()
    n_missing_test_after = test_data[numeric_cols].isnull().sum().sum()
    print(f"  Missing values after - Discovery: {n_missing_disc_after}, Test: {n_missing_test_after}")
    
    # =========================================================================
    # PHASE 6: Z-score Standardization (Fit on Discovery ONLY)
    # =========================================================================
    print("\n[6/7] Z-score Standardization")
    print("  CRITICAL: Fitting scaler on DISCOVERY set ONLY (Methods 2.3)")
    
    # Fit StandardScaler on Discovery set ONLY
    scaler = StandardScaler()
    
    print("  Fitting StandardScaler on Discovery set...")
    scaler.fit(discovery_data[numeric_cols])
    
    # Transform Discovery set
    print("  Transforming Discovery set...")
    discovery_scaled = scaler.transform(discovery_data[numeric_cols])
    discovery_data[numeric_cols] = discovery_scaled
    
    # Transform Test set using Discovery-fitted scaler (NO REFITTING!)
    print("  Transforming Test set (using Discovery-fitted scaler)...")
    test_scaled = scaler.transform(test_data[numeric_cols])
    test_data[numeric_cols] = test_scaled
    
    # Verify standardization
    disc_means = discovery_data[numeric_cols].mean().mean()
    disc_stds = discovery_data[numeric_cols].std().mean()
    test_means = test_data[numeric_cols].mean().mean()
    test_stds = test_data[numeric_cols].std().mean()
    
    print(f"  Discovery - Mean of means: {disc_means:.6f}, Mean of stds: {disc_stds:.4f}")
    print(f"  Test - Mean of means: {test_means:.4f}, Mean of stds: {test_stds:.4f}")
    print("  NOTE: Test set means/stds may differ from 0/1 - this is CORRECT behavior")
    
    # =========================================================================
    # PHASE 7: Column Renaming for Step 8+ Compatibility
    # =========================================================================
    print("\n[7/7] Column renaming for Step 8+ compatibility...")
    
    # Column rename map: Step 1-7 output names -> Step 8 VAE expected names
    column_rename_map = {
        'PTAU181': 'CSF_PTAU181',
        'ABETA42_ABETA40_RATIO': 'CSF_ABETA42_ABETA40_RATIO',
        'TAU_TOTAL': 'CSF_TAU_TOTAL',
        'APOE4_STATUS': 'APOE_VAR',
    }
    
    # Apply renaming to both datasets
    renamed_cols = []
    for old_name, new_name in column_rename_map.items():
        if old_name in discovery_data.columns:
            renamed_cols.append(f"{old_name} -> {new_name}")
    
    discovery_data = discovery_data.rename(columns=column_rename_map)
    test_data = test_data.rename(columns=column_rename_map)
    
    if renamed_cols:
        print(f"  Renamed columns for Step 8+ compatibility:")
        for col in renamed_cols:
            print(f"    {col}")
    else:
        print("  No columns needed renaming (already compatible)")
    
    # =========================================================================
    # PHASE 8: Save outputs and fitted processors
    # =========================================================================
    print("\n[8/8] Saving outputs and fitted processors...")
    
    # Save Discovery cohort
    discovery_path = os.path.join(output_dir, discovery_output)
    discovery_data.to_csv(discovery_path, index=False)
    print(f"  Saved Discovery cohort: {discovery_path}")
    print(f"    Subjects: {len(discovery_data)}, Features: {len(discovery_data.columns)}")
    
    # Save Test set
    test_path = os.path.join(output_dir, test_output)
    test_data.to_csv(test_path, index=False)
    print(f"  Saved Test set: {test_path}")
    print(f"    Subjects: {len(test_data)}, Features: {len(test_data.columns)}")
    
    # Save fitted processors for future use (e.g., external validation)
    imputer_path = os.path.join(output_dir, 'pipeline_imputer.pkl')
    with open(imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    print(f"  Saved MICE imputer: {imputer_path}")
    
    scaler_path = os.path.join(output_dir, 'pipeline_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved StandardScaler: {scaler_path}")
    
    # Save feature column list
    feature_cols_path = os.path.join(output_dir, 'feature_columns.pkl')
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(numeric_cols, f)
    print(f"  Saved feature columns: {feature_cols_path}")
    
    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("FROZEN PIPELINE STRATEGY - COMPLIANCE SUMMARY")
    print("=" * 70)
    print("\nMethods 2.3 Compliance:")
    print("  ✓ StandardScaler fitted on Discovery (Training) set ONLY")
    print("  ✓ Test set transformed using Discovery-fitted parameters")
    print("  ✓ No data leakage from Test to Training")
    
    print("\nMethods 2.6 Compliance:")
    print(f"  ✓ MICE imputation: {mice_iterations} iterations")
    print("  ✓ MICE fitted on Discovery (Training) set ONLY")
    print("  ✓ Test set imputed using Discovery-fitted equations")
    print(f"  ✓ Independent test set: {len(test_data)} MCI patients")
    
    print("\nStep 8+ Compatibility:")
    print("  ✓ Column names mapped to VAE expected format")
    print("  ✓ CSF biomarkers: PTAU181 -> CSF_PTAU181, etc.")
    print("  ✓ APOE: APOE4_STATUS -> APOE_VAR")
    
    print("\nOutput Files:")
    print(f"  - {discovery_output}: Discovery cohort (standardized)")
    print(f"  - {test_output}: Independent test set (standardized)")
    print(f"  - pipeline_imputer.pkl: Fitted MICE imputer")
    print(f"  - pipeline_scaler.pkl: Fitted StandardScaler")
    print(f"  - feature_columns.pkl: List of feature columns")
    
    print("\nUsage for External Validation:")
    print("  1. Load external cohort data")
    print("  2. Load pipeline_imputer.pkl and pipeline_scaler.pkl")
    print("  3. imputer.transform(external_data) - DO NOT refit!")
    print("  4. scaler.transform(external_data) - DO NOT refit!")
    
    print("\n" + "=" * 70)
    print("Step 7: Cohort Integration Complete")
    print("DATA LEAKAGE PREVENTION GUARANTEED")
    print("=" * 70)
    
    return discovery_data, test_data


def main():
    """Main entry point."""
    args = parse_args()
    integrate_and_process_strictly(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        mice_iterations=args.mice_iterations,
        seed=args.seed,
        discovery_output=args.discovery_output,
        test_output=args.test_output
    )


if __name__ == "__main__":
    main()
