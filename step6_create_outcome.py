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
        description='Outcome Variable Creation'
    )
    parser.add_argument('--input_file', type=str,
                        default='./ADNI_Raw_Data/Clinical/Diagnostic_Summary.csv',
                        help='Path to diagnostic summary CSV')
    parser.add_argument('--output_file', type=str,
                        default='./processed_data/AD_conversion_outcomes.csv',
                        help='Output file path')
    parser.add_argument('--output_dir', type=str,
                        default='./processed_data',
                        help='Output directory')
    parser.add_argument('--followup_months', type=int, default=36,
                        help='Follow-up period in months ')
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


def is_mci_diagnosis(dx_value):
    """Check if diagnosis indicates MCI."""
    if pd.isna(dx_value):
        return False
    
    # Numeric coding
    if dx_value == 2:
        return True
    
    # String coding
    dx_str = str(dx_value).upper()
    mci_keywords = ['MCI', 'MILD COGNITIVE IMPAIRMENT', 'EMCI', 'LMCI']
    return any(kw in dx_str for kw in mci_keywords)


def is_ad_diagnosis(dx_value):
    """Check if diagnosis indicates AD/Dementia."""
    if pd.isna(dx_value):
        return False
    
    # Numeric coding
    if dx_value == 3:
        return True
    
    # String coding
    dx_str = str(dx_value).upper()
    ad_keywords = ['AD', 'ALZHEIMER', 'DEMENTIA']
    exclude_keywords = ['MCI', 'MILD']
    
    if any(ex in dx_str for ex in exclude_keywords):
        return False
    
    return any(kw in dx_str for kw in ad_keywords)


def parse_viscode_to_months(viscode):
    """Parse VISCODE to months from baseline."""
    if pd.isna(viscode):
        return None
    
    viscode = str(viscode).lower().strip()
    
    # Baseline codes
    if viscode in ['bl', 'sc', 'scmri']:
        return 0
    
    # Month codes (m06, m12, m24, etc.)
    if viscode.startswith('m'):
        try:
            return int(viscode[1:])
        except ValueError:
            pass
    
    # Year codes (y1, y2, etc.)
    if viscode.startswith('y'):
        try:
            return int(viscode[1:]) * 12
        except ValueError:
            pass
    
    return None


def calculate_followup_time(baseline_date, event_date):
    """Calculate follow-up time in months."""
    try:
        bl_date = pd.to_datetime(baseline_date)
        ev_date = pd.to_datetime(event_date)
        days = (ev_date - bl_date).days
        return days / 30.44  # Average days per month
    except Exception:
        return None


def create_outcome(input_file, output_file, output_dir, followup_months=36):
    """
    Main function to create outcome variables.

    Args:
        input_file: Path to diagnostic summary CSV
        output_file: Output file path
        output_dir: Output directory
        followup_months: Follow-up period in months
    """
    print("=" * 70)
    print("Step 6: Outcome Variable Creation ")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/5] Loading diagnostic data from: {input_file}")
    diag_data = read_csv_robust(input_file)
    print(f"  Total records: {len(diag_data)}")
    print(f"  Unique subjects: {diag_data['PTID'].nunique()}")
    
    # Identify diagnosis column
    diagnosis_col = 'DIAGNOSIS'
    if diagnosis_col not in diag_data.columns:
        for col in ['DX', 'DXCURREN', 'DXCHANGE']:
            if col in diag_data.columns:
                diagnosis_col = col
                break
    
    print(f"  Diagnosis column: {diagnosis_col}")
    
    # Extract baseline MCI patients
    print(f"\n[2/5] Identifying baseline MCI patients...")
    
    # Get baseline records
    baseline_codes = ['bl', 'BL', 'sc', 'SC']
    if 'VISCODE' in diag_data.columns:
        baseline_mask = diag_data['VISCODE'].isin(baseline_codes)
        baseline_data = diag_data[baseline_mask].copy()
    else:
        baseline_data = diag_data.sort_values(['PTID', 'EXAMDATE']).groupby('PTID').first().reset_index()
    
    # Filter for MCI at baseline
    mci_baseline = baseline_data[baseline_data[diagnosis_col].apply(is_mci_diagnosis)].copy()
    mci_ptid_list = mci_baseline['PTID'].unique()
    print(f"  Baseline MCI patients: {len(mci_ptid_list)}")
    
    # Get follow-up visits
    print(f"\n[3/5] Extracting follow-up visits (up to {followup_months} months)...")
    
    # Get follow-up data for MCI patients
    if 'VISCODE' in diag_data.columns:
        followup_data = diag_data[
            (diag_data['PTID'].isin(mci_ptid_list)) &
            (~diag_data['VISCODE'].isin(baseline_codes))
        ].copy()
    else:
        followup_data = diag_data[diag_data['PTID'].isin(mci_ptid_list)].copy()
    
    print(f"  Follow-up records: {len(followup_data)}")
    
    # Determine conversion status and time
    print(f"\n[4/5] Determining conversion status...")
    
    conversion_results = []
    
    for ptid in mci_ptid_list:
        patient_baseline = mci_baseline[mci_baseline['PTID'] == ptid].iloc[0]
        patient_followup = followup_data[followup_data['PTID'] == ptid].copy()
        
        converted = False
        time_to_event = None
        censored = True
        
        # Get baseline date
        baseline_date = patient_baseline.get('EXAMDATE', None)
        
        if len(patient_followup) > 0:
            # Sort by visit code or date
            if 'VISCODE' in patient_followup.columns:
                patient_followup['MONTHS'] = patient_followup['VISCODE'].apply(parse_viscode_to_months)
                patient_followup = patient_followup.dropna(subset=['MONTHS'])
                patient_followup = patient_followup.sort_values('MONTHS')
            elif 'EXAMDATE' in patient_followup.columns:
                patient_followup = patient_followup.sort_values('EXAMDATE')
            
            # Check for conversion
            for idx, row in patient_followup.iterrows():
                if is_ad_diagnosis(row[diagnosis_col]):
                    converted = True
                    censored = False
                    
                    # Calculate time to event
                    if 'MONTHS' in patient_followup.columns:
                        time_to_event = row['MONTHS']
                    elif 'EXAMDATE' in row and baseline_date:
                        time_to_event = calculate_followup_time(baseline_date, row['EXAMDATE'])
                    break
            
            # If not converted, get last follow-up time (censoring time)
            if not converted and len(patient_followup) > 0:
                last_visit = patient_followup.iloc[-1]
                if 'MONTHS' in patient_followup.columns:
                    time_to_event = last_visit['MONTHS']
                elif 'EXAMDATE' in last_visit and baseline_date:
                    time_to_event = calculate_followup_time(baseline_date, last_visit['EXAMDATE'])
        
        # Default time if not calculated
        if time_to_event is None:
            time_to_event = followup_months if not converted else followup_months / 2
        
        conversion_results.append({
            'ID': ptid,
            'AD_Conversion': 1 if converted else 0,
            'Time_to_Event': time_to_event,
            'Censored': 1 if censored else 0,
            'Followup_Months': min(time_to_event, followup_months) if time_to_event else followup_months
        })
    
    outcome_df = pd.DataFrame(conversion_results)
    
    # Summary statistics
    print(f"\n[5/5] Summary statistics...")
    n_converters = (outcome_df['AD_Conversion'] == 1).sum()
    n_non_converters = (outcome_df['AD_Conversion'] == 0).sum()
    conversion_rate = 100 * n_converters / len(outcome_df)
    
    print(f"  Total MCI patients: {len(outcome_df)}")
    print(f"  Converters (MCI→AD): {n_converters} ({conversion_rate:.1f}%)")
    print(f"  Non-converters: {n_non_converters} ({100-conversion_rate:.1f}%)")
    print(f"  Mean follow-up time: {outcome_df['Followup_Months'].mean():.1f} months")
    
    if n_converters > 0:
        converters = outcome_df[outcome_df['AD_Conversion'] == 1]
        print(f"  Mean time to conversion: {converters['Time_to_Event'].mean():.1f} months")
    
    # Save output
    output_path = output_file if os.path.isabs(output_file) else os.path.join(
        output_dir, os.path.basename(output_file)
    )
    outcome_df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("Step 6: Outcome Creation Complete")
    print("=" * 70)
    
    return outcome_df


def main():
    """Main entry point."""
    args = parse_args()
    create_outcome(
        input_file=args.input_file,
        output_file=args.output_file,
        output_dir=args.output_dir,
        followup_months=args.followup_months
    )


if __name__ == "__main__":
    main()

