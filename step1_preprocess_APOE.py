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
        description='APOE Genotype Preprocessing (Methods 2.2)'
    )
    parser.add_argument('--input_file', type=str,
                        default='./ADNI_Raw_Data/APOE/ApoE_Genotyping_Results.csv',
                        help='Path to APOE genotyping results CSV')
    parser.add_argument('--output_file', type=str,
                        default='./processed_data/APOE_genetics.csv',
                        help='Output file path')
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


def extract_apoe4_features(genotype):
    """
    Extract APOE ε4 features from genotype string.
    
    Methods 2.2: "individuals carrying ≥1 ε4 allele were classified as APOE ε4-positive"
    
    Args:
        genotype: APOE genotype string (e.g., "3/4", "4/4", "3/3")
    
    Returns:
        tuple: (apoe4_dosage, apoe4_status)
            - apoe4_dosage: Number of ε4 alleles (0, 1, or 2)
            - apoe4_status: Binary indicator (1 if ≥1 ε4 allele, else 0)
    """
    geno_clean = str(genotype).strip().lower().replace(" ", "")
    apoe4_dosage = geno_clean.count("4")
    apoe4_status = 1 if apoe4_dosage >= 1 else 0
    return apoe4_dosage, apoe4_status


def preprocess_apoe(input_file, output_file, output_dir):
    """
    Main preprocessing function for APOE genotype data.
    
    Methods 2.2 Implementation:
    - Load APOE genotyping results
    - Extract APOE ε4 dosage (0, 1, or 2 alleles)
    - Classify APOE ε4 status (positive if ≥1 ε4 allele)
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        output_dir: Output directory
    """
    print("=" * 70)
    print("Step 1: APOE Genotype Preprocessing (Methods 2.2)")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/4] Loading APOE genotype data from: {input_file}")
    apoe_raw = read_csv_robust(input_file)
    print(f"  Raw records: {len(apoe_raw)}")
    
    # Extract core columns
    print("\n[2/4] Extracting and deduplicating...")
    
    # Handle different column naming conventions
    id_col = None
    geno_col = None
    
    for col in apoe_raw.columns:
        col_lower = col.lower()
        if col_lower in ['ptid', 'rid', 'id', 'subject_id']:
            id_col = col
        if col_lower in ['genotype', 'apoe_genotype', 'apoe']:
            geno_col = col
    
    if id_col is None:
        raise ValueError("Could not find ID column (PTID/RID/ID)")
    if geno_col is None:
        raise ValueError("Could not find GENOTYPE column")
    
    print(f"  ID column: {id_col}")
    print(f"  Genotype column: {geno_col}")
    
    apoe_core = apoe_raw[[id_col, geno_col]].copy()
    apoe_core.columns = ['ID', 'GENOTYPE']
    
    # Deduplicate by ID (keep first occurrence)
    apoe_dedup = apoe_core.drop_duplicates(subset='ID', keep='first')
    print(f"  After deduplication: {len(apoe_dedup)}")
    
    # Remove missing genotypes
    apoe_clean = apoe_dedup.dropna(subset=['GENOTYPE']).reset_index(drop=True)
    print(f"  After removing missing: {len(apoe_clean)}")
    
    # Extract APOE4 features
    print("\n[3/4] Extracting APOE ε4 features (Methods 2.2)...")
    apoe_features = apoe_clean['GENOTYPE'].apply(extract_apoe4_features)
    apoe_clean[['APOE4_DOSAGE', 'APOE4_STATUS']] = pd.DataFrame(
        apoe_features.tolist(), index=apoe_clean.index
    )
    
    # Summary statistics
    n_e4_positive = (apoe_clean['APOE4_STATUS'] == 1).sum()
    n_e4_negative = (apoe_clean['APOE4_STATUS'] == 0).sum()
    n_homozygous = (apoe_clean['APOE4_DOSAGE'] == 2).sum()
    n_heterozygous = (apoe_clean['APOE4_DOSAGE'] == 1).sum()
    
    print(f"\n  APOE ε4 Status Distribution:")
    print(f"    ε4-positive: {n_e4_positive} ({100*n_e4_positive/len(apoe_clean):.1f}%)")
    print(f"    ε4-negative: {n_e4_negative} ({100*n_e4_negative/len(apoe_clean):.1f}%)")
    
    print(f"\n  APOE ε4 Dosage Distribution:")
    print(f"    Homozygous (ε4/ε4): {n_homozygous} ({100*n_homozygous/len(apoe_clean):.1f}%)")
    print(f"    Heterozygous (ε4/x): {n_heterozygous} ({100*n_heterozygous/len(apoe_clean):.1f}%)")
    print(f"    Non-carrier: {n_e4_negative} ({100*n_e4_negative/len(apoe_clean):.1f}%)")
    
    # Prepare final output
    print("\n[4/4] Saving processed data...")
    apoe_final = apoe_clean[['ID', 'APOE4_DOSAGE', 'APOE4_STATUS']].copy()
    
    # Save to output file
    output_path = output_file if os.path.isabs(output_file) else os.path.join(
        output_dir, os.path.basename(output_file)
    )
    apoe_final.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Total subjects: {len(apoe_final)}")
    
    print("\n" + "=" * 70)
    print("Step 1: APOE Preprocessing Complete")
    print("=" * 70)
    
    return apoe_final


def main():
    """Main entry point."""
    args = parse_args()
    preprocess_apoe(
        input_file=args.input_file,
        output_file=args.output_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
