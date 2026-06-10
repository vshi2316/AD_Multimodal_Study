import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="CSF biomarker preprocessing")
    parser.add_argument(
        "--roche_file",
        type=str,
        default="./ADNI_Raw_Data/CSF/CSF_Roche_Elecsys.csv",
        help="Path to Roche Elecsys CSF file",
    )
    parser.add_argument(
        "--alzbio3_file",
        type=str,
        default="./ADNI_Raw_Data/CSF/CSF_Alzbio3.csv",
        help="Path to AlzBio3 CSF file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./processed_data/CSF_biomarkers_raw.csv",
        help="Output file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Output directory",
    )
    return parser.parse_args()


def read_csv_robust(filepath):
    encodings = ["utf_8_sig", "utf_8", "gbk", "latin_1"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read file: {filepath}") from last_error


def first_existing(columns, candidates):
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def build_roche_table(roche_data):
    rid_col = first_existing(roche_data.columns, ["RID"])
    ptid_col = first_existing(roche_data.columns, ["PTID"])
    abeta42_col = first_existing(roche_data.columns, ["ABETA42"])
    abeta40_col = first_existing(roche_data.columns, ["ABETA40"])
    tau_col = first_existing(roche_data.columns, ["TAU", "TAU_TOTAL"])
    ptau_col = first_existing(roche_data.columns, ["PTAU", "PTAU181"])

    if rid_col is None:
        raise ValueError("RID column not found in Roche file")

    roche = pd.DataFrame()
    roche["RID"] = roche_data[rid_col].astype(str)

    if ptid_col is not None:
        roche["PTID"] = roche_data[ptid_col].astype(str)
    else:
        roche["PTID"] = np.nan

    if abeta42_col is not None:
        roche["ABETA42_ROCHE"] = pd.to_numeric(roche_data[abeta42_col], errors="coerce")
    if abeta40_col is not None:
        roche["ABETA40_ROCHE"] = pd.to_numeric(roche_data[abeta40_col], errors="coerce")
    if tau_col is not None:
        roche["TAU_TOTAL_ROCHE"] = pd.to_numeric(roche_data[tau_col], errors="coerce")
    if ptau_col is not None:
        roche["PTAU181_ROCHE"] = pd.to_numeric(roche_data[ptau_col], errors="coerce")

    return roche


def build_alzbio3_table(alzbio3_data):
    rid_col = first_existing(alzbio3_data.columns, ["RID"])
    abeta_col = first_existing(alzbio3_data.columns, ["ABETA", "ABETA42"])
    tau_col = first_existing(alzbio3_data.columns, ["TAU", "TAU_TOTAL"])
    ptau_col = first_existing(alzbio3_data.columns, ["PTAU", "PTAU181"])

    if rid_col is None:
        raise ValueError("RID column not found in AlzBio3 file")

    alzbio3 = pd.DataFrame()
    alzbio3["RID"] = alzbio3_data[rid_col].astype(str)

    if abeta_col is not None:
        alzbio3["ABETA42_ALZBIO3"] = pd.to_numeric(alzbio3_data[abeta_col], errors="coerce")
    if tau_col is not None:
        alzbio3["TAU_TOTAL_ALZBIO3"] = pd.to_numeric(alzbio3_data[tau_col], errors="coerce")
    if ptau_col is not None:
        alzbio3["PTAU181_ALZBIO3"] = pd.to_numeric(alzbio3_data[ptau_col], errors="coerce")

    return alzbio3


def coalesce_columns(frame, ordered_columns):
    available = [col for col in ordered_columns if col in frame.columns]
    if not available:
        return pd.Series([np.nan] * len(frame), index=frame.index)
    series = frame[available[0]].copy()
    for col in available[1:]:
        series = series.fillna(frame[col])
    return series


def harmonize_platforms(roche_data, alzbio3_data):
    roche = build_roche_table(roche_data)
    alzbio3 = build_alzbio3_table(alzbio3_data)

    merged = pd.merge(roche, alzbio3, on="RID", how="outer")

    merged["PTID"] = coalesce_columns(merged, ["PTID"])
    merged["ABETA42"] = coalesce_columns(merged, ["ABETA42_ROCHE", "ABETA42_ALZBIO3"])
    merged["ABETA40"] = coalesce_columns(merged, ["ABETA40_ROCHE"])
    merged["TAU_TOTAL"] = coalesce_columns(merged, ["TAU_TOTAL_ROCHE", "TAU_TOTAL_ALZBIO3"])
    merged["PTAU181"] = coalesce_columns(merged, ["PTAU181_ROCHE", "PTAU181_ALZBIO3"])

    merged["ABETA42_ABETA40_RATIO"] = np.where(
        merged["ABETA42"].notna() & merged["ABETA40"].notna() & (merged["ABETA40"] != 0),
        merged["ABETA42"] / merged["ABETA40"],
        np.nan,
    )

    merged["ID"] = merged["PTID"].fillna(merged["RID"]).astype(str)

    keep_cols = [
        "ID",
        "RID",
        "PTID",
        "ABETA42",
        "ABETA40",
        "ABETA42_ABETA40_RATIO",
        "TAU_TOTAL",
        "PTAU181",
    ]
    out = merged[keep_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.drop_duplicates(subset=["ID"])

    return out


def report_missingness(df, variables):
    print("\nMissingness report")
    for variable in variables:
        if variable not in df.columns:
            continue
        n_missing = int(df[variable].isna().sum())
        pct_missing = 100 * n_missing / len(df)
        print(f"  {variable}: {n_missing} missing values ({pct_missing:.1f} percent)")


def report_summary(df, variables):
    print("\nRaw value summary")
    for variable in variables:
        if variable not in df.columns:
            continue
        values = pd.to_numeric(df[variable], errors="coerce").dropna()
        if values.empty:
            print(f"  {variable}: no observed values")
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        print(
            f"  {variable}: median={values.median():.4f}, "
            f"IQR=({q1:.4f}, {q3:.4f}), n={len(values)}"
        )


def preprocess_csf(roche_file, alzbio3_file, output_file, output_dir):
    print("=" * 70)
    print("Step 2: CSF biomarker preprocessing")
    print("Raw values are preserved")
    print("Standardization is deferred to later analysis steps")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/4] Loading files")
    roche_data = read_csv_robust(roche_file)
    alzbio3_data = read_csv_robust(alzbio3_file)
    print(f"  Roche records: {len(roche_data)}")
    print(f"  AlzBio3 records: {len(alzbio3_data)}")

    print("\n[2/4] Harmonizing platforms")
    csf_final = harmonize_platforms(roche_data, alzbio3_data)
    print(f"  Subjects after merge: {len(csf_final)}")

    print("\n[3/4] Reporting missingness and raw summaries")
    report_missingness(
        csf_final,
        ["ABETA42", "ABETA40", "ABETA42_ABETA40_RATIO", "TAU_TOTAL", "PTAU181"],
    )
    report_summary(
        csf_final,
        ["ABETA42", "ABETA40", "ABETA42_ABETA40_RATIO", "TAU_TOTAL", "PTAU181"],
    )

    print("\n[4/4] Writing output")
    output_path = output_file
    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, os.path.basename(output_file))
    csf_final.to_csv(output_path, index=False)
    print(f"  Saved file: {output_path}")

    print("\nCompleted step 2")
    return csf_final


def main():
    args = parse_args()
    preprocess_csf(
        roche_file=args.roche_file,
        alzbio3_file=args.alzbio3_file,
        output_file=args.output_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
