from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
STEPS = [
    "01_define_36m_endpoints.py",
    "02_rulec_statistics_core.py",
    "03_extract_aligned_features.py",
    "04_fit_leakage_controlled_models.py",
    "05_validate_aibl_clinical_proxy.py",
    "06_vae_sensitivity_analysis.py",
    "07_build_nonoverlap_adni_validation.py",
    "08_crossfit_five_reader_benchmark.py",
    "09_multireader_statistics.py",
    "10_generate_figures.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete multimodal analysis pipeline.")
    parser.add_argument("--data-root", type=Path, required=True, help="Local restricted-data root.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs", help="Tabular output directory.")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "submission_figures", help="Figure output directory.")
    parser.add_argument("--start-step", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--stop-step", type=int, default=10, choices=range(1, 11))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    output_dir = args.output_dir.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["AD_MULTIMODAL_DATA_ROOT"] = str(data_root)
    environment["AD_MULTIMODAL_OUTPUT_DIR"] = str(output_dir)
    environment["AD_MULTIMODAL_FIGURE_DIR"] = str(figure_dir)

    if args.start_step > args.stop_step:
        raise ValueError("--start-step cannot exceed --stop-step")
    for index, filename in enumerate(STEPS, start=1):
        if index < args.start_step or index > args.stop_step:
            continue
        print(f"[{index:02d}/10] {filename}", flush=True)
        subprocess.run([sys.executable, str(HERE / filename)], check=True, env=environment)

    print(f"Outputs: {output_dir}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
