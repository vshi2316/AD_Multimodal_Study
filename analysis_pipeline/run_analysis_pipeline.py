"""Run the manuscript analyses from restricted inputs in fresh subprocesses."""
from __future__ import annotations
import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "analysis_config.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rscript")
    parser.add_argument("--component", choices=["all", "core", "vae", "extensions", "figures"], default="all")
    parser.add_argument("--check-inputs", action="store_true", help="Check required file paths; do not fit models")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    base = args.config.resolve().parent
    def resolve(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base/path).resolve()
    root = resolve(args.data_root or config["data_root"])
    results = resolve(args.output_dir or config["output_dir"])
    core = results / "core"
    env = os.environ.copy()
    env.update(AD_MULTIMODAL_DATA_ROOT=str(root), AD_MULTIMODAL_OUTPUT_DIR=str(core),
        AD_MULTIMODAL_RESULTS_DIR=str(results), AD_MULTIMODAL_PIPELINE_DIR=str(HERE))
    env["PYTHONUNBUFFERED"] = "1"
    rscript = args.rscript or config["rscript"]
    run_core = args.component in ["all", "core"]
    run_vae = args.component in ["all", "vae"]
    run_external = args.component in ["all", "extensions"]
    run_figures = args.component in ["all", "figures"]
    inputs = json.loads((HERE / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    required_roles = set()
    if run_core: required_roles.add("core")
    if run_vae: required_roles.add("vae")
    if run_external: required_roles.add("extensions")
    if run_figures: required_roles.add("figures")
    missing = [x["path"] for x in inputs if set(x["components"]) & required_roles
               and not (root/x["path"]).is_file()]
    if missing:
        raise SystemExit("Missing restricted inputs under " + str(root) + ":\n" + "\n".join(missing))
    if args.check_inputs:
        print("All required input files exist. No analysis has been run.")
        return
    if not (Path(rscript).is_file() or shutil.which(rscript)):
        raise SystemExit("Rscript was not found. Set rscript in analysis_config.json or use run_all.R.")
    core.mkdir(parents=True, exist_ok=True)
    logs = results / "logs"
    logs.mkdir(exist_ok=True)
    marker = results / ("COMPLETED_" + args.component + ".txt")
    marker.unlink(missing_ok=True)
    observed_inputs = []
    for item in inputs:
        if not set(item["components"]) & required_roles: continue
        digest = hashlib.sha256()
        with (root/item["path"]).open("rb") as handle:
            for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
        observed_inputs.append({"path":item["path"],"sha256":digest.hexdigest(),
                                "matches_frozen_input":digest.hexdigest()==item["sha256"]})
    (results/"input_versions.json").write_text(json.dumps(observed_inputs,indent=2),encoding="utf-8")
    versions = {"Python": sys.version, "platform": platform.platform(), "executable": sys.executable}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "torch", "matplotlib", "seaborn"]:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = "not installed"
    (results/"python_environment.json").write_text(json.dumps(versions,indent=2),encoding="utf-8")
    commands = []
    def run(label, command):
        print("Running " + label, flush=True)
        commands.append({"stage":label,"command":list(map(str,command))})
        (results/"run_commands.json").write_text(json.dumps(commands,indent=2),encoding="utf-8")
        with (logs/(label+".log")).open("w",encoding="utf-8") as log:
            status = subprocess.run(list(map(str,command)),cwd=REPO,env=env,
                                    stdout=log,stderr=subprocess.STDOUT).returncode
        if status:
            raise SystemExit(f"{label} failed ({status}); inspect {logs/(label+'.log')}")
    def py(label, script, *argv): run(label,[sys.executable,HERE/script,*argv])
    def r(label, script, *argv): run(label,[rscript,REPO/script,*argv])
    if run_core:
        for filename in ["01_define_36m_endpoints.py","03_extract_aligned_features.py",
                         "04_fit_leakage_controlled_models.py","05_validate_aibl_clinical_proxy.py",
                         "07_build_nonoverlap_adni_validation.py","08_crossfit_five_reader_benchmark.py",
                         "09_multireader_statistics.py"]:
            py(Path(filename).stem,filename)
        py("model_stability","final/model_stability.py",core/"development_strict126.csv",results/"model_stability")
        survival = results/"reader_survival"
        survival.mkdir(exist_ok=True)
        py("frozen_all196","final/predict_all_reader_cases.py",core/"development_strict126.csv",
           core/"adni_holdout_raw_aligned_features.csv",survival/"01_frozen_AI_all196.csv")
        py("missing_indicator_ablation","final/missing_indicator_ablation.py",
           core/"development_strict126.csv",core/"nonoverlapping_adni_validation_predictions.csv",
           core/"leakage_free_model_summary.json",results/"missing_indicator_ablation")
        r("final_reader_and_calibration","analysis_pipeline/final/run_statistics.R")
        py("frozen_aggregate_comparison","final/check_frozen_results.py")
    if run_vae:
        if not (core/"adni_discovery_endpoints.csv").is_file():
            raise SystemExit("Run core before the VAE endpoint analyses")
        py("prepare_fixed_vae_inputs","final/prepare_vae_inputs.py")
        vae = results/"vae_inputs"
        env["AD_MULTIMODAL_SUBTYPE_FILE"] = str(vae/"subtype_assignments.csv")
        py("vae_adjusted_sensitivity","06_vae_sensitivity_analysis.py")
        discovery = root/"Derived_Inputs/Discovery_CSF_Cohort"
        r("vae_signatures","2_discovery_characterization/Cluster_signatures.R",
          "--vae_dir",vae,"--data_dir",discovery,"--output_dir",results/"vae_signatures")
        r("vae_cluster_validation","2_discovery_characterization/Cluster_validation.R",
          "--vae_dir",vae,"--data_dir",discovery,"--output_dir",results/"vae_validation")
        r("vae_neuroimaging","2_discovery_characterization/Neuroimaging_endotypes.R",
          "--vae_dir",vae,"--data_dir",discovery,"--output_dir",results/"vae_neuroimaging")
        r("vae_conversion_contrasts","2_discovery_characterization/Conversion_differential.R",
          "--outcome_file",discovery/"Womac_score_pain_function.csv",
          "--clinical_file",discovery/"Clinical_data.csv","--smri_file",discovery/"RNA_plasma.csv",
          "--csf_file",discovery/"metabolites.csv","--output_dir",results/"conversion_contrasts")
        run("vae_biomarkers",[sys.executable,REPO/"2_discovery_characterization/Biomarker_validation.py",
          "--subtype_file",vae/"subtype_assignments.csv","--csf_file",discovery/"metabolites.csv",
          "--output_dir",results/"vae_biomarkers"])
        r("vae_real_followup","analysis_pipeline/final/vae_followup.R")
        r("vae_bsi","2_discovery_characterization/Cross_modal_validation.R",
          "--vae_dir",vae,"--clinical_file",discovery/"Clinical_data.csv",
          "--bsi_file",root/"ADNI_Raw_Data/BSI/FOXLABBSI_02Mar2026.csv",
          "--output_dir",results/"vae_bsi")
    if run_external:
        r("aibl_three_feature","4_external_contextualization/AIBL reduced-feature external validation.R")
        r("a4_nominal_classes","4_external_contextualization/A4_validation.R")
        r("habs_hd","4_external_contextualization/HABS_validation.R")
        r("extension_missingness_precision","analysis_pipeline/final/external_context_summary.R")
        py("extension_aggregate_comparison","final/check_frozen_results.py","--component","extensions")
    if run_figures:
        r("figure3","analysis_pipeline/final/figure3.R")
        r("reader_figures","analysis_pipeline/final/reader_figures.R")
    marker.write_text(
        f"Completed component: {args.component}\nSee run_commands.json and logs.\n",encoding="utf-8")
    print("Completed. Results: " + str(results))


if __name__ == "__main__":
    main()
