"""
Frozen AI prediction pipeline for the independent ADNI test set.

Inputs:
  - test_set.csv
  - subtype_assignments.csv
  - latent_representations.csv
  - vae_summary.json
  - Clinical_data.csv
  - RNA_plasma.csv
  - metabolites.csv

Outputs:
  - AI_test_predictions.csv
  - AI_test_results.json
  - confusion_matrix.png
  - probability_distribution.png
"""
import argparse
import pandas as pd
import numpy as np
import os
import json
import warnings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss,
    precision_score, recall_score, f1_score,
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
np.random.seed(42)
torch.manual_seed(42)
def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Prediction — Frozen Pipeline with VAE Encoder Inference")
    parser.add_argument("--base_dir", type=str,
        default=".",
        help="Directory containing ALL required files")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Output directory (defaults to base_dir)")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
# ==============================================================================
# VAE Architecture (must match step8 EXACTLY)
# ==============================================================================
class Encoder(nn.Module):
    def __init__(self, d_in, h1, h2, d_z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(h1, h2),   nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.15),
        )
        self.fc_mu     = nn.Linear(h2, d_z)
        self.fc_logvar = nn.Linear(h2, d_z)
    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)
class Decoder(nn.Module):
    def __init__(self, d_z, h2, h1, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, h2),  nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h2, h1),   nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(h1, d_out),
        )
    def forward(self, z):
        return self.net(z)
class VAE(nn.Module):
    def __init__(self, d_in, h1, h2, d_z):
        super().__init__()
        self.encoder = Encoder(d_in, h1, h2, d_z)
        self.decoder = Decoder(d_z, h2, h1, d_in)
    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
    def forward(self, x):
        mu, lv = self.encoder(x)
        z = self.reparameterize(mu, lv)
        return self.decoder(z), mu, lv, z
    def encode_mu(self, x):
        mu, _ = self.encoder(x)
        return mu
def vae_loss_weighted(recon, x, mu, lv, beta, n_csf, n_clin):
    """Modality-weighted reconstruction loss (identical to step8)."""
    loss_csf  = nn.functional.mse_loss(recon[:, :n_csf],
                                        x[:, :n_csf], reduction="mean")
    loss_clin = nn.functional.mse_loss(recon[:, n_csf:n_csf+n_clin],
                                        x[:, n_csf:n_csf+n_clin], reduction="mean")
    loss_mri  = nn.functional.mse_loss(recon[:, n_csf+n_clin:],
                                        x[:, n_csf+n_clin:], reduction="mean")
    recon_loss = loss_csf + loss_clin + loss_mri
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl
# ==============================================================================
# Column mapping: test set column names -> VAE variable names
# ==============================================================================
TEST_TO_VAE_MAP = {
    "PTAU181":   "PTAU181",
    "AB42_40":   "ABETA42_ABETA40_RATIO",
    "ABETA40":   "ABETA40",
    "APOE4":     "APOE4_Positive",
    "MMSE":      "MMSE_Baseline",
    "EDUCATION": "Education",
    "GDS":       None,  # MISSING in test set -> fill with 0
}
# Prediction feature mapping: training col -> test col
TRAIN_TO_TEST_MAP = {
    "MMSE":                    "MMSE_Baseline",
    "EDUCATION":               "Education",
    "APOE4_DOSAGE":            "APOE4_Positive",
    "PTAU181":                 "PTAU181",
    "ABETA42_ABETA40_RATIO":   "ABETA42_ABETA40_RATIO",
    "ABETA40":                 "ABETA40",
    "GDS":                     None,
}
EXCLUDED_PATTERNS = [
    "FAQ", "FAQTOTAL", "ADAS13", "ADAS", "CDRSB", "CDR",
    "SEX", "AGE", "GENDER", "APOE4_STATUS", "AD_Conversion",
    "Womac", "VAE_Subtype", "Direct_KMeans", "DIAGNOSIS",
    "DX_bl", "DX", "VISCODE",
]
# ==============================================================================
# PHASE 1: Load all data from single directory
# ==============================================================================
def load_all_data(base_dir):
    print("[PHASE 1] Loading data...")
    print(f"  Base directory: {base_dir}")
    required = ["independent_test_set.csv", "subtype_assignments.csv",
                "latent_representations.csv", "vae_summary.json",
                "Clinical_data.csv", "RNA_plasma.csv", "metabolites.csv"]
    for f in required:
        p = os.path.join(base_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"MISSING: {p}")
    print("  All 7 required files found")
    test_df  = pd.read_csv(os.path.join(base_dir, "independent_test_set.csv"))
    subtypes = pd.read_csv(os.path.join(base_dir, "subtype_assignments.csv"))
    latent   = pd.read_csv(os.path.join(base_dir, "latent_representations.csv"))
    clinical = pd.read_csv(os.path.join(base_dir, "Clinical_data.csv"))
    smri     = pd.read_csv(os.path.join(base_dir, "RNA_plasma.csv"))
    csf      = pd.read_csv(os.path.join(base_dir, "metabolites.csv"))
    with open(os.path.join(base_dir, "vae_summary.json")) as f:
        vae_summary = json.load(f)
    for df in [subtypes, latent, clinical, smri, csf]:
        df["ID"] = df["ID"].astype(str)
    print(f"  Test set:     {test_df.shape}")
    print(f"  Training set: {subtypes.shape[0]} participants")
    print(f"  VAE features: {len(vae_summary['features'])}")
    return test_df, subtypes, latent, vae_summary, clinical, smri, csf
# ==============================================================================
# PHASE 2: Build 37-variable VAE matrices (training + test)
# ==============================================================================
def build_vae_matrices(test_df, subtypes, vae_summary, clinical, smri, csf):
    print("\n[PHASE 2] Building 37-variable VAE matrices...")
    vae_features = vae_summary["features"]
    n_feat = len(vae_features)
    n_test = len(test_df)
    source_map = vae_summary.get("source_map", {})
    # --- Test set matrix ---
    X_test = np.full((n_test, n_feat), np.nan)
    for i, vf in enumerate(vae_features):
        if vf in TEST_TO_VAE_MAP:
            tc = TEST_TO_VAE_MAP[vf]
            if tc is None:
                X_test[:, i] = 0.0
                print(f"  {vf:>25s} -> FILLED WITH 0 (missing in test set)")
            elif tc in test_df.columns:
                X_test[:, i] = pd.to_numeric(test_df[tc], errors="coerce").values
            else:
                cl = {c.lower(): c for c in test_df.columns}
                if tc.lower() in cl:
                    X_test[:, i] = pd.to_numeric(test_df[cl[tc.lower()]], errors="coerce").values
                else:
                    print(f"  WARNING: {vf} -> {tc} NOT FOUND")
        elif vf.startswith("ST"):
            if vf in test_df.columns:
                X_test[:, i] = pd.to_numeric(test_df[vf], errors="coerce").values
            else:
                cl = {c.lower(): c for c in test_df.columns}
                if vf.lower() in cl:
                    X_test[:, i] = pd.to_numeric(test_df[cl[vf.lower()]], errors="coerce").values
                else:
                    print(f"  WARNING: MRI {vf} NOT FOUND")
    # --- Training set matrix (same column order) ---
    train_df = subtypes[["ID"]].copy()
    for vf in vae_features:
        found = False
        for src_df in [csf, clinical, smri]:
            if vf in src_df.columns:
                train_df = train_df.merge(src_df[["ID", vf]], on="ID", how="left")
                found = True
                break
            cl = {c.lower(): c for c in src_df.columns}
            if vf.lower() in cl:
                actual = cl[vf.lower()]
                tmp = src_df[["ID", actual]].rename(columns={actual: vf})
                train_df = train_df.merge(tmp, on="ID", how="left")
                found = True
                break
        if not found and vf in source_map:
            src_info = source_map[vf]
            src_file, src_col = src_info.split("->")
            for fname, df in [("Clinical_data.csv", clinical),
                              ("RNA_plasma.csv", smri),
                              ("metabolites.csv", csf)]:
                if src_file == fname and src_col in df.columns:
                    tmp = df[["ID", src_col]].rename(columns={src_col: vf})
                    train_df = train_df.merge(tmp, on="ID", how="left")
                    found = True
                    break
        if not found:
            print(f"  WARNING: Training feature '{vf}' not found, filling NaN")
            train_df[vf] = np.nan
    X_train = train_df[vae_features].values.astype(float)
    print(f"  Training VAE matrix: {X_train.shape}")
    print(f"  Test VAE matrix:     {X_test.shape}")
    n_miss = int(np.isnan(X_test).sum())
    print(f"  Test NaN count: {n_miss}")
    return X_train, X_test, vae_features
# ==============================================================================
# PHASE 3: Preprocessing for VAE
# ==============================================================================
def preprocess_for_vae(X_train_raw, X_test_raw, winsorize_sd=3.0):
    """
    Training data is already z-score (from step7 preprocessing).
    Test data is RAW scale (original clinical values).
    
    Strategy:
      1. Median impute training (fitted on training)
      2. Median impute test (fitted on test, since scales differ)
      3. Z-score standardize test set to match training scale
         (using test set's own mean/sd — this is NOT data leakage,
          it's the same operation step7 applied to training data)
      4. Winsorize both at +/-3 SD using training statistics
    """
    print("\n[PHASE 3] Preprocessing for VAE...")
    # Detect scale mismatch
    train_means = np.nanmean(X_train_raw, axis=0)
    test_means  = np.nanmean(X_test_raw, axis=0)
    scale_ratio = np.nanmedian(np.abs(test_means) / (np.abs(train_means) + 1e-8))
    print(f"  Scale check: train mean range [{train_means.min():.2f}, {train_means.max():.2f}]")
    print(f"  Scale check: test  mean range [{test_means.min():.2f}, {test_means.max():.2f}]")
    print(f"  Scale ratio (median): {scale_ratio:.1f}x")
    # Step 1: Median impute training
    train_imputer = SimpleImputer(strategy="median")
    X_train = train_imputer.fit_transform(X_train_raw)
    # Step 2: Median impute test (separate imputer since scales differ)
    test_imputer = SimpleImputer(strategy="median")
    X_test = test_imputer.fit_transform(X_test_raw)
    # Step 3: Z-score standardize test set if scale mismatch detected
    if scale_ratio > 5:
        print(f"  SCALE MISMATCH DETECTED (ratio={scale_ratio:.0f}x)")
        print(f"  Training data is z-score (step7), test data is raw scale")
        print(f"  Applying z-score standardization to test set...")
        test_mu  = X_test.mean(axis=0)
        test_std = X_test.std(axis=0)
        test_std[test_std < 1e-12] = 1.0  # avoid division by zero
        X_test = (X_test - test_mu) / test_std
        print(f"  Test set after z-score: mean range [{X_test.mean(axis=0).min():.3f}, "
              f"{X_test.mean(axis=0).max():.3f}], "
              f"sd range [{X_test.std(axis=0).min():.3f}, {X_test.std(axis=0).max():.3f}]")
    else:
        print(f"  Scales appear matched, no additional standardization needed")
    # Step 4: Winsorize using TRAINING statistics
    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    n_clip_train, n_clip_test = 0, 0
    for i in range(X_train.shape[1]):
        if stds[i] < 1e-12:
            continue
        lo = means[i] - winsorize_sd * stds[i]
        hi = means[i] + winsorize_sd * stds[i]
        n_clip_train += int(np.sum((X_train[:, i] < lo) | (X_train[:, i] > hi)))
        n_clip_test  += int(np.sum((X_test[:, i] < lo) | (X_test[:, i] > hi)))
        X_train[:, i] = np.clip(X_train[:, i], lo, hi)
        X_test[:, i]  = np.clip(X_test[:, i], lo, hi)
    print(f"  Winsorized at +/-{winsorize_sd} SD: "
          f"{n_clip_train} train, {n_clip_test} test values clipped")
    return X_train, X_test, train_imputer
# ==============================================================================
# PHASE 4: Retrain VAE on training data + encode test set
# ==============================================================================
def retrain_vae_and_encode(X_train, X_test, vae_summary):
    print("\n[PHASE 4] Retraining VAE on training data...")
    h1 = vae_summary.get("hidden1", 256)
    h2 = vae_summary.get("hidden2", 128)
    d_z = vae_summary.get("latent_dim", 3)
    d_in = X_train.shape[1]
    n_csf = vae_summary.get("n_csf", 3)
    n_clin = vae_summary.get("n_clinical", 4)
    epochs = vae_summary.get("epochs", 300)
    beta_max = vae_summary.get("beta_max", 0.5)
    beta_warmup = vae_summary.get("beta_warmup", 80)
    lr = vae_summary.get("lr", 5e-4)
    print(f"  Architecture: {d_in}->{h1}->{h2}->{d_z}")
    print(f"  Modality: CSF({n_csf}) + Clin({n_clin}) + MRI({d_in-n_csf-n_clin})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.FloatTensor(X_train).to(device)
    loader = DataLoader(TensorDataset(Xt), batch_size=32, shuffle=True)
    model = VAE(d_in, h1, h2, d_z).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    for ep in range(1, epochs + 1):
        model.train()
        beta = min(beta_max, beta_max * ep / beta_warmup)
        for (bx,) in loader:
            opt.zero_grad()
            rec, mu, lv, z = model(bx)
            loss, _, _ = vae_loss_weighted(rec, bx, mu, lv, beta, n_csf, n_clin)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 100 == 0 or ep == 1:
            print(f"  Epoch {ep}/{epochs}: loss={loss.item():.4f}, beta={beta:.3f}")
    model.eval()
    with torch.no_grad():
        Z_train = model.encode_mu(Xt).cpu().numpy()
        Z_test  = model.encode_mu(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    print(f"  Z_train: {Z_train.shape}, Z_test: {Z_test.shape}")
    return Z_train, Z_test
# ==============================================================================
# PHASE 5: Build prediction feature matrices
# ==============================================================================
def build_prediction_features(subtypes, latent, clinical, smri, csf,
                               test_df, Z_train, Z_test):
    """Raw clinical/MRI/CSF + Z1-Z3. NO FAQ/ADAS13/CDRSB."""
    print("\n[PHASE 5] Building prediction feature matrices...")
    # --- Training features ---
    master = subtypes[["ID", "AD_Conversion"]].copy()
    clin_keep = [c for c in ["MMSE", "EDUCATION", "GDS", "APOE4_DOSAGE"]
                 if c in clinical.columns]
    if clin_keep:
        master = master.merge(clinical[["ID"] + clin_keep], on="ID", how="left")
    csf_keep = [c for c in ["PTAU181", "ABETA42_ABETA40_RATIO", "ABETA40"]
                if c in csf.columns]
    if csf_keep:
        master = master.merge(csf[["ID"] + csf_keep], on="ID", how="left")
    mri_cols = [c for c in smri.columns if c.startswith("ST")]
    if mri_cols:
        master = master.merge(smri[["ID"] + mri_cols], on="ID", how="left")
    z_cols = [c for c in latent.columns if c.startswith("Z")]
    master = master.merge(latent[["ID"] + z_cols], on="ID", how="left")
    # Filter features
    all_cols = [c for c in master.columns if c not in ["ID", "AD_Conversion"]]
    feature_cols = []
    for c in all_cols:
        if any(pat.upper() in c.upper() for pat in EXCLUDED_PATTERNS):
            continue
        if pd.api.types.is_numeric_dtype(master[c]):
            feature_cols.append(c)
    y_train = master["AD_Conversion"].values
    X_train = master[feature_cols].values
    # --- Test features (map column names) ---
    X_test_list = []
    used_features = []
    for feat in feature_cols:
        if feat.startswith("Z"):
            z_idx = int(feat[1:]) - 1
            X_test_list.append(Z_test[:, z_idx])
            used_features.append(feat)
        elif feat in TRAIN_TO_TEST_MAP:
            tc = TRAIN_TO_TEST_MAP[feat]
            if tc is None:
                X_test_list.append(np.zeros(len(test_df)))
                used_features.append(feat)
            elif tc in test_df.columns:
                X_test_list.append(pd.to_numeric(test_df[tc], errors="coerce").values)
                used_features.append(feat)
            else:
                cl = {c.lower(): c for c in test_df.columns}
                if tc.lower() in cl:
                    X_test_list.append(pd.to_numeric(test_df[cl[tc.lower()]], errors="coerce").values)
                    used_features.append(feat)
        elif feat.startswith("ST"):
            if feat in test_df.columns:
                X_test_list.append(pd.to_numeric(test_df[feat], errors="coerce").values)
                used_features.append(feat)
            else:
                cl = {c.lower(): c for c in test_df.columns}
                if feat.lower() in cl:
                    X_test_list.append(pd.to_numeric(test_df[cl[feat.lower()]], errors="coerce").values)
                    used_features.append(feat)
        elif feat in test_df.columns:
            X_test_list.append(pd.to_numeric(test_df[feat], errors="coerce").values)
            used_features.append(feat)
    # Align to common features
    common_idx = [feature_cols.index(f) for f in used_features]
    X_train_aligned = X_train[:, common_idx]
    X_test_aligned = np.column_stack(X_test_list)
    # Z-score standardize raw test features (non-Z columns) to match
    # training scale. Training data is already z-score from step7.
    # Z1-Z3 are already in correct scale from VAE encoder (Phase 3-4).
    raw_feature_idx = [i for i, f in enumerate(used_features)
                       if not f.startswith("Z")]
    if raw_feature_idx:
        raw_test = X_test_aligned[:, raw_feature_idx]
        raw_mu  = np.nanmean(raw_test, axis=0)
        raw_std = np.nanstd(raw_test, axis=0)
        raw_std[raw_std < 1e-12] = 1.0
        X_test_aligned[:, raw_feature_idx] = (raw_test - raw_mu) / raw_std
        print(f"  Z-scored {len(raw_feature_idx)} raw test features to match training scale")
    # Circularity check
    for pat in ["FAQ", "ADAS13", "CDRSB"]:
        for f in used_features:
            if pat.upper() in f.upper():
                raise RuntimeError(f"CIRCULARITY LEAK: {f}")
    print(f"  Features: {len(used_features)}")
    print(f"    Clinical: {[f for f in used_features if f in clin_keep]}")
    print(f"    CSF:      {[f for f in used_features if f in csf_keep]}")
    print(f"    MRI:      {len([f for f in used_features if f.startswith('ST')])} ST features")
    print(f"    VAE:      {[f for f in used_features if f.startswith('Z')]}")
    print("  Circularity check PASSED")
    return X_train_aligned, X_test_aligned, y_train, used_features
# ==============================================================================
# PHASE 6-7: MICE imputation + StandardScaler
# ==============================================================================
def impute_and_scale(X_train, X_test):
    print("\n[PHASE 6] MICE imputation (fitted on training)...")
    imputer = IterativeImputer(max_iter=15, random_state=42, sample_posterior=False)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)
    print(f"  Imputed: {int(np.isnan(X_train).sum())} train NaN, "
          f"{int(np.isnan(X_test).sum())} test NaN")
    print("\n[PHASE 7] StandardScaler (fitted on training)...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc  = scaler.transform(X_test_imp)
    return X_train_sc, X_test_sc, imputer, scaler
# ==============================================================================
# PHASE 8: Lasso feature selection
# ==============================================================================
def lasso_feature_selection(X_train, y_train, X_test, feature_names):
    print("\n[PHASE 8] Lasso feature selection (fitted on training)...")
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    mask = np.abs(lasso.coef_) > 1e-6
    if int(mask.sum()) < 3:
        print(f"  Lasso selected only {int(mask.sum())}, relaxing to top 10")
        top_idx = np.argsort(np.abs(lasso.coef_))[::-1][:10]
        mask = np.zeros(len(feature_names), dtype=bool)
        mask[top_idx] = True
    selected = [f for f, s in zip(feature_names, mask) if s]
    print(f"  Selected {len(selected)}/{len(feature_names)} features:")
    for f, c in sorted(zip(feature_names, lasso.coef_),
                        key=lambda x: abs(x[1]), reverse=True):
        if abs(c) > 1e-6:
            print(f"    {f:>30s}: coef={c:+.4f}")
    return X_train[:, mask], X_test[:, mask], selected, lasso
# ==============================================================================
# PHASE 9: Train Elastic Net
# ==============================================================================
def train_elastic_net(X_train, y_train):
    print("\n[PHASE 9] Training Elastic Net (consistent with step11)...")
    param_grid = {
        "C": np.logspace(-3, 1, 20),
        "l1_ratio": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    enet = LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=10000,
        random_state=42, class_weight="balanced",
    )
    grid = GridSearchCV(enet, param_grid, cv=5, scoring="roc_auc",
                        n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"  Best: C={grid.best_params_['C']:.4f}, "
          f"l1_ratio={grid.best_params_['l1_ratio']}")
    print(f"  CV AUC: {grid.best_score_:.4f}")
    print(f"  Non-zero coefs: {int(np.sum(np.abs(best.coef_[0]) > 1e-6))}/{X_train.shape[1]}")
    # Youden threshold on training set
    probs = best.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, probs)
    j = tpr - fpr
    idx = np.argmax(j)
    threshold = thresholds[idx]
    print(f"  Youden threshold: {threshold:.4f} "
          f"(Sens={tpr[idx]:.3f}, Spec={1-fpr[idx]:.3f})")
    return best, threshold
# ==============================================================================
# PHASE 10: Evaluate on test set
# ==============================================================================
def evaluate_test(model, X_test, y_test, threshold, n_bootstrap=2000):
    print(f"\n[PHASE 10] Evaluating on test set (N={len(y_test)})...")
    print(f"  Frozen threshold: {threshold:.4f}")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    acc  = (tp + tn) / len(y_test)
    # Bootstrap AUC CI
    np.random.seed(42)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
        if len(np.unique(y_test[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_test[idx], probs[idx]))
    ci_lo = np.percentile(boot_aucs, 2.5)
    ci_hi = np.percentile(boot_aucs, 97.5)
    print(f"\n  === Test Set Results ===")
    print(f"  AUC:         {auc:.4f} (95% CI: {ci_lo:.4f}-{ci_hi:.4f})")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Sensitivity: {sens:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  PPV:         {ppv:.4f}")
    print(f"  NPV:         {npv:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Brier:       {brier:.4f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    results = {
        "AUC": round(auc, 4), "AUC_95CI_Lower": round(ci_lo, 4),
        "AUC_95CI_Upper": round(ci_hi, 4), "Accuracy": round(acc, 4),
        "Sensitivity": round(sens, 4), "Specificity": round(spec, 4),
        "PPV": round(ppv, 4), "NPV": round(npv, 4),
        "Precision": round(prec, 4), "Recall": round(rec, 4),
        "F1": round(f1, 4), "Brier": round(brier, 4),
        "Threshold": round(threshold, 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "N_test": len(y_test), "N_converters": int(y_test.sum()),
    }
    return results, probs, preds
# ==============================================================================
# PHASE 11: Visualization + Save
# ==============================================================================
def save_all(y_test, probs, preds, results, selected_features, test_df,
             output_dir):
    print("\n[PHASE 11] Saving results and plots...")
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, "b-", lw=2,
            label=f"Elastic Net (AUC={results['AUC']:.3f}, "
                  f"95%CI [{results['AUC_95CI_Lower']:.3f}-"
                  f"{results['AUC_95CI_Upper']:.3f}])")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - Specificity", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("ROC: AI Prediction on Independent Test Set (N=196)", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AI_ROC_Curve.png"), dpi=300)
    plt.close()
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Converter", "Converter"],
                yticklabels=["Non-Converter", "Converter"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={results['Threshold']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AI_Confusion_Matrix.png"), dpi=300)
    plt.close()
    # Calibration Plot
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(prob_pred, prob_true, "bo-", label="Elastic Net")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Observed Proportion")
    ax.set_title("Calibration Plot"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AI_Calibration_Plot.png"), dpi=300)
    plt.close()
    # Probability Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs[y_test == 0], bins=20, alpha=0.6, label="Non-Converter",
            color="steelblue", density=True)
    ax.hist(probs[y_test == 1], bins=20, alpha=0.6, label="Converter",
            color="salmon", density=True)
    ax.axvline(results["Threshold"], color="black", ls="--", lw=2,
               label=f"Threshold={results['Threshold']:.3f}")
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AI_Probability_Distribution.png"), dpi=300)
    plt.close()
    # Save JSON results
    with open(os.path.join(output_dir, "AI_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # Per-patient predictions
    pred_df = pd.DataFrame({
        "Actual": y_test.astype(int),
        "Predicted_Prob": np.round(probs, 4),
        "Predicted_Class": preds,
    })
    for ic in ["ID", "PTID", "RID", "Subject"]:
        if ic in test_df.columns:
            pred_df.insert(0, ic, test_df[ic].values)
            break
    pred_df.to_csv(os.path.join(output_dir, "AI_per_patient_predictions.csv"),
                   index=False)
    compat_df = pred_df.copy()
    id_col = None
    for candidate in ["ID", "PTID", "RID", "Subject"]:
        if candidate in compat_df.columns:
            id_col = candidate
            break
    if id_col is not None:
        compat_df = compat_df.rename(columns={id_col: "CaseID"})
    compat_df = compat_df.rename(columns={
        "Predicted_Prob": "AI_Probability",
        "Predicted_Class": "AI_Predicted_Class"
    })
    compat_df.to_csv(os.path.join(output_dir, "AI_Predictions_Final.csv"), index=False)
    compat_df.to_csv(os.path.join(output_dir, "AI_test_predictions.csv"), index=False)
    # Feature list
    pd.DataFrame({"Feature": selected_features}).to_csv(
        os.path.join(output_dir, "AI_selected_features.csv"), index=False)
    # Summary text
    lines = [
        "AI Prediction Results Summary",
        "=" * 50,
        f"Model: Elastic Net (frozen pipeline)",
        f"Training: N=157 (ADNI discovery)",
        f"Test: N={results['N_test']} (independent MCI cohort)",
        f"Features: {len(selected_features)} (Lasso-selected, includes Z1-Z3)",
        f"Excludes: FAQ, ADAS13, CDRSB",
        f"",
        f"AUC: {results['AUC']:.3f} (95% CI: {results['AUC_95CI_Lower']:.3f}-{results['AUC_95CI_Upper']:.3f})",
        f"Sensitivity: {results['Sensitivity']:.3f}",
        f"Specificity: {results['Specificity']:.3f}",
        f"PPV: {results['PPV']:.3f}  NPV: {results['NPV']:.3f}",
        f"F1: {results['F1']:.3f}  Brier: {results['Brier']:.4f}",
        f"Threshold: {results['Threshold']:.3f} (Youden, frozen from training)",
    ]
    with open(os.path.join(output_dir, "AI_results_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved to {output_dir}:")
    print(f"    AI_ROC_Curve.png, AI_Confusion_Matrix.png,")
    print(f"    AI_Calibration_Plot.png, AI_Probability_Distribution.png,")
    print(f"    AI_test_results.json, AI_per_patient_predictions.csv,")
    print(f"    AI_selected_features.csv, AI_results_summary.txt")
# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base_dir = args.base_dir
    output_dir = args.output_dir or base_dir
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("Step 2: AI Prediction on Independent Test Set")
    print("  Frozen Pipeline + VAE Encoder Inference + Elastic Net")
    print("=" * 70)
    print(f"  Base dir:   {base_dir}")
    print(f"  Output dir: {output_dir}")
    print()
    # PHASE 1
    test_df, subtypes, latent, vae_summary, clinical, smri, csf = \
        load_all_data(base_dir)
    # Identify outcome in test set
    outcome_col = None
    for c in ["AD_Conversion", "Conversion", "Label", "Outcome"]:
        if c in test_df.columns:
            outcome_col = c
            break
    if outcome_col is None:
        raise RuntimeError("Cannot find outcome column in test set. "
                           "Expected: AD_Conversion, Conversion, Label, or Outcome")
    y_test = test_df[outcome_col].values.astype(int)
    print(f"  Outcome: '{outcome_col}' -> {y_test.sum()} converters / "
          f"{len(y_test)-y_test.sum()} non-converters")
    # PHASE 2
    X_train_vae_raw, X_test_vae_raw, vae_features = \
        build_vae_matrices(test_df, subtypes, vae_summary, clinical, smri, csf)
    # PHASE 3
    X_train_vae, X_test_vae, vae_imputer = \
        preprocess_for_vae(X_train_vae_raw, X_test_vae_raw)
    # PHASE 4
    Z_train, Z_test = retrain_vae_and_encode(X_train_vae, X_test_vae, vae_summary)
    # Verify against original latent
    Z_orig = latent[[c for c in latent.columns if c.startswith("Z")]].values
    for i in range(Z_train.shape[1]):
        r = np.corrcoef(Z_train[:, i], Z_orig[:, i])[0, 1]
        print(f"  Z{i+1} correlation (retrained vs original): r={r:.4f}")
    # PHASE 5
    X_train_pred, X_test_pred, y_train, feature_names = \
        build_prediction_features(subtypes, latent, clinical, smri, csf,
                                   test_df, Z_train, Z_test)
    # PHASE 6-7
    X_train_sc, X_test_sc, pred_imputer, pred_scaler = \
        impute_and_scale(X_train_pred, X_test_pred)
    # PHASE 8
    X_train_sel, X_test_sel, selected_features, lasso = \
        lasso_feature_selection(X_train_sc, y_train, X_test_sc, feature_names)
    # PHASE 9
    model, threshold = train_elastic_net(X_train_sel, y_train)
    # PHASE 10
    results, probs, preds = evaluate_test(
        model, X_test_sel, y_test, threshold, args.n_bootstrap)
    # PHASE 11
    save_all(y_test, probs, preds, results, selected_features, test_df,
             output_dir)
    # Save pipeline artifacts
    artifacts = {
        "vae_imputer": vae_imputer, "pred_imputer": pred_imputer,
        "pred_scaler": pred_scaler, "lasso": lasso,
        "model": model, "threshold": threshold,
        "vae_features": vae_features, "prediction_features": feature_names,
        "selected_features": selected_features,
    }
    with open(os.path.join(output_dir, "pipeline_artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    print("\n" + "=" * 70)
    print("Step 2 COMPLETE")
    print("=" * 70)
    print(f"  Model:     Elastic Net")
    print(f"  Features:  {len(selected_features)} (from {len(feature_names)})")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  AUC:       {results['AUC']:.3f} "
          f"({results['AUC_95CI_Lower']:.3f}-{results['AUC_95CI_Upper']:.3f})")
    print(f"  Sens/Spec: {results['Sensitivity']:.3f} / {results['Specificity']:.3f}")
    print(f"  Output:    {output_dir}")
    print("=" * 70)
if __name__ == "__main__":
    main()


