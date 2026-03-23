"""
Modality-weighted multimodal VAE clustering for AD subtype discovery.

Inputs:
  - Clinical_data.csv
  - metabolites.csv
  - RNA_plasma.csv

Outputs:
  - subtype_assignments.csv
  - latent_representations.csv
  - vae_summary.json
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, os, warnings, pickle, json
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# ===================== Arguments =====================
parser = argparse.ArgumentParser(description="Modality-Weighted Multimodal VAE Clustering")
parser.add_argument("--cohort",      type=str,   default="A")
parser.add_argument("--input_dir",   type=str,   default=".")
parser.add_argument("--output_dir",  type=str,   default=None)
parser.add_argument("--n_clusters",  type=int,   default=3)
parser.add_argument("--latent_dim",  type=int,   default=3)
parser.add_argument("--epochs",      type=int,   default=300)
parser.add_argument("--batch_size",  type=int,   default=32)
parser.add_argument("--hidden1",     type=int,   default=256)
parser.add_argument("--hidden2",     type=int,   default=128)
parser.add_argument("--lr",          type=float, default=5e-4)
parser.add_argument("--beta_max",    type=float, default=0.5)
parser.add_argument("--beta_warmup", type=int,   default=80)
parser.add_argument("--winsorize_sd",type=float, default=3.0)
args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.output_dir is None:
    args.output_dir = os.path.join(args.input_dir, "vae_revised_output")
os.makedirs(args.output_dir, exist_ok=True)
# ===================== Variable Definitions =====================
# Feature ordering in VAE matrix: CSF first, then Clinical, then MRI.
# This ordering is critical for the modality-weighted loss function.
# --- CSF biomarkers retained after quality control ---
# Exclusion criterion: variables with >50% missingness are removed because
# median imputation at such rates produces near-constant values that
# contribute no meaningful variance to latent space learning.
# Excluded: ABETA42 (72.0%), TAU_TOTAL (52.9%), STREM2 (68.8%), PGRN (63.7%)
# Retained: PTAU181 (22.3%), AB42_40 (0%), ABETA40 (44.6%)
CSF_ALIASES = {
    "PTAU181":   ["ptau181", "ptau", "p_tau181"],
    "AB42_40":   ["abeta42_abeta40_ratio", "ab42_40", "abeta42_40"],
    "ABETA40":   ["abeta40", "ab40"],
}
# CSF variables excluded due to >50% missingness (kept for post-hoc reporting)
CSF_EXCLUDED_HIGH_MISS = {
    "ABETA42":   (["abeta42", "ab42"], 72.0),
    "TAU_TOTAL": (["tau_total", "tau", "total_tau", "t_tau"], 52.9),
    "STREM2":    (["strem2", "trem2"], 68.8),
    "PGRN":      (["pgrn", "progranulin"], 63.7),
}
CLINICAL_ALIASES = {
    "APOE4":     ["apoe4_dosage"],
    "MMSE":      ["mmse"],
    "EDUCATION": ["education", "pteducat", "years_education"],
    "GDS":       ["gds", "gdstotal", "gds_total"],
}
# MRI: all ST*** columns from RNA_plasma.csv (discovered dynamically)
DEMO_ALIASES = {
    "SEX": ["sex", "ptgender", "gender"],
    "AGE": ["age", "age_at_visit", "age_bl"],
}
EXCLUDED_VARS = {"ADAS13", "CDRSB", "FAQTOTAL", "FAQ", "SEX", "AGE",
                 "APOE4_STATUS", "AD_Conversion", "Womac.Pain", "Womac.Function",
                 "ABETA42", "TAU_TOTAL", "STREM2", "PGRN"}
# ===================== Helpers =====================
def find_col(df, candidates):
    """Case-insensitive column lookup."""
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None
def get_id_col(df):
    for c in ["ID", "PTID", "RID", "Subject", "id"]:
        if c in df.columns:
            return c
    return None
# ===================== Data Loading =====================
def load_data(input_dir):
    print("=" * 70)
    print("MODALITY-WEIGHTED MULTIMODAL VAE CLUSTERING (37 variables)")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Device: {DEVICE}  |  Latent: {args.latent_dim}  |  K: {args.n_clusters}")
    n_total = len(CSF_ALIASES) + len(CLINICAL_ALIASES) + 30  # approximate
    print(f"Arch:   {n_total}->{args.hidden1}->{args.hidden2}->{args.latent_dim}")
    print(f"Beta:   warmup={args.beta_warmup}, max={args.beta_max}")
    print(f"Winsorize: {args.winsorize_sd} SD  |  Log1p: DISABLED (z-score input)")
    print(f"StandardScaler: DISABLED (z-score input)")
    print(f"Loss:   MODALITY-WEIGHTED (CSF:Clin:MRI = 1:1:1)")
    print(f"CSF QC: excluded 4 variables with >50% missingness")
    print()
    print("[1/8] Loading CSV files...")
    csvs = {}
    for f in sorted(os.listdir(input_dir)):
        if not f.endswith(".csv"):
            continue
        path = os.path.join(input_dir, f)
        try:
            df = pd.read_csv(path)
            csvs[f] = df
            print(f"  {f}: {df.shape[0]}x{df.shape[1]}  {list(df.columns)}")
        except Exception as e:
            print(f"  WARNING: {f} -> {e}")
    # ---- Locate CSF biomarkers ----
    csf_found = {}
    for var, aliases in CSF_ALIASES.items():
        for fname, df in csvs.items():
            col = find_col(df, aliases)
            idc = get_id_col(df)
            if col and idc:
                csf_found[var] = (fname, col, idc)
                print(f"  CSF  {var}: {fname} -> '{col}'")
                break
        if var not in csf_found:
            raise SystemExit(f"FATAL: CSF variable {var} not found in any CSV")
    # Report excluded CSF variables
    print(f"\n  CSF excluded (>50% missing):")
    for var, (aliases, miss_pct) in CSF_EXCLUDED_HIGH_MISS.items():
        print(f"    {var}: {miss_pct:.1f}% missing -> EXCLUDED")
    # ---- Locate clinical vars ----
    clin_found = {}
    for var, aliases in CLINICAL_ALIASES.items():
        for fname, df in csvs.items():
            col = find_col(df, aliases)
            idc = get_id_col(df)
            if col and idc:
                clin_found[var] = (fname, col, idc)
                print(f"  CLIN {var}: {fname} -> '{col}'")
                break
        if var not in clin_found:
            raise SystemExit(f"FATAL: Clinical variable {var} not found")
    # ---- Locate MRI features (all ST*** columns) ----
    mri_file = "RNA_plasma.csv"
    if mri_file not in csvs:
        raise SystemExit(f"FATAL: {mri_file} not found")
    mri_df_raw = csvs[mri_file]
    mri_id = get_id_col(mri_df_raw)
    mri_cols = [c for c in mri_df_raw.columns if c.startswith("ST")]
    print(f"  MRI  {len(mri_cols)} features from {mri_file}: {mri_cols[:5]}...{mri_cols[-2:]}")
    # ---- Demographics (post-hoc) ----
    demo_found = {}
    for var, aliases in DEMO_ALIASES.items():
        for fname, df in csvs.items():
            col = find_col(df, aliases)
            idc = get_id_col(df)
            if col and idc:
                demo_found[var] = (fname, col, idc)
                break
    # ---- Outcome ----
    outcome_df = None
    if "Womac_score_pain_function.csv" in csvs:
        outcome_df = csvs["Womac_score_pain_function.csv"].copy()
        oc = get_id_col(outcome_df)
        if oc:
            outcome_df = outcome_df.rename(columns={oc: "ID"})
    # ---- Build unified VAE DataFrame ----
    # CRITICAL: Column order = CSF first, then Clinical, then MRI
    # This ordering is required by the modality-weighted loss function.
    print(f"\n[2/8] Building VAE matrix (ordered: CSF -> Clin -> MRI)...")
    vae_df = None
    source_map = {}
    # CSF first
    for var, (fname, col, idc) in csf_found.items():
        tmp = csvs[fname][[idc, col]].rename(columns={idc: "ID", col: var})
        vae_df = tmp if vae_df is None else vae_df.merge(tmp, on="ID", how="inner")
        source_map[var] = f"{fname}->{col}"
    # Clinical second
    for var, (fname, col, idc) in clin_found.items():
        tmp = csvs[fname][[idc, col]].rename(columns={idc: "ID", col: var})
        vae_df = vae_df.merge(tmp, on="ID", how="inner")
        source_map[var] = f"{fname}->{col}"
    # MRI last
    mri_subset = mri_df_raw[[mri_id] + mri_cols].rename(columns={mri_id: "ID"})
    vae_df = vae_df.merge(mri_subset, on="ID", how="inner")
    for mc in mri_cols:
        source_map[mc] = f"{mri_file}->{mc}"
    n_feat = vae_df.shape[1] - 1  # minus ID
    feat_names = [c for c in vae_df.columns if c != "ID"]
    csf_names = list(CSF_ALIASES.keys())
    clin_names = list(CLINICAL_ALIASES.keys())
    # Verify ordering
    n_csf = len(csf_names)
    n_clin = len(clin_names)
    n_mri = len(mri_cols)
    assert feat_names[:n_csf] == csf_names, "CSF columns not in expected position"
    assert feat_names[n_csf:n_csf+n_clin] == clin_names, "Clinical columns not in expected position"
    print(f"  VAE input: {vae_df.shape[0]} participants x {n_feat} variables")
    print(f"  CSF ({n_csf}):  {csf_names}  [cols 0:{n_csf}]")
    print(f"  Clin ({n_clin}): {clin_names}  [cols {n_csf}:{n_csf+n_clin}]")
    print(f"  MRI ({n_mri}):  {mri_cols[:3]}...{mri_cols[-1]}  [cols {n_csf+n_clin}:{n_feat}]")
    print(f"  EXCLUDED: {sorted(EXCLUDED_VARS)}")
    print(f"  Modality weights: CSF(1/{n_csf}) + Clin(1/{n_clin}) + MRI(1/{n_mri}) = 1:1:1")
    # Demographics df
    demo_df = vae_df[["ID"]].copy()
    for var, (fname, col, idc) in demo_found.items():
        tmp = csvs[fname][[idc, col]].rename(columns={idc: "ID", col: var})
        demo_df = demo_df.merge(tmp, on="ID", how="left")
    # Fix SEX encoding
    if "SEX" in demo_df.columns:
        n_odd = int(((demo_df["SEX"] != 0) & (demo_df["SEX"] != 1) &
                      demo_df["SEX"].notna()).sum())
        if n_odd > 0:
            print(f"\n  WARNING: {n_odd} non-binary SEX values detected, rounding to 0/1")
            print(f"    Before: {demo_df['SEX'].value_counts().to_dict()}")
            demo_df["SEX"] = demo_df["SEX"].round().astype(int)
            print(f"    After:  {demo_df['SEX'].value_counts().to_dict()}")
    return vae_df, demo_df, outcome_df, source_map, csf_names, clin_names, mri_cols
# ===================== Preprocessing =====================
def preprocess(vae_df, csf_names, winsorize_sd=3.0):
    """
    Data is ALREADY z-score standardized. Preprocessing:
      1. Median imputation
      2. Report missingness per variable
      3. Winsorize at +/-winsorize_sd (clip extreme z-scores)
      4. NO log1p (would distort z-score distribution)
      5. NO StandardScaler (already standardized)
    """
    print(f"\n[3/8] Preprocessing (z-score input detected)...")
    ids = vae_df["ID"].values
    feat = [c for c in vae_df.columns if c != "ID"]
    X = vae_df[feat].values.astype(np.float64)
    # Report missingness
    print(f"\n  Missing data report:")
    high_miss_vars = []
    for i, f in enumerate(feat):
        n_miss = int(np.isnan(X[:,i]).sum())
        pct = 100 * n_miss / X.shape[0]
        if n_miss > 0:
            print(f"    {f}: {n_miss}/{X.shape[0]} ({pct:.1f}%) missing")
        if pct > 50:
            high_miss_vars.append((f, pct))
    if high_miss_vars:
        print(f"\n  WARNING: {len(high_miss_vars)} variables have >50% missing:")
        for v, p in high_miss_vars:
            print(f"    {v}: {p:.1f}%")
        print(f"  Median imputation will fill these with constant values.")
        print(f"  Consider excluding high-missingness variables.\n")
    # Median imputation
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)
    # Verify data is z-score (sanity check)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    print(f"\n  Post-imputation stats:")
    print(f"    Mean range: [{means.min():.3f}, {means.max():.3f}]")
    print(f"    SD range:   [{stds.min():.3f}, {stds.max():.3f}]")
    n_const = int(np.sum(stds < 0.01))
    if n_const > 0:
        const_vars = [feat[i] for i in range(len(feat)) if stds[i] < 0.01]
        print(f"    WARNING: {n_const} near-constant variables (sd<0.01): {const_vars}")
        print(f"    These are likely high-missingness vars filled with median.")
    # NO log1p
    print(f"\n  Skipping log1p (data is pre-standardized z-scores)")
    # Winsorize ALL columns at +/-winsorize_sd
    n_clip = 0
    for i in range(X.shape[1]):
        mu, sd = X[:,i].mean(), X[:,i].std()
        if sd < 1e-12:
            continue
        lo, hi = mu - winsorize_sd * sd, mu + winsorize_sd * sd
        n_clip += int(np.sum((X[:,i] < lo) | (X[:,i] > hi)))
        X[:,i] = np.clip(X[:,i], lo, hi)
    print(f"  Winsorized {n_clip} values at +/-{winsorize_sd} SD")
    # NO StandardScaler
    print(f"  SkippingStandardScaler (data is pre-standardized)")
    print(f"  Final matrix: {X.shape[0]} x {X.shape[1]}")
    return X, ids, feat, None, imp
# ===================== VAE Model =====================
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
# ===================== Modality-Weighted Loss =====================
def vae_loss_weighted(recon, x, mu, lv, beta, n_csf, n_clin):
    """
    Modality-weighted reconstruction loss.
    Instead of computing a single MSE across all 41 features (which lets
    the 30 MRI features dominate), we compute the MEAN MSE within each
    modality and then SUM the three modality losses. This gives each
    modality equal weight (1:1:1) regardless of its dimensionality.
    Feature layout (columns): [CSF_0 .. CSF_{n_csf-1}] [Clin_0 .. Clin_{n_clin-1}] [MRI_0 ..MRI_rest]
    Args:
        recon: reconstructed output (batch x d_in)
        x:     original input (batch x d_in)
        mu:    latent mean (batch x d_z)
        lv:    latent log-variance (batch x d_z)
        beta:  KL weight (annealed)
        n_csf: number of CSF features (7)
        n_clin: number of clinical features (4)
    Returns:
        total_loss, recon_loss, kl_loss, (loss_csf, loss_clin, loss_mri)
    """
    # Per-modality mean MSE (averaged over features AND batch)
    loss_csf  =nn.functional.mse_loss(recon[:, :n_csf],
                                        x[:, :n_csf], reduction="mean")
    loss_clin = nn.functional.mse_loss(recon[:,n_csf:n_csf+n_clin],
                                        x[:,n_csf:n_csf+n_clin], reduction="mean")
    loss_mri  =nn.functional.mse_loss(recon[:,n_csf+n_clin:],
                                        x[:,n_csf+n_clin:], reduction="mean")
    # Equal-weight sum: each modality contributes 1/3 of reconstruction signal
    recon_loss = loss_csf + loss_clin + loss_mri
    # KL divergence (unchanged)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    total = recon_loss + beta * kl
    return total, recon_loss, kl, (loss_csf, loss_clin, loss_mri)
# ===================== Training =====================
def train_vae(X, args, n_csf, n_clin):
    d_in = X.shape[1]
    n_mri = d_in - n_csf - n_clin
    print(f"\n[4/8] Training Modality-Weighted VAE  (d_in={d_in}, latent={args.latent_dim})")
    print(f"  {d_in}->{args.hidden1}->{args.hidden2}->{args.latent_dim}")
    print(f"  beta 0->{args.beta_max} over {args.beta_warmup} ep, "
          f"lr={args.lr}, epochs={args.epochs}")
    print(f"  Modality weighting: CSF({n_csf}) + Clin({n_clin}) + MRI({n_mri}) = 1:1:1")
    Xt = torch.FloatTensor(X).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)
    model = VAE(d_in, args.hidden1, args.hidden2, args.latent_dim).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    hist = {"total": [], "recon": [], "kl": [], "beta": [],
            "loss_csf": [], "loss_clin": [], "loss_mri": []}
    for ep in range(1, args.epochs + 1):
        model.train()
        beta = min(args.beta_max, args.beta_max * ep / args.beta_warmup)
        s_t, s_r, s_k, nb = 0, 0, 0, 0
        s_csf, s_clin, s_mri = 0, 0, 0
        for (bx,) in loader:
            opt.zero_grad()
            rec, mu, lv, z = model(bx)
            loss, rl, kl, (lc, lcl, lm) = vae_loss_weighted(
                rec, bx, mu, lv, beta, n_csf, n_clin)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s_t += loss.item(); s_r += rl.item(); s_k += kl.item(); nb += 1
            s_csf += lc.item(); s_clin += lcl.item(); s_mri += lm.item()
        sched.step()
        hist["total"].append(s_t / nb)
        hist["recon"].append(s_r / nb)
        hist["kl"].append(s_k / nb)
        hist["beta"].append(beta)
        hist["loss_csf"].append(s_csf / nb)
        hist["loss_clin"].append(s_clin / nb)
        hist["loss_mri"].append(s_mri / nb)
        if ep % 50 == 0 or ep == 1:
            print(f"  Ep {ep:>3}/{args.epochs}: total={s_t/nb:.4f} "
                  f"recon={s_r/nb:.4f} kl={s_k/nb:.4f} beta={beta:.3f} "
                  f"| CSF={s_csf/nb:.4f} Clin={s_clin/nb:.4f} MRI={s_mri/nb:.4f}")
    model.eval()
    with torch.no_grad():
        Z = model.encode_mu(Xt).cpu().numpy()
    print(f"  Latent: {Z.shape}")
    # Report final per-modality loss balance
    print(f"\n  Final loss balance:")
    print(f"    CSF:  {hist['loss_csf'][-1]:.4f}")
    print(f"    Clin: {hist['loss_clin'][-1]:.4f}")
    print(f"    MRI:  {hist['loss_mri'][-1]:.4f}")
    ratio_csf = hist['loss_csf'][-1] / (hist['loss_csf'][-1] + hist['loss_clin'][-1] + hist['loss_mri'][-1])
    ratio_clin = hist['loss_clin'][-1] / (hist['loss_csf'][-1] + hist['loss_clin'][-1] + hist['loss_mri'][-1])
    ratio_mri = hist['loss_mri'][-1] / (hist['loss_csf'][-1] + hist['loss_clin'][-1] + hist['loss_mri'][-1])
    print(f"    Ratio: CSF={ratio_csf:.1%} Clin={ratio_clin:.1%} MRI={ratio_mri:.1%}")
    return model, Z, hist
# ===================== Clustering =====================
def run_kmeans(data, n_clusters, label):
    km = KMeans(n_clusters=n_clusters, n_init=50, max_iter=500, random_state=42)
    lab = km.fit_predict(data)
    sil = silhouette_score(data, lab)
    db  =davies_bouldin_score(data, lab)
    ch  =calinski_harabasz_score(data, lab)
    print(f"  {label}: Sil={sil:.4f}  DB={db:.4f}  CH={ch:.4f}")
    for k in range(n_clusters):
        print(f"    Cluster {k}: n={int(np.sum(lab == k))}")
    return lab, km, {"sil": sil, "db": db, "ch": ch}
def compare_clusterings(lab_a, lab_b):
    ari = adjusted_rand_score(lab_a, lab_b)
    nmi = normalized_mutual_info_score(lab_a, lab_b)
    n = len(lab_a)
    a11 = sum(1 for i in range(n) for j in range(i+1, n)
              if (lab_a[i] == lab_a[j]) and (lab_b[i] == lab_b[j]))
    a10 = sum(1 for i in range(n) for j in range(i+1, n)
              if (lab_a[i] == lab_a[j]) and (lab_b[i] != lab_b[j]))
    a01 = sum(1 for i in range(n) for j in range(i+1, n)
              if (lab_a[i] != lab_a[j]) and (lab_b[i] == lab_b[j]))
    jac = a11 / (a11 + a10 + a01) if (a11 + a10 + a01) else 0
    print(f"  VAE vs Direct: ARI={ari:.4f}  NMI={nmi:.4f}  Jaccard={jac:.4f}")
    return {"ARI": ari, "NMI": nmi, "Jaccard": jac}
def check_sex(labels, demo_df, n_clusters):
    print("\n  Sex distribution across subtypes:")
    if "SEX" not in demo_df.columns:
        print("    SEX not available"); return None
    tmp = pd.DataFrame({"ID": demo_df["ID"], "SEX": demo_df["SEX"],
                         "Subtype": labels + 1})
    ct = pd.crosstab(tmp["Subtype"], tmp["SEX"])
    print(ct)
    try:
        chi2, p, dof, _ = chi2_contingency(ct)
        print(f"    Chi2={chi2:.3f}  p={p:.4f}")
        return {"chi2": float(chi2), "p": float(p), "dof": int(dof)}
    except Exception as e:
        print(f"    Chi2 failed: {e}"); return None
# ===================== Visualization =====================
def plot_latent(Z, labels, out):
    if Z.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for k in sorted(set(labels)):
            m = labels == k
            ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2],
                       label=f"Subtype {k+1} (n={m.sum()})", alpha=.7, s=40)
        ax.set_xlabel("Z1"); ax.set_ylabel("Z2"); ax.set_zlabel("Z3")
        ax.legend(); ax.set_title("VAE Latent Space (3D)")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        for k in sorted(set(labels)):
            m = labels == k
            ax.scatter(Z[m, 0], Z[m, 1],
                       label=f"Subtype {k+1} (n={m.sum()})", alpha=.7, s=40)
        ax.set_xlabel("Z1"); ax.set_ylabel("Z2")
        ax.legend(); ax.set_title("VAE Latent Space")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "latent_3d.png"), dpi=150); plt.close()
    # PCA 2D
    if Z.shape[1] > 2:
        pca = PCA(2); Z2 = pca.fit_transform(Z)
        fig, ax = plt.subplots(figsize=(10, 8))
        for k in sorted(set(labels)):
            m = labels == k
            ax.scatter(Z2[m, 0], Z2[m, 1],
                       label=f"Subtype {k+1} (n={m.sum()})", alpha=.7, s=40)
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.legend(); ax.set_title("Latent PCA 2D")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "latent_pca2d.png"), dpi=150); plt.close()
    print(f"  Saved latent plots")
def plot_history(hist, out):
    """Training history with per-modality loss breakdown."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    # Row 1: standard plots
    axes[0][0].plot(hist["total"], label="Total")
    axes[0][0].plot(hist["recon"], label="Recon")
    axes[0][0].set_title("Loss"); axes[0][0].legend()
    axes[0][1].plot(hist["kl"], color="orange"); axes[0][1].set_title("KL")
    axes[0][2].plot(hist["beta"], color="green"); axes[0][2].set_title("Beta")
    # Row 2: per-modality loss
    axes[1][0].plot(hist["loss_csf"], color="red", label="CSF")
    axes[1][0].plot(hist["loss_clin"], color="blue", label="Clinical")
    axes[1][0].plot(hist["loss_mri"], color="green", label="MRI")
    axes[1][0].set_title("Per-Modality Recon Loss"); axes[1][0].legend()
    # Modality loss ratio over time
    csf_arr = np.array(hist["loss_csf"])
    clin_arr = np.array(hist["loss_clin"])
    mri_arr = np.array(hist["loss_mri"])
    total_arr = csf_arr + clin_arr + mri_arr
    total_arr[total_arr < 1e-12] = 1e-12
    axes[1][1].plot(csf_arr / total_arr, color="red", label="CSF %")
    axes[1][1].plot(clin_arr / total_arr, color="blue", label="Clin %")
    axes[1][1].plot(mri_arr / total_arr, color="green", label="MRI %")
    axes[1][1].axhline(1/3, color="grey", ls="--", lw=1, label="33.3%")
    axes[1][1].set_title("Modality Loss Ratio"); axes[1][1].legend()
    axes[1][1].set_ylim(0, 1)
    # Final bar chart
    final_vals = [hist["loss_csf"][-1], hist["loss_clin"][-1], hist["loss_mri"][-1]]
    axes[1][2].bar(["CSF", "Clinical", "MRI"], final_vals,
                   color=["red", "blue", "green"])
    axes[1][2].set_title("Final Epoch Loss by Modality")
    axes[1][2].set_ylabel("MSE")
    for row in axes:
        for a in row:
            a.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "training_history.png"), dpi=150); plt.close()
    print(f"  Saved training_history.png")
def plot_profiles(X, labels, feat, csf_names, clin_names, mri_cols, out):
    """Heatmap: mean z-score per subtype, grouped by modality."""
    nk = len(set(labels))
    prof = np.zeros((nk, len(feat)))
    for k in range(nk):
        prof[k] = X[labels == k].mean(axis=0)
    fig, ax = plt.subplots(figsize=(max(14, len(feat) * 0.4), 5))
    sns.heatmap(prof, xticklabels=feat,
                yticklabels=[f"Subtype {k+1}" for k in range(nk)],
                cmap="RdBu_r", center=0, annot=False, ax=ax)
    ax.set_title("Subtype Profiles (standardized mean)")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "subtype_profiles.png"), dpi=150); plt.close()
    # Modality-grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    groups = [("CSF", csf_names), ("Clinical", clin_names), ("MRI", mri_cols)]
    for ax, (gname, gcols) in zip(axes, groups):
        idx = [i for i, f in enumerate(feat) if f in gcols]
        if not idx:
            continue
        sub_prof = prof[:,idx]
        x = np.arange(len(idx))
        w = 0.25
        for k in range(nk):
            ax.bar(x + k * w, sub_prof[k], w, label=f"Subtype {k+1}")
        ax.set_xticks(x + w)
        ax.set_xticklabels([feat[i] for i in idx], rotation=90, fontsize=6)
        ax.set_title(gname)
        ax.legend(fontsize=7)
        ax.axhline(0, color="grey", lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "subtype_profiles_grouped.png"), dpi=150); plt.close()
    print(f"  Saved subtype profile plots")
def plot_k_selection(X, Z, out, k_range=range(2, 8)):
    print(f"  K selection (k={list(k_range)})...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (data, lbl) in enumerate([(X, "Raw"), (Z, "VAE latent")]):
        inertias, sils = [], []
        for k in k_range:
            km = KMeans(k, n_init=30, random_state=42).fit(data)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(data, km.labels_))
        axes[0][idx].plot(list(k_range), inertias, "bo-")
        axes[0][idx].set_title(f"Elbow ({lbl})"); axes[0][idx].set_xlabel("K")
        axes[1][idx].plot(list(k_range), sils, "ro-")
        axes[1][idx].set_title(f"Silhouette ({lbl})"); axes[1][idx].set_xlabel("K")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "k_selection.png"), dpi=150); plt.close()
    print(f"  Saved k_selection.png")
# ===================== Export =====================
def export_all(ids, lab_vae, lab_dir, demo_df, outcome_df, Z, model, km, imp,
               comp, vae_m, dir_m, sex_res, feat, source_map,
               csf_names, clin_names, mri_cols, out, args, hist):
    print(f"\n[8/8] Exporting...")
    # Assignments
    adf = pd.DataFrame({"ID": ids,
                         "VAE_Subtype": lab_vae + 1,
                         "Direct_KMeans_Subtype": lab_dir + 1})
    if demo_df is not None:
        adf = adf.merge(demo_df, on="ID", how="left")
    if outcome_df is not None and "AD_Conversion" in outcome_df.columns:
        adf = adf.merge(outcome_df[["ID", "AD_Conversion"]], on="ID", how="left")
    adf.to_csv(os.path.join(out, "subtype_assignments.csv"), index=False)
    # Conversion rates
    if "AD_Conversion" in adf.columns:
        print("\n  === AD Conversion by VAE Subtype ===")
        for st in sorted(adf["VAE_Subtype"].unique()):
            sub = adf[adf["VAE_Subtype"] == st]
            nc = int(sub["AD_Conversion"].sum())
            print(f"    Subtype {st}: n={len(sub)}, conv={nc}, "
                  f"rate={nc/len(sub)*100:.1f}%")
        print("\n  === AD Conversion by Direct K-means ===")
        for st in sorted(adf["Direct_KMeans_Subtype"].unique()):
            sub = adf[adf["Direct_KMeans_Subtype"] == st]
            nc = int(sub["AD_Conversion"].sum())
            print(f"    Cluster {st}: n={len(sub)}, conv={nc}, "
                  f"rate={nc/len(sub)*100:.1f}%")
    # Latent
    zdf = pd.DataFrame(Z, columns=[f"Z{i+1}" for i in range(Z.shape[1])])
    zdf.insert(0, "ID", ids)
    zdf.to_csv(os.path.join(out, "latent_representations.csv"), index=False)
    # Summary JSON
    summary = {
        "method": "Modality-Weighted VAE (37 variables)",
        "loss_type": "per-modality mean MSE, 1:1:1 weighting",
        "csf_qc": "excluded 4 CSF variables with >50% missingness",
        "csf_excluded": {k: f"{v[1]:.1f}% missing" for k, v in CSF_EXCLUDED_HIGH_MISS.items()},
        "n_participants": int(len(ids)),
        "n_features": len(feat),
        "n_csf": len(csf_names),
        "n_clinical": len(clin_names),
        "n_mri": len(mri_cols),
        "features": feat,
        "csf_features": csf_names,
        "clinical_features": clin_names,
        "mri_features": mri_cols,
        "source_map": source_map,
        "latent_dim": args.latent_dim,
        "n_clusters": args.n_clusters,
        "epochs": args.epochs,
        "beta_max": args.beta_max,
        "beta_warmup": args.beta_warmup,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "lr": args.lr,
        "winsorize_sd": args.winsorize_sd,
        "vae_metrics": {k: float(v) for k, v in vae_m.items()},
        "direct_metrics": {k: float(v) for k, v in dir_m.items()},
        "comparison": {k: float(v) for k, v in comp.items()},
        "sex_test": sex_res,
        "vae_sizes": {f"Subtype_{k+1}": int(np.sum(lab_vae == k))
                      for k in range(args.n_clusters)},
        "final_modality_loss": {
            "CSF": float(hist["loss_csf"][-1]),
            "Clinical": float(hist["loss_clin"][-1]),
            "MRI": float(hist["loss_mri"][-1]),
        },
    }
    if "AD_Conversion" in adf.columns:
        cr = {}
        for st in sorted(adf["VAE_Subtype"].unique()):
            sub = adf[adf["VAE_Subtype"] == st]
            cr[f"Subtype_{st}"] = {
                "n": int(len(sub)),
                "converters": int(sub["AD_Conversion"].sum()),
                "rate": round(float(sub["AD_Conversion"].mean() * 100), 1),
            }
        summary["conversion_rates"] = cr
    with open(os.path.join(out, "vae_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # Pickle artifacts
    with open(os.path.join(out, "vae_artifacts.pkl"), "wb") as f:
        pickle.dump({"features": feat, "source_map": source_map,
                      "args": vars(args),
                      "imputer_statistics": imp.statistics_.tolist(),
                      "cluster_centers": km.cluster_centers_.tolist()}, f)
    centroids_df = pd.DataFrame(
        km.cluster_centers_,
        columns=[f"Z{i+1}" for i in range(km.cluster_centers_.shape[1])]
    )
    centroids_df.insert(0, "Subtype", np.arange(1, km.cluster_centers_.shape[0] + 1))
    centroids_df.to_csv(os.path.join(out, "subtype_centroids.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(out, "vae_model.pt"))
    print(f"  All files saved to {out}")
# ===================== Main =====================
def main():
    vae_df, demo_df, outcome_df, source_map, csf_names, clin_names, mri_cols = \
        load_data(args.input_dir)
    X, ids, feat, scaler, imp = preprocess(vae_df, csf_names, args.winsorize_sd)
    # Modality boundary indices (critical for weighted loss)
    n_csf = len(csf_names)
    n_clin = len(clin_names)
    n_mri = len(mri_cols)
    print(f"\n  Modality boundaries: CSF[0:{n_csf}] Clin[{n_csf}:{n_csf+n_clin}] "
          f"MRI[{n_csf+n_clin}:{n_csf+n_clin+n_mri}]")
    # Direct K-means baseline
    print(f"\n[5/8] Direct K-means baseline...")
    lab_dir, _, dir_m = run_kmeans(X, args.n_clusters, "Direct")
    # VAE with modality-weighted loss
    model, Z, hist = train_vae(X, args, n_csf, n_clin)
    # VAE K-means
    print(f"\n[6/8] VAE K-means...")
    lab_vae, km, vae_m = run_kmeans(Z, args.n_clusters, "VAE")
    # Compare
    comp = compare_clusterings(lab_dir, lab_vae)
    # Sex
    sex_res = check_sex(lab_vae, demo_df, args.n_clusters)
    # Plots
    print(f"\n[7/8] Plots...")
    plot_latent(Z, lab_vae, args.output_dir)
    plot_history(hist, args.output_dir)
    plot_profiles(X, lab_vae, feat, csf_names, clin_names, mri_cols,
                  args.output_dir)
    plot_k_selection(X, Z, args.output_dir)
    # Export
    export_all(ids, lab_vae, lab_dir, demo_df, outcome_df, Z, model, km, imp,
               comp, vae_m, dir_m, sex_res, feat, source_map,
               csf_names, clin_names, mri_cols, args.output_dir, args, hist)
    print("\n" + "=" * 70)
    print("DONE:", args.output_dir)
    print("=" * 70)
if __name__ == "__main__":
    main()



