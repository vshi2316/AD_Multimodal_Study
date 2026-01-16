
"""

import os
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Utilities
# =========================
def _as_str_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _find_col_case_insensitive(df: pd.DataFrame, candidates):
    col_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _one_hot_from_binary(x: np.ndarray):
    x = x.astype(int)
    if not set(np.unique(x)).issubset({0, 1}):
        raise ValueError("Binary variable contains values outside {0,1}.")
    oh = np.zeros((len(x), 2), dtype=np.float32)
    oh[np.arange(len(x)), x] = 1.0
    return oh

def _one_hot_from_multiclass(x: np.ndarray, classes_sorted):
    class_to_index = {c: i for i, c in enumerate(classes_sorted)}
    oh = np.zeros((len(x), len(classes_sorted)), dtype=np.float32)
    for i, v in enumerate(x):
        oh[i, class_to_index[v]] = 1.0
    return oh

def _gap_statistic(X: np.ndarray, k_list, n_refs=10, random_state=42):
    """
    Simple Gap Statistic:
    Gap(k) = E_ref[log(Wk_ref)] - log(Wk_data)
    where Wk is KMeans inertia (within-cluster dispersion proxy).
    """
    rng = np.random.RandomState(random_state)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    gaps = []
    sds = []
    for k in k_list:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        km.fit(X)
        wk = km.inertia_
        log_wk = np.log(wk + 1e-12)
        
        ref_logs = []
        for _ in range(n_refs):
            X_ref = rng.uniform(mins, maxs, size=X.shape)
            km_ref = KMeans(n_clusters=k, n_init=10, random_state=rng.randint(0, 10**9))
            km_ref.fit(X_ref)
            ref_logs.append(np.log(km_ref.inertia_ + 1e-12))
        
        ref_logs = np.array(ref_logs)
        gap = ref_logs.mean() - log_wk
        sd = ref_logs.std() * np.sqrt(1 + 1.0 / n_refs)
        gaps.append(float(gap))
        sds.append(float(sd))
    
    return np.array(gaps), np.array(sds)

# =========================
# PyTorch Dataset
# =========================
class VAEDataset(Dataset):
    def __init__(self, X, y_dict):
        self.X = torch.FloatTensor(X)
        self.y_dict = {k: torch.FloatTensor(v) for k, v in y_dict.items()}
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.y_dict.items()}

# =========================
# Model Components (PyTorch)
# =========================
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        return z_mean, z_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, output_activation='linear'):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc_out = nn.Linear(128, output_dim)
        self.output_activation = output_activation
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        out = self.fc_out(h)
        
        if self.output_activation == 'softmax':
            out = F.softmax(out, dim=-1)
        # linear: no activation
        return out

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, decoder_dims):
        super(BetaVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        
        # Create 6 decoders
        self.decoders = nn.ModuleDict({
            'csf_ptau181': Decoder(latent_dim, decoder_dims['csf_ptau181'], 'linear'),
            'csf_abeta_ratio': Decoder(latent_dim, decoder_dims['csf_abeta_ratio'], 'linear'),
            'mmse': Decoder(latent_dim, decoder_dims['mmse'], 'linear'),
            'age': Decoder(latent_dim, decoder_dims['age'], 'linear'),
            'sex': Decoder(latent_dim, decoder_dims['sex'], 'softmax'),
            'apoe': Decoder(latent_dim, decoder_dims['apoe'], 'softmax'),
        })
        
        self.beta = 0.001  # Will be updated during training
    
    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        
        reconstructions = {}
        for name, decoder in self.decoders.items():
            reconstructions[name] = decoder(z)
        
        return reconstructions, z_mean, z_logvar
    
    def loss_function(self, x, y_dict, reconstructions, z_mean, z_logvar):
        # Reconstruction losses
        recon_loss = 0.0
        recon_losses = {}
        
        # Continuous domains (MSE)
        for domain in ['csf_ptau181', 'csf_abeta_ratio', 'mmse', 'age']:
            mse = F.mse_loss(reconstructions[domain], y_dict[domain], reduction='sum')
            recon_losses[domain] = mse.item() / len(x)
            recon_loss += mse
        
        # Categorical domains (Cross-Entropy)
        for domain in ['sex', 'apoe']:
            # Use cross_entropy which expects (N, C) for predictions and (N, C) for one-hot targets
            ce = -torch.sum(y_dict[domain] * torch.log(reconstructions[domain] + 1e-10))
            recon_losses[domain] = ce.item() / len(x)
            recon_loss += ce
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss, recon_losses

# =========================
# Training Functions
# =========================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0
    train_recon = 0.0
    train_kl = 0.0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = {k: v.to(device) for k, v in batch_y.items()}
        
        optimizer.zero_grad()
        reconstructions, z_mean, z_logvar = model(batch_x)
        loss, recon, kl, _ = model.loss_function(batch_x, batch_y, reconstructions, z_mean, z_logvar)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon += recon.item()
        train_kl += kl.item()
    
    n_batches = len(dataloader)
    return train_loss / n_batches, train_recon / n_batches, train_kl / n_batches

def validate_epoch(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    val_recon = 0.0
    val_kl = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = {k: v.to(device) for k, v in batch_y.items()}
            
            reconstructions, z_mean, z_logvar = model(batch_x)
            loss, recon, kl, _ = model.loss_function(batch_x, batch_y, reconstructions, z_mean, z_logvar)
            
            val_loss += loss.item()
            val_recon += recon.item()
            val_kl += kl.item()
    
    n_batches = len(dataloader)
    return val_loss / n_batches, val_recon / n_batches, val_kl / n_batches

def encode_data(model, X, device, batch_size=64):
    model.eval()
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    z_means = []
    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0].to(device)
            z_mean, _ = model.encoder(batch_x)
            z_means.append(z_mean.cpu().numpy())
    
    return np.concatenate(z_means, axis=0)

# =========================
# Main
# =========================
def main():
    # ========== Parse Arguments ==========
    parser = argparse.ArgumentParser(
        description="Multi-modal VAE Clustering for AD Subtyping (PyTorch)"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Path to integrated CSV file with columns: ID, AGE, SEX, MMSE, CSF_PTAU181, CSF_ABETA42_ABETA40_RATIO, APOE_VAR, AD_Conversion"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Directory to save results (default: ./results)"
    )
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=3,
        help="Final K (default: 3)"
    )
    parser.add_argument(
        "--latent_dim", 
        type=int, 
        default=10,
        help="Latent dim (default: 10)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=200,
        help="Max epochs (default: 200)"
    )
    args = parser.parse_args()
    
    # Methods-aligned training settings
    learning_rate = 0.001
    batch_size = 64
    early_stop_patience = 20
    beta_start = 0.001
    beta_end = 1.0
    beta_anneal_epochs = max(1, int(args.epochs * 0.5))
    k_list = list(range(2, 7))  # 2-6
    
    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("=" * 70)
    print("VAE Clustering for AD Subtyping (PyTorch 1.12.0)".center(70))
    print("=" * 70)
    print(f"\n📂 Input File: {args.input_file}")
    print(f"📂 Output Directory: {args.output_dir}")
    print(f"🖥️  Device: {device}\n")
    
    # ========== Step 1: Load Data ==========
    print("[Step 1/10] Loading integrated data...")
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    df = pd.read_csv(args.input_file)
    print(f"  Loaded: {df.shape}")
    
    # Required columns
    required_cols = ["ID", "AGE", "SEX", "MMSE", "CSF_PTAU181", "CSF_ABETA42_ABETA40_RATIO", "APOE_VAR", "AD_Conversion"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df["ID"] = _as_str_id(df["ID"])
    df = df.drop_duplicates(subset=["ID"], keep="first")
    
    # Convert to numeric
    df["AGE"] = _safe_numeric(df["AGE"])
    df["MMSE"] = _safe_numeric(df["MMSE"])
    df["CSF_PTAU181"] = _safe_numeric(df["CSF_PTAU181"])
    df["CSF_ABETA42_ABETA40_RATIO"] = _safe_numeric(df["CSF_ABETA42_ABETA40_RATIO"])
    df["SEX"] = _safe_numeric(df["SEX"])
    df["APOE_VAR"] = _safe_numeric(df["APOE_VAR"])
    df["AD_Conversion"] = _safe_numeric(df["AD_Conversion"])
    
    # Exclude >20% missing baseline data 
    baseline_cols = ["AGE", "SEX", "MMSE", "CSF_PTAU181", "CSF_ABETA42_ABETA40_RATIO", "APOE_VAR"]
    miss_frac = df[baseline_cols].isna().mean(axis=1)
    excluded_ids = df.loc[miss_frac > 0.2, "ID"].tolist()
    if len(excluded_ids) > 0:
        print(f"  ℹ Excluding {len(excluded_ids)} participants with >20% missing baseline data .")
    df = df[miss_frac <= 0.2].reset_index(drop=True)
    
    n_samples = len(df)
    n_converters = int((df["AD_Conversion"] == 1).sum())
    print(f"  ✓ Final: {n_samples} samples, {n_converters} converters")
    
    # Validate SEX/APOE
    if df["SEX"].isna().any():
        raise ValueError("SEX contains missing values.")
    if not set(sorted(df["SEX"].unique().tolist())).issubset({0.0, 1.0}):
        raise ValueError("SEX must be binary (0/1).")
    
    if df["APOE_VAR"].isna().any():
        raise ValueError("APOE contains missing values.")
    apoe_unique = sorted(df["APOE_VAR"].unique().tolist())
    if not set(apoe_unique).issubset({0.0, 1.0, 2.0}):
        raise ValueError("APOE variable must be in {0,1,2} (dosage) or {0,1} (status).")
    
    # ========== Step 2: Leakage-safe standardization + split ==========
    print("\n[Step 2/10] Standardization (train-fit only) and Train/Val split...")
    idx_all = np.arange(n_samples)
    idx_train, idx_val = train_test_split(idx_all, test_size=0.2, random_state=42)
    
    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    
    # Train-set median imputation for continuous vars
    cont_cols = ["CSF_PTAU181", "CSF_ABETA42_ABETA40_RATIO", "MMSE", "AGE"]
    cont_medians = {c: float(train_df[c].median()) for c in cont_cols}
    
    for c in cont_cols:
        train_df[c] = train_df[c].fillna(cont_medians[c])
        val_df[c] = val_df[c].fillna(cont_medians[c])
        df[c] = df[c].fillna(cont_medians[c])
    
    scaler_cont = StandardScaler()
    X_train_cont = scaler_cont.fit_transform(train_df[cont_cols].values.astype(np.float32))
    X_val_cont = scaler_cont.transform(val_df[cont_cols].values.astype(np.float32))
    X_all_cont = scaler_cont.transform(df[cont_cols].values.astype(np.float32))
    
    # One-hot categorical
    sex_train_oh = _one_hot_from_binary(train_df["SEX"].values.astype(int))
    sex_val_oh = _one_hot_from_binary(val_df["SEX"].values.astype(int))
    sex_all_oh = _one_hot_from_binary(df["SEX"].values.astype(int))
    
    apoe_vals_all = df["APOE_VAR"].values.astype(int)
    apoe_classes = sorted(np.unique(apoe_vals_all).tolist())
    
    if apoe_classes == [0, 1]:
        apoe_train_oh = _one_hot_from_binary(train_df["APOE_VAR"].values.astype(int))
        apoe_val_oh = _one_hot_from_binary(val_df["APOE_VAR"].values.astype(int))
        apoe_all_oh = _one_hot_from_binary(df["APOE_VAR"].values.astype(int))
    else:
        apoe_train_oh = _one_hot_from_multiclass(train_df["APOE_VAR"].values.astype(int), apoe_classes)
        apoe_val_oh = _one_hot_from_multiclass(val_df["APOE_VAR"].values.astype(int), apoe_classes)
        apoe_all_oh = _one_hot_from_multiclass(df["APOE_VAR"].values.astype(int), apoe_classes)
    
    # Input features to encoder: continuous(z) + sex(onehot) + apoe(onehot)
    X_train = np.concatenate([X_train_cont, sex_train_oh, apoe_train_oh], axis=1).astype(np.float32)
    X_val = np.concatenate([X_val_cont, sex_val_oh, apoe_val_oh], axis=1).astype(np.float32)
    X_all = np.concatenate([X_all_cont, sex_all_oh, apoe_all_oh], axis=1).astype(np.float32)
    
    # Decoder targets: use standardized continuous targets
    y_train = {
        "csf_ptau181": X_train_cont[:, [0]],
        "csf_abeta_ratio": X_train_cont[:, [1]],
        "mmse": X_train_cont[:, [2]],
        "age": X_train_cont[:, [3]],
        "sex": sex_train_oh.astype(np.float32),
        "apoe": apoe_train_oh.astype(np.float32),
    }
    y_val = {
        "csf_ptau181": X_val_cont[:, [0]],
        "csf_abeta_ratio": X_val_cont[:, [1]],
        "mmse": X_val_cont[:, [2]],
        "age": X_val_cont[:, [3]],
        "sex": sex_val_oh.astype(np.float32),
        "apoe": apoe_val_oh.astype(np.float32),
    }
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Encoder input dim: {X_all.shape[1]}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Batch size: {batch_size}, Max epochs: {args.epochs}, EarlyStop patience: {early_stop_patience}")
    print(f"  Beta annealing: {beta_start} -> {beta_end} over {beta_anneal_epochs} epochs")
    
    # Create PyTorch datasets
    train_dataset = VAEDataset(X_train, y_train)
    val_dataset = VAEDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ========== Step 3: Build beta-VAE (6 decoders) ==========
    print("\n[Step 3/10] Building beta-VAE (1 encoder + 6 decoders)...")
    input_dim = X_all.shape[1]
    latent_dim = int(args.latent_dim)
    
    decoder_dims = {
        'csf_ptau181': 1,
        'csf_abeta_ratio': 1,
        'mmse': 1,
        'age': 1,
        'sex': sex_train_oh.shape[1],
        'apoe': apoe_train_oh.shape[1],
    }
    
    model = BetaVAE(input_dim=input_dim, latent_dim=latent_dim, decoder_dims=decoder_dims)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("  ✓ VAE built and moved to device")
    
    # ========== Step 4: Train (β anneal + EarlyStopping) ==========
    print(f"\n[Step 4/10] Training (max {args.epochs} epochs)...")
    
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_loss': [],
        'val_recon': [],
        'val_kl': [],
        'beta': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Beta annealing
        t = min(epoch / beta_anneal_epochs, 1.0)
        current_beta = beta_start + (beta_end - beta_start) * t
        model.beta = current_beta
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)
        history['beta'].append(current_beta)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Beta: {current_beta:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("  ✓ Training completed")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label="Train Loss", linewidth=2)
    plt.plot(history['val_loss'], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("beta-VAE Training (PyTorch)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "training_history.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== Step 5: Encode ==========
    print("\n[Step 5/10] Encoding...")
    z_mean_encoded = encode_data(model, X_all, device, batch_size=batch_size)
    print(f"  ✓ Latent: {z_mean_encoded.shape}")
    
    # ========== Step 6: Cluster evaluation (K=2-6 + gap) ==========
    print("\n[Step 6/10] Cluster evaluation (K=2-6, Silhouette/DB/CH/Gap/Inertia)...")
    gaps, gap_sds = _gap_statistic(z_mean_encoded, k_list=k_list, n_refs=10, random_state=42)
    
    metrics = {"K": [], "Silhouette": [], "DB": [], "CH": [], "Gap": [], "Gap_SD": [], "Inertia": []}
    for i, k in enumerate(k_list):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
        labels = km.fit_predict(z_mean_encoded)
        sil = silhouette_score(z_mean_encoded, labels)
        db = davies_bouldin_score(z_mean_encoded, labels)
        ch = calinski_harabasz_score(z_mean_encoded, labels)
        
        metrics["K"].append(k)
        metrics["Silhouette"].append(float(sil))
        metrics["DB"].append(float(db))
        metrics["CH"].append(float(ch))
        metrics["Gap"].append(float(gaps[i]))
        metrics["Gap_SD"].append(float(gap_sds[i]))
        metrics["Inertia"].append(float(km.inertia_))
        
        print(f"    K={k}: Sil={sil:.3f}, DB={db:.3f}, CH={ch:.1f}, Gap={gaps[i]:.3f}")
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(args.output_dir, "cluster_metrics_table.csv"), index=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(metrics["K"], metrics["Silhouette"], "o-", linewidth=2)
    axes[0, 0].set_title("Silhouette (Higher Better)")
    axes[0, 0].set_xlabel("K")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(metrics["K"], metrics["DB"], "o-", linewidth=2, color="red")
    axes[0, 1].set_title("Davies-Bouldin (Lower Better)")
    axes[0, 1].set_xlabel("K")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(metrics["K"], metrics["CH"], "o-", linewidth=2, color="green")
    axes[1, 0].set_title("Calinski-Harabasz (Higher Better)")
    axes[1, 0].set_xlabel("K")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].errorbar(metrics["K"], metrics["Gap"], yerr=metrics["Gap_SD"], fmt="o-", linewidth=2, color="purple")
    axes[1, 1].set_title("Gap Statistic (Higher Better)")
    axes[1, 1].set_xlabel("K")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "cluster_evaluation.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ========== Step 7: Final clustering ==========
    print("\n[Step 7/10] Clustering...")
    final_k = int(args.n_clusters)
    kmeans_final = KMeans(n_clusters=final_k, init="k-means++", n_init=50, random_state=42)
    cluster_labels = kmeans_final.fit_predict(z_mean_encoded)
    
    print(f"\n  ✓ Final K={final_k}:")
    for i in range(final_k):
        count = int(np.sum(cluster_labels == i))
        print(f"    Cluster {i}: {count} ({count/len(cluster_labels)*100:.1f}%)")
    
    # ========== Step 8: Save Results ==========
    print("\n[Step 8/10] Saving...")
    
    # cluster_results.csv
    result_df = pd.DataFrame({
        "ID": df["ID"].values,
        "Cluster_Labels": cluster_labels
    })
    result_df = pd.merge(result_df, df[["ID", "AD_Conversion"]], on="ID", how="left")
    result_df.to_csv(os.path.join(args.output_dir, "cluster_results.csv"), index=False)
    
    # latent_encoded.csv
    latent_cols = [f"Latent_{i+1}" for i in range(latent_dim)]
    latent_df = pd.DataFrame(z_mean_encoded, columns=latent_cols)
    latent_df.insert(0, "ID", df["ID"].values)
    latent_df["Cluster_Labels"] = cluster_labels
    latent_df = pd.merge(latent_df, df[["ID", "AD_Conversion"]], on="ID", how="left")
    latent_df.to_csv(os.path.join(args.output_dir, "latent_encoded.csv"), index=False)
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'decoder_dims': decoder_dims,
    }, os.path.join(args.output_dir, "vae_model.pth"))
    
    # Save scalers.pkl / metadata.pkl
    scalers = {
        "continuous_scaler": scaler_cont,
        "continuous_train_medians": cont_medians,
    }
    with open(os.path.join(args.output_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    
    feature_dims = {
        "continuous": int(X_all_cont.shape[1]),
        "sex_onehot": int(sex_all_oh.shape[1]),
        "apoe_onehot": int(apoe_all_oh.shape[1]),
    }
    with open(os.path.join(args.output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({
            "feature_dims": feature_dims,
            "excluded_missing_gt_20pct": len(excluded_ids),
            "required_baseline_columns": baseline_cols,
            "apoe_classes": apoe_classes,
            "latent_dim": latent_dim,
            "final_k": final_k,
            "framework": "PyTorch 1.12.0",
        }, f)
    
    # ========== Step 9: Latent visualization (PCA) ==========
    print("\n[Step 9/10] Visualization...")
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_mean_encoded)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, final_k))
    for i in range(final_k):
        mask = cluster_labels == i
        plt.scatter(
            z_pca[mask, 0],
            z_pca[mask, 1],
            c=[colors[i]],
            label=f"Cluster {i}",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Clusters (latent PCA)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if n_converters > 0:
        conv_mask = df["AD_Conversion"].values == 1
        plt.scatter(z_pca[~conv_mask, 0], z_pca[~conv_mask, 1], c="blue", label="Stable", alpha=0.5, s=50)
        plt.scatter(z_pca[conv_mask, 0], z_pca[conv_mask, 1], c="red", label="Converter", alpha=0.7, s=50, marker="^")
        plt.title("Colored by Outcome")
    else:
        plt.scatter(z_pca[:, 0], z_pca[:, 1], c="gray", alpha=0.5, s=50)
        plt.title("All Stable (No Converters)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "latent_visualization.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Cluster-outcome summary
    if n_converters > 0:
        print("\n📊 Cluster-Outcome Association:")
        for i in range(final_k):
            mask = cluster_labels == i
            conv = int((df.loc[mask, "AD_Conversion"] == 1).sum())
            total = int(mask.sum())
            print(f"    Cluster {i}: {conv}/{total} converters ({(conv/total*100 if total>0 else 0):.1f}%)")
    
    # ========== Step 10: Summary report ==========
    print("\n[Step 10/10] Summary report...")
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("=" * 70 + "\n")
        f.write("beta-VAE Clustering Summary (PyTorch 1.12.0)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input: {args.input_file}\n")
        f.write(f"Samples: {n_samples}\n")
        f.write(f"Converters: {n_converters}/{n_samples} ({(n_converters/n_samples*100 if n_samples>0 else 0):.1f}%)\n")
        f.write(f"Latent dim: {latent_dim}\n")
        f.write(f"Clusters (final K): {final_k}\n")
        f.write(f"Excluded (>20% missing baseline): {len(excluded_ids)}\n")
        f.write(f"Framework: PyTorch 1.12.0\n")
        f.write(f"Device: {device}\n\n")
        f.write("K Evaluation (2-6):\n")
        for _, row in metrics_df.iterrows():
            f.write(f"  K={int(row['K'])}: Sil={row['Silhouette']:.3f}, "
                   f"DB={row['DB']:.3f}, CH={row['CH']:.1f}, Gap={row['Gap']:.3f}\n")
        f.write("\nCluster Distribution:\n")
        for i in range(final_k):
            count = int(np.sum(cluster_labels == i))
            f.write(f"  Cluster {i}: {count} ({count/n_samples*100:.1f}%)\n")
        f.write("\nInterpretation:\n")
        f.write("- Continuous domains are z-score standardized using training-set parameters to prevent leakage.\n")
        f.write("- Categorical domains (SEX/APOE) are one-hot encoded and reconstructed using cross-entropy loss.\n")

    
    print("\n" + "=" * 70)
    print("VAE Clustering Completed!".center(70))
    print("=" * 70)
    print(f"✓ Output: {args.output_dir}")
    print(f"✓ Samples: {n_samples}, Converters: {n_converters}")
    print(f"✓ Framework: PyTorch 1.12.0 (matches manuscript)")
    print("✓ Saved: cluster_results.csv, latent_encoded.csv, vae_model.pth, scalers.pkl, metadata.pkl, figures, summary.txt")
    print("=" * 70)

if __name__ == "__main__":
    main()

