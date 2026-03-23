"""
Step 9A: External cohort projection into the ADNI VAE latent space.

This script applies the ADNI-trained encoder to an external cohort and assigns
subtypes using Euclidean distance to discovery-cohort subtype centroids.
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


class Encoder(nn.Module):
    def __init__(self, d_in, h1, h2, d_z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.15),
        )
        self.fc_mu = nn.Linear(h2, d_z)
        self.fc_logvar = nn.Linear(h2, d_z)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, d_z, h2, h1, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(h1, d_out),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, d_in, h1, h2, d_z):
        super().__init__()
        self.encoder = Encoder(d_in, h1, h2, d_z)
        self.decoder = Decoder(d_z, h2, h1, d_in)

    def encode_mu(self, x):
        mu, _ = self.encoder(x)
        return mu


def parse_args():
    parser = argparse.ArgumentParser(description="Project an external cohort into the ADNI VAE latent space")
    parser.add_argument("--external_file", required=True, help="External integrated cohort CSV")
    parser.add_argument("--vae_dir", required=True, help="Directory containing vae_model.pt, vae_summary.json, and subtype_centroids.csv")
    parser.add_argument("--output_dir", default="./step9A_results", help="Output directory")
    parser.add_argument("--cohort_name", default="external", help="Cohort label")
    parser.add_argument("--id_col", default="ID", help="ID column in the external cohort file")
    return parser.parse_args()


def first_existing(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def build_feature_matrix(df, features, source_map, imputer_statistics):
    out = pd.DataFrame({"ID": df["ID"].astype(str)})
    missing_features = []

    for idx, feature in enumerate(features):
        source_candidates = [feature]
        if isinstance(source_map, dict) and feature in source_map:
            if isinstance(source_map[feature], list):
                source_candidates.extend(source_map[feature])
            else:
                source_candidates.append(source_map[feature])

        source_col = first_existing(df, source_candidates)
        if source_col is None:
            out[feature] = imputer_statistics[idx]
            missing_features.append(feature)
        else:
            out[feature] = pd.to_numeric(df[source_col], errors="coerce")
            out[feature] = out[feature].fillna(imputer_statistics[idx])

    return out, missing_features


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    external = pd.read_csv(args.external_file)
    if args.id_col not in external.columns:
        raise ValueError(f"ID column not found: {args.id_col}")
    external = external.rename(columns={args.id_col: "ID"})
    external["ID"] = external["ID"].astype(str)

    with open(os.path.join(args.vae_dir, "vae_summary.json"), "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(os.path.join(args.vae_dir, "vae_artifacts.pkl"), "rb") as handle:
        artifacts = pickle.load(handle)

    centroids = pd.read_csv(os.path.join(args.vae_dir, "subtype_centroids.csv"))
    feature_list = summary["features"]
    source_map = summary.get("source_map", artifacts.get("source_map", {}))
    imputer_statistics = np.array(artifacts.get("imputer_statistics", [0.0] * len(feature_list)), dtype=float)

    matrix_df, missing_features = build_feature_matrix(external, feature_list, source_map, imputer_statistics)
    x = matrix_df[feature_list].values.astype(np.float32)

    winsorize_sd = float(summary.get("winsorize_sd", 3.0))
    x = np.clip(x, -winsorize_sd, winsorize_sd)

    d_in = len(feature_list)
    h1 = int(summary.get("hidden1", 256))
    h2 = int(summary.get("hidden2", 128))
    d_z = int(summary.get("latent_dim", 3))

    model = VAE(d_in, h1, h2, d_z)
    state_dict = torch.load(os.path.join(args.vae_dir, "vae_model.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        latent = model.encode_mu(torch.tensor(x, dtype=torch.float32)).numpy()

    centroid_cols = [column for column in centroids.columns if column.startswith("Z")]
    centroid_matrix = centroids[centroid_cols].values.astype(float)

    distances = np.linalg.norm(latent[:, None, :] - centroid_matrix[None, :, :], axis=2)
    subtype_idx = np.argmin(distances, axis=1)
    subtype = centroids.iloc[subtype_idx]["Subtype"].astype(int).values
    min_distance = distances[np.arange(distances.shape[0]), subtype_idx]

    latent_df = pd.DataFrame(latent, columns=[f"Z{i+1}" for i in range(latent.shape[1])])
    latent_df.insert(0, "ID", matrix_df["ID"])
    latent_df.to_csv(os.path.join(args.output_dir, f"{args.cohort_name}_latent_representations.csv"), index=False)

    assign_df = pd.DataFrame({
        "ID": matrix_df["ID"],
        "Projected_Subtype": subtype,
        "Projection_Distance": np.round(min_distance, 6),
        "Cohort": args.cohort_name,
    })
    assign_df.to_csv(os.path.join(args.output_dir, f"{args.cohort_name}_projected_subtypes.csv"), index=False)

    summary_out = {
        "cohort": args.cohort_name,
        "n_participants": int(len(assign_df)),
        "n_missing_features_filled": int(len(missing_features)),
        "missing_features_filled": missing_features,
        "subtype_counts": assign_df["Projected_Subtype"].value_counts().sort_index().to_dict(),
    }
    with open(os.path.join(args.output_dir, f"{args.cohort_name}_projection_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_out, handle, indent=2)

    print(f"Saved projection outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
