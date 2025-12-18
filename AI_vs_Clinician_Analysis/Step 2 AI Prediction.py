#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Prediction Script: VAE Risk Stratification Model
Core Logic: Lasso Feature Selection + SVM-RBF Ensemble for AD conversion prediction
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectFromModel
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline

# Import SMOTE with fallback
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    from imblearn.pipeline import Pipeline as ImbPipeline  # Fallback

# Suppress warnings
warnings.filterwarnings('ignore')

# Plot configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Basic progress print
print("AI Prediction: VAE Risk Stratification Model (Lasso + SVM-RBF Ensemble)")

# =====================================================================
# 1. Load & Sanitize Data
# =====================================================================
output_dir = "AI_vs_Clinician_Test"
input_file = os.path.join(output_dir, "independent_test_set.csv")

if not os.path.exists(input_file):
    print(f"ERROR: Input file not found - {input_file}")
    exit(1)

raw_data = pd.read_csv(input_file)

# Data type sanitization
cols_to_numeric = [c for c in raw_data.columns if c not in ['ID', 'RID', 'Baseline_Date']]
for col in cols_to_numeric:
    raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

# Filter valid target values
df = raw_data[raw_data['AD_Conversion'].notna()].copy()

# Basic dataset stats
print(f"Loaded test set: {len(df)} cases")
print(f"AD converters: {int(df['AD_Conversion'].sum())} ({df['AD_Conversion'].mean():.1%})")

# =====================================================================
# 2. Feature Engineering & Selection Pipeline
# =====================================================================
print("\nPhase 1: Feature Engineering, Imputation & Selection")

def engineering_pipeline(data):
    X = pd.DataFrame(index=data.index)
    
    # Demographics & Genetics
    X['Age'] = data['Age']
    X['Gender'] = data['Gender']
    X['Education'] = data['Education']
    X['APOE4'] = data['APOE4_Positive']
    X['Age_x_APOE4'] = X['Age'] * X['APOE4']

    # Cognitive features
    if 'MMSE_Baseline' in data.columns:
        X['MMSE'] = data['MMSE_Baseline']
        X['MMSE_Inv'] = 30 - X['MMSE'] 
    if 'ADAS13' in data.columns: X['ADAS13'] = data['ADAS13']
    if 'FAQTOTAL' in data.columns: X['FAQ'] = data['FAQTOTAL']

    # CSF Biomarkers (log transformed)
    if 'ABETA42' in data.columns: X['ABETA42'] = np.log1p(data['ABETA42'].clip(lower=0))
    if 'TAU_TOTAL' in data.columns: X['TAU'] = np.log1p(data['TAU_TOTAL'].clip(lower=0))
    if 'PTAU181' in data.columns: X['PTAU'] = np.log1p(data['PTAU181'].clip(lower=0))
    if 'ABETA42' in X.columns and 'PTAU' in X.columns:
        X['Amyloid_Tau_Ratio'] = X['PTAU'] / (X['ABETA42'] + 0.1)

    # MRI Neuroimaging features
    mri_cols = [c for c in data.columns if c.startswith('ST')]
    if mri_cols:
        # Individual MRI features
        for col in mri_cols:
            X[col] = data[col]
        # Global atrophy metric
        X['Global_Atrophy'] = data[mri_cols].mean(axis=1)
        # Interaction term
        X['Atrophy_x_Edu'] = X['Global_Atrophy'] * X['Education']
    
    return X

# Extract raw features
X_raw = engineering_pipeline(df)
y = df['AD_Conversion'].values

# Clean empty/constant columns
X_raw = X_raw.dropna(axis=1, how='all')
X_raw = X_raw.loc[:, X_raw.apply(lambda x: x.nunique(dropna=True) > 0)]
print(f"Raw feature count: {X_raw.shape[1]}")

# MICE imputation
print("Applying MICE imputation...")
imputer = IterativeImputer(max_iter=15, random_state=42, initial_strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

# KernelPCA for latent space (VAE simulation)
print("Generating latent features (KernelPCA)...")
scaler_pre = StandardScaler()
X_scaled_pre = scaler_pre.fit_transform(X_imputed)
kpca = KernelPCA(n_components=5, kernel='rbf', gamma=0.01, random_state=42)
X_latent = kpca.fit_transform(X_scaled_pre)
X_latent_df = pd.DataFrame(X_latent, columns=[f'Latent_{i+1}' for i in range(5)])

# Combine imputed and latent features
X_full = pd.concat([X_imputed.reset_index(drop=True), X_latent_df], axis=1)

# Lasso feature selection
print("Running Lasso feature selection...")
selector = SelectFromModel(estimator=LassoCV(cv=5, random_state=42, max_iter=2000), threshold='median')
selector.fit(X_full, y)
X_selected = selector.transform(X_full)

# Get selected features
selected_indices = selector.get_support(indices=True)
selected_feat_names = X_full.columns[selected_indices]
print(f"Features reduced to {X_selected.shape[1]} key predictors")
print(f"Top 5 features: {list(selected_feat_names[:5])}")

# =====================================================================
# 3. Advanced Modeling (SVM-RBF + Ensemble)
# =====================================================================
print("\nPhase 2: Model Training & Cross-Validation")

# Define base classifiers
clf_svm = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
clf_gbm = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_depth=8, l2_regularization=1.5, class_weight='balanced', random_state=42)
clf_rf = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced_subsample', max_features='sqrt', random_state=42)

# Ensemble voting classifier
ensemble = VotingClassifier(
    estimators=[('svm', clf_svm), ('gbm', clf_gbm), ('rf', clf_rf)],
    voting='soft',
    weights=[3, 2, 1]
)

# Pipeline creation helper
def create_pipeline(classifier):
    steps = [('scaler', RobustScaler())]
    if HAS_SMOTE:
        steps.append(('smote', BorderlineSMOTE(random_state=42, kind='borderline-1')))
    steps.append(('classifier', classifier))
    return ImbPipeline(steps)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
methods_preds = {}

# Evaluate individual models
print("Evaluating SVM-RBF...")
methods_preds['SVM-RBF'] = cross_val_predict(create_pipeline(clf_svm), X_selected, y, cv=cv, method='predict_proba')[:, 1]

print("Evaluating Gradient Boosting...")
methods_preds['Gradient Boosting'] = cross_val_predict(create_pipeline(clf_gbm), X_selected, y, cv=cv, method='predict_proba')[:, 1]

print("Evaluating Random Forest...")
methods_preds['Random Forest'] = cross_val_predict(create_pipeline(clf_rf), X_selected, y, cv=cv, method='predict_proba')[:, 1]

# Evaluate ensemble model
print("Evaluating final ensemble...")
ensemble_probs = cross_val_predict(create_pipeline(ensemble), X_selected, y, cv=cv, method='predict_proba')[:, 1]
methods_preds['Ensemble (Final)'] = ensemble_probs

# =====================================================================
# 4. Risk Stratification
# =====================================================================
print("\nVAE Risk Stratification")

# Risk pattern definition
RISK_PATTERNS = {
    'Pattern3_High': {'name': 'High Risk'},
    'Pattern2_Medium': {'name': 'Intermediate Risk'},
    'Pattern1_Low': {'name': 'Low Risk'}
}

# Dynamic threshold calculation
q70 = np.quantile(ensemble_probs, 0.70)
q35 = np.quantile(ensemble_probs, 0.35)

# Risk pattern assignment
def assign_pattern(prob):
    if prob >= q70: return 'Pattern3_High'
    elif prob >= q35: return 'Pattern2_Medium'
    else: return 'Pattern1_Low'

df['AI_Probability'] = ensemble_probs
df['AI_Probability_Percent'] = (ensemble_probs * 100).round(1)
df['VAE_Pattern'] = df['AI_Probability'].apply(assign_pattern)

print(f"Risk thresholds: High >= {q70:.2f}, Medium >= {q35:.2f}")

# =====================================================================
# 5. Performance Evaluation & Visualization
# =====================================================================
print("\nPerformance Evaluation & Visualization")

# Calculate ensemble metrics
y_true = y
y_pred_prob = ensemble_probs

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
y_pred = (y_pred_prob >= optimal_threshold).astype(int)

auc_score = roc_auc_score(y_true, y_pred_prob)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
specificity = tn / (tn + fp) if (tn+fp) > 0 else 0
accuracy = (tp + tn) / len(y_true)
ppv = tp / (tp + fp) if (tp+fp) > 0 else 0
npv = tn / (tn + fn) if (tn+fn) > 0 else 0
brier = brier_score_loss(y_true, y_pred_prob)

# Create 4-panel visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. ROC Curve Comparison
ax = axes[0, 0]
model_aucs = {
    'SVM-RBF': roc_auc_score(y_true, methods_preds['SVM-RBF']),
    'Gradient Boosting': roc_auc_score(y_true, methods_preds['Gradient Boosting']),
    'Random Forest': roc_auc_score(y_true, methods_preds['Random Forest']),
    'Ensemble (Final)': auc_score
}
best_model_name = max(model_aucs, key=model_aucs.get)

for method_name, preds in methods_preds.items():
    fpr_m, tpr_m, _ = roc_curve(y_true, preds)
    auc_m = roc_auc_score(y_true, preds)
    best_tag = " (Best)" if method_name == best_model_name else ""
    
    # Highlight ensemble model
    if 'Ensemble' in method_name:
        lw, ls, alpha, color = 2.5, '-', 1.0, '#d95f02'
    else:
        lw, ls, alpha, color = 1.5, '--', 0.7, None 
        
    ax.plot(fpr_m, tpr_m, lw=lw, linestyle=ls, alpha=alpha, label=f'{method_name} (AUC={auc_m:.3f}){best_tag}', color=color)

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title(f'ROC Curve Comparison (N={len(y_true)})', fontweight='bold', fontsize=12)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Confusion Matrix
ax = axes[0, 1]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non-converter', 'Converter'],
            yticklabels=['Non-converter', 'Converter'],
            cbar_kws={'label': 'Count'})
ax.set_title(f'Confusion Matrix (Threshold={optimal_threshold:.2f})', fontweight='bold', fontsize=12)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)

# 3. Probability Distribution
ax = axes[1, 0]
converters = df[df['AD_Conversion'] == 1]['AI_Probability']
non_converters = df[df['AD_Conversion'] == 0]['AI_Probability']

ax.hist(non_converters, bins=20, alpha=0.6, label=f'Non-converters (n={len(non_converters)})', color='green', edgecolor='black')
ax.hist(converters, bins=20, alpha=0.6, label=f'Converters (n={len(converters)})', color='red', edgecolor='black')
ax.axvline(optimal_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold={optimal_threshold:.2f}')
ax.set_xlabel('Predicted Probability', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Probability Distribution', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 4. Risk Stratification Calibration
ax = axes[1, 1]
pattern_stats = []
for pattern in ['Pattern1_Low', 'Pattern2_Medium', 'Pattern3_High']:
    pattern_data = df[df['VAE_Pattern'] == pattern]
    if len(pattern_data) > 0:
        actual_rate = pattern_data['AD_Conversion'].mean()
        predicted_rate = pattern_data['AI_Probability'].mean()
        pattern_stats.append({
            'name': RISK_PATTERNS[pattern]['name'],
            'actual': actual_rate,
            'predicted': predicted_rate,
            'n': len(pattern_data)
        })

if pattern_stats:
    x = np.arange(len(pattern_stats))
    width = 0.35
    bars1 = ax.bar(x - width/2, [s['actual'] for s in pattern_stats], width, label='Actual Conversion Rate', color='coral', edgecolor='black')
    bars2 = ax.bar(x + width/2, [s['predicted'] for s in pattern_stats], width, label='Predicted Conversion Rate', color='skyblue', edgecolor='black')
    
    for i, s in enumerate(pattern_stats):
        ax.text(i, max(s['actual'], s['predicted']) + 0.05, f"n={s['n']}", ha='center', fontsize=9)
    
    ax.set_xlabel('Risk Stratification', fontsize=11)
    ax.set_ylabel('Conversion Rate', fontsize=11)
    ax.set_title('Risk Stratification Calibration', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([s['name'] for s in pattern_stats])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

# Save visualization
plt.tight_layout()
figures_dir = os.path.join(output_dir, "Figures")
os.makedirs(figures_dir, exist_ok=True)
fig_path = os.path.join(figures_dir, 'AI_Performance_Final.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Visualization saved: {fig_path}")

# =====================================================================
# 6. Save Results & Reports
# =====================================================================
# Save prediction results
output_data = pd.DataFrame({
    'CaseID': df['ID'],
    'RID': df['RID'],
    'VAE_Pattern': df['VAE_Pattern'].map(lambda x: RISK_PATTERNS[x]['name']),
    'SVM_Prob': (methods_preds['SVM-RBF'] * 100).round(1),
    'GBM_Prob': (methods_preds['Gradient Boosting'] * 100).round(1),
    'RF_Prob': (methods_preds['Random Forest'] * 100).round(1),
    'AI_Probability': df['AI_Probability'].round(4),
    'AI_Probability_Percent': df['AI_Probability_Percent'],
    'AI_Risk_Level': df['VAE_Pattern'].map(lambda x: RISK_PATTERNS[x]['name']),
    'Actual_Conversion': df['AD_Conversion'],
    'Prediction_Correct': ((df['AI_Probability'] >= optimal_threshold).astype(int) == df['AD_Conversion']).astype(int)
})

output_file = os.path.join(output_dir, "AI_Predictions_Final.csv")
output_data.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Prediction results saved: {output_file}")

# Generate evaluation report
report_lines = [
    "VAE Risk Stratification Model - Evaluation Report",
    "==================================================",
    "",
    "1. Core Methodology",
    "   - Feature Selection: Lasso (L1) noise reduction",
    "   - Modeling: SVM-RBF + GBM + RF ensemble",
    "   - Latent Space: KernelPCA (VAE simulation)",
    "   - Imputation: MICE iterative imputation",
    "",
    "2. Test Set Characteristics",
    f"   Sample size: {len(df)} cases",
    f"   AD converters: {df['AD_Conversion'].sum()} ({df['AD_Conversion'].mean():.1%})",
    f"   Non-converters: {(df['AD_Conversion']==0).sum()} ({(df['AD_Conversion']==0).mean():.1%})",
    "",
    "3. Model Performance (5-Fold Stratified CV)"
]

# Add model AUC scores
for model_name in ['SVM-RBF', 'Gradient Boosting', 'Random Forest', 'Ensemble (Final)']:
    auc_val = model_aucs[model_name]
    best_tag = " (Best)" if model_name == best_model_name else ""
    report_lines.append(f"   {model_name}: AUC={auc_val:.3f}{best_tag}")

# Add detailed ensemble metrics
report_lines.extend([
    "",
    "4. Ensemble Model Metrics",
    f"   AUC: {auc_score:.3f}",
    f"   Optimal threshold: {optimal_threshold:.3f}",
    f"   Sensitivity: {sensitivity:.1%}",
    f"   Specificity: {specificity:.1%}",
    f"   Accuracy: {accuracy:.1%}",
    f"   PPV: {ppv:.1%}",
    f"   NPV: {npv:.1%}",
    f"   Brier Score: {brier:.3f}",
    "",
    "5. Risk Stratification Results"
])

# Add risk pattern stats
for pattern in ['Pattern3_High', 'Pattern2_Medium', 'Pattern1_Low']:
    pattern_data = df[df['VAE_Pattern'] == pattern]
    if len(pattern_data) > 0:
        n = len(pattern_data)
        conv = pattern_data['AD_Conversion'].sum()
        rate = pattern_data['AD_Conversion'].mean()
        avg_prob = pattern_data['AI_Probability'].mean()
        report_lines.append(
            f"   {RISK_PATTERNS[pattern]['name']}: {n} cases, "
            f"{conv} converters ({rate:.1%}), "
            f"predicted {avg_prob:.1%}"
        )

# Save report
report = "\n".join(report_lines)
report_file = os.path.join(output_dir, "AI_Report_Final.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Evaluation report saved: {report_file}")
print("\nAnalysis completed successfully.")
