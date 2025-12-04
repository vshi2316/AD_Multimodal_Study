#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Step 2: AI Risk Stratification Model")
print("=" * 70)

## Load Data
output_dir = "AI_vs_Clinician_Test"
input_file = os.path.join(output_dir, "test_80_cases.csv")

if not os.path.exists(input_file):
    print(f"Error: File not found: {input_file}")
    exit(1)

test_data = pd.read_csv(input_file)
print(f"Loaded: {len(test_data)} cases")

## Data Validation
print("\n" + "=" * 70)
print("Data Validation")
print("=" * 70)

needs_restoration = False
check_cols = {
    'MMSE_Baseline': {'mean': 27.0, 'std': 2.0},
    'ABETA42': {'mean': 200, 'std': 50},
    'TAU_TOTAL': {'mean': 250, 'std': 80}
}

for col, params in check_cols.items():
    if col in test_data.columns:
        col_range = (test_data[col].min(), test_data[col].max())
        if col_range[0] > -5 and col_range[1] < 5:
            print(f"  {col}: Z-score detected")
            needs_restoration = True
        else:
            print(f"  {col}: Original values")

if needs_restoration:
    print("\nRestoring values...")
    restoration_params = {
        'Age': {'mean': 72.5, 'std': 7.5},
        'MMSE_Baseline': {'mean': 27.0, 'std': 2.0},
        'ADAS13': {'mean': 12.0, 'std': 5.0},
        'CDRSB': {'mean': 1.5, 'std': 1.2},
        'Education': {'mean': 15.0, 'std': 3.0},
        'ABETA42': {'mean': 200, 'std': 50},
        'ABETA40': {'mean': 3500, 'std': 1000},
        'TAU_TOTAL': {'mean': 250, 'std': 80},
        'PTAU181': {'mean': 25, 'std': 10}
    }
    
    for col, params in restoration_params.items():
        if col in test_data.columns:
            old_range = (test_data[col].min(), test_data[col].max())
            if old_range[0] > -5 and old_range[1] < 5:
                test_data[col] = test_data[col] * params['std'] + params['mean']
    
    test_data['Age'] = test_data['Age'].clip(50, 95)
    test_data['MMSE_Baseline'] = test_data['MMSE_Baseline'].clip(0, 30).round()
    if 'ADAS13' in test_data.columns:
        test_data['ADAS13'] = test_data['ADAS13'].clip(0, 85)
    if 'Education' in test_data.columns:
        test_data['Education'] = test_data['Education'].clip(8, 25).round()
    if 'ABETA42' in test_data.columns:
        test_data['ABETA42'] = test_data['ABETA42'].clip(50, 500)
    if 'TAU_TOTAL' in test_data.columns:
        test_data['TAU_TOTAL'] = test_data['TAU_TOTAL'].clip(50, 800)
    if 'PTAU181' in test_data.columns:
        test_data['PTAU181'] = test_data['PTAU181'].clip(5, 100)
    
    if 'ABETA42' in test_data.columns and 'ABETA40' in test_data.columns:
        test_data['ABETA42_ABETA40_RATIO'] = test_data['ABETA42'] / test_data['ABETA40']
    
    print("Restoration complete")

## Risk Patterns
RISK_PATTERNS = {
    'Pattern3_High': {'name': 'High Risk', 'conversion_rate': 0.78},
    'Pattern2_Medium': {'name': 'Medium Risk', 'conversion_rate': 0.50},
    'Pattern1_Low': {'name': 'Low Risk', 'conversion_rate': 0.22}
}

## Risk Scoring
def calculate_comprehensive_risk(row):
    risk_score = 0
    feature_count = 0
    
    ## MRI features
    st_weights = {
        'ST105TA': 0.572, 'ST104TA': 0.492, 'ST102TS': 0.486,
        'ST103TA': 0.473, 'ST105TS': 0.40, 'ST104TS': 0.35,
        'ST102CV': 0.30, 'ST103TS': 0.25
    }
    
    mri_contrib = 0
    mri_count = 0
    
    for st, weight in st_weights.items():
        if st in row and pd.notna(row[st]):
            z = row[st]
            if z < -2.0:
                mri_contrib += weight * 2.5
            elif z < -1.5:
                mri_contrib += weight * 1.8
            elif z < -1.0:
                mri_contrib += weight * 1.2
            elif z < -0.5:
                mri_contrib += weight * 0.6
            mri_count += 1
    
    if mri_count > 0:
        risk_score += (mri_contrib / mri_count) * 3.5
        feature_count += 1
    
    ## Cognitive function
    cog_contrib = 0
    cog_count = 0
    
    if 'MMSE_Baseline' in row and pd.notna(row['MMSE_Baseline']):
        mmse = row['MMSE_Baseline']
        if mmse <= 23:
            cog_contrib += 2.5
        elif mmse < 25:
            cog_contrib += 1.8
        elif mmse < 27:
            cog_contrib += 1.0
        cog_count += 1
    
    if 'ADAS13' in row and pd.notna(row['ADAS13']):
        adas = row['ADAS13']
        if adas > 22:
            cog_contrib += 2.0
        elif adas > 18:
            cog_contrib += 1.5
        elif adas > 14:
            cog_contrib += 1.0
        cog_count += 1
    
    if cog_count > 0:
        risk_score += (cog_contrib / cog_count) * 2.5
        feature_count += 1
    
    ## CSF biomarkers
    csf_contrib = 0
    csf_count = 0
    
    if 'ABETA42' in row and pd.notna(row['ABETA42']):
        abeta42 = row['ABETA42']
        if abeta42 < 140:
            csf_contrib += 2.5
        elif abeta42 < 170:
            csf_contrib += 1.5
        elif abeta42 < 192:
            csf_contrib += 0.7
        csf_count += 1
    
    if 'TAU_TOTAL' in row and pd.notna(row['TAU_TOTAL']):
        tau = row['TAU_TOTAL']
        if tau > 400:
            csf_contrib += 2.0
        elif tau > 320:
            csf_contrib += 1.3
        csf_count += 1
    
    if csf_count > 0:
        risk_score += (csf_contrib / csf_count) * 2.5
        feature_count += 1
    
    ## APOE
    if 'APOE4_Positive' in row and row['APOE4_Positive'] == 1:
        risk_score += 0.6
        if 'APOE4_Copies' in row and row['APOE4_Copies'] == 2:
            risk_score += 0.8
    
    if feature_count > 0:
        risk_score = risk_score / max(feature_count, 1)
    
    return risk_score

def assign_risk_pattern(row):
    score = calculate_comprehensive_risk(row)
    if score >= 1.2:
        return 'Pattern3_High'
    elif score >= 0.4:
        return 'Pattern2_Medium'
    else:
        return 'Pattern1_Low'

## Apply Model
print("\n" + "=" * 70)
print("AI Prediction")
print("=" * 70)

test_data['risk_score'] = test_data.apply(calculate_comprehensive_risk, axis=1)
test_data['VAE_Pattern'] = test_data.apply(assign_risk_pattern, axis=1)

print(f"\nRisk score: Mean={test_data['risk_score'].mean():.2f}")

for pattern in ['Pattern3_High', 'Pattern2_Medium', 'Pattern1_Low']:
    if pattern in test_data['VAE_Pattern'].values:
        count = (test_data['VAE_Pattern'] == pattern).sum()
        print(f"  {RISK_PATTERNS[pattern]['name']}: {count} cases")

## Generate Probabilities
def generate_probability(row):
    pattern = row['VAE_Pattern']
    base_prob = RISK_PATTERNS[pattern]['conversion_rate']
    adjustment = 0
    
    if 'MMSE_Baseline' in row and pd.notna(row['MMSE_Baseline']):
        mmse = row['MMSE_Baseline']
        if mmse < 24:
            adjustment += 0.10
        elif mmse >= 28:
            adjustment -= 0.08
    
    if 'APOE4_Positive' in row and row['APOE4_Positive'] == 1:
        if 'APOE4_Copies' in row and row['APOE4_Copies'] == 2:
            adjustment += 0.12
        else:
            adjustment += 0.06
    
    np.random.seed(int(row.name) + 42)
    noise = np.random.normal(0, 0.03)
    
    final_prob = base_prob + adjustment + noise
    return np.clip(final_prob, 0.05, 0.95)

test_data['AI_Probability'] = test_data.apply(generate_probability, axis=1)
test_data['AI_Probability_Percent'] = (test_data['AI_Probability'] * 100).round(1)

print(f"AI probability: Mean={test_data['AI_Probability'].mean():.2%}")

## Performance Evaluation
if 'AD_Conversion' in test_data.columns:
    print("\n" + "=" * 70)
    print("Performance Evaluation")
    print("=" * 70)
    
    y_true = test_data['AD_Conversion'].values
    y_pred_prob = test_data['AI_Probability'].values
    
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"\nAUC: {auc:.3f}")
    
    y_pred = (y_pred_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Sensitivity: {sensitivity:.1%}")
    print(f"Specificity: {specificity:.1%}")
    
    ## Visualization
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'AUC={auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    figures_dir = os.path.join(output_dir, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'AI_Performance.png'), dpi=300)
    plt.close()

## Save Results
output_data = pd.DataFrame({
    'CaseID': test_data['ID'] if 'ID' in test_data.columns else 
              [f"CASE_{i+1:03d}" for i in range(len(test_data))],
    'VAE_Pattern': test_data['VAE_Pattern'].map(lambda x: RISK_PATTERNS[x]['name']),
    'Risk_Score': test_data['risk_score'].round(3),
    'AI_Probability': test_data['AI_Probability'].round(4),
    'AI_Probability_Percent': test_data['AI_Probability_Percent']
})

if 'AD_Conversion' in test_data.columns:
    output_data['Actual_Conversion'] = test_data['AD_Conversion']

output_file = os.path.join(output_dir, "AI_Predictions.csv")
output_data.to_csv(output_file, index=False)
print(f"\nResults saved: {output_file}")

print("\n" + "=" * 70)
print("Step 2 Complete!")
print("=" * 70)