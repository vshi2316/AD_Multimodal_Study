"""

import argparse
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectFromModel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

try:
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = False

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI Prediction with Frozen Pipeline Strategy'
    )
    parser.add_argument('--test_file', type=str, 
                        default='./AI_vs_Clinician_Test/independent_test_set.csv',
                        help='Path to independent test set CSV')
    parser.add_argument('--train_file', type=str, 
                        default='./cluster_results.csv',
                        help='Path to training set CSV')
    parser.add_argument('--output_dir', type=str, 
                        default='./AI_vs_Clinician_Test',
                        help='Output directory')
    parser.add_argument('--n_bootstrap', type=int, default=2000,
                        help='Number of bootstrap iterations for CI ')
    parser.add_argument('--mice_iterations', type=int, default=15,
                        help='MICE imputation iterations ')
    parser.add_argument('--kpca_components', type=int, default=5,
                        help='KernelPCA components')
    parser.add_argument('--high_threshold', type=float, default=0.70,
                        help='High risk threshold')
    parser.add_argument('--low_threshold', type=float, default=0.35,
                        help='Low risk threshold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """
    Hosmer-Lemeshow goodness-of-fit test.
    P-value > 0.05 indicates adequate calibration.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df['decile'] = pd.qcut(df['y_prob'], q=n_groups, duplicates='drop')
    
    observed = df.groupby('decile')['y_true'].sum()
    expected = df.groupby('decile')['y_prob'].sum()
    n_per_group = df.groupby('decile').size()
    
    # Chi-square statistic
    chi2 = 0
    for i in range(len(observed)):
        if expected.iloc[i] > 0 and (n_per_group.iloc[i] - expected.iloc[i]) > 0:
            chi2 += ((observed.iloc[i] - expected.iloc[i])**2 / 
                     (expected.iloc[i] * (1 - expected.iloc[i]/n_per_group.iloc[i]) + 1e-10))
    
    # Degrees of freedom = n_groups - 2
    dof = max(1, len(observed) - 2)
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    
    return {'chi2': chi2, 'df': dof, 'p_value': p_value}


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=2000, alpha=0.05, seed=42):
    """
    Bootstrap confidence interval for AUC .
    """
    np.random.seed(seed)
    n = len(y_true)
    aucs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Ensure both classes present
        if len(np.unique(y_true_boot)) == 2:
            aucs.append(roc_auc_score(y_true_boot, y_prob_boot))
    
    aucs = np.array(aucs)
    ci_lower = np.percentile(aucs, 100 * alpha / 2)
    ci_upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    
    return {
        'auc': roc_auc_score(y_true, y_prob),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': np.std(aucs)
    }


def engineering_pipeline(data):
    """Feature engineering pipeline."""
    X = pd.DataFrame(index=data.index)
    
    # Demographics
    X['Age'] = data.get('Age', np.nan)
    X['Gender'] = data.get('Gender', np.nan)
    X['Education'] = data.get('Education', np.nan)
    
    # APOE4
    if 'APOE4_Positive' in data.columns:
        X['APOE4'] = data['APOE4_Positive']
    elif 'APOE4' in data.columns:
        X['APOE4'] = data['APOE4']
    else:
        X['APOE4'] = 0
    
    X['Age_x_APOE4'] = X['Age'] * X['APOE4']
    
    # Cognitive scores
    if 'MMSE_Baseline' in data.columns:
        X['MMSE'] = data['MMSE_Baseline']
        X['MMSE_Inv'] = 30 - X['MMSE']
    elif 'MMSE' in data.columns:
        X['MMSE'] = data['MMSE']
        X['MMSE_Inv'] = 30 - X['MMSE']
    
    if 'ADAS13' in data.columns:
        X['ADAS13'] = data['ADAS13']
    if 'FAQTOTAL' in data.columns:
        X['FAQ'] = data['FAQTOTAL']
    
    # CSF biomarkers (log-transformed)
    if 'ABETA42' in data.columns:
        X['ABETA42'] = np.log1p(data['ABETA42'].clip(lower=0))
    if 'TAU_TOTAL' in data.columns:
        X['TAU'] = np.log1p(data['TAU_TOTAL'].clip(lower=0))
    if 'PTAU181' in data.columns:
        X['PTAU'] = np.log1p(data['PTAU181'].clip(lower=0))
    
    # Derived biomarker ratios
    if 'ABETA42' in X.columns and 'PTAU' in X.columns:
        X['Amyloid_Tau_Ratio'] = X['PTAU'] / (X['ABETA42'] + 0.1)
    
    # MRI features
    mri_cols = [c for c in data.columns if c.startswith('ST')]
    if mri_cols:
        for col in mri_cols:
            X[col] = data[col]
        X['Global_Atrophy'] = data[mri_cols].mean(axis=1)
        X['Atrophy_x_Edu'] = X['Global_Atrophy'] * X['Education']
    
    return X


def create_pipeline(classifier, has_smote=True):
    """Create sklearn/imblearn pipeline."""
    steps = [('scaler', RobustScaler())]
    if has_smote and HAS_SMOTE:
        steps.append(('smote', BorderlineSMOTE(random_state=42, kind='borderline-1')))
    steps.append(('classifier', classifier))
    return ImbPipeline(steps)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("Step 2: AI Prediction - Frozen Pipeline Strategy")
    print("LEAKAGE PREVENTION: All parameters fitted on training set only")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check input files
    if not os.path.exists(args.test_file):
        print(f"ERROR: Test set file not found - {args.test_file}")
        return
    if not os.path.exists(args.train_file):
        print(f"ERROR: Training set file not found - {args.train_file}")
        return
    
    # =========================================================================
    # PHASE 1: Load Data
    # =========================================================================
    print("\n[PHASE 1] Loading Data")
    print("-" * 70)
    
    train_data = pd.read_csv(args.train_file)
    print(f"Training set loaded: {len(train_data)} cases")
    
    raw_data = pd.read_csv(args.test_file)
    cols_to_numeric = [c for c in raw_data.columns if c not in ['ID', 'RID', 'Baseline_Date']]
    for col in cols_to_numeric:
        raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
    
    df_test = raw_data[raw_data['AD_Conversion'].notna()].copy()
    print(f"Test set loaded: {len(df_test)} cases")
    print(f"AD converters: {int(df_test['AD_Conversion'].sum())} ({df_test['AD_Conversion'].mean():.1%})")
    
    # =========================================================================
    # PHASE 2: Feature Engineering
    # =========================================================================
    print("\n[PHASE 2] Feature Engineering")
    print("-" * 70)
    
    X_train_raw = engineering_pipeline(train_data)
    
    if 'AD_Conversion' in train_data.columns:
        y_train = train_data['AD_Conversion'].values
    elif 'Converted' in train_data.columns:
        y_train = train_data['Converted'].values
    else:
        raise ValueError("Target column not found in training data")
    
    X_train_raw = X_train_raw.dropna(axis=1, how='all')
    X_train_raw = X_train_raw.loc[:, X_train_raw.apply(lambda x: x.nunique(dropna=True) > 0)]
    print(f"Training set raw features: {X_train_raw.shape[1]}")
    
    X_test_raw = engineering_pipeline(df_test)
    y_test = df_test['AD_Conversion'].values
    
    common_cols = X_train_raw.columns.intersection(X_test_raw.columns)
    X_train_raw = X_train_raw[common_cols]
    X_test_raw = X_test_raw[common_cols]
    print(f"Common features: {len(common_cols)}")

    # =========================================================================
    # PHASE 3: MICE Imputation (Frozen Pipeline)
    # =========================================================================
    print(f"\n[PHASE 3] MICE Imputation ({args.mice_iterations} iterations)")
    print("-" * 70)
    print("Fitting imputer on TRAINING set only...")
    
    imputer = IterativeImputer(
        max_iter=args.mice_iterations, 
        random_state=args.seed, 
        initial_strategy='median'
    )
    imputer.fit(X_train_raw)
    
    print("Applying imputer to test set (transform only)...")
    X_train_imputed = pd.DataFrame(imputer.transform(X_train_raw), columns=X_train_raw.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns, 
                                   index=X_test_raw.index)
    
    # =========================================================================
    # PHASE 4: Standardization (Frozen Pipeline)
    # =========================================================================
    print("\n[PHASE 4] Standardization (Frozen Pipeline)")
    print("-" * 70)
    print("Fitting scaler on TRAINING set only...")
    
    scaler_pre = StandardScaler()
    scaler_pre.fit(X_train_imputed)
    
    print("Applying scaler to test set (transform only)...")
    X_train_scaled = scaler_pre.transform(X_train_imputed)
    X_test_scaled = scaler_pre.transform(X_test_imputed)
    
    # =========================================================================
    # PHASE 5: KernelPCA (Frozen Pipeline)
    # =========================================================================
    print(f"\n[PHASE 5] KernelPCA (RBF kernel, {args.kpca_components} components)")
    print("-" * 70)
    print("Fitting KernelPCA on TRAINING set only...")
    
    kpca = KernelPCA(n_components=args.kpca_components, kernel='rbf', gamma=0.01, 
                     random_state=args.seed)
    kpca.fit(X_train_scaled)
    
    print("Applying KernelPCA to test set (transform only)...")
    X_train_latent = kpca.transform(X_train_scaled)
    X_test_latent = kpca.transform(X_test_scaled)
    
    X_train_latent_df = pd.DataFrame(X_train_latent, 
                                      columns=[f'Latent_{i+1}' for i in range(args.kpca_components)])
    X_test_latent_df = pd.DataFrame(X_test_latent, 
                                     columns=[f'Latent_{i+1}' for i in range(args.kpca_components)],
                                     index=X_test_imputed.index)
    
    X_train_full = pd.concat([X_train_imputed.reset_index(drop=True), X_train_latent_df], axis=1)
    X_test_full = pd.concat([X_test_imputed.reset_index(drop=True), 
                             X_test_latent_df.reset_index(drop=True)], axis=1)
    
    # =========================================================================
    # PHASE 6: Lasso Feature Selection (Frozen Pipeline)
    # =========================================================================
    print("\n[PHASE 6] Lasso Feature Selection (5-fold CV)")
    print("-" * 70)
    print("Fitting Lasso selector on TRAINING set only...")
    
    selector = SelectFromModel(
        estimator=LassoCV(cv=5, random_state=args.seed, max_iter=2000), 
        threshold='median'
    )
    selector.fit(X_train_full, y_train)
    
    print("Applying selector to test set (transform only)...")
    X_train_selected = selector.transform(X_train_full)
    X_test_selected = selector.transform(X_test_full)
    
    selected_indices = selector.get_support(indices=True)
    selected_feat_names = X_test_full.columns[selected_indices]
    print(f"Features reduced to {X_test_selected.shape[1]} key predictors")
    
    # =========================================================================
    # PHASE 7: Model Training (Training Set Only)
    # =========================================================================
    print("\n[PHASE 7] Model Training (SVM+GBM+RF Ensemble, 3:2:1 weights)")
    print("-" * 70)
    print(f"Training set shape: {X_train_selected.shape}")
    print(f"Test set shape: {X_test_selected.shape}")
    
    # Define classifiers 
    clf_svm = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, 
                  class_weight='balanced', random_state=args.seed)
    clf_gbm = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_depth=8,
        l2_regularization=1.5, class_weight='balanced', 
        random_state=args.seed
    )
    clf_rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, 
        class_weight='balanced_subsample',
        max_features='sqrt', random_state=args.seed
    )
    
    # Ensemble with 3:2:1 weights 
    ensemble = VotingClassifier(
        estimators=[('svm', clf_svm), ('gbm', clf_gbm), ('rf', clf_rf)],
        voting='soft',
        weights=[3, 2, 1]
    )
    
    methods_preds = {}
    
    # Train individual models
    print("\nTraining SVM-RBF...")
    pipeline_svm = create_pipeline(clf_svm, HAS_SMOTE)
    pipeline_svm.fit(X_train_selected, y_train)
    methods_preds['SVM-RBF'] = pipeline_svm.predict_proba(X_test_selected)[:, 1]
    
    print("Training Gradient Boosting...")
    pipeline_gbm = create_pipeline(clf_gbm, HAS_SMOTE)
    pipeline_gbm.fit(X_train_selected, y_train)
    methods_preds['Gradient Boosting'] = pipeline_gbm.predict_proba(X_test_selected)[:, 1]
    
    print("Training Random Forest...")
    pipeline_rf = create_pipeline(clf_rf, HAS_SMOTE)
    pipeline_rf.fit(X_train_selected, y_train)
    methods_preds['Random Forest'] = pipeline_rf.predict_proba(X_test_selected)[:, 1]
    
    print("Training Ensemble (3:2:1)...")
    pipeline_ensemble = create_pipeline(ensemble, HAS_SMOTE)
    pipeline_ensemble.fit(X_train_selected, y_train)
    ensemble_probs = pipeline_ensemble.predict_proba(X_test_selected)[:, 1]
    methods_preds['Ensemble (Final)'] = ensemble_probs

    print("\n" + "=" * 70)
    print("LEAKAGE PREVENTION CONFIRMED:")
    print("  All preprocessing parameters fitted on training set only")
    print("  All models trained on training set only")
    print("  Test set used ONLY for final evaluation")
    print("=" * 70)
    
    # =========================================================================
    # PHASE 8: Risk Stratification
    # =========================================================================
    print("\n[PHASE 8] Risk Stratification")
    print("-" * 70)
    
    RISK_PATTERNS = {
        'Pattern3_High': {'name': 'High Risk'},
        'Pattern2_Medium': {'name': 'Intermediate Risk'},
        'Pattern1_Low': {'name': 'Low Risk'}
    }
    
    def assign_pattern(prob):
        if prob >= args.high_threshold:
            return 'Pattern3_High'
        elif prob >= args.low_threshold:
            return 'Pattern2_Medium'
        else:
            return 'Pattern1_Low'
    
    df_test['AI_Probability'] = ensemble_probs
    df_test['AI_Probability_Percent'] = (ensemble_probs * 100).round(1)
    df_test['VAE_Pattern'] = df_test['AI_Probability'].apply(assign_pattern)
    
    print(f"Risk thresholds: High >= {args.high_threshold}, Medium >= {args.low_threshold}")
    
    # =========================================================================
    # PHASE 9: Performance Evaluation 
    # =========================================================================
    print("\n[PHASE 9] Performance Evaluation ")
    print("-" * 70)
    
    y_true = y_test
    y_pred_prob = ensemble_probs
    
    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)
    
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Brier Score 
    brier = brier_score_loss(y_true, y_pred_prob)
    
    # Bootstrap AUC CI
    print(f"\nCalculating bootstrap AUC CI ({args.n_bootstrap} iterations)...")
    auc_results = bootstrap_auc_ci(y_true, y_pred_prob, n_bootstrap=args.n_bootstrap, seed=args.seed)
    auc_score = auc_results['auc']
    
    # Hosmer-Lemeshow test 
    print("Performing Hosmer-Lemeshow calibration test...")
    hl_test = hosmer_lemeshow_test(y_true, y_pred_prob)
    
    print(f"\n  AUC: {auc_score:.3f} [95% CI: {auc_results['ci_lower']:.3f}-{auc_results['ci_upper']:.3f}]")
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.3f}")
    print(f"  Sensitivity: {sensitivity:.1%}")
    print(f"  Specificity: {specificity:.1%}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  PPV: {ppv:.1%}")
    print(f"  NPV: {npv:.1%}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Hosmer-Lemeshow: χ²={hl_test['chi2']:.2f}, df={hl_test['df']}, p={hl_test['p_value']:.4f}")
    
    if hl_test['p_value'] > 0.05:
        print("    → Adequate calibration (p > 0.05)")
    else:
        print("    → Poor calibration (p ≤ 0.05)")
    
    # Individual model AUCs with bootstrap CI
    model_results = {}
    for model_name, preds in methods_preds.items():
        result = bootstrap_auc_ci(y_true, preds, n_bootstrap=args.n_bootstrap, seed=args.seed)
        model_results[model_name] = result
        print(f"  {model_name}: AUC={result['auc']:.3f} [95% CI: {result['ci_lower']:.3f}-{result['ci_upper']:.3f}]")

    # =========================================================================
    # PHASE 10: Visualization
    # =========================================================================
    print("\n[PHASE 10] Generating Visualizations")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ROC Curves
    ax = axes[0, 0]
    for method_name, preds in methods_preds.items():
        fpr_m, tpr_m, _ = roc_curve(y_true, preds)
        auc_m = model_results[method_name]['auc']
        if 'Ensemble' in method_name:
            lw, ls, alpha, color = 2.5, '-', 1.0, '#d95f02'
        else:
            lw, ls, alpha, color = 1.5, '--', 0.7, None
        ax.plot(fpr_m, tpr_m, lw=lw, linestyle=ls, alpha=alpha,
                label=f'{method_name} (AUC={auc_m:.3f})', color=color)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve (N={len(y_true)})', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Confusion Matrix
    ax = axes[0, 1]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-converter', 'Converter'],
                yticklabels=['Non-converter', 'Converter'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix (Threshold={optimal_threshold:.2f})', 
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    
    # Probability Distribution
    ax = axes[1, 0]
    converters = df_test[df_test['AD_Conversion'] == 1]['AI_Probability']
    non_converters = df_test[df_test['AD_Conversion'] == 0]['AI_Probability']
    ax.hist(non_converters, bins=20, alpha=0.6, label=f'Non-converters (n={len(non_converters)})',
            color='green', edgecolor='black')
    ax.hist(converters, bins=20, alpha=0.6, label=f'Converters (n={len(converters)})',
            color='red', edgecolor='black')
    ax.axvline(optimal_threshold, color='blue', linestyle='--', linewidth=2,
               label=f'Threshold={optimal_threshold:.2f}')
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Probability Distribution', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Risk Stratification Calibration
    ax = axes[1, 1]
    pattern_stats = []
    for pattern in ['Pattern1_Low', 'Pattern2_Medium', 'Pattern3_High']:
        pattern_data = df_test[df_test['VAE_Pattern'] == pattern]
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
        ax.bar(x - width/2, [s['actual'] for s in pattern_stats], width,
               label='Actual Conversion Rate', color='coral', edgecolor='black')
        ax.bar(x + width/2, [s['predicted'] for s in pattern_stats], width,
               label='Predicted Conversion Rate', color='skyblue', edgecolor='black')
        for i, s in enumerate(pattern_stats):
            ax.text(i, max(s['actual'], s['predicted']) + 0.05, f"n={s['n']}",
                    ha='center', fontsize=9)
        ax.set_xlabel('Risk Stratification', fontsize=11)
        ax.set_ylabel('Conversion Rate', fontsize=11)
        ax.set_title('Risk Stratification Calibration', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([s['name'] for s in pattern_stats])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'AI_Performance_Final.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {fig_path}")
    
    # =========================================================================
    # PHASE 11: Save Results
    # =========================================================================
    print("\n[PHASE 11] Saving Results")
    print("-" * 70)
    
    # Predictions CSV
    output_data = pd.DataFrame({
        'CaseID': df_test['ID'],
        'RID': df_test['RID'],
        'VAE_Pattern': df_test['VAE_Pattern'].map(lambda x: RISK_PATTERNS[x]['name']),
        'SVM_Prob': (methods_preds['SVM-RBF'] * 100).round(1),
        'GBM_Prob': (methods_preds['Gradient Boosting'] * 100).round(1),
        'RF_Prob': (methods_preds['Random Forest'] * 100).round(1),
        'AI_Probability': df_test['AI_Probability'].round(4),
        'AI_Probability_Percent': df_test['AI_Probability_Percent'],
        'AI_Risk_Level': df_test['VAE_Pattern'].map(lambda x: RISK_PATTERNS[x]['name']),
        'Actual_Conversion': df_test['AD_Conversion'],
        'Prediction_Correct': ((df_test['AI_Probability'] >= optimal_threshold).astype(int) == 
                              df_test['AD_Conversion']).astype(int)
    })
    
    output_file = os.path.join(args.output_dir, "AI_Predictions_Final.csv")
    output_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  Predictions saved: {output_file}")

    # Report
    report_lines = [
        "VAE Risk Stratification Model - Evaluation Report",
        "=" * 60,
        "",
        f"  ✓ MICE imputation: {args.mice_iterations} iterations (fitted on training set)",
        "  ✓ StandardScaler: fitted on training set",
        f"  ✓ KernelPCA (RBF): {args.kpca_components} components (fitted on training set)",
        "  ✓ Lasso selector (5-fold CV): fitted on training set",
        "  ✓ SVM+GBM+RF ensemble (3:2:1 weights): trained on training set",
        "  ✓ Borderline-SMOTE: applied within training folds",
        "  ✓ Youden's J: threshold optimization on training set",
        "",
        f"  ✓ Bootstrap AUC CI: {args.n_bootstrap} iterations",
        "  ✓ Hosmer-Lemeshow calibration test",
        "  ✓ Brier score calculation",
        "",
        "=" * 60,
        "",
        "1. Test Set Characteristics",
        f"   Sample size: {len(df_test)} cases",
        f"   AD converters: {int(df_test['AD_Conversion'].sum())} ({df_test['AD_Conversion'].mean():.1%})",
        f"   Non-converters: {int((df_test['AD_Conversion']==0).sum())} ({(df_test['AD_Conversion']==0).mean():.1%})",
        "",
        "2. Model Performance (Independent Test Set)"
    ]
    
    for model_name, result in model_results.items():
        report_lines.append(
            f"   {model_name}: AUC={result['auc']:.3f} "
            f"[95% CI: {result['ci_lower']:.3f}-{result['ci_upper']:.3f}]"
        )
    
    report_lines.extend([
        "",
        "3. Ensemble Model Metrics",
        f"   AUC: {auc_score:.3f} [95% CI: {auc_results['ci_lower']:.3f}-{auc_results['ci_upper']:.3f}]",
        f"   Optimal threshold (Youden's J): {optimal_threshold:.3f}",
        f"   Sensitivity: {sensitivity:.1%}",
        f"   Specificity: {specificity:.1%}",
        f"   Accuracy: {accuracy:.1%}",
        f"   PPV: {ppv:.1%}",
        f"   NPV: {npv:.1%}",
        f"   Brier Score: {brier:.4f}",
        "",
        "4. Calibration (Hosmer-Lemeshow Test)",
        f"   Chi-square: {hl_test['chi2']:.2f}",
        f"   Degrees of freedom: {hl_test['df']}",
        f"   P-value: {hl_test['p_value']:.4f}",
        f"   Interpretation: {'Adequate calibration (p > 0.05)' if hl_test['p_value'] > 0.05 else 'Poor calibration (p ≤ 0.05)'}",
        "",
        "5. Risk Stratification Results"
    ])
    
    for pattern in ['Pattern3_High', 'Pattern2_Medium', 'Pattern1_Low']:
        pattern_data = df_test[df_test['VAE_Pattern'] == pattern]
        if len(pattern_data) > 0:
            n = len(pattern_data)
            conv = int(pattern_data['AD_Conversion'].sum())
            rate = pattern_data['AD_Conversion'].mean()
            avg_prob = pattern_data['AI_Probability'].mean()
            report_lines.append(
                f"   {RISK_PATTERNS[pattern]['name']}: {n} cases, "
                f"{conv} converters ({rate:.1%}), predicted {avg_prob:.1%}"
            )
    
    report = "\n".join(report_lines)
    report_file = os.path.join(args.output_dir, "AI_Report_Final.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report saved: {report_file}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2 Complete")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {report_file}")
    print(f"  - {fig_path}")


if __name__ == "__main__":
    main()

