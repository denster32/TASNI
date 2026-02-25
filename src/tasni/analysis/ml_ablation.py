#!/usr/bin/env python3
"""
TASNI ML Ablation Study
========================

Quantifies the contribution of variability features to the ML classification
pipeline by comparing model performance with and without them.

Two-stage analysis:
  1. Full-sample ablation (3375 tier-5 sources, 95 golden overlap):
     Train XGBoost to predict golden-sample membership using all available
     features vs. non-variability features only. Because lightcurve-derived
     features are NaN for non-golden sources (filled to 0), the presence of
     variability data itself acts as a near-perfect predictor -- we quantify
     this leakage and report it transparently.

  2. Feature-importance decomposition:
     Train on the full feature set and report gain-based importances to show
     which feature groups drive the ranking.

Outputs:
  - output/ml_ablation/ablation_results.json
  - output/ml_ablation/appendix_tables.tex
"""

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project root (two levels up from this file, or use cwd)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Variability feature prefixes/substrings to ablate
VARIABILITY_PREFIXES = (
    "trend_w1",
    "trend_w2",
    "rms_w1",
    "rms_w2",
    "chi2_w1",
    "chi2_w2",
    "is_fading",
    "is_variable",
    "variability_score",
    "slope_w1",
    "slope_w2",
    "fading",
    "var_",
    # Also lightcurve-derived columns in the feature parquet
    "w1_rms",
    "w2_rms",
    "w1_std",
    "w2_std",
    "w1_mean",
    "w2_mean",
    "w1_range",
    "w2_range",
    "w1_n_epochs",
    "w2_n_epochs",
)


def _is_variability_feature(col_name):
    """Check whether a column name matches a variability feature prefix."""
    lower = col_name.lower()
    return any(lower.startswith(p) or lower == p for p in VARIABILITY_PREFIXES)


def load_data(features_path, golden_csv):
    """Load feature matrix and create binary golden-membership labels.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric columns only, NaN filled to 0).
    y : np.ndarray
        Binary labels (1 = golden, 0 = not golden).
    feature_cols : list[str]
        Column names used as features.
    """
    feat = pd.read_parquet(features_path)
    golden = pd.read_csv(golden_csv)

    golden_set = set(golden["designation"].tolist())
    y = np.array([1 if d in golden_set else 0 for d in feat.index], dtype=int)

    logger.info(
        "Loaded %d sources (%d golden, %d non-golden)",
        len(feat),
        y.sum(),
        (y == 0).sum(),
    )

    # Keep all numeric columns; exclude ph_qual_value (categorical encoding)
    exclude = {"ph_qual_value"}
    feature_cols = [
        c
        for c in feat.columns
        if feat[c].dtype in ("float64", "float32", "int64", "int32", "bool") and c not in exclude
    ]

    X = feat[feature_cols].fillna(0).copy()
    return X, y, feature_cols


def run_with_features(X, y, feature_cols, model_type="xgboost", n_splits=5):
    """Train a classifier via stratified k-fold CV and return AUC stats.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : np.ndarray
        Binary labels.
    feature_cols : list[str]
        Subset of columns to use.
    model_type : str
        One of 'xgboost', 'lightgbm', 'random_forest'.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    dict with roc_auc_mean, roc_auc_std, and per-fold scores.
    """
    Xsub = X[feature_cols].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for train_idx, test_idx in skf.split(Xsub, y):
        X_train, X_test = Xsub[train_idx], Xsub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_type == "xgboost":
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "seed": 42,
                "verbosity": 0,
            }
            model = xgb.train(params, dtrain, num_boost_round=100)
            preds = model.predict(dtest)
        elif model_type == "lightgbm":
            train_data = lgb.Dataset(X_train, label=y_train)
            params = {
                "objective": "binary",
                "metric": "auc",
                "max_depth": 4,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1,
                "seed": 42,
            }
            model = lgb.train(params, train_data, num_boost_round=100)
            preds = model.predict(X_test)
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        auc = roc_auc_score(y_test, preds)
        aucs.append(auc)

    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "fold_aucs": [float(a) for a in aucs],
        "n_features": len(feature_cols),
    }


def get_feature_importances(X, y, feature_cols):
    """Train XGBoost on full data and extract gain-based importances."""
    dtrain = xgb.DMatrix(X[feature_cols].values, label=y, feature_names=feature_cols)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "seed": 42,
        "verbosity": 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    importance = model.get_score(importance_type="gain")

    # Normalise to sum to 1
    total = sum(importance.values())
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    result = []
    for rank_idx, (feat, gain) in enumerate(ranked, 1):
        result.append(
            {
                "feature": feat,
                "importance": round(gain / total, 4) if total > 0 else 0.0,
                "rank": rank_idx,
            }
        )
    return result


def compute_golden_overlap(X, y, all_cols, novar_cols):
    """Train models on ALL and NO_VAR features, predict top 100, measure overlap."""
    # ALL features model
    dtrain_all = xgb.DMatrix(X[all_cols].values, label=y, feature_names=all_cols)
    params = {
        "objective": "binary:logistic",
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "seed": 42,
        "verbosity": 0,
    }
    model_all = xgb.train(params, dtrain_all, num_boost_round=100)
    preds_all = model_all.predict(dtrain_all)

    # NO_VAR features model
    dtrain_novar = xgb.DMatrix(X[novar_cols].values, label=y, feature_names=novar_cols)
    model_novar = xgb.train(params, dtrain_novar, num_boost_round=100)
    preds_novar = model_novar.predict(dtrain_novar)

    # Top 100 by each
    top100_all = set(np.argsort(preds_all)[-100:])
    top100_novar = set(np.argsort(preds_novar)[-100:])
    golden_idx = set(np.where(y == 1)[0])

    overlap_all = len(top100_all & golden_idx)
    overlap_novar = len(top100_novar & golden_idx)
    top100_intersection = len(top100_all & top100_novar)
    jaccard = (
        top100_intersection / len(top100_all | top100_novar) if (top100_all | top100_novar) else 0
    )

    return {
        "all_features_top100": int(overlap_all),
        "no_var_features_top100": int(overlap_novar),
        "top100_overlap_both_lists": int(top100_intersection),
        "jaccard_similarity": round(jaccard, 3),
    }


def run_ablation(
    features_path=None,
    golden_csv=None,
    variability_prefixes=VARIABILITY_PREFIXES,
):
    """Run the full ablation study.

    Returns a dict suitable for JSON serialisation.
    """
    if features_path is None:
        features_path = PROJECT_ROOT / "output" / "features" / "tier5_features.parquet"
    if golden_csv is None:
        golden_csv = PROJECT_ROOT / "data" / "processed" / "final" / "golden_improved.csv"

    X, y, all_feature_cols = load_data(features_path, golden_csv)

    # Split into variability vs non-variability
    var_cols = [c for c in all_feature_cols if _is_variability_feature(c)]
    novar_cols = [c for c in all_feature_cols if not _is_variability_feature(c)]

    logger.info(
        "Feature split: %d all, %d variability, %d non-variability",
        len(all_feature_cols),
        len(var_cols),
        len(novar_cols),
    )
    logger.info("Variability features: %s", var_cols)

    # --- Stage 1: XGBoost ablation ---
    logger.info("Running XGBoost CV with ALL features...")
    res_all = run_with_features(X, y, all_feature_cols, model_type="xgboost")

    logger.info("Running XGBoost CV with NO_VAR features...")
    res_novar = run_with_features(X, y, novar_cols, model_type="xgboost")

    auc_drop = res_all["roc_auc_mean"] - res_novar["roc_auc_mean"]
    auc_drop_pct = (auc_drop / res_all["roc_auc_mean"] * 100) if res_all["roc_auc_mean"] > 0 else 0

    # --- Additional models for robustness ---
    logger.info("Running LightGBM CV with ALL features...")
    res_lgb_all = run_with_features(X, y, all_feature_cols, model_type="lightgbm")

    logger.info("Running LightGBM CV with NO_VAR features...")
    res_lgb_novar = run_with_features(X, y, novar_cols, model_type="lightgbm")

    logger.info("Running RandomForest CV with ALL features...")
    res_rf_all = run_with_features(X, y, all_feature_cols, model_type="random_forest")

    logger.info("Running RandomForest CV with NO_VAR features...")
    res_rf_novar = run_with_features(X, y, novar_cols, model_type="random_forest")

    # --- Feature importances ---
    logger.info("Computing feature importances...")
    importances = get_feature_importances(X, y, all_feature_cols)
    top20 = importances[:20]

    # Compute variability share of total importance
    var_importance_total = sum(
        item["importance"] for item in importances if _is_variability_feature(item["feature"])
    )

    # --- Golden overlap ---
    logger.info("Computing golden overlap...")
    overlap = compute_golden_overlap(X, y, all_feature_cols, novar_cols)

    # --- NaN leakage check ---
    # Count how many variability features are non-zero for golden vs non-golden
    golden_nonzero = X.loc[y == 1, var_cols].ne(0).any(axis=1).sum() if var_cols else 0
    nongolden_nonzero = X.loc[y == 0, var_cols].ne(0).any(axis=1).sum() if var_cols else 0

    results = {
        "method": "XGBoost_ablation",
        "n_sources_total": int(len(X)),
        "n_golden": int(y.sum()),
        "feature_sets": {
            "all_features": {
                "n_features": res_all["n_features"],
                "roc_auc_mean": round(res_all["roc_auc_mean"], 4),
                "roc_auc_std": round(res_all["roc_auc_std"], 4),
                "fold_aucs": res_all["fold_aucs"],
            },
            "no_variability_features": {
                "n_features": res_novar["n_features"],
                "n_removed": len(var_cols),
                "removed_features": var_cols,
                "roc_auc_mean": round(res_novar["roc_auc_mean"], 4),
                "roc_auc_std": round(res_novar["roc_auc_std"], 4),
                "fold_aucs": res_novar["fold_aucs"],
            },
        },
        "auc_drop": round(auc_drop, 4),
        "auc_drop_percent": round(auc_drop_pct, 1),
        "robustness_check": {
            "lightgbm": {
                "all_auc": round(res_lgb_all["roc_auc_mean"], 4),
                "novar_auc": round(res_lgb_novar["roc_auc_mean"], 4),
                "auc_drop": round(res_lgb_all["roc_auc_mean"] - res_lgb_novar["roc_auc_mean"], 4),
            },
            "random_forest": {
                "all_auc": round(res_rf_all["roc_auc_mean"], 4),
                "novar_auc": round(res_rf_novar["roc_auc_mean"], 4),
                "auc_drop": round(res_rf_all["roc_auc_mean"] - res_rf_novar["roc_auc_mean"], 4),
            },
        },
        "variability_importance_share": round(var_importance_total, 4),
        "top20_features": top20,
        "golden_overlap": overlap,
        "nan_leakage_check": {
            "note": (
                "Variability features are non-zero almost exclusively for golden "
                "sources because lightcurve analysis was only performed on the "
                "golden sample. This creates information leakage: the model can "
                "distinguish golden from non-golden by the presence/absence of "
                "variability data rather than by the feature values themselves."
            ),
            "golden_with_nonzero_var": int(golden_nonzero),
            "nongolden_with_nonzero_var": int(nongolden_nonzero),
        },
        "hyperparameters": {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
            },
            "isolation_forest": {
                "n_estimators": 200,
                "contamination": 0.001,
                "max_features": 0.8,
            },
            "lightgbm": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.05,
                "num_leaves": 31,
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_leaf": 2,
            },
        },
    }

    return results


def format_latex_feature_table(importances):
    """Generate AASTeX deluxetable* for top-20 feature importances."""
    var_total = sum(
        item["importance"] for item in importances if _is_variability_feature(item["feature"])
    )
    var_pct = var_total * 100

    lines = [
        r"\begin{deluxetable*}{lcc}",
        r"\tablecaption{Top-20 XGBoost Feature Importances\label{tbl:features}}",
        r"\tablehead{",
        r"  \colhead{Feature} & \colhead{Importance (Gain)} & \colhead{Rank}",
        r"}",
        r"\startdata",
    ]

    top20 = importances[:20]
    for i, item in enumerate(top20):
        feat_name = item["feature"].replace("_", r"\_")
        imp = f'{item["importance"]:.4f}'
        rank = str(item["rank"])
        sep = r" \\" if i < len(top20) - 1 else ""
        lines.append(f"{feat_name} & {imp} & {rank}{sep}")

    lines.append(r"\enddata")
    lines.append(
        r"\tablecomments{Feature importances from XGBoost trained on all features "
        r"to predict golden-sample membership. Gain-based importance measures the "
        r"total improvement in the loss function attributed to each feature. "
        f"Variability features (rms, std, range, epochs) collectively account for "
        f"{var_pct:.1f}\\% of total feature importance. "
        r"Note that lightcurve-derived variability features are available only for "
        r"the golden sample (Section~\ref{subsec:fading}), so their discriminative "
        r"power partly reflects information leakage rather than intrinsic physical "
        r"differences (Section~\ref{subsec:limitations}).}"
    )
    lines.append(r"\end{deluxetable*}")

    return "\n".join(lines)


def format_latex_hyperparameter_table():
    """Generate AASTeX deluxetable* for ML hyperparameter settings."""
    lines = [
        r"\begin{deluxetable*}{lll}",
        r"\tablecaption{Machine Learning Hyperparameter Settings\label{tbl:hyperparams}}",
        r"\tablehead{",
        r"  \colhead{Model} & \colhead{Parameter} & \colhead{Value}",
        r"}",
        r"\startdata",
        r"XGBoost & n\_estimators & 100 \\",
        r"XGBoost & max\_depth & 4 \\",
        r"XGBoost & learning\_rate & 0.1 \\",
        r"XGBoost & subsample & 0.8 \\",
        r"Isolation Forest & n\_estimators & 200 \\",
        r"Isolation Forest & contamination & 0.001 \\",
        r"Isolation Forest & max\_features & 0.8 \\",
        r"LightGBM & n\_estimators & 100 \\",
        r"LightGBM & max\_depth & 4 \\",
        r"LightGBM & learning\_rate & 0.05 \\",
        r"LightGBM & num\_leaves & 31 \\",
        r"Random Forest & n\_estimators & 100 \\",
        r"Random Forest & max\_depth & 6 \\",
        r"Random Forest & min\_samples\_leaf & 2",
        r"\enddata",
        r"\tablecomments{Hyperparameters used for each classifier in the ablation "
        r"study. XGBoost and LightGBM values match those used in the production "
        r"pipeline (Section~\ref{subsec:pipeline}). Isolation Forest parameters follow "
        r"the contamination rate estimated from the WISE orphan catalog.}",
        r"\end{deluxetable*}",
    ]
    return "\n".join(lines)


def main():
    """Run ablation and write outputs."""
    output_dir = PROJECT_ROOT / "output" / "ml_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_ablation()

    # Write JSON
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", json_path)

    # Write LaTeX
    tex_path = output_dir / "appendix_tables.tex"
    feature_table = format_latex_feature_table(results["top20_features"])
    hyperparam_table = format_latex_hyperparameter_table()

    with open(tex_path, "w") as f:
        f.write("% Auto-generated by ml_ablation.py\n")
        f.write("% Feature importance table (Table B1)\n\n")
        f.write(feature_table)
        f.write("\n\n")
        f.write("% Hyperparameter table (Table B2)\n\n")
        f.write(hyperparam_table)
        f.write("\n")
    logger.info("Wrote %s", tex_path)

    # Summary
    logger.info(
        "ABLATION SUMMARY: AUC all=%.4f, AUC novar=%.4f, drop=%.4f (%.1f%%)",
        results["feature_sets"]["all_features"]["roc_auc_mean"],
        results["feature_sets"]["no_variability_features"]["roc_auc_mean"],
        results["auc_drop"],
        results["auc_drop_percent"],
    )
    if results["top20_features"]:
        logger.info(
            "Top feature: %s (importance=%.4f)",
            results["top20_features"][0]["feature"],
            results["top20_features"][0]["importance"],
        )
    logger.info(
        "Golden overlap: Jaccard=%.3f",
        results["golden_overlap"]["jaccard_similarity"],
    )


if __name__ == "__main__":
    main()
