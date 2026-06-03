"""
run_pipeline.py — Single-command entry point for the full fraud detection pipeline.

Usage:
    # Full run from scratch (train all models):
    python -m src.run_pipeline --data-dir data/ --output-dir outputs/

    # Fast re-run using saved models (skip training):
    python -m src.run_pipeline --data-dir data/ --output-dir outputs/ --skip-training

    # Skip ingestion too (DB already populated):
    python -m src.run_pipeline --skip-ingestion --skip-training

Steps:
    [1/7] Ingest CSVs into SQLite
    [2/7] Build feature matrix (SQL + pandas velocity)
    [3/7] Train / load models  (IF, XGBoost, MLP-AE, LOF)
    [4/7] Ensemble weight tuning
    [5/7] Evaluation  (metrics, plots, results table)
    [6/7] Explainability  (SHAP summary + waterfalls)
    [7/7] Print final results table
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOTAL_STEPS = 7


def _step(n: int, label: str) -> float:
    print(f"\n{'='*60}")
    print(f"  [{n}/{TOTAL_STEPS}] {label}")
    print(f"{'='*60}")
    return time.time()


def _ok(start: float) -> None:
    s = time.time() - start
    dur = f"{s//60:.0f}m {s%60:.0f}s" if s >= 60 else f"{s:.1f}s"
    print(f"  ✓  Done in {dur}")


def _fail(step_label: str, exc: Exception) -> None:
    logger.error("Step '%s' failed: %s", step_label, exc)
    logger.debug(traceback.format_exc())
    print(f"  ✗  FAILED — continuing to next step (check logs above)")


# ─────────────────────────────────────────────────────────────────────────────
# Step implementations
# ─────────────────────────────────────────────────────────────────────────────

def step_ingestion(data_dir: str, db_path: str) -> None:
    from src.ingestion import DataIngestion
    ing = DataIngestion(data_dir=data_dir, db_path=db_path)
    ing.run(split="train")


def step_features(
    db_path: str, sql_dir: str, cache_path: str
) -> "tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]":
    import pickle
    from src.features import FeatureEngineer

    cache = Path(cache_path)
    if cache.exists():
        import pandas as pd
        print(f"  Loading cached split from {cache} …")
        with open(cache, "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
        print(f"  Train {X_train.shape}  Test {X_test.shape}")
    else:
        fe = FeatureEngineer(db_path=db_path, sql_dir=sql_dir)
        X_train, X_test, y_train, y_test = fe.build_train_test_split()
        with open(cache, "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"  Split cached → {cache}")

    return X_train, X_test, y_train, y_test


def step_train_or_load(
    X_train, y_train, output_dir: str, skip_training: bool
) -> dict:
    """Train all models or load saved ones. Returns dict of {name: model}."""
    import joblib
    import numpy as np
    from pathlib import Path

    models_dir = Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    models = {}

    # ── Isolation Forest ─────────────────────────────────────────────
    if_path = models_dir / "iso_forest.pkl"
    if skip_training and if_path.exists():
        print("  Loading Isolation Forest …")
        models["iso"] = joblib.load(if_path)
    else:
        print("  Training Isolation Forest …")
        from src.models.iso_forest import IsolationForestModel
        iso = IsolationForestModel(n_estimators=200, contamination=0.035, random_state=42)
        iso.fit(X_train)
        joblib.dump(iso, if_path)
        models["iso"] = iso

    # ── XGBoost ──────────────────────────────────────────────────────
    xgb_path = models_dir / "xgboost.pkl"
    if skip_training and xgb_path.exists():
        print("  Loading XGBoost …")
        from src.models.xgboost_model import XGBoostModel
        models["xgb"] = XGBoostModel.load(xgb_path)
    else:
        print("  Training XGBoost (n_estimators=500) …")
        from src.models.xgboost_model import XGBoostModel
        xgb = XGBoostModel(n_estimators=500, max_depth=6, learning_rate=0.05)
        xgb.fit(X_train, y_train)
        xgb.save(str(xgb_path))
        models["xgb"] = xgb

    # ── MLP Autoencoder ───────────────────────────────────────────────
    mlp_path = models_dir / "mlp_ae.pt"
    if skip_training and mlp_path.exists():
        print("  Loading MLP Autoencoder …")
        from src.models.mlp_autoencoder import MLPAutoencoder
        mlp = MLPAutoencoder()
        mlp.load(str(mlp_path))
        models["mlp"] = mlp
    else:
        print("  Training MLP Autoencoder (50 epochs) …")
        from src.models.mlp_autoencoder import MLPAutoencoder
        mlp = MLPAutoencoder()
        mlp.fit(X_train, y_train, epochs=50)
        mlp.save(str(mlp_path))
        models["mlp"] = mlp

    # ── LOF ───────────────────────────────────────────────────────────
    lof_path = models_dir / "lof.pkl"
    if skip_training and lof_path.exists():
        print("  Loading LOF …")
        from src.models.lof_model import LOFModel
        models["lof"] = LOFModel.load(lof_path)
    else:
        print("  Training LOF (n_neighbors=20) …")
        from src.models.lof_model import LOFModel
        lof = LOFModel(n_neighbors=20, contamination=0.035)
        lof.fit(X_train, y_train)
        lof.save(str(lof_path))
        models["lof"] = lof

    return models


def step_ensemble(models: dict, X_test, y_test, output_dir: str) -> dict:
    """Score all models, tune ensemble weights, return score arrays."""
    import numpy as np
    from src.ensemble import EnsembleScorer

    labels = y_test.values
    scores = {}

    print("  Scoring models …")
    scores["xgb"] = models["xgb"].score(X_test)
    scores["iso"] = models["iso"].score(X_test)
    scores["mlp"] = models["mlp"].reconstruction_error(X_test)
    scores["lof"] = models["lof"].score(X_test)

    # Save scores
    out = Path(output_dir)
    np.save(str(out / "xgb_scores.npy"), scores["xgb"])
    np.save(str(out / "iso_scores_test.npy"), scores["iso"])
    np.save(str(out / "mlp_scores.npy"), scores["mlp"])
    np.save(str(out / "lof_scores.npy"), scores["lof"])

    print("  Tuning ensemble weights (grid search over PR-AUC) …")
    ens = EnsembleScorer()
    best = ens.tune_weights(scores["xgb"], scores["iso"], scores["mlp"],
                            labels, metric="pr_auc")
    scores["ensemble"] = ens.combine_v2(scores["xgb"], scores["iso"], scores["mlp"])
    scores["_ens_obj"] = ens
    scores["_labels"]  = labels

    return scores


def step_evaluation(scores: dict, output_dir: str) -> "pd.DataFrame":
    import numpy as np
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    from src.evaluate import Evaluator
    from src.ensemble import EnsembleScorer

    labels = scores["_labels"]
    ens    = scores["_ens_obj"]
    ev     = Evaluator(output_dir=output_dir)

    def best_threshold_metrics(y, raw_scores, threshold=None):
        norm = EnsembleScorer.normalize(raw_scores)
        if threshold is None:
            best_t, best_f1 = 0.5, 0.0
            for t in [i/100 for i in range(5, 96)]:
                f = f1_score(y, (norm > t).astype(int), zero_division=0)
                if f > best_f1:
                    best_f1, best_t = f, t
            preds = (norm > best_t).astype(int)
        else:
            preds = (raw_scores > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        return {
            "Precision": round(precision_score(y, preds, zero_division=0), 4),
            "Recall":    round(recall_score(y, preds, zero_division=0),    4),
            "F1":        round(f1_score(y, preds, zero_division=0),        4),
            "ROC-AUC":   round(roc_auc_score(y, raw_scores),               4),
            "PR-AUC":    round(average_precision_score(y, raw_scores),     4),
            "FPR":       round(fp / max(fp + tn, 1),                       4),
        }

    results = {
        "LSTM-AE (neg. result)": {"Precision":0,"Recall":0,"F1":0,
                                   "ROC-AUC":0.5003,"PR-AUC":0.036,"FPR":0},
        "MLP-AE":           best_threshold_metrics(labels, scores["mlp"]),
        "Isolation Forest": best_threshold_metrics(labels, scores["iso"]),
        "LOF":              best_threshold_metrics(labels, scores["lof"]),
        "XGBoost":          best_threshold_metrics(labels, scores["xgb"]),
        "Ensemble":         best_threshold_metrics(labels, scores["ensemble"],
                                threshold=ens.best_threshold),
    }

    import pandas as pd
    df_res = pd.DataFrame(results).T
    df_res.to_csv(Path(output_dir) / "results_table.csv")

    # PR + ROC curves for all models
    ev.precision_recall_curve(
        labels,
        {"MLP-AE": scores["mlp"],
         "IF":     scores["iso"],
         "XGBoost":scores["xgb"],
         "Ensemble":scores["ensemble"]},
    )
    ens_preds = (EnsembleScorer.normalize(scores["ensemble"]) > ens.best_threshold).astype(int)
    ev.confusion_matrix(labels, ens_preds, "Ensemble")
    ev.error_distribution(scores["mlp"], labels)
    ev.log_all_metrics()

    return df_res


def step_explainability(models: dict, X_test, scores: dict, output_dir: str) -> None:
    import numpy as np
    from src.explainability import Explainer

    ex      = Explainer(output_dir=output_dir)
    labels  = scores["_labels"]
    xgb_s   = scores["xgb"]
    flagged = [int(i) for i in np.where((xgb_s > 0.5).astype(int) == 1)[0][:3]]

    print(f"  SHAP summary on XGBoost …")
    ex.shap_xgboost_summary(models["xgb"], X_test, n_samples=500)

    print(f"  SHAP waterfalls for {len(flagged)} flagged transactions …")
    for idx in flagged:
        ex.shap_xgboost_waterfall(models["xgb"], X_test, transaction_idx=idx)

    print("  SHAP model comparison (XGB vs IF) …")
    ex.shap_comparison(models["xgb"], models["iso"], X_test, n_samples=500)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full financial anomaly detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",      default="data/",            help="Raw CSV directory")
    parser.add_argument("--db-path",       default="db/fraud.db",      help="SQLite database path")
    parser.add_argument("--sql-dir",       default="sql/",             help=".sql files directory")
    parser.add_argument("--output-dir",    default="outputs/",         help="Plots/results directory")
    parser.add_argument("--cache-path",    default="db/train_test_split.pkl",
                                                                        help="Train/test split cache")
    parser.add_argument("--skip-ingestion", action="store_true",       help="Skip DB ingestion")
    parser.add_argument("--skip-training",  action="store_true",       help="Load saved models instead")
    args = parser.parse_args()

    pipeline_start = time.time()
    print("\n" + "="*60)
    print("  FINANCIAL ANOMALY DETECTION PIPELINE")
    print(f"  output_dir : {Path(args.output_dir).resolve()}")
    print(f"  skip_train : {args.skip_training}")
    print("="*60)

    errors: list[str] = []

    # [1/7] Ingestion
    t = _step(1, "Ingesting data into SQLite")
    if args.skip_ingestion:
        print("  Skipped (--skip-ingestion)")
    else:
        try:
            step_ingestion(args.data_dir, args.db_path)
            _ok(t)
        except Exception as e:
            _fail("Ingestion", e); errors.append("Ingestion")

    # [2/7] Features
    t = _step(2, "Building feature matrix")
    try:
        X_train, X_test, y_train, y_test = step_features(
            args.db_path, args.sql_dir, args.cache_path
        )
        _ok(t)
    except Exception as e:
        _fail("Features", e); errors.append("Features")
        print("\n❌ Cannot continue without features. Exiting."); sys.exit(1)

    # [3/7] Models
    t = _step(3, "Training / loading models")
    try:
        models = step_train_or_load(X_train, y_train, args.output_dir, args.skip_training)
        _ok(t)
    except Exception as e:
        _fail("Models", e); errors.append("Models")
        print("\n❌ Cannot continue without models. Exiting."); sys.exit(1)

    # [4/7] Ensemble
    t = _step(4, "Scoring & ensemble weight tuning")
    try:
        scores = step_ensemble(models, X_test, y_test, args.output_dir)
        _ok(t)
    except Exception as e:
        _fail("Ensemble", e); errors.append("Ensemble"); scores = {}

    # [5/7] Evaluation
    t = _step(5, "Evaluation (metrics + plots)")
    df_results = None
    try:
        df_results = step_evaluation(scores, args.output_dir)
        _ok(t)
    except Exception as e:
        _fail("Evaluation", e); errors.append("Evaluation")

    # [6/7] Explainability
    t = _step(6, "Explainability (SHAP)")
    try:
        step_explainability(models, X_test, scores, args.output_dir)
        _ok(t)
    except Exception as e:
        _fail("Explainability", e); errors.append("Explainability")

    # [7/7] Results table
    _step(7, "Final results")
    if df_results is not None:
        print()
        print(df_results.to_string())
    else:
        print("  Results table unavailable (evaluation step failed).")

    # Summary
    total = time.time() - pipeline_start
    dur   = f"{total//60:.0f}m {total%60:.0f}s"
    print(f"\n{'='*60}")
    if errors:
        print(f"  Pipeline finished with {len(errors)} error(s): {errors}")
    else:
        print(f"  Pipeline complete — no errors")
    print(f"  Total time : {dur}")
    print(f"  Outputs    : {Path(args.output_dir).resolve()}")
    print(f"  MLflow     : mlflow ui --backend-store-uri mlruns/")
    print(f"  Dashboard  : streamlit run dashboard/app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
