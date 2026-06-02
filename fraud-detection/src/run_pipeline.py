"""
run_pipeline.py — End-to-end fraud detection pipeline runner.

Executes the full pipeline in sequence:
  1. Data ingestion (CSV → SQLite)
  2. Feature engineering (SQL → feature matrix)
  3. Isolation Forest training + scoring
  4. Evaluation (plots, results table)
  5. SHAP explainability

Usage:
    python -m src.run_pipeline --data-dir data/ --output-dir outputs/
    python -m src.run_pipeline --data-dir data/ --output-dir outputs/ --skip-ingestion
    python -m src.run_pipeline --help
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _banner(title: str) -> None:
    width = 60
    logger.info("=" * width)
    logger.info(f"  {title}")
    logger.info("=" * width)


def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s//60:.0f}m {s%60:.0f}s" if s >= 60 else f"{s:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def step_ingestion(data_dir: str, db_path: str) -> None:
    _banner("STEP 1 — Data Ingestion")
    t = time.time()
    from src.ingestion import DataIngestion
    ing = DataIngestion(data_dir=data_dir, db_path=db_path)
    ing.run(split="train")
    logger.info("Ingestion completed in %s", _elapsed(t))


def step_features(db_path: str, sql_dir: str, cache_path: str) -> "pd.DataFrame":
    _banner("STEP 2 — Feature Engineering")
    t = time.time()
    import pandas as pd
    cache = Path(cache_path)

    from src.features import FeatureEngineer
    fe = FeatureEngineer(db_path=db_path, sql_dir=sql_dir)
    df = fe.build_feature_matrix()
    fe.get_feature_summary(df)
    df.to_pickle(str(cache))
    logger.info("Feature matrix cached at %s", cache)
    logger.info("Features completed in %s", _elapsed(t))
    return df


def step_isolation_forest(
    df: "pd.DataFrame",
    output_dir: str,
    n_estimators: int,
    contamination: float,
) -> tuple:
    _banner("STEP 3 — Isolation Forest")
    t = time.time()
    import numpy as np
    import joblib
    from src.models.iso_forest import IsolationForestModel
    from src.ensemble import EnsembleScorer
    from sklearn.metrics import f1_score

    Path(output_dir, "models").mkdir(parents=True, exist_ok=True)

    iso = IsolationForestModel(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
    )
    iso.fit(df)
    iso_scores = iso.score(df)
    np.save(str(Path(output_dir, "iso_scores.npy")), iso_scores)

    # Best-F1 threshold
    labels   = df["isFraud"].values
    iso_norm = EnsembleScorer.normalize(iso_scores)
    best_t, best_f1 = 0.5, 0.0
    for t_val in [i / 100 for i in range(5, 96)]:
        s = f1_score(labels, (iso_norm > t_val).astype(int), zero_division=0)
        if s > best_f1:
            best_f1, best_t = s, t_val
    iso_preds = (iso_norm > best_t).astype(int)
    np.save(str(Path(output_dir, "iso_preds.npy")), iso_preds)

    joblib.dump(iso, str(Path(output_dir, "models", "iso_forest.pkl")))
    logger.info("IF saved  |  best F1=%.4f at threshold=%.2f", best_f1, best_t)
    logger.info("Isolation Forest completed in %s", _elapsed(t))
    return iso_scores, iso_preds


def step_evaluation(
    df: "pd.DataFrame",
    iso_scores: "np.ndarray",
    iso_preds: "np.ndarray",
    output_dir: str,
) -> None:
    _banner("STEP 4 — Evaluation")
    t = time.time()
    import numpy as np
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    from src.evaluate import Evaluator

    labels = df["isFraud"].values
    ev = Evaluator(output_dir=output_dir)

    ev.precision_recall_curve(labels, {"IsolationForest": iso_scores})
    ev.confusion_matrix(labels, iso_preds, "Isolation_Forest")
    ev.threshold_analysis(
        labels,
        EnsembleScorer_normalize(iso_scores),
        "Isolation_Forest",
    )

    tn, fp, fn, tp = confusion_matrix(labels, iso_preds).ravel()
    results = {
        "Isolation Forest": {
            "precision":           precision_score(labels, iso_preds, zero_division=0),
            "recall":              recall_score(labels, iso_preds, zero_division=0),
            "f1":                  f1_score(labels, iso_preds, zero_division=0),
            "roc_auc":             roc_auc_score(labels, iso_scores),
            "pr_auc":              average_precision_score(labels, iso_scores),
            "false_positive_rate": fp / max(fp + tn, 1),
        }
    }
    df_results = ev.results_table(results)
    ev.log_all_metrics()

    print()
    print("=" * 65)
    print("  RESULTS TABLE")
    print("=" * 65)
    print(df_results.to_string())
    print("=" * 65)
    logger.info("Evaluation completed in %s", _elapsed(t))


def EnsembleScorer_normalize(scores):
    """Inline min-max normalise (avoids import-order issues)."""
    import numpy as np
    s_min, s_max = scores.min(), scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-9)


def step_explainability(
    iso,
    df: "pd.DataFrame",
    iso_preds: "np.ndarray",
    output_dir: str,
    n_waterfall: int = 3,
) -> None:
    _banner("STEP 5 — SHAP Explainability")
    t = time.time()
    from src.explainability import Explainer

    ex = Explainer(output_dir=output_dir)
    ex.shap_isolation_forest(iso, df, n_samples=500, n_top_features=15)

    flagged = [int(i) for i in __import__("numpy").where(iso_preds == 1)[0][:n_waterfall]]
    for idx in flagged:
        ex.shap_waterfall(iso, df, transaction_idx=idx)

    logger.info("Explainability completed in %s", _elapsed(t))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full financial anomaly detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",        default="data/",    help="Directory with raw CSVs")
    parser.add_argument("--db-path",         default="db/fraud.db", help="SQLite database path")
    parser.add_argument("--sql-dir",         default="sql/",     help="Directory with .sql files")
    parser.add_argument("--output-dir",      default="outputs/", help="Output directory for plots/models")
    parser.add_argument("--cache-path",      default="db/feature_matrix.pkl",
                        help="Path to cache the feature matrix")
    parser.add_argument("--n-estimators",    type=int,   default=200)
    parser.add_argument("--contamination",   type=float, default=0.035)
    parser.add_argument("--skip-ingestion",  action="store_true",
                        help="Skip ingestion (DB already populated)")
    parser.add_argument("--skip-features",   action="store_true",
                        help="Skip feature engineering (use cached matrix)")
    args = parser.parse_args()

    pipeline_start = time.time()
    _banner("FINANCIAL ANOMALY DETECTION PIPELINE")
    logger.info("data_dir    : %s", args.data_dir)
    logger.info("db_path     : %s", args.db_path)
    logger.info("output_dir  : %s", args.output_dir)

    # Step 1 — Ingestion
    if not args.skip_ingestion:
        step_ingestion(args.data_dir, args.db_path)
    else:
        logger.info("STEP 1 — Ingestion SKIPPED (--skip-ingestion)")

    # Step 2 — Features
    cache = Path(args.cache_path)
    if args.skip_features and cache.exists():
        logger.info("STEP 2 — Features SKIPPED (loading cache from %s)", cache)
        import pandas as pd
        df = pd.read_pickle(str(cache))
        logger.info("Loaded feature matrix: %s", df.shape)
    else:
        df = step_features(args.db_path, args.sql_dir, args.cache_path)

    # Step 3 — Isolation Forest
    import joblib
    iso_model_path = Path(args.output_dir, "models", "iso_forest.pkl")
    iso_scores_path = Path(args.output_dir, "iso_scores.npy")
    iso_preds_path  = Path(args.output_dir, "iso_preds.npy")

    import numpy as np
    iso_scores, iso_preds = step_isolation_forest(
        df, args.output_dir, args.n_estimators, args.contamination
    )
    iso = joblib.load(str(iso_model_path))

    # Step 4 — Evaluation
    step_evaluation(df, iso_scores, iso_preds, args.output_dir)

    # Step 5 — Explainability
    step_explainability(iso, df, iso_preds, args.output_dir)

    logger.info("")
    _banner(f"PIPELINE COMPLETE  —  total time: {_elapsed(pipeline_start)}")
    logger.info("  Outputs saved to : %s", Path(args.output_dir).resolve())
    logger.info("  MLflow UI        : mlflow ui --backend-store-uri mlruns/")
    logger.info("  Dashboard        : streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
