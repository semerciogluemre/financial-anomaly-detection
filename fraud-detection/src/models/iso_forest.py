"""
Isolation Forest anomaly detector for financial fraud detection.

IsolationForest is label-free: it isolates anomalies by randomly partitioning
the feature space and measuring how few splits are needed to isolate a point.
Fraudulent transactions tend to be isolated quickly (anomaly score → 1).

Anomaly score convention used here:
    score = -1 * decision_function(X)
    → higher score = more anomalous (consistent with LSTM reconstruction error)
    → threshold: flag samples where score > threshold

MLflow integration logs:
    Params : n_estimators, max_samples, contamination, random_state, threshold
    Metrics: score distribution (mean, std, p50, p95, p99),
             predicted_anomaly_rate
"""

import logging
from typing import Optional, Union

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class IsolationForestModel:
    """
    Sklearn IsolationForest wrapper with MLflow logging and a consistent
    anomaly-score API matching the LSTM autoencoder interface.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees (default 200).
    max_samples : int | float | str
        Subsample size per tree.  'auto' = min(256, n_samples).
    contamination : float | str
        Expected fraction of anomalies in training data.  'auto' uses the
        sklearn default decision boundary.
    random_state : int
        Reproducibility seed.
    scale_features : bool
        If True, apply StandardScaler before fitting/scoring.
    """

    # Identifier and label columns to drop automatically
    DROP_COLS = {"TransactionID", "isFraud", "TransactionDT"}

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Union[int, float, str] = "auto",
        contamination: Union[float, str] = "auto",
        random_state: int = 42,
        scale_features: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.scale_features = scale_features

        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Drop non-feature columns and return a float32 numpy array."""
        cols = [c for c in df.columns if c not in self.DROP_COLS]
        if self.feature_cols:
            # At scoring time use the same columns seen at fit time
            cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].astype(np.float32).values

        # Replace any remaining NaNs with column means (fallback)
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        run_name: str = "isolation_forest_fit",
    ) -> "IsolationForestModel":
        """
        Train IsolationForest on the full feature matrix (no labels needed).

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix from FeatureEngineer.build_feature_matrix().
        run_name : str
            MLflow run display name.

        Returns
        -------
        self
        """
        self.feature_cols = [c for c in df.columns if c not in self.DROP_COLS]
        X = self._extract_features(df)

        if self.scale_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
            logger.info("Features standardized: mean≈0, std≈1")

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )

        logger.info(
            "Fitting IsolationForest  n_estimators=%d  contamination=%s  n=%d rows",
            self.n_estimators, self.contamination, len(X),
        )
        self._model.fit(X)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model":          "IsolationForest",
                "n_estimators":   self.n_estimators,
                "max_samples":    str(self.max_samples),
                "contamination":  str(self.contamination),
                "random_state":   self.random_state,
                "scale_features": self.scale_features,
                "n_features":     len(self.feature_cols),
                "n_train_rows":   len(X),
            })

        logger.info("IsolationForest fitted successfully.")
        return self

    def score(
        self,
        df: pd.DataFrame,
        run_name: str = "isolation_forest_score",
    ) -> np.ndarray:
        """
        Return anomaly scores for each row.

        Score = -decision_function(X), so higher = more anomalous.
        This matches the LSTM reconstruction error convention.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        scores : np.ndarray  shape (N,)
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X = self._extract_features(df)
        if self.scale_features and self._scaler is not None:
            X = self._scaler.transform(X)

        # sklearn decision_function: negative = anomalous, positive = normal
        # We negate so high score = more anomalous
        scores = -self._model.decision_function(X).astype(np.float32)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics({
                "score_mean":  float(scores.mean()),
                "score_std":   float(scores.std()),
                "score_min":   float(scores.min()),
                "score_max":   float(scores.max()),
                "score_p50":   float(np.percentile(scores, 50)),
                "score_p95":   float(np.percentile(scores, 95)),
                "score_p99":   float(np.percentile(scores, 99)),
            })

        logger.info(
            "Scored %d samples  |  score range [%.4f, %.4f]  mean=%.4f",
            len(scores), scores.min(), scores.max(), scores.mean(),
        )
        return scores

    def predict(
        self,
        df: pd.DataFrame,
        threshold: float,
        run_name: str = "isolation_forest_predict",
    ) -> np.ndarray:
        """
        Return binary fraud predictions (1 = fraud, 0 = normal).

        Parameters
        ----------
        df : pd.DataFrame
        threshold : float
            Samples with score > threshold are labelled as fraud.

        Returns
        -------
        predictions : np.ndarray  shape (N,)  dtype int8
        """
        scores = self.score(df, run_name="_internal_score")
        predictions = (scores > threshold).astype(np.int8)

        anomaly_rate = predictions.mean()
        logger.info(
            "Predicted %d anomalies / %d total  (%.2f%%)",
            predictions.sum(), len(predictions), anomaly_rate * 100,
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"threshold": threshold})
            mlflow.log_metrics({
                "predicted_anomaly_rate": float(anomaly_rate),
                "predicted_anomaly_count": int(predictions.sum()),
            })

        return predictions

    def feature_importances(self) -> Optional[pd.Series]:
        """
        Approximate feature importance via mean tree depth reduction.
        Only available after fitting; returns None otherwise.

        Note: IsolationForest does not expose feature_importances_ directly
        in all sklearn versions, so we approximate via the underlying trees.
        """
        if self._model is None:
            return None
        try:
            importances = np.mean(
                [tree.feature_importances_ for tree in self._model.estimators_],
                axis=0,
            )
            return pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)
        except AttributeError:
            logger.warning("feature_importances_ not available for this sklearn version.")
            return None
