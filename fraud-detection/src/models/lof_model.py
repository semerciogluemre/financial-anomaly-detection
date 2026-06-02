"""
LOFModel — Local Outlier Factor for fraud anomaly detection.

Fitted on non-fraud training rows with novelty=True so it can score
unseen test data. Uses StandardScaler (LOF is distance-based — scale matters).

Anomaly score = -decision_function, consistent with IF and MLP-AE convention
(higher = more anomalous).
"""

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class LOFModel:
    """
    Local Outlier Factor wrapper for tabular fraud detection.

    Parameters
    ----------
    n_neighbors   : int    Number of neighbours for local density estimation.
    contamination : float  Expected fraction of anomalies.
    n_jobs        : int    Parallel jobs (-1 = all cores).
    """

    DROP_COLS = {"TransactionID", "isFraud", "TransactionDT"}

    def __init__(
        self,
        n_neighbors:   int   = 20,
        contamination: float = 0.035,
        n_jobs:        int   = -1,
    ):
        self.n_neighbors   = n_neighbors
        self.contamination = contamination
        self.n_jobs        = n_jobs
        self._model: LocalOutlierFactor | None = None
        self._scaler: StandardScaler   | None = None

    # ------------------------------------------------------------------
    def _extract(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.drop(columns=[c for c in self.DROP_COLS if c in X.columns])
            X = X.select_dtypes(include="number")
        arr = np.array(X, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series    | np.ndarray,
        run_name: str = "lof_fit",
    ) -> "LOFModel":
        """
        Fit on non-fraud training rows only (novelty=True).

        Parameters
        ----------
        X_train : feature matrix (full training split)
        y_train : labels (used to filter to normal rows)
        """
        X_arr = self._extract(X_train)
        y_arr = np.array(y_train, dtype=np.int32)

        normal_mask = y_arr == 0
        X_normal    = X_arr[normal_mask]
        logger.info(
            "LOF fitting on %d normal rows (from %d total)", len(X_normal), len(X_arr)
        )

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_normal)

        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X_scaled)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model":           "LOF",
                "n_neighbors":     self.n_neighbors,
                "contamination":   self.contamination,
                "n_train_normal":  len(X_normal),
                "novelty":         True,
            })

        logger.info("LOF fitted.")
        return self

    # ------------------------------------------------------------------
    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        run_name: str = "lof_score",
    ) -> np.ndarray:
        """
        Return anomaly scores (higher = more anomalous).

        Score = -decision_function (consistent with IF convention).
        """
        if self._model is None:
            raise RuntimeError("Not fitted. Call fit() first.")

        X_scaled = self._scaler.transform(self._extract(X))
        scores   = -self._model.decision_function(X_scaled).astype(np.float32)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics({
                "lof_score_mean": float(scores.mean()),
                "lof_score_std":  float(scores.std()),
                "lof_score_p95":  float(np.percentile(scores, 95)),
            })

        logger.info(
            "LOF scored %d rows  mean=%.4f  max=%.4f",
            len(scores), scores.mean(), scores.max(),
        )
        return scores

    # ------------------------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """
        Binary predictions.  If threshold is None uses the LOF decision boundary.
        """
        if threshold is not None:
            return (self.score(X) > threshold).astype(np.int8)
        # Default: use fitted decision boundary (novelty=True sklearn convention)
        X_scaled = self._scaler.transform(self._extract(X))
        raw = self._model.predict(X_scaled)  # +1 = normal, -1 = anomaly
        return ((raw == -1).astype(np.int8))

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("LOFModel saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "LOFModel":
        return joblib.load(path)
