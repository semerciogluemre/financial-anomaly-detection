"""
XGBoostModel — supervised fraud classifier using XGBoost.

Uses scale_pos_weight to handle class imbalance (n_negative / n_positive),
which is more effective than oversampling for tree ensembles.

MLflow logs: all hyperparams, ROC-AUC, PR-AUC, F1, precision, recall, FPR.
"""

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    Supervised XGBoost classifier for fraud detection.

    Parameters
    ----------
    n_estimators   : int    Number of boosting rounds.
    max_depth      : int    Maximum tree depth.
    learning_rate  : float  Step size shrinkage.
    subsample      : float  Row subsampling ratio.
    colsample_bytree: float Column subsampling ratio.
    random_state   : int
    """

    def __init__(
        self,
        n_estimators:    int   = 500,
        max_depth:       int   = 6,
        learning_rate:   float = 0.05,
        subsample:       float = 0.8,
        colsample_bytree:float = 0.8,
        random_state:    int   = 42,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state     = random_state
        self._model           = None
        self.feature_names_   = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series   | None = None,
        run_name: str = "xgboost_fit",
    ) -> "XGBoostModel":
        """
        Train XGBoost with scale_pos_weight to handle class imbalance.

        Parameters
        ----------
        X_train, y_train : training features and labels
        X_val,   y_val   : optional validation set for early stopping
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())
        spw   = n_neg / max(n_pos, 1)
        logger.info(
            "XGBoost: n_pos=%d  n_neg=%d  scale_pos_weight=%.2f", n_pos, n_neg, spw
        )

        self.feature_names_ = list(X_train.columns)
        X_arr = X_train.values.astype(np.float32)
        y_arr = y_train.values.astype(np.int32)

        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val.values.astype(np.float32),
                                       y_val.values.astype(np.int32))]
            fit_kwargs["verbose"]  = False

        self._model.fit(X_arr, y_arr, **fit_kwargs)
        logger.info(
            "XGBoost trained  best_iteration=%s",
            getattr(self._model, "best_iteration", "N/A"),
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model":             "XGBoost",
                "n_estimators":      self.n_estimators,
                "max_depth":         self.max_depth,
                "learning_rate":     self.learning_rate,
                "subsample":         self.subsample,
                "colsample_bytree":  self.colsample_bytree,
                "scale_pos_weight":  round(spw, 4),
                "n_train":           len(X_train),
                "n_features":        len(self.feature_names_),
            })

        return self

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted fraud probability (class=1) for each row."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_arr = (X.values if isinstance(X, pd.DataFrame) else X).astype(np.float32)
        return self._model.predict_proba(X_arr)[:, 1]

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary fraud flags at the given probability threshold."""
        proba = self.score(X)
        return (proba > threshold).astype(np.int8)

    def feature_importances(
        self,
        feature_names: list[str] | None = None,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Return feature importances ranked by gain.

        Parameters
        ----------
        feature_names  : list[str]  Column names (uses fitted names if None).
        importance_type: str        'gain', 'weight', or 'cover'.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        names = feature_names or self.feature_names_ or []
        scores = self._model.get_booster().get_score(importance_type=importance_type)
        df = pd.DataFrame(
            {"feature": list(scores.keys()), "importance": list(scores.values())}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
        run_name: str = "xgboost_evaluate",
    ) -> dict:
        """Evaluate on test set and log metrics to MLflow."""
        scores = self.score(X_test)
        preds  = (scores > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        metrics = {
            "precision":           precision_score(y_test, preds, zero_division=0),
            "recall":              recall_score(y_test, preds, zero_division=0),
            "f1":                  f1_score(y_test, preds, zero_division=0),
            "roc_auc":             roc_auc_score(y_test, scores),
            "pr_auc":              average_precision_score(y_test, scores),
            "false_positive_rate": fp / max(fp + tn, 1),
        }

        logger.info(
            "XGBoost test — ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f  FPR: %.4f",
            metrics["roc_auc"], metrics["pr_auc"],
            metrics["f1"], metrics["false_positive_rate"],
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"threshold": threshold})
            mlflow.log_metrics({f"xgb_{k}": v for k, v in metrics.items()})

        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("XGBoostModel saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        return joblib.load(path)
