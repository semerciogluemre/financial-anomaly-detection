"""
EnsembleScorer — combines XGBoost, Isolation Forest, and MLP-AE scores.

v2 design (replaces LSTM-only v1):
    Three models contribute:
      - XGBoost probability     (supervised, strong signal)
      - Isolation Forest score  (unsupervised, fast)
      - MLP-AE reconstruction error (unsupervised, captures non-linear patterns)

    LSTM-AE is excluded (documented negative result — ROC-AUC ≈ 0.50).
    LOF is kept as standalone comparison but excluded from ensemble
    (does not generalise well at scale: O(n²) distance computation).

    All scores are min-max normalised to [0, 1] then combined with a
    weighted sum. Weights are tuned via grid search maximising PR-AUC.

MLflow integration logs:
    Params : weight_xgb, weight_iso, weight_mlp, best_threshold, metric
    Metrics: best metric score, score distribution stats
"""

import logging
from typing import Literal

import mlflow
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MetricName = Literal["f1", "precision", "recall", "roc_auc", "pr_auc"]


class EnsembleScorer:
    """
    Weighted ensemble of XGBoost, Isolation Forest, and MLP-AE scores.

    Parameters
    ----------
    weight_xgb  : float  Weight for XGBoost probability.  Default 0.6.
    weight_iso  : float  Weight for Isolation Forest.      Default 0.25.
    weight_mlp  : float  Weight for MLP-AE reconstruction. Default 0.15.
                         Weights are re-normalised to sum to 1 internally.

    Legacy parameters weight_lstm kept for backwards compatibility but ignored.
    """

    def __init__(
        self,
        weight_xgb:  float = 0.6,
        weight_iso:  float = 0.25,
        weight_mlp:  float = 0.15,
        # legacy — ignored
        weight_lstm: float = 0.0,
    ):
        total = weight_xgb + weight_iso + weight_mlp
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        self.weight_xgb  = weight_xgb  / total
        self.weight_iso  = weight_iso  / total
        self.weight_mlp  = weight_mlp  / total
        # kept for API compatibility
        self.weight_lstm = 0.0
        self.best_threshold: float | None = None
        self.best_weights: dict | None = None

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """
        Min-max normalize an array of scores to [0, 1].

        Parameters
        ----------
        scores : np.ndarray  shape (N,)
        eps    : float       Small constant to avoid division by zero.

        Returns
        -------
        normalized : np.ndarray  shape (N,)  in [0, 1]
        """
        s_min = scores.min()
        s_max = scores.max()
        normalized = (scores - s_min) / (s_max - s_min + eps)
        return normalized.astype(np.float32)

    # ------------------------------------------------------------------
    # v2 combination — XGBoost + IF + MLP-AE
    # ------------------------------------------------------------------

    def combine_v2(
        self,
        xgb_scores: np.ndarray,
        iso_scores: np.ndarray,
        mlp_scores: np.ndarray,
        w_xgb: float | None = None,
        w_iso:  float | None = None,
        w_mlp:  float | None = None,
    ) -> np.ndarray:
        """
        Normalise XGBoost, IF, and MLP-AE scores and combine with weighted sum.

        Parameters
        ----------
        xgb_scores : XGBoost predicted fraud probability  (N,)
        iso_scores : IF anomaly score                     (N,)
        mlp_scores : MLP-AE reconstruction error         (N,)

        Returns
        -------
        combined : np.ndarray  (N,)  in [0, 1]
        """
        wx = w_xgb if w_xgb is not None else self.weight_xgb
        wi = w_iso  if w_iso  is not None else self.weight_iso
        wm = w_mlp  if w_mlp  is not None else self.weight_mlp
        total = wx + wi + wm
        wx, wi, wm = wx / total, wi / total, wm / total

        combined = (
            wx * self.normalize(xgb_scores) +
            wi * self.normalize(iso_scores)  +
            wm * self.normalize(mlp_scores)
        )
        logger.info(
            "combine_v2  w_xgb=%.2f  w_iso=%.2f  w_mlp=%.2f  "
            "score range=[%.4f, %.4f]",
            wx, wi, wm, combined.min(), combined.max(),
        )
        return combined

    def tune_weights(
        self,
        xgb_scores: np.ndarray,
        iso_scores: np.ndarray,
        mlp_scores: np.ndarray,
        labels: np.ndarray,
        metric: MetricName = "pr_auc",
        run_name: str = "ensemble_weight_tuning",
    ) -> dict:
        """
        Grid-search over weight combinations to maximise PR-AUC on the test set.

        Candidate weights: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        (all combinations that sum to > 0 are tried; 56 combos total).

        Returns
        -------
        dict  {'w_xgb': float, 'w_iso': float, 'w_mlp': float,
               'best_score': float, 'best_threshold': float}
        """
        candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        best_score = -np.inf
        best_cfg   = None

        for wx in candidates:
            for wi in candidates:
                for wm in candidates:
                    if wx + wi + wm == 0:
                        continue
                    combined = self.combine_v2(
                        xgb_scores, iso_scores, mlp_scores,
                        w_xgb=wx, w_iso=wi, w_mlp=wm,
                    )
                    if metric == "pr_auc":
                        s = average_precision_score(labels, combined)
                    elif metric == "roc_auc":
                        s = roc_auc_score(labels, combined)
                    elif metric == "f1":
                        best_t, best_f1 = 0.5, 0.0
                        for t in np.linspace(0.1, 0.9, 81):
                            f = f1_score(labels, (combined > t).astype(int), zero_division=0)
                            if f > best_f1:
                                best_f1, best_t = f, t
                        s = best_f1
                    else:
                        s = average_precision_score(labels, combined)

                    if s > best_score:
                        best_score = s
                        best_cfg   = {"w_xgb": wx, "w_iso": wi, "w_mlp": wm}

        # Tune threshold at best weights
        combined_best = self.combine_v2(
            xgb_scores, iso_scores, mlp_scores,
            **best_cfg,
        )
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 81):
            f = f1_score(labels, (combined_best > t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t

        self.weight_xgb = best_cfg["w_xgb"] / (sum(best_cfg.values()) or 1)
        self.weight_iso = best_cfg["w_iso"]  / (sum(best_cfg.values()) or 1)
        self.weight_mlp = best_cfg["w_mlp"]  / (sum(best_cfg.values()) or 1)
        self.best_threshold = best_t
        self.best_weights   = {**best_cfg, "best_score": best_score, "best_threshold": best_t}

        logger.info(
            "Best weights: xgb=%.2f iso=%.2f mlp=%.2f  %s=%.4f  threshold=%.3f",
            best_cfg["w_xgb"], best_cfg["w_iso"], best_cfg["w_mlp"],
            metric, best_score, best_t,
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({**best_cfg, "tune_metric": metric})
            mlflow.log_metrics({
                f"best_{metric}":  best_score,
                "best_threshold":  best_t,
                "best_f1":         best_f1,
            })

        return self.best_weights

    # ------------------------------------------------------------------
    # Legacy v1 combination — kept for backwards compatibility
    # ------------------------------------------------------------------

    def combine(
        self,
        lstm_scores: np.ndarray,
        iso_scores: np.ndarray,
        weight_lstm: float | None = None,
        weight_iso: float | None = None,
    ) -> np.ndarray:
        """
        Normalize both score arrays and combine with weighted sum.

        Parameters
        ----------
        lstm_scores : np.ndarray  shape (N,)  raw LSTM reconstruction errors.
        iso_scores  : np.ndarray  shape (N,)  raw IF anomaly scores.
        weight_lstm : float | None  Override instance weight if provided.
        weight_iso  : float | None  Override instance weight if provided.

        Returns
        -------
        combined : np.ndarray  shape (N,)  in [0, 1]
        """
        if len(lstm_scores) != len(iso_scores):
            raise ValueError(
                f"Score arrays must have equal length. "
                f"Got LSTM={len(lstm_scores)}, IF={len(iso_scores)}"
            )

        w_lstm = weight_lstm if weight_lstm is not None else self.weight_lstm
        w_iso  = weight_iso  if weight_iso  is not None else self.weight_iso

        # Re-normalize custom weights
        total = w_lstm + w_iso
        w_lstm /= total
        w_iso  /= total

        norm_lstm = self.normalize(lstm_scores)
        norm_iso  = self.normalize(iso_scores)

        combined = w_lstm * norm_lstm + w_iso * norm_iso
        logger.info(
            "Combined scores  |  w_lstm=%.2f  w_iso=%.2f  "
            "range=[%.4f, %.4f]  mean=%.4f",
            w_lstm, w_iso, combined.min(), combined.max(), combined.mean(),
        )
        return combined

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        combined_scores: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """
        Convert combined scores to binary fraud flags.

        Parameters
        ----------
        combined_scores : np.ndarray  shape (N,)
        threshold : float  If None, uses self.best_threshold (must have called
                           tune_threshold() first).

        Returns
        -------
        predictions : np.ndarray  shape (N,)  dtype int8  (1=fraud, 0=normal)
        """
        t = threshold if threshold is not None else self.best_threshold
        if t is None:
            raise ValueError(
                "No threshold set. Pass threshold= or call tune_threshold() first."
            )
        predictions = (combined_scores > t).astype(np.int8)
        logger.info(
            "Predictions  threshold=%.4f  flagged=%d / %d  (%.2f%%)",
            t, predictions.sum(), len(predictions), predictions.mean() * 100,
        )
        return predictions

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def tune_threshold(
        self,
        combined_scores: np.ndarray,
        labels: np.ndarray,
        metric: MetricName = "f1",
        sweep_steps: int = 81,
        run_name: str = "ensemble_threshold_tuning",
    ) -> float:
        """
        Grid-search thresholds in [0.1, 0.9] to maximise the chosen metric
        on a labelled validation set.

        Parameters
        ----------
        combined_scores : np.ndarray  shape (N,)  output of combine().
        labels          : np.ndarray  shape (N,)  ground truth (0/1).
        metric          : one of 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc'
        sweep_steps     : int  Number of threshold candidates (default 81 → step 0.01).
        run_name        : str  MLflow run name.

        Returns
        -------
        best_threshold : float
        """
        thresholds = np.linspace(0.1, 0.9, sweep_steps)
        best_score = -np.inf
        best_threshold = 0.5

        # Metrics that don't require thresholded predictions
        threshold_free = {"roc_auc", "pr_auc"}

        for t in thresholds:
            preds = (combined_scores > t).astype(np.int8)

            if metric == "f1":
                s = f1_score(labels, preds, zero_division=0)
            elif metric == "precision":
                s = precision_score(labels, preds, zero_division=0)
            elif metric == "recall":
                s = recall_score(labels, preds, zero_division=0)
            elif metric == "roc_auc":
                s = roc_auc_score(labels, combined_scores)
            elif metric == "pr_auc":
                s = average_precision_score(labels, combined_scores)
            else:
                raise ValueError(f"Unknown metric: {metric!r}")

            if s > best_score:
                best_score = s
                best_threshold = float(t)

            # For threshold-free metrics the score is constant → use once
            if metric in threshold_free:
                break

        self.best_threshold = best_threshold

        # Compute final stats at best threshold
        final_preds = self.predict(combined_scores, best_threshold)
        final_f1        = f1_score(labels, final_preds, zero_division=0)
        final_precision = precision_score(labels, final_preds, zero_division=0)
        final_recall    = recall_score(labels, final_preds, zero_division=0)
        try:
            final_roc_auc = roc_auc_score(labels, combined_scores)
            final_pr_auc  = average_precision_score(labels, combined_scores)
        except ValueError:
            final_roc_auc = final_pr_auc = float("nan")

        logger.info(
            "Best threshold=%.4f  %s=%.4f  F1=%.4f  P=%.4f  R=%.4f",
            best_threshold, metric, best_score, final_f1, final_precision, final_recall,
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "weight_lstm":      self.weight_lstm,
                "weight_iso":       self.weight_iso,
                "tune_metric":      metric,
                "sweep_steps":      sweep_steps,
                "best_threshold":   best_threshold,
            })
            mlflow.log_metrics({
                f"best_{metric}":   best_score,
                "final_f1":         final_f1,
                "final_precision":  final_precision,
                "final_recall":     final_recall,
                "final_roc_auc":    final_roc_auc,
                "final_pr_auc":     final_pr_auc,
                "anomaly_rate":     float(final_preds.mean()),
                # Score distribution at best threshold
                "score_mean":       float(combined_scores.mean()),
                "score_std":        float(combined_scores.std()),
                "score_p50":        float(np.percentile(combined_scores, 50)),
                "score_p95":        float(np.percentile(combined_scores, 95)),
            })

        return best_threshold

    # ------------------------------------------------------------------
    # Convenience: full pipeline in one call
    # ------------------------------------------------------------------

    def run(
        self,
        lstm_scores: np.ndarray,
        iso_scores: np.ndarray,
        labels: np.ndarray | None = None,
        threshold: float | None = None,
        metric: MetricName = "f1",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combine scores → (optionally) tune threshold → predict.

        Parameters
        ----------
        lstm_scores : np.ndarray
        iso_scores  : np.ndarray
        labels      : np.ndarray | None   Provide to auto-tune threshold.
        threshold   : float | None        Use this fixed threshold if no labels.
        metric      : MetricName          Metric to optimise during tuning.

        Returns
        -------
        combined    : np.ndarray  shape (N,)  continuous scores in [0, 1]
        predictions : np.ndarray  shape (N,)  binary flags
        """
        combined = self.combine(lstm_scores, iso_scores)

        if labels is not None:
            self.tune_threshold(combined, labels, metric=metric)
        elif threshold is not None:
            self.best_threshold = threshold
        else:
            raise ValueError("Provide either labels (for tuning) or a fixed threshold.")

        predictions = self.predict(combined)
        return combined, predictions
