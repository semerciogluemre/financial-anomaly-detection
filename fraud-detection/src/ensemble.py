"""
EnsembleScorer — combines LSTM Autoencoder and Isolation Forest anomaly scores.

Design:
    Both models produce raw scores on different scales (MSE vs negated
    decision function).  We min-max normalize each to [0, 1] then take
    a weighted sum.  The threshold that maximises a chosen sklearn metric
    (default F1) on a labelled validation set is found via grid search.

MLflow integration logs:
    Params : weight_lstm, weight_iso, best_threshold, metric
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
    Weighted ensemble of LSTM reconstruction errors and Isolation Forest scores.

    Parameters
    ----------
    weight_lstm : float   Weight for LSTM scores in [0, 1].  Default 0.6.
    weight_iso  : float   Weight for IF scores.  Default 0.4.
                          Weights are re-normalized to sum to 1 internally.
    """

    def __init__(self, weight_lstm: float = 0.6, weight_iso: float = 0.4):
        total = weight_lstm + weight_iso
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        self.weight_lstm = weight_lstm / total
        self.weight_iso = weight_iso / total
        self.best_threshold: float | None = None

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
    # Combination
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
