"""
Evaluator — model evaluation, visualisations, and MLflow metric logging.

All plots are saved to outputs/ (created automatically if absent).
All final metrics are logged to a single MLflow run.

Typical usage:
    ev = Evaluator(output_dir="outputs/")

    # PR curve for all three models
    ev.precision_recall_curve(
        labels,
        {"LSTM": lstm_scores, "IsolationForest": iso_scores, "Ensemble": ens_scores},
    )

    # Per-model confusion matrix
    ev.confusion_matrix(labels, lstm_preds, model_name="LSTM")

    # Results table
    df_results = ev.results_table({
        "LSTM":          {"precision": ..., "recall": ..., ...},
        "IsolationForest": {...},
        "Ensemble":       {...},
    })

    # Threshold tradeoff
    ev.threshold_analysis(labels, ens_scores, model_name="Ensemble")

    # LSTM error distribution
    ev.error_distribution(lstm_errors, labels)
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")   # headless rendering — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Centralises all evaluation, visualisation, and MLflow logging logic.

    Parameters
    ----------
    output_dir : str
        Directory where plots and CSVs are saved.  Created if absent.
    mlflow_run_name : str
        Name of the MLflow run that aggregates all final metrics.
    dpi : int
        Figure resolution for saved plots.
    """

    COLORS = {
        "LSTM":            "#E63946",
        "IsolationForest": "#457B9D",
        "Ensemble":        "#2A9D8F",
    }
    DEFAULT_COLOR = "#6C757D"

    def __init__(
        self,
        output_dir: str = "outputs/",
        mlflow_run_name: str = "model_evaluation",
        dpi: int = 150,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_run_name = mlflow_run_name
        self.dpi = dpi
        self._mlflow_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    @staticmethod
    def _color(model_name: str, default: str = "#6C757D") -> str:
        return Evaluator.COLORS.get(model_name, default)

    # ------------------------------------------------------------------
    # 1. Precision-Recall Curve
    # ------------------------------------------------------------------

    def precision_recall_curve(
        self,
        labels: np.ndarray,
        scores_dict: Dict[str, np.ndarray],
        run_name: str = "pr_curve",
    ) -> Path:
        """
        Plot PR curves for multiple models on the same axes.

        Parameters
        ----------
        labels      : np.ndarray  Ground-truth binary labels (0/1).
        scores_dict : dict        {model_name: score_array}
        run_name    : str         MLflow run name.

        Returns
        -------
        Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#F8F9FA")
        fig.patch.set_facecolor("#FFFFFF")

        # Baseline: random classifier
        base_rate = labels.mean()
        ax.axhline(base_rate, color="#ADB5BD", linestyle="--", linewidth=1,
                   label=f"Random (precision={base_rate:.3f})")

        pr_aucs: Dict[str, float] = {}
        for model_name, scores in scores_dict.items():
            prec, rec, _ = precision_recall_curve(labels, scores)
            auc_pr = average_precision_score(labels, scores)
            pr_aucs[model_name] = auc_pr
            color = self._color(model_name)
            ax.plot(rec, prec, color=color, linewidth=2,
                    label=f"{model_name}  (AUC-PR = {auc_pr:.4f})")
            self._mlflow_metrics[f"{model_name}_pr_auc"] = auc_pr

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision–Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.4)

        path = self._save(fig, "pr_curve.png")

        with mlflow.start_run(run_name=run_name):
            for name, auc in pr_aucs.items():
                mlflow.log_metric(f"{name}_pr_auc", auc)
            mlflow.log_artifact(str(path))

        return path

    # ------------------------------------------------------------------
    # 2. Confusion Matrix
    # ------------------------------------------------------------------

    def confusion_matrix(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        model_name: str,
        run_name: str | None = None,
    ) -> Path:
        """
        Plot confusion matrix with absolute counts and row-normalised percentages.

        Parameters
        ----------
        labels     : np.ndarray  Ground-truth.
        preds      : np.ndarray  Binary predictions.
        model_name : str
        run_name   : str | None

        Returns
        -------
        Path to saved figure.
        """
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

        for ax, data, title, fmt in zip(
            axes,
            [cm, cm_norm],
            ["Absolute counts", "Row-normalised (%)"],
            ["d", ".2%"],
        ):
            disp = ConfusionMatrixDisplay(confusion_matrix=data,
                                          display_labels=["Normal", "Fraud"])
            disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=fmt)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)

        plt.tight_layout()
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        path = self._save(fig, filename)

        # Derived metrics
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)

        for key, val in {"fpr": fpr, "precision": prec, "recall": rec, "f1": f1}.items():
            self._mlflow_metrics[f"{model_name}_{key}"] = val

        _run = run_name or f"cm_{model_name}"
        with mlflow.start_run(run_name=_run):
            mlflow.log_metrics({f"{model_name}_{k}": v
                                 for k, v in {"fpr": fpr, "precision": prec,
                                              "recall": rec, "f1": f1}.items()})
            mlflow.log_artifact(str(path))

        logger.info("%s  TP=%d  FP=%d  FN=%d  TN=%d  FPR=%.4f",
                    model_name, tp, fp, fn, tn, fpr)
        return path

    # ------------------------------------------------------------------
    # 3. Results Table
    # ------------------------------------------------------------------

    def results_table(
        self,
        results_dict: Dict[str, Dict[str, float]],
        run_name: str = "results_table",
    ) -> pd.DataFrame:
        """
        Build a formatted results DataFrame and save to CSV.

        Parameters
        ----------
        results_dict : dict
            {model_name: {precision, recall, f1, roc_auc, pr_auc,
                          false_positive_rate}}

        Returns
        -------
        pd.DataFrame  with columns = metrics, index = model names.
        """
        COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate"]

        rows = []
        for model_name, metrics in results_dict.items():
            row = {"Model": model_name}
            for col in COLS:
                row[col] = round(metrics.get(col, float("nan")), 4)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")
        df.columns = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR"]

        csv_path = self.output_dir / "results_table.csv"
        df.to_csv(csv_path)
        logger.info("Results table saved → %s", csv_path)
        logger.info("\n%s", df.to_string())

        with mlflow.start_run(run_name=run_name):
            for model_name, row in df.iterrows():
                for metric, val in row.items():
                    if not np.isnan(val):
                        mlflow.log_metric(f"{model_name}_{metric}", val)
            mlflow.log_artifact(str(csv_path))

        return df

    # ------------------------------------------------------------------
    # 4. Threshold Analysis
    # ------------------------------------------------------------------

    def threshold_analysis(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        model_name: str,
        steps: int = 100,
        run_name: str | None = None,
    ) -> Path:
        """
        Plot Precision, Recall, and F1 vs threshold on the same axes.

        Parameters
        ----------
        labels     : np.ndarray  Ground-truth.
        scores     : np.ndarray  Continuous anomaly scores.
        model_name : str
        steps      : int         Number of threshold candidates.

        Returns
        -------
        Path to saved figure.
        """
        thresholds = np.linspace(scores.min(), scores.max(), steps)
        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            preds = (scores > t).astype(int)
            precisions.append(precision_score(labels, preds, zero_division=0))
            recalls.append(recall_score(labels, preds, zero_division=0))
            f1s.append(f1_score(labels, preds, zero_division=0))

        best_idx = int(np.argmax(f1s))
        best_t = thresholds[best_idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#F8F9FA")

        ax.plot(thresholds, precisions, color="#457B9D", linewidth=2, label="Precision")
        ax.plot(thresholds, recalls,   color="#E63946", linewidth=2, label="Recall")
        ax.plot(thresholds, f1s,       color="#2A9D8F", linewidth=2.5, label="F1", linestyle="--")
        ax.axvline(best_t, color="#F4A261", linewidth=1.5, linestyle=":",
                   label=f"Best F1 threshold = {best_t:.3f}")

        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Precision / Recall / F1 vs Threshold — {model_name}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.4)

        filename = f"threshold_analysis_{model_name.lower().replace(' ', '_')}.png"
        path = self._save(fig, filename)

        _run = run_name or f"threshold_{model_name}"
        with mlflow.start_run(run_name=_run):
            mlflow.log_params({"best_f1_threshold": best_t, "model": model_name})
            mlflow.log_metrics({
                f"{model_name}_best_f1":        max(f1s),
                f"{model_name}_best_precision": precisions[best_idx],
                f"{model_name}_best_recall":    recalls[best_idx],
            })
            mlflow.log_artifact(str(path))

        logger.info("%s  best F1=%.4f at threshold=%.4f", model_name, max(f1s), best_t)
        return path

    # ------------------------------------------------------------------
    # 5. Error Distribution
    # ------------------------------------------------------------------

    def error_distribution(
        self,
        lstm_errors: np.ndarray,
        labels: np.ndarray,
        run_name: str = "error_distribution",
        clip_percentile: float = 99.5,
    ) -> Path:
        """
        Plot reconstruction error distributions for fraud vs normal transactions.

        This is the key diagnostic for the LSTM autoencoder: if the model has
        learned a normal manifold, fraud errors will be visibly right-shifted.

        Parameters
        ----------
        lstm_errors     : np.ndarray  Per-sample MSE from ModelTrainer.score().
        labels          : np.ndarray  Ground-truth binary labels.
        clip_percentile : float       Clip x-axis to reduce visual distortion
                                      from extreme outliers.

        Returns
        -------
        Path to saved figure.
        """
        normal_errors = lstm_errors[labels == 0]
        fraud_errors  = lstm_errors[labels == 1]

        # Clip x-axis so outliers don't compress the bulk of the distribution
        x_max = float(np.percentile(lstm_errors, clip_percentile))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "LSTM Reconstruction Error — Fraud vs Normal",
            fontsize=14, fontweight="bold",
        )

        for ax, errors, label, color in [
            (axes[0], normal_errors, "Normal", "#457B9D"),
            (axes[1], fraud_errors,  "Fraud",  "#E63946"),
        ]:
            clipped = errors[errors <= x_max]
            ax.hist(clipped, bins=80, color=color, alpha=0.75, edgecolor="white",
                    linewidth=0.3)
            ax.axvline(np.mean(clipped), color="black", linewidth=1.5,
                       linestyle="--", label=f"Mean = {np.mean(clipped):.4f}")
            ax.axvline(np.median(clipped), color="orange", linewidth=1.5,
                       linestyle=":", label=f"Median = {np.median(clipped):.4f}")
            ax.set_title(f"{label}  (n={len(errors):,})", fontsize=12)
            ax.set_xlabel("Reconstruction Error (MSE)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.legend(fontsize=9)
            ax.set_facecolor("#F8F9FA")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self._save(fig, "error_distribution.png")

        # Separation metric: how far apart are the means?
        separation = float(np.mean(fraud_errors) - np.mean(normal_errors))

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics({
                "normal_error_mean":   float(np.mean(normal_errors)),
                "normal_error_std":    float(np.std(normal_errors)),
                "fraud_error_mean":    float(np.mean(fraud_errors)),
                "fraud_error_std":     float(np.std(fraud_errors)),
                "error_separation":    separation,
            })
            mlflow.log_artifact(str(path))

        logger.info(
            "Error distribution  |  normal mean=%.4f  fraud mean=%.4f  separation=%.4f",
            np.mean(normal_errors), np.mean(fraud_errors), separation,
        )
        return path

    # ------------------------------------------------------------------
    # 6. Flush all metrics to a single MLflow run
    # ------------------------------------------------------------------

    def log_all_metrics(self, run_name: str | None = None) -> None:
        """Log every metric accumulated across all evaluation calls in one run."""
        if not self._mlflow_metrics:
            logger.warning("No metrics accumulated — nothing to log.")
            return
        _run = run_name or self.mlflow_run_name
        with mlflow.start_run(run_name=_run):
            mlflow.log_metrics(self._mlflow_metrics)
        logger.info("Logged %d metrics to MLflow run '%s'",
                    len(self._mlflow_metrics), _run)
