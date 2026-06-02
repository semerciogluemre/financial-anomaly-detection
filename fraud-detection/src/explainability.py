"""
Explainer — SHAP-based feature attribution and LSTM latent space visualisation.

Two complementary explanation strategies:
  1. SHAP TreeExplainer on IsolationForest:
       - Global: summary bar chart of top-N features by mean |SHAP|
       - Local:  waterfall plot explaining a single flagged transaction
  2. LSTM latent space:
       - Encode all sequences → PCA to 2D → scatter coloured by fraud label
       - A clean separation in latent space confirms the encoder is capturing
         meaningful anomaly signal

All plots are saved to outputs/ (created if absent).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mlflow
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.models.iso_forest import IsolationForestModel
from src.models.lstm_ae import LSTMAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Explainer:
    """
    SHAP attribution and latent-space diagnostics for fraud models.

    Parameters
    ----------
    output_dir : str  Directory for saved plots.
    dpi        : int  Figure resolution.
    """

    DROP_COLS = {"TransactionID", "isFraud", "TransactionDT"}

    def __init__(self, output_dir: str = "outputs/", dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    def _extract_features(
        self, df: pd.DataFrame, feature_cols: Optional[list[str]] = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """Drop identifier columns and return feature-only DataFrame + col list."""
        cols = feature_cols or [c for c in df.columns if c not in self.DROP_COLS]
        return df[cols].astype(np.float32), cols

    # ------------------------------------------------------------------
    # 1. SHAP Summary (global importance)
    # ------------------------------------------------------------------

    def shap_isolation_forest(
        self,
        model: IsolationForestModel,
        df: pd.DataFrame,
        n_samples: int = 500,
        n_top_features: int = 15,
        run_name: str = "shap_isolation_forest",
    ) -> Path:
        """
        Run SHAP TreeExplainer on Isolation Forest and plot a global summary
        bar chart of the top-N most important features.

        Parameters
        ----------
        model          : IsolationForestModel  Must be fitted.
        df             : pd.DataFrame          Feature matrix.
        n_samples      : int                   Rows to subsample for SHAP
                                               (full dataset can be slow).
        n_top_features : int                   Number of features to display.
        run_name       : str

        Returns
        -------
        Path to saved figure.
        """
        if model._model is None:
            raise RuntimeError("IsolationForestModel must be fitted before SHAP analysis.")

        X, feature_cols = self._extract_features(df, model.feature_cols or None)

        # Subsample for speed
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[idx].reset_index(drop=True)
        else:
            X_sample = X.reset_index(drop=True)

        logger.info("Running SHAP TreeExplainer on %d samples …", len(X_sample))
        explainer = shap.TreeExplainer(model._model)
        shap_values = explainer.shap_values(X_sample)  # (n_samples, n_features)

        # Mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:n_top_features]
        top_features = [feature_cols[i] for i in top_idx]
        top_values   = mean_abs_shap[top_idx]

        fig, ax = plt.subplots(figsize=(9, max(5, n_top_features * 0.45)))
        ax.set_facecolor("#F8F9FA")

        bars = ax.barh(
            range(len(top_features)),
            top_values[::-1],
            color="#457B9D",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(
            f"SHAP Feature Importance — Isolation Forest\n(top {n_top_features}, n={len(X_sample)})",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, axis="x", alpha=0.4)
        plt.tight_layout()

        path = self._save(fig, "shap_summary.png")

        # Log top features to MLflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"n_samples": n_samples, "n_top_features": n_top_features})
            for feat, val in zip(top_features, top_values):
                mlflow.log_metric(f"shap_{feat}", float(val))
            mlflow.log_artifact(str(path))

        return path

    # ------------------------------------------------------------------
    # 2. SHAP Waterfall (local explanation)
    # ------------------------------------------------------------------

    def shap_waterfall(
        self,
        model: IsolationForestModel,
        df: pd.DataFrame,
        transaction_idx: int,
        run_name: str | None = None,
    ) -> Path:
        """
        Waterfall plot explaining why a single transaction was flagged.

        Parameters
        ----------
        model           : IsolationForestModel  Fitted model.
        df              : pd.DataFrame          Full feature matrix.
        transaction_idx : int                   Row index of the transaction.

        Returns
        -------
        Path to saved figure.
        """
        if model._model is None:
            raise RuntimeError("IsolationForestModel must be fitted before SHAP analysis.")

        X, feature_cols = self._extract_features(df, model.feature_cols or None)
        X_sample = X.iloc[[transaction_idx]]

        explainer  = shap.TreeExplainer(model._model)
        shap_vals  = explainer.shap_values(X_sample)[0]   # (n_features,)
        base_val   = explainer.expected_value

        # Sort by absolute contribution
        order    = np.argsort(np.abs(shap_vals))[::-1][:15]
        features = [feature_cols[i] for i in order]
        values   = shap_vals[order]
        raw_vals = X_sample.iloc[0].values[order]

        colors = ["#E63946" if v > 0 else "#457B9D" for v in values]

        fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.5)))
        ax.set_facecolor("#F8F9FA")

        bars = ax.barh(range(len(features)), values[::-1],
                       color=colors[::-1], edgecolor="white", alpha=0.85)

        # Annotate with raw feature values
        for i, (bar, raw) in enumerate(zip(bars, raw_vals[::-1])):
            x_pos = bar.get_width()
            ax.text(
                x_pos + (0.002 if x_pos >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{raw:.3g}",
                va="center",
                ha="left" if x_pos >= 0 else "right",
                fontsize=7,
                color="#495057",
            )

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value  (impact on anomaly score)", fontsize=11)
        ax.set_title(
            f"SHAP Waterfall — Transaction index {transaction_idx}\n"
            f"Base value = {float(np.atleast_1d(base_val)[0]):.4f}",
            fontsize=12, fontweight="bold",
        )

        red_patch  = mpatches.Patch(color="#E63946", alpha=0.85, label="Increases risk")
        blue_patch = mpatches.Patch(color="#457B9D", alpha=0.85, label="Decreases risk")
        ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="lower right")
        ax.grid(True, axis="x", alpha=0.4)
        plt.tight_layout()

        filename = f"shap_waterfall_{transaction_idx}.png"
        path = self._save(fig, filename)

        _run = run_name or f"shap_waterfall_{transaction_idx}"
        with mlflow.start_run(run_name=_run):
            mlflow.log_param("transaction_idx", transaction_idx)
            mlflow.log_artifact(str(path))

        return path

    # ------------------------------------------------------------------
    # 3. Top Features DataFrame
    # ------------------------------------------------------------------

    def top_features(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of top-N features ranked by mean absolute SHAP value.

        Parameters
        ----------
        shap_values   : np.ndarray  shape (n_samples, n_features)
        feature_names : list[str]
        n             : int

        Returns
        -------
        pd.DataFrame  columns: feature, mean_abs_shap, rank
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        order    = np.argsort(mean_abs)[::-1][:n]

        df = pd.DataFrame({
            "feature":        [feature_names[i] for i in order],
            "mean_abs_shap":  mean_abs[order].round(6),
            "rank":           range(1, len(order) + 1),
        })
        logger.info("Top %d features by SHAP:\n%s", n, df.to_string(index=False))
        return df

    # ------------------------------------------------------------------
    # 4. LSTM Latent Space Visualisation
    # ------------------------------------------------------------------

    def latent_space_viz(
        self,
        encoder: LSTMAutoencoder,
        dataloader: DataLoader,
        labels: np.ndarray,
        device: str = "cpu",
        run_name: str = "latent_space_viz",
    ) -> Path:
        """
        Encode all sequences, reduce to 2D via PCA, and plot fraud vs normal.

        A clean cluster separation shows the LSTM encoder has learned a
        meaningful normal manifold that separates anomalous transactions.

        Parameters
        ----------
        encoder    : LSTMAutoencoder  Trained model.
        dataloader : DataLoader       Yields (batch_tensor,) tuples.
        labels     : np.ndarray       shape (N,) ground-truth binary labels.
                                      Must match total samples in dataloader.
        device     : str
        run_name   : str

        Returns
        -------
        Path to saved figure.
        """
        encoder.eval()
        device_ = torch.device(device)
        encoder = encoder.to(device_)

        latent_vecs = []
        with torch.no_grad():
            for (batch,) in dataloader:
                batch = batch.to(device_)
                z = encoder.encode(batch)          # (batch, latent_dim)
                latent_vecs.append(z.cpu().numpy())

        Z = np.concatenate(latent_vecs, axis=0)   # (N, latent_dim)

        # Align label length with sequence count (sliding window reduces N)
        n = min(len(Z), len(labels))
        Z      = Z[:n]
        labels_ = labels[:n]

        # PCA → 2D
        pca = PCA(n_components=2, random_state=42)
        Z2  = pca.fit_transform(Z)
        var_explained = pca.explained_variance_ratio_

        # Plot
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_facecolor("#F8F9FA")

        colors_map = {0: "#457B9D", 1: "#E63946"}
        label_map  = {0: "Normal", 1: "Fraud"}
        alphas     = {0: 0.3, 1: 0.7}
        sizes      = {0: 8,   1: 20}

        for cls in [0, 1]:
            mask = labels_ == cls
            ax.scatter(
                Z2[mask, 0], Z2[mask, 1],
                c=colors_map[cls],
                label=f"{label_map[cls]}  (n={mask.sum():,})",
                alpha=alphas[cls],
                s=sizes[cls],
                edgecolors="none",
            )

        ax.set_xlabel(
            f"PC1  ({var_explained[0]*100:.1f}% variance)", fontsize=11
        )
        ax.set_ylabel(
            f"PC2  ({var_explained[1]*100:.1f}% variance)", fontsize=11
        )
        ax.set_title(
            "LSTM Encoder — Latent Space (PCA 2D projection)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10, markerscale=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self._save(fig, "latent_space.png")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "latent_dim":   encoder.latent_dim,
                "pca_components": 2,
            })
            mlflow.log_metrics({
                "pca_var_pc1": float(var_explained[0]),
                "pca_var_pc2": float(var_explained[1]),
                "pca_var_total": float(sum(var_explained)),
            })
            mlflow.log_artifact(str(path))

        logger.info(
            "Latent space: PCA explains %.1f%% variance (PC1=%.1f%% PC2=%.1f%%)",
            sum(var_explained) * 100, var_explained[0] * 100, var_explained[1] * 100,
        )
        return path
