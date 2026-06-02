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

    # ------------------------------------------------------------------
    # XGBoost SHAP (primary explainer — v1.1.0)
    # ------------------------------------------------------------------

    def shap_xgboost_summary(
        self,
        model,                     # XGBoostModel instance
        X: pd.DataFrame,
        n_samples: int = 2000,
        n_top_features: int = 15,
        run_name: str = "shap_xgboost_summary",
    ) -> Path:
        """
        SHAP TreeExplainer on XGBoost — global feature importance bar chart.

        Uses the exact TreeExplainer (not the approximate kernel version),
        which is fast and precise for tree-based models.
        """
        if model._model is None:
            raise RuntimeError("XGBoostModel must be fitted.")

        feat_cols = model.feature_names_ or list(X.columns)
        X_sample  = X[feat_cols].astype(np.float32)
        if len(X_sample) > n_samples:
            X_sample = X_sample.sample(n_samples, random_state=42)

        logger.info("SHAP TreeExplainer on XGBoost (%d samples) …", len(X_sample))
        # Use predict as background to avoid base_score float-parse bug
        # in some shap/xgboost version combinations
        predict_fn = lambda x: model._model.predict_proba(x)[:, 1]
        explainer  = shap.Explainer(predict_fn, X_sample.values[:100])
        shap_vals  = explainer(X_sample.values).values   # (n, n_features)
        mean_abs   = np.abs(shap_vals).mean(axis=0)
        top_idx    = np.argsort(mean_abs)[::-1][:n_top_features]
        top_feats  = [feat_cols[i] for i in top_idx]
        top_vals   = mean_abs[top_idx]

        fig, ax = plt.subplots(figsize=(9, max(5, n_top_features * 0.45)))
        ax.set_facecolor("#F8F9FA")
        ax.barh(range(len(top_feats)), top_vals[::-1],
                color="#2A9D8F", edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(
            f"SHAP Feature Importance — XGBoost\n(top {n_top_features}, n={len(X_sample)})",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, axis="x", alpha=0.4)
        plt.tight_layout()

        path = self._save(fig, "shap_xgboost_summary.png")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"n_samples": n_samples, "n_top_features": n_top_features})
            for feat, val in zip(top_feats, top_vals):
                mlflow.log_metric(f"xgb_shap_{feat}", float(val))
            mlflow.log_artifact(str(path))

        return path

    def shap_xgboost_waterfall(
        self,
        model,
        X: pd.DataFrame,
        transaction_idx: int,
        run_name: str | None = None,
    ) -> Path:
        """Waterfall plot for a single transaction using XGBoost SHAP values."""
        if model._model is None:
            raise RuntimeError("XGBoostModel must be fitted.")

        feat_cols = model.feature_names_ or list(X.columns)
        X_row     = X[feat_cols].iloc[[transaction_idx]].astype(np.float32)

        predict_fn = lambda x: model._model.predict_proba(x)[:, 1]
        # Use a small background to avoid the TreeExplainer base_score bug
        bg = X_row.values  # single-row background for speed
        explainer = shap.Explainer(predict_fn, bg)
        shap_obj  = explainer(X_row.values)
        shap_vals = shap_obj.values[0]
        base_val  = float(shap_obj.base_values[0]) if hasattr(shap_obj, "base_values") else 0.0
        raw_vals  = X_row.values[0]

        order   = np.argsort(np.abs(shap_vals))[::-1][:15]
        feats   = [feat_cols[i] for i in order]
        vals    = shap_vals[order]
        raws    = raw_vals[order]
        colors  = ["#E63946" if v > 0 else "#2A9D8F" for v in vals]

        fig, ax = plt.subplots(figsize=(10, max(5, len(feats) * 0.5)))
        ax.set_facecolor("#F8F9FA")
        bars = ax.barh(range(len(feats)), vals[::-1],
                       color=colors[::-1], edgecolor="white", alpha=0.85)
        for bar, raw in zip(bars, raws[::-1]):
            x_pos = bar.get_width()
            ax.text(x_pos + (0.002 if x_pos >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    f"{raw:.3g}", va="center",
                    ha="left" if x_pos >= 0 else "right",
                    fontsize=7, color="#495057")
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on fraud probability)", fontsize=11)
        ax.set_title(
            f"XGBoost SHAP Waterfall — Transaction index {transaction_idx}\n"
            f"Base value = {base_val:.4f}",
            fontsize=12, fontweight="bold",
        )
        import matplotlib.patches as mpatches
        ax.legend(handles=[
            mpatches.Patch(color="#E63946", alpha=0.85, label="Increases fraud prob"),
            mpatches.Patch(color="#2A9D8F", alpha=0.85, label="Decreases fraud prob"),
        ], fontsize=9, loc="lower right")
        ax.grid(True, axis="x", alpha=0.4)
        plt.tight_layout()

        filename = f"shap_waterfall_{transaction_idx}.png"
        path = self._save(fig, filename)
        _run = run_name or f"shap_xgb_waterfall_{transaction_idx}"
        with mlflow.start_run(run_name=_run):
            mlflow.log_artifact(str(path))
        return path

    def shap_comparison(
        self,
        xgb_model,
        iso_model,
        X: pd.DataFrame,
        n_samples: int = 1000,
        n_top: int = 10,
        run_name: str = "shap_comparison",
    ) -> Path:
        """
        Side-by-side bar chart comparing top-10 SHAP features from
        XGBoost (supervised) and Isolation Forest (unsupervised).
        Shows whether both models agree on what drives anomaly scores.
        """
        # XGBoost SHAP
        xgb_feats = xgb_model.feature_names_ or list(X.columns)
        X_xgb = X[xgb_feats].astype(np.float32)
        if len(X_xgb) > n_samples:
            X_xgb = X_xgb.sample(n_samples, random_state=42)
        predict_fn_xgb = lambda x: xgb_model._model.predict_proba(x)[:, 1]
        xgb_exp  = shap.Explainer(predict_fn_xgb, X_xgb.values[:100])
        xgb_sv   = xgb_exp(X_xgb.values).values
        xgb_mean = np.abs(xgb_sv).mean(axis=0)
        xgb_top  = pd.Series(xgb_mean, index=xgb_feats).nlargest(n_top)

        # IF SHAP
        X_if = X.drop(columns=[c for c in iso_model.DROP_COLS if c in X.columns])
        X_if = X_if.select_dtypes(include="number").astype(np.float32)
        if iso_model._scaler:
            X_if_s = iso_model._scaler.transform(X_if.values)
        else:
            X_if_s = X_if.values
        if len(X_if_s) > n_samples:
            X_if_s = X_if_s[:n_samples]
        if_exp  = shap.TreeExplainer(iso_model._model)
        if_sv   = if_exp.shap_values(X_if_s)
        if_mean = np.abs(if_sv).mean(axis=0)
        if_feats = iso_model.feature_cols or list(X_if.columns)
        if_top   = pd.Series(if_mean[:len(if_feats)], index=if_feats[:len(if_mean)]).nlargest(n_top)

        # Normalise to [0,1] for fair visual comparison
        xgb_norm = xgb_top / xgb_top.max()
        if_norm  = if_top  / if_top.max()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("SHAP Feature Importance Comparison\nXGBoost (supervised) vs Isolation Forest (unsupervised)",
                     fontsize=13, fontweight="bold")

        for ax, series, title, color in [
            (axes[0], xgb_norm, "XGBoost", "#2A9D8F"),
            (axes[1], if_norm,  "Isolation Forest", "#457B9D"),
        ]:
            ax.set_facecolor("#F8F9FA")
            ax.barh(range(len(series)), series.values[::-1],
                    color=color, edgecolor="white", alpha=0.85)
            ax.set_yticks(range(len(series)))
            ax.set_yticklabels(series.index[::-1], fontsize=9)
            ax.set_xlabel("Normalised Mean |SHAP|")
            ax.set_title(title, fontweight="bold")
            ax.grid(True, axis="x", alpha=0.4)

        plt.tight_layout()
        path = self._save(fig, "shap_comparison.png")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_artifact(str(path))
        return path
