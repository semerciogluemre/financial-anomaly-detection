"""
Financial Anomaly Detection — Streamlit Dashboard
==================================================
4-page interactive dashboard:
  Page 1 · Overview          — results summary, metrics cards, PR curve
  Page 2 · Model Comparison  — interactive confusion matrices, threshold slider
  Page 3 · Transaction Explorer — upload CSV, score live, SHAP waterfall
  Page 4 · Feature Insights  — SHAP summary, latent space, error distribution

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Resolve project root so src/ imports work whether launched from
#    fraud-detection/ or from a parent directory ───────────────────────
ROOT = Path(__file__).resolve().parent.parent   # fraud-detection/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
SQL_DIR = ROOT / "sql"
DB_PATH = ROOT / "db" / "fraud.db"

# ─────────────────────────────────────────────────────────────────────
# Global page config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────

def _load_image(path: Path) -> Image.Image | None:
    if path.exists():
        return Image.open(path)
    return None


def _results_table() -> pd.DataFrame | None:
    csv = OUTPUTS / "results_table.csv"
    if csv.exists():
        return pd.read_csv(csv, index_col=0)
    return None


@st.cache_resource(show_spinner="Loading feature engineering pipeline …")
def _load_feature_engineer():
    """Lazy-load FeatureEngineer so the dashboard works even without a DB."""
    try:
        from src.features import FeatureEngineer
        return FeatureEngineer(db_path=str(DB_PATH), sql_dir=str(SQL_DIR))
    except Exception as e:
        return None


@st.cache_resource(show_spinner="Loading Isolation Forest model …")
def _load_iso_model():
    try:
        import pickle
        model_path = ROOT / "db" / "iso_forest.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def _missing_output_warning(filename: str) -> None:
    st.info(
        f"📂 **`outputs/{filename}`** not found.  "
        "Run the training pipeline first to generate this plot.",
        icon="ℹ️",
    )


# ─────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Fraud Detection")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🤖 Model Comparison",
            "🔎 Transaction Explorer",
            "💡 Feature Insights",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "IEEE-CIS Fraud Detection  \n"
        "LSTM Autoencoder · Isolation Forest · Ensemble  \n"
        "[GitHub](https://github.com/semerciogluemre/financial-anomaly-detection)"
    )


# ═════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📊 Pipeline Overview")
    st.markdown(
        "End-to-end anomaly detection on the **IEEE-CIS Fraud Detection** dataset. "
        "590k+ transactions · 3.5% fraud rate · unsupervised + ensemble approach."
    )
    st.markdown("---")

    # ── Summary metric cards ────────────────────────────────────────
    df_res = _results_table()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Transactions Analysed",
            value="590,540",
            help="Total rows in the IEEE-CIS training set.",
        )
    with col2:
        ens_row = df_res.loc["Ensemble"] if (df_res is not None and "Ensemble" in df_res.index) else None
        fraud_pct = f"{ens_row['Recall'] * 3.5:.1f}%" if ens_row is not None else "~3.5%"
        st.metric(
            label="Fraud Flagged",
            value=fraud_pct,
            help="Approximate % of total transactions flagged as fraudulent by the Ensemble.",
        )
    with col3:
        fpr_val = f"{ens_row['FPR']:.4f}" if ens_row is not None else "—"
        st.metric(
            label="False Positive Rate (Ensemble)",
            value=fpr_val,
            help="FPR = FP / (FP + TN). Lower is better — fewer legitimate transactions wrongly blocked.",
        )

    st.markdown("---")

    # ── Results Table ────────────────────────────────────────────────
    st.subheader("Model Performance Summary")

    if df_res is not None:
        # Highlight best value in each column (max for P/R/F1/ROC/PR, min for FPR)
        def highlight_best(df: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            higher_better = [c for c in df.columns if c != "FPR"]
            lower_better  = ["FPR"] if "FPR" in df.columns else []

            for col in higher_better:
                if df[col].notna().any():
                    best = df[col].idxmax()
                    styles.loc[best, col] = "background-color: #d4edda; font-weight: bold"
            for col in lower_better:
                if df[col].notna().any():
                    best = df[col].idxmin()
                    styles.loc[best, col] = "background-color: #d4edda; font-weight: bold"
            return styles

        styled = df_res.style.apply(highlight_best, axis=None).format("{:.4f}", na_rep="—")
        st.dataframe(styled, use_container_width=True)
        st.caption("✅ Green cells = best value in each column.  FPR: lower is better.  All others: higher is better.")
    else:
        st.info(
            "📂 **`outputs/results_table.csv`** not found.  "
            "Run `src/evaluate.py` after training to generate the results table.",
            icon="ℹ️",
        )
        # Show schema preview so the user knows what to expect
        placeholder = pd.DataFrame(
            [["—"] * 6] * 3,
            index=["LSTM Autoencoder", "Isolation Forest", "Ensemble"],
            columns=["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR"],
        )
        st.dataframe(placeholder, use_container_width=True)

    st.markdown("---")

    # ── PR Curve ─────────────────────────────────────────────────────
    st.subheader("Precision–Recall Curve")

    pr_img = _load_image(OUTPUTS / "pr_curve.png")
    if pr_img:
        st.image(pr_img, use_container_width=True)
    else:
        _missing_output_warning("pr_curve.png")

    with st.expander("📖 Why PR-AUC, not ROC-AUC?"):
        st.markdown(
            """
**ROC-AUC** measures performance across all classification thresholds using TPR vs FPR.
On imbalanced datasets (like fraud, where ~96.5% of transactions are normal),
the enormous number of true negatives keeps FPR artificially low — making a model
that flags almost nothing look deceptively good.

**PR-AUC** (Precision–Recall AUC, also called Average Precision) measures the tradeoff
between Precision (how often a flagged transaction is actually fraud) and Recall (what
fraction of all frauds are caught). It focuses entirely on the **positive class**, so
it is not inflated by the majority of easy-to-classify normal transactions.

A random classifier on a 3.5% fraud dataset achieves a PR-AUC of ~0.035.
Any meaningful model should substantially exceed this baseline — the gap is a clean
measure of how much the model actually learned.

> **Rule of thumb:** if the classes are imbalanced, report PR-AUC.
> If a reviewer only asks for ROC-AUC, mention PR-AUC anyway.
"""
        )


# ═════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown("---")

    df_res = _results_table()

    # ── Interactive Confusion Matrices (Plotly) ──────────────────────
    st.subheader("Confusion Matrices")

    if df_res is not None:
        MODELS = [m for m in ["LSTM Autoencoder", "Isolation Forest", "Ensemble"]
                  if m in df_res.index]

        if MODELS:
            cols = st.columns(len(MODELS))
            TOTAL    = 590_540
            FRAUD    = int(TOTAL * 0.035)
            NORMAL   = TOTAL - FRAUD

            for col, model in zip(cols, MODELS):
                row = df_res.loc[model]
                recall    = row.get("Recall",    0.0)
                precision = row.get("Precision", 0.0)
                fpr       = row.get("FPR",       0.0)

                tp = int(FRAUD  * recall)
                fn = FRAUD  - tp
                fp = int(NORMAL * fpr)
                tn = NORMAL - fp

                z    = [[tn, fp], [fn, tp]]
                text = [
                    [f"TN\n{tn:,}", f"FP\n{fp:,}"],
                    [f"FN\n{fn:,}", f"TP\n{tp:,}"],
                ]

                fig = go.Figure(go.Heatmap(
                    z=z,
                    text=text,
                    texttemplate="%{text}",
                    colorscale=[[0, "#EBF5FB"], [1, "#1A5276"]],
                    showscale=False,
                    xgap=3, ygap=3,
                ))
                fig.update_layout(
                    title=dict(text=model, font=dict(size=14)),
                    xaxis=dict(tickvals=[0, 1], ticktext=["Pred Normal", "Pred Fraud"],
                               title="Predicted"),
                    yaxis=dict(tickvals=[0, 1], ticktext=["Actual Normal", "Actual Fraud"],
                               title="Actual", autorange="reversed"),
                    height=320,
                    margin=dict(l=10, r=10, t=50, b=10),
                    font=dict(size=11),
                )
                col.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model rows not found in results_table.csv.")
    else:
        st.info("📂 Run the evaluation pipeline to generate results_table.csv.", icon="ℹ️")

    st.markdown("---")

    # ── Threshold Slider ─────────────────────────────────────────────
    st.subheader("Ensemble Threshold Explorer")
    st.markdown(
        "Drag the slider to see how changing the decision threshold shifts "
        "Precision, Recall, and F1 for the Ensemble model.  "
        "This makes the precision–recall tradeoff tangible."
    )

    threshold = st.slider(
        "Decision threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Transactions with a combined score above this value are flagged as fraud.",
    )

    # Simulate P/R/F1 as a function of threshold using a sigmoid-shaped curve
    # (in a real deployment these would be computed from actual model scores)
    t   = np.linspace(0, 1, 200)
    sim_precision = 1 / (1 + np.exp(-8 * (t - 0.3)))
    sim_recall    = 1 / (1 + np.exp( 8 * (t - 0.6)))
    sim_f1        = (
        2 * sim_precision * sim_recall /
        np.maximum(sim_precision + sim_recall, 1e-9)
    )

    # Interpolate at selected threshold
    idx       = np.argmin(np.abs(t - threshold))
    p_at_t    = float(sim_precision[idx])
    r_at_t    = float(sim_recall[idx])
    f1_at_t   = float(sim_f1[idx])

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Precision",  f"{p_at_t:.3f}",  help="Of flagged transactions, what fraction are actually fraud?")
    mc2.metric("Recall",     f"{r_at_t:.3f}",  help="Of all frauds, what fraction did we catch?")
    mc3.metric("F1",         f"{f1_at_t:.3f}", help="Harmonic mean of Precision and Recall.")

    st.caption(
        "⚠️ Metrics shown here are **illustrative simulations** based on a sigmoid approximation.  "
        "Replace `sim_precision/recall` arrays with your actual scored validation set "
        "(`EnsembleScorer.combine()` output) for live, data-driven curves."
    )

    st.markdown("---")

    # ── Threshold Analysis Plot ───────────────────────────────────────
    st.subheader("Threshold Analysis (Ensemble)")
    img = _load_image(OUTPUTS / "threshold_analysis_ensemble.png")
    if img:
        st.image(img, use_container_width=True)
    else:
        _missing_output_warning("threshold_analysis_ensemble.png")


# ═════════════════════════════════════════════════════════════════════
# PAGE 3 — TRANSACTION EXPLORER
# ═════════════════════════════════════════════════════════════════════

elif page == "🔎 Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown(
        "Upload a CSV of new transactions to score them live through the "
        "feature engineering pipeline and ensemble model."
    )
    st.markdown("---")

    # ── File Uploader ────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload transactions CSV",
        type=["csv"],
        help="Must include at least the columns expected by the feature pipeline "
             "(TransactionID, TransactionAmt, card1, TransactionDT, etc.)",
    )

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded, low_memory=False)
            st.success(f"✅ Loaded **{len(raw_df):,}** transactions.")

            with st.expander("Raw data preview"):
                st.dataframe(raw_df.head(10), use_container_width=True)

            # ── Score ────────────────────────────────────────────────
            with st.spinner("Running feature engineering …"):
                fe = _load_feature_engineer()
                iso = _load_iso_model()

                if fe is None or iso is None:
                    st.warning(
                        "⚠️ Feature engineer or model not available. "
                        "Showing a **demo mode** — random scores for illustration.",
                        icon="⚠️",
                    )
                    np.random.seed(42)
                    scores     = np.random.beta(2, 20, size=len(raw_df))
                    feature_df = raw_df.copy()
                else:
                    # Write upload to a temp CSV so ingestion can read it
                    import tempfile, shutil
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        raw_df.to_csv(tmp.name, index=False)
                        tmp_path = tmp.name
                    feature_df = fe.build_feature_matrix()
                    scores     = iso.score(feature_df)
                    os.unlink(tmp_path)

            THRESHOLD = 0.5
            results = raw_df.copy()
            results["anomaly_score"] = scores[:len(results)]
            results["fraud_flag"]    = (results["anomaly_score"] > THRESHOLD).astype(int)

            n_flagged = results["fraud_flag"].sum()
            flag_rate = n_flagged / len(results) * 100

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Transactions scored",  f"{len(results):,}")
            rc2.metric("Flagged as fraud",      f"{n_flagged:,}")
            rc3.metric("Flag rate",             f"{flag_rate:.2f}%")

            st.markdown("---")

            # ── Colour-coded results table ────────────────────────────
            st.subheader("Scored Results")

            def _row_color(row):
                if row["fraud_flag"] == 1:
                    return ["background-color: #fce4e4"] * len(row)
                return [""] * len(row)

            display_cols = ["TransactionID", "anomaly_score", "fraud_flag"] + [
                c for c in ["TransactionAmt", "card1", "ProductCD"]
                if c in results.columns
            ]
            st.dataframe(
                results[display_cols]
                .style.apply(_row_color, axis=1)
                .format({"anomaly_score": "{:.4f}"}),
                use_container_width=True,
                height=400,
            )

            # ── SHAP Waterfall for flagged transactions ───────────────
            flagged_ids = results[results["fraud_flag"] == 1].index.tolist()

            if flagged_ids:
                st.markdown("---")
                st.subheader("SHAP Explanation — Flagged Transactions")

                selected = st.selectbox(
                    "Select a flagged transaction to explain",
                    options=flagged_ids[:50],   # cap at 50 for performance
                    format_func=lambda i: (
                        f"Index {i}"
                        + (f"  (ID: {results.loc[i, 'TransactionID']})"
                           if "TransactionID" in results.columns else "")
                    ),
                )

                if iso is not None:
                    with st.spinner("Computing SHAP values …"):
                        try:
                            import shap
                            from src.explainability import Explainer
                            ex = Explainer(output_dir=str(OUTPUTS))
                            path = ex.shap_waterfall(iso, feature_df, transaction_idx=selected)
                            st.image(str(path), use_container_width=True)
                            st.caption(
                                "Red bars increase the fraud risk score; "
                                "blue bars decrease it. "
                                "Raw feature values are annotated on each bar."
                            )
                        except Exception as e:
                            st.error(f"SHAP computation failed: {e}")
                else:
                    # Demo: show a placeholder waterfall from outputs/ if it exists
                    wf = _load_image(OUTPUTS / f"shap_waterfall_0.png")
                    if wf:
                        st.image(wf, use_container_width=True,
                                 caption="Example SHAP waterfall (demo mode).")
                    else:
                        st.info("Train and save the model to enable live SHAP explanations.", icon="ℹ️")

            # ── Download button ───────────────────────────────────────
            st.markdown("---")
            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download scored results as CSV",
                data=csv_bytes,
                file_name="scored_transactions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error processing upload: {e}")

    else:
        st.info(
            "👆 Upload a CSV file to score it.  \n\n"
            "**Expected columns** (minimum): `TransactionID`, `TransactionAmt`, "
            "`card1`, `TransactionDT`.  \n\n"
            "The pipeline will attempt to engineer features for any columns "
            "matching the training schema.",
            icon="ℹ️",
        )


# ═════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE INSIGHTS
# ═════════════════════════════════════════════════════════════════════

elif page == "💡 Feature Insights":
    st.title("💡 Feature Insights")
    st.markdown(
        "Explainability outputs from SHAP and the LSTM encoder — showing *which* "
        "signals drive fraud predictions and *what* the models learned."
    )
    st.markdown("---")

    # ── SHAP Summary ─────────────────────────────────────────────────
    st.subheader("SHAP Feature Importance — Isolation Forest")

    shap_img = _load_image(OUTPUTS / "shap_summary.png")
    if shap_img:
        st.image(shap_img, use_container_width=True)
    else:
        _missing_output_warning("shap_summary.png")

    with st.expander("📖 Interpreting this chart + top 3 features"):
        st.markdown(
            """
Each bar shows the **mean absolute SHAP value** for a feature across 500 sampled
transactions. Higher = that feature has a larger average impact on the Isolation
Forest's anomaly score.

**Top 3 features (typical findings on IEEE-CIS):**

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | `amt_zscore` | A transaction that is an extreme outlier relative to the card's own history is the strongest single signal. Fraudsters frequently make one large purchase after gaining access. |
| 2 | `txn_count_1h` | Velocity in the past hour. A card that transacts 10× in 60 minutes when its normal rate is once a day is almost always compromised. |
| 3 | `device_risk_score` | Devices with a historically high fraud rate are strong prior evidence. Combined with an unusual amount, they form the most common fraud signature in this dataset. |

SHAP values here are computed via `TreeExplainer`, which uses the tree structure
directly rather than sampling — making them fast and exact for tree-based models.
"""
        )

    st.markdown("---")

    # ── Latent Space ─────────────────────────────────────────────────
    st.subheader("LSTM Encoder — Latent Space (PCA 2D)")

    latent_img = _load_image(OUTPUTS / "latent_space.png")
    if latent_img:
        st.image(latent_img, use_container_width=True)
    else:
        _missing_output_warning("latent_space.png")

    with st.expander("📖 What did the LSTM learn?"):
        st.markdown(
            """
The LSTM Autoencoder was trained on **normal transactions only**. After training,
we pass *all* transactions (including fraud) through the encoder and project the
resulting latent vectors to 2D using PCA.

**What a good plot looks like:**
- Normal transactions (blue) form a **dense central cluster** — the model has
  compressed them into a compact region it understands well.
- Fraud transactions (red) scatter **outside or at the periphery** of that cluster —
  because the encoder never saw them during training, it cannot compress them
  efficiently, and they land in unexpected regions of latent space.

This scatter plot is a visual sanity check that the encoder is doing something
meaningful. If fraud and normal points are completely mixed, the model has not
learned a useful normal manifold and reconstruction error will not be a reliable
anomaly signal.

The variance explained by PC1 and PC2 is annotated on the axes — together they
should capture at least 30–50% of the latent variance for the projection to be
informative.
"""
        )

    st.markdown("---")

    # ── Error Distribution ────────────────────────────────────────────
    st.subheader("LSTM Reconstruction Error Distribution")

    err_img = _load_image(OUTPUTS / "error_distribution.png")
    if err_img:
        st.image(err_img, use_container_width=True)
    else:
        _missing_output_warning("error_distribution.png")

    with st.expander("📖 Reading the reconstruction error chart"):
        st.markdown(
            """
The LSTM Autoencoder outputs a **reconstruction** of each input sequence.
The anomaly score is the mean squared error (MSE) between input and output.

**Why this works:**
The model was trained to minimise reconstruction error on *normal* transactions.
It became good at reconstructing normal sequences — so normal transactions have
**low MSE**. Fraudulent transactions have patterns the model never trained on,
so it reconstructs them poorly — resulting in **high MSE**.

**What to look for in this plot:**
- The **normal distribution** (left panel) should be tightly concentrated near
  zero with a low mean.
- The **fraud distribution** (right panel) should be right-shifted — higher mean
  and broader spread.
- The `error_separation` metric (logged to MLflow) quantifies this gap:
  `fraud_mean − normal_mean`. A larger separation means the LSTM has learned
  a better normal manifold and the threshold will be more stable.

If the two distributions overlap heavily, consider increasing model capacity
(`hidden_dim`, `latent_dim`) or training for more epochs.
"""
        )
