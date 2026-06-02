# Dashboard

Interactive 4-page Streamlit app for the financial anomaly detection pipeline.

## Pages

| Page | What it shows |
|---|---|
| **Overview** | Results table with conditional formatting, summary metric cards, PR curve with AUC-PR explanation |
| **Model Comparison** | Interactive Plotly confusion matrices for all 3 models, threshold slider with live P/R/F1, threshold analysis plot |
| **Transaction Explorer** | CSV upload → live scoring → colour-coded results table → SHAP waterfall for flagged rows → download |
| **Feature Insights** | SHAP summary chart, LSTM latent space PCA scatter, reconstruction error distribution — each with explanatory captions |

## Requirements

Install the project dependencies from the root `requirements.txt`, plus:

```bash
pip install streamlit plotly Pillow
```

## Running

From the `fraud-detection/` directory:

```bash
streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Notes

- **Plots** are read from `outputs/`. Run `src/evaluate.py` and `src/explainability.py`
  after training to generate them.
- **Live scoring** on the Transaction Explorer page requires a fitted `IsolationForestModel`
  saved to `db/iso_forest.pkl` and a populated SQLite database at `db/fraud.db`.
  Without these, the page runs in **demo mode** with random scores for illustration.
- **SHAP waterfalls** on uploaded transactions require the fitted model to be available.
