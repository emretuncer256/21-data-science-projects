## 21 Data Science Projects

A curated collection of end‑to‑end data science projects spanning regression, classification, clustering, NLP, computer vision, recommendations, time series, visualization, and deep learning. Each project is self‑contained with code, analysis, and (when applicable) an interactive Streamlit app.

### What you’ll find in each project
-   **Problem overview** and clear objectives
-   **Notebook(s)** with data preparation, modeling, and evaluation
-   **Reusable code** (pipelines, utilities, saved artefacts when useful)
-   **Interactive demo** via Streamlit and/or **Hugging Face Space** (when available)

### Getting started (Poetry)
1. Install Poetry (if not installed)
   - `curl -sSL https://install.python-poetry.org | python3 -`
   - Ensure Poetry is on your PATH (restart shell or follow installer notes)
2. Clone the repository
   - `git clone https://github.com/emretuncer256/21-data-science-projects.git`
   - `cd 21-data-science-projects`
3. Install shared dependencies at the repository root
   - `poetry install`
   - Optional: `poetry shell` to enter the virtual environment

### How to run
-   **Notebooks**: open the `.ipynb` files directly, or launch Jupyter with Poetry from the repo root (works in subfolders too)
    - `poetry run jupyter lab`
    - Optional kernel: `poetry run python -m ipykernel install --user --name ds-projects`
-   **Streamlit apps**: from the specific project folder run `poetry run streamlit run app.py`
-   **Model artefacts**: some projects include serialized models (e.g., `*.pkl`) for quick reuse

### Conventions
-   Small example data or dataset links are referenced inside each project
-   Random seeds are used where appropriate for reproducibility
-   Clear naming for features, targets, and pipeline steps

### Tech stack
-   Python, pandas, NumPy, scikit‑learn
-   Visualization: Matplotlib, Seaborn, Plotly
-   Apps: Streamlit, Hugging Face Spaces
-   Packaging/utility: joblib, Poetry

### Current projects

<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <div style="width:340px; border:1px solid rgba(255,255,255,0.25); border-radius:16px; padding:16px; background:linear-gradient(135deg,#1f6feb 0%,#a371f7 50%,#f778ba 100%); box-shadow:0 8px 24px rgba(31,111,235,0.25); text-align:left; color:#fff;">
      <h3 style="margin:0 0 8px 0; line-height:1.3;">1. Retail Price Optimization — Regression</h3>
      <p style="margin:0 0 8px 0; opacity:0.95;"><b>Description:</b> Model demand and price elasticity from sales data to recommend profit-maximizing prices across products and segments.</p>
      <p style="margin:0 0 12px 0; opacity:0.95;"><b>Topics:</b> Regression, Price Elasticity, Feature Engineering, Regularization, Cross-Validation</p>
      <div style="display:flex; gap:12px; flex-wrap:wrap;">
        <a href="1.%20Retail%20Price%20Optimization%20-%20Regression/" style="background:rgba(255,255,255,0.18); color:#ffffff; padding:6px 12px; border-radius:10px; text-decoration:none; border:1px solid rgba(255,255,255,0.35);">Code</a>
        <a href="1.%20Retail%20Price%20Optimization%20-%20Regression/Retail%20Price%20Optimization.ipynb" style="background:rgba(255,255,255,0.18); color:#ffffff; padding:6px 12px; border-radius:10px; text-decoration:none; border:1px solid rgba(255,255,255,0.35);">Notebook</a>
        <a href="https://huggingface.co/spaces/etuncer/retail-price-optimization-regression" style="background:rgba(255,255,255,0.18); color:#ffffff; padding:6px 12px; border-radius:10px; text-decoration:none; border:1px solid rgba(255,255,255,0.35);">Live App</a>
      </div>
    </div>
  </div>
</div>
<hr>
<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <div style="width:340px; border:1px solid rgba(255,255,255,0.25); border-radius:16px; padding:16px; background:linear-gradient(135deg,#0ea5e9 0%,#10b981 50%,#f59e0b 100%); box-shadow:0 8px 24px rgba(14,165,233,0.25); text-align:left; color:#fff;">
      <h3 style="margin:0 0 8px 0; line-height:1.3;">2. Car Price Prediction — Regression</h3>
      <p style="margin:0 0 8px 0; opacity:0.95;"><b>Description:</b> Predict car prices from technical specs using a scikit-learn pipeline with preprocessing and one-hot encoding.</p>
      <p style="margin:0 0 12px 0; opacity:0.95;"><b>Topics:</b> Regression, scikit-learn Pipeline, OneHotEncoder, Feature Engineering, Streamlit</p>
      <div style="display:flex; gap:12px; flex-wrap:wrap;">
        <a href="2.%20Car%20Price%20Prediction%20-%20Regression/" style="background:rgba(255,255,255,0.18); color:#ffffff; padding:6px 12px; border-radius:10px; text-decoration:none; border:1px solid rgba(255,255,255,0.35);">Code</a>
        <a href="2.%20Car%20Price%20Prediction%20-%20Regression/Car%20Price%20Prediction.ipynb" style="background:rgba(255,255,255,0.18); color:#ffffff; padding:6px 12px; border-radius:10px; text-decoration:none; border:1px solid rgba(255,255,255,0.35);">Notebook</a>
        <a href="https://huggingface.co/spaces/etuncer/car-price-prediction-regression" style="background:rgba(255,255,255,0.18); color:#ffffff; padding:6px 12px; border-radius:10px; text-decoration:none; border:1px solid rgba(255,255,255,0.35);">Live App</a>
      </div>
    </div>
  </div>
</div>

### Contact
For questions or suggestions, please open an issue or reach out my social accounts via GitHub.

---

This README will evolve as new projects are added.