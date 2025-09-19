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
    <a href="1.%20Retail%20Price%20Optimization%20-%20Regression/">
      <img src="assets/cards/retail-price-optimization.svg" alt="Retail Price Optimization — Regression" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="1.%20Retail%20Price%20Optimization%20-%20Regression/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="1.%20Retail%20Price%20Optimization%20-%20Regression/Retail%20Price%20Optimization.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/retail-price-optimization-regression"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>
<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="2.%20Car%20Price%20Prediction%20-%20Regression/">
      <img src="assets/cards/car-price-prediction.svg" alt="Car Price Prediction — Regression" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="2.%20Car%20Price%20Prediction%20-%20Regression/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="2.%20Car%20Price%20Prediction%20-%20Regression/Car%20Price%20Prediction.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/car-price-prediction-regression"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>
<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="3.%20Food%20Delivery%20Time%20Prediction%20-%20Regression/">
      <img src="assets/cards/food-delivery-time-prediction.svg" alt="Food Delivery Time Prediction — Regression" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="3.%20Food%20Delivery%20Time%20Prediction%20-%20Regression/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="3.%20Food%20Delivery%20Time%20Prediction%20-%20Regression/Food%20Delivery%20Time%20Prediction.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/food-delivery-time-prediction-regression"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>
<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="4.%20Iris%20Flower%20-%20Classification/">
      <img src="assets/cards/iris-flower-classification.svg" alt="Iris Flower Classification — Classification" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="4.%20Iris%20Flower%20-%20Classification/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="4.%20Iris%20Flower%20-%20Classification/Iris%20Flower%20Classification.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/iris-flower-classification"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>
<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="5.%20Text%20Emotions%20-%20Classification/">
      <img src="assets/cards/text-emotions-classification.svg" alt="Text Emotions Classification — Classification" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="5.%20Text%20Emotions%20-%20Classification/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="5.%20Text%20Emotions%20-%20Classification/Text%20Emotions%20Classification.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/text-emotions-classification"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>

<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="6.%20Mobile%20Price%20-%20Classification/">
      <img src="assets/cards/mobile-price-classification.svg" alt="Mobile Price — Classification" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="6.%20Mobile%20Price%20-%20Classification/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="6.%20Mobile%20Price%20-%20Classification/Mobile%20Price%20Classification.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/mobile-price-classification"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>

<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="7.%20Music%20Genres%20-%20Clustering/">
      <img src="assets/cards/music-genres-clustering.svg" alt="Music Genres — Clustering" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="7.%20Music%20Genres%20-%20Clustering/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="7.%20Music%20Genres%20-%20Clustering/Music%20Genres%20Clustering.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/music-genres-clustering"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>

<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="8.%20Credit%20Card%20-%20Clustering/">
      <img src="assets/cards/credit-card-clustering.svg" alt="Credit Card Clustering — Customer Segmentation" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="8.%20Credit%20Card%20-%20Clustering/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="8.%20Credit%20Card%20-%20Clustering/Credit%20Card%20Clustering.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/credit-card-clustering"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>

<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="9.%20Customer%20RFM%20Analysis%20-%20Clustering/">
      <img src="assets/cards/customer-rfm-analysis.svg" alt="Customer RFM Analysis — Clustering" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="9.%20Customer%20RFM%20Analysis%20-%20Clustering/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="9.%20Customer%20RFM%20Analysis%20-%20Clustering/Customer%20RFM%20Analysis.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/customer-rfm-analysis-clustering"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>

<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="10.%20MNIST%20Digits%20Classification%20-%20Computer%20Vision/">
      <img src="assets/cards/mnist-digits-classification.svg" alt="MNIST Digits Classification — Computer Vision" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="10.%20MNIST%20Digits%20Classification%20-%20Computer%20Vision/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="10.%20MNIST%20Digits%20Classification%20-%20Computer%20Vision/MNIST%20Digits%20Classification.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/mnist-digits-classification-CV"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>

<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="11.%20Fashion%20Recommendation%20System%20-%20Computer%20Vision/">
      <img src="assets/cards/fashion-recommendation-system.svg" alt="Fashion Recommendation System — Computer Vision" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="11.%20Fashion%20Recommendation%20System%20-%20Computer%20Vision/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="11.%20Fashion%20Recommendation%20System%20-%20Computer%20Vision/Fashion%20Recommendation%20System.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>

<div align="center">
  <div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
    <a href="12.%20Real-time%20OCR%20-%20Computer%20Vision/">
      <img src="assets/cards/real-time-ocr.svg" alt="Real-time OCR — Computer Vision" width="560" />
    </a>
    <div align="center" style="margin-top:8px;">
      <a href="12.%20Real-time%20OCR%20-%20Computer%20Vision/"><img alt="Code" src="https://img.shields.io/badge/Code-1f6feb?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="12.%20Real-time%20OCR%20-%20Computer%20Vision/Optical%20Character%20Recognition.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a>
      <a href="https://huggingface.co/spaces/etuncer/real-time-ocr"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a>
    </div>
  </div>
</div>
<hr>


### Contact
For questions or suggestions, please open an issue or reach out my social accounts via GitHub.

---

This README will evolve as new projects are added.