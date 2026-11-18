# PathSnap: Distributed Map-Matching & Route Reconstruction

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-blue.svg)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Tuning-Optuna-brightgreen.svg)](https://optuna.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

**Course:** UE24CS352A: Machine Learning  
**Institution:** Department of Computer Science and Engineering, RR Campus, PES University  

---

## 📌 Executive Overview
GPS tracking systems frequently suffer from multipath distortions, urban canyon reflections, signal attenuation, and low sampling frequencies (~1 Hz). As a result, observed trajectories appear to jump across buildings or onto adjacent roads. 

**PathSnap** is an end-to-end distributed machine learning system that accurately reconstructs realistic vehicle trajectories by matching noisy GPS observations to a directed road network. It combines:
1. **Planar Geodesic Metric Projections:** Eliminates angular coordinate distortion via WGS-84 metric transformations.
2. **STRtree Spatial Indexing:** Sub-40ms spatial candidate arc queries.
3. **Multi-Model Machine Learning:** Evaluates 5 distinct model families with GroupKFold cross-validation on 80 training trips.
4. **Bayesian Hyperparameter Optimization:** Optuna TPE search improving Macro F1 and Log Loss.
5. **Topology-Aware Hidden Markov Model (HMM) & Viterbi Decoding:** Reconstructs continuous road routes with **88.70% Top-1 Accuracy** and **98.14% Route Continuity**.
6. **Two-Laptop Distributed Deployment:** FastAPI Backend on Laptop 1 and Streamlit Evaluator UI on Laptop 2.

---

## 📁 Repository Structure

```text
├── pathsnap/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── loader.py                   # Data ingestion, coordinate projections & trip dataclasses
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── extractor.py                # Candidate generation, spatial index & 10 engineered features
│   │   └── build_dataset.py            # Feature dataset generation for train and test partitions
│   ├── models/
│   │   ├── __init__.py
│   │   └── train_models.py             # 5 model families, GroupKFold CV, metrics & serialized weights
│   ├── hyperparameter_tuning/
│   │   ├── __init__.py
│   │   └── bayesian_opt.py             # Optuna Bayesian hyperparameter search (20 trials)
│   ├── hmm/
│   │   ├── __init__.py
│   │   ├── viterbi.py                  # Topology graph builder, emission/transition scoring & Viterbi
│   │   └── evaluate_hmm.py             # Benchmark comparing Classifier vs Classifier + HMM
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                     # FastAPI server (Laptop 1) with evaluation endpoints & live logs
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py                      # Interactive Streamlit dashboard (Laptop 2) with Folium maps
│   ├── visualizations/
│   │   ├── __init__.py
│   │   ├── folium_maps.py              # Interactive HTML Folium map generation
│   │   ├── generate_plots.py           # Publication-quality benchmark figures & training curves
│   │   └── generate_deployment_screenshots.py # UI & API terminal mockups for report
│   ├── artifacts/
│   │   ├── trained_models/             # Serialized model weights (*.joblib) & scaler
│   │   ├── results/                    # CSV & JSON evaluation records
│   │   ├── figures/                    # Generated high-resolution charts (*.png)
│   │   └── maps/                       # Interactive HTML Folium maps (*.html)
│   └── report/
│       ├── generate_report.py          # ReportLab script generating Deliverable 2
│       └── PathSnap_Technical_Report.pdf # Final Consolidated Technical Report (PDF)
├── requirements.txt                    # Python package dependencies
└── README.md                           # Documentation & execution guide
```

---

## 🔬 Experimental Results

### 1. Model Comparison across 5 Families (Student-Testing Dataset)
| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Test Macro F1 | Log Loss | Inference Latency |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Linear / Probabilistic | $C=1.0$, balanced | 0.7501 | 76.32% | 88.07% | 0.7501 | 0.2766 | 0.05 ms / 1k |
| **K-Nearest Neighbors** | Instance-Based | $k=15$, distance-weighted | 0.8208 | 76.14% | 86.96% | 0.8452 | 0.1956 | 115.7 ms / 1k |
| **Random Forest** | Ensemble (Bagging) | 100 trees, depth=12 | 0.8286 | 80.05% | 89.69% | 0.8342 | 0.1803 | 1.93 ms / 1k |
| **LightGBM** | Boosting (GBDT) | 150 trees, depth=7, lr=0.08 | **0.8381** | **80.44%** | **90.08%** | **0.8478** | **0.1526** | **1.41 ms / 1k** |
| **Multi-Layer Perceptron** | Neural Networks | $(64, 32)$, ReLU, Adam | 0.8495 | 79.85% | 89.40% | 0.8703 | 0.1170 | 0.56 ms / 1k |

### 2. Bayesian Hyperparameter Optimization (LightGBM)
* **Search Space:** `num_leaves` [15, 95], `max_depth` [4, 10], `learning_rate` [0.03, 0.20], `n_estimators` [80, 220], `min_child_samples` [20, 150], `subsample` [0.65, 1.0], `colsample_bytree` [0.65, 1.0], `scale_pos_weight` [2.0, 7.0].
* **Optimal Parameters:** `num_leaves=18`, `max_depth=7`, `learning_rate=0.1075`, `n_estimators=150`, `min_child_samples=120`, `subsample=0.734`, `colsample_bytree=0.670`, `scale_pos_weight=2.082`.
* **Gains:** CV Macro F1: $0.8381 \to \mathbf{0.8561}$ (+0.0180) | Test Macro F1: $0.8478 \to \mathbf{0.8663}$ (+0.0185) | Log Loss: $0.1526 \to \mathbf{0.1258}$ (-0.0268).

### 3. Route Reconstruction: Classifier-Only vs. Classifier + HMM
| Metric | Classifier-Only | Classifier + HMM (Viterbi) | Absolute Gain |
|---|:---:|:---:|:---:|
| **Mean Top-1 Accuracy** | 82.67% | **88.70%** | **+6.03%** |
| **Mean Route Continuity** | 89.68% | **98.14%** | **+8.46%** |

---

## 🚀 Setup & Execution Guide

### 1. Environment Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/<your-username>/PathSnap.git
cd PathSnap
python -m pip install -r requirements.txt
```

### 2. Feature Extraction & Dataset Generation
Extract candidate segments and engineer geometric/kinematic features:
```bash
python -m pathsnap.feature_engineering.build_dataset
```

### 3. Model Training & Evaluation
Train 5 model families with GroupKFold CV and save trained weights:
```bash
python -m pathsnap.models.train_models
```

### 4. Bayesian Hyperparameter Optimization
Execute Optuna Bayesian optimization:
```bash
python -m pathsnap.hyperparameter_tuning.bayesian_opt
```

### 5. HMM Route Reconstruction Benchmark
Benchmark HMM Viterbi decoding against classifier-only predictions:
```bash
python -m pathsnap.hmm.evaluate_hmm
```

### 6. Generate Figures, Folium Maps & PDF Report
```bash
python -m pathsnap.visualizations.generate_plots
python -m pathsnap.visualizations.folium_maps
python -m pathsnap.report.generate_report
```

---

## 💻 Two-Laptop Deployment Architecture

During evaluation, PathSnap runs across two interconnected machines over local Wi-Fi / Hotspot:

```
+------------------------------------+           +------------------------------------+
|             LAPTOP 2               |           |             LAPTOP 1               |
|      (Evaluator Interface)         |           |       (Model & Inference API)      |
|                                    |   HTTP    |                                    |
|   - Selects any evaluation trip    |  Request  |   - Hosts tuned LightGBM model     |
|   - Sends Trip ID to Laptop 1      | --------> |   - HMM Viterbi Decoder engine     |
|   - Renders interactive Folium map |           |   - Server-side Ground Truth       |
|   - Displays evaluation metrics    | <-------- |   - Evaluates Top-1 & Continuity   |
|                                    |  Result   |   - Prints live request CMD logs   |
+------------------------------------+           +------------------------------------+
```

### Step 1: Start Backend Server on Laptop 1
Find Laptop 1's local IP address (e.g., `ipconfig` &rarr; `192.168.1.10`), then run:
```bash
python -m uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```
*Laptop 1 logs all incoming inference requests, candidate extraction timing, Viterbi execution latency, and server-side ground-truth metrics in real-time.*

### Step 2: Start Evaluator UI on Laptop 2
On Laptop 2, launch the Streamlit frontend:
```bash
streamlit run pathsnap/ui/app.py
```
1. In the sidebar, set the **Laptop 1 Backend API URL** (e.g. `http://192.168.1.10:8000` or `http://localhost:8000`).
2. Select any trip from the 10 evaluation options.
3. Click **"🚀 Reconstruct & Evaluate Route"**.
4. View real-time KPIs (Top-1 Accuracy, Route Continuity, Latency) and the interactive Folium map displaying raw GPS points, ground truth, and the reconstructed path.

---

## ⚖️ Academic Integrity & AI Use Disclosure
* All software, feature extractions, model pipelines, and decoders were independently written.
* Evaluator testing data and ground-truth routes remained strictly server-side inside Laptop 1 and were never used in training, feature selection, or exposed to the client.
* No hardcoded answers or predicted values are used.
* All reported metrics reflect actual program outputs reproducible from the submitted repository.
