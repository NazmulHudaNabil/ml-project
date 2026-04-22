# 🎓 Student Math Score Predictor

> An end-to-end Machine Learning web application that predicts a student's **math score** based on demographic and academic features, served via a **FastAPI** REST API with a clean HTML frontend.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This project builds a complete ML pipeline — from raw data ingestion through preprocessing, model training with hyperparameter tuning, to real-time inference via a web API. It compares **7 regression algorithms** and automatically selects the best-performing model for deployment.

**Target variable:** `math score` (0–100)

**Input features:**
| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | `male` / `female` |
| `race/ethnicity` | Categorical | Group A–E |
| `parental level of education` | Categorical | Highest education attained |
| `lunch` | Categorical | `standard` / `free/reduced` |
| `test preparation course` | Categorical | `completed` / `none` |
| `reading score` | Numerical | Score out of 100 |
| `writing score` | Numerical | Score out of 100 |

---

## 🏗️ Project Architecture

```
ml-project/
│
├── src/                          # Core source package
│   ├── component/
│   │   ├── data_ingestion.py     # Loads raw data, splits into train/test
│   │   ├── data_transformation.py# Feature engineering & preprocessing pipeline
│   │   └── model_trainer.py      # Trains & evaluates multiple models, saves best
│   │
│   ├── pipline/
│   │   ├── training_pipline.py   # Orchestrates the full training workflow
│   │   └── predict_pipline.py    # Loads model & preprocessor for inference
│   │
│   ├── exception.py              # Custom exception handling
│   ├── loger.py                  # Centralized logging setup
│   └── utils.py                  # Shared utilities (save/load objects, evaluation)
│
├── artifacts/                    # Auto-generated model & data artifacts
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── notebook/                     # Exploratory Data Analysis notebooks
├── templates/                    # HTML frontend templates (Jinja2)
│   ├── index.html
│   └── home.html
│
├── app.py                        # FastAPI application entry point
├── setup.py                      # Package installation script
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🤖 Models Evaluated

The training pipeline benchmarks all of the following regressors using **GridSearchCV** and selects the one with the highest **R² score** on the test set:

| Model | Hyperparameters Tuned |
|---|---|
| Linear Regression | — |
| Decision Tree | `criterion` |
| Random Forest | `n_estimators` |
| Gradient Boosting | `learning_rate`, `subsample`, `n_estimators` |
| XGBoost | `learning_rate`, `n_estimators` |
| CatBoost | `depth`, `learning_rate`, `iterations` |
| AdaBoost | `learning_rate`, `n_estimators` |

> ✅ A model is only accepted if its R² score exceeds **0.60**. Otherwise, the pipeline raises an exception.

---

## ⚙️ ML Pipeline

```
Raw CSV Data
     │
     ▼
┌─────────────────┐
│  Data Ingestion  │  → Loads data, performs 80/20 train-test split
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Data Transformation  │  → Numerical: Median Imputer + Standard Scaler
└────────┬─────────────┘     Categorical: Mode Imputer + OneHotEncoder + Scaler
         │
         ▼
┌──────────────────┐
│  Model Training   │  → GridSearchCV over 7 algorithms → saves best model.pkl
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  Prediction Service   │  → FastAPI loads model.pkl + preprocessor.pkl → /predict
└──────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/NazmulHudaNabil/ml-project.git
cd ml-project
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Run the Training Pipeline

This step ingests the raw data, preprocesses it, trains all models, and saves the best one:

```bash
python src/component/data_ingestion.py
```

Artifacts will be generated under the `artifacts/` directory.

### 5. Start the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Navigate to **http://localhost:8080** in your browser to use the web UI.

---

## 📡 API Reference

### `GET /`
Returns the home page (HTML).

---

### `POST /predict`

Predicts the math score for a given student profile.

**Request Body (JSON):**
```json
{
  "gender": "female",
  "race_ethnicity": "group b",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "completed",
  "reading_score": 72,
  "writing_score": 74
}
```

**Response:**
```json
{
  "preds": 78.4
}
```

---

## 🧪 Interactive API Docs

FastAPI auto-generates interactive documentation:

| Interface | URL |
|---|---|
| Swagger UI | `http://localhost:8080/docs` |
| ReDoc | `http://localhost:8080/redoc` |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `scikit-learn` | ML algorithms & preprocessing |
| `xgboost` | Gradient boosting |
| `catboost` | Gradient boosting (categorical support) |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `pydantic` | Input validation & schema |
| `jinja2` | HTML templating |
| `dill` | Object serialization |

---

## 🗂️ Logging & Error Handling

- All pipeline steps are logged to the `logs/` directory with timestamps.
- A `CustomException` class wraps Python exceptions with detailed traceback info (file, line number) for easier debugging.

---

## 👤 Author

**Md Nazmul Huda Nabil**
📧 nabil648777@gmail.com
🐙 [GitHub](https://github.com/NazmulHudaNabil)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
