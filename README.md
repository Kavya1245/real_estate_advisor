# 🏠 Real Estate Investment Advisor
### Predicting Property Profitability & Future Value

> An end-to-end Machine Learning project built on **2,50,000 Indian housing records** — predicts whether a property is a good investment and forecasts its price 5 years into the future.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Demo & Screenshots](#-demo--screenshots)
3. [Features](#-features)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Dataset](#-dataset)
7. [ML Models](#-ml-models)
8. [Installation & Setup](#-installation--setup)
9. [How to Run](#-how-to-run)
10. [App Pages](#-app-pages)
11. [MLflow Tracking](#-mlflow-tracking)
12. [Results](#-results)
13. [Author](#-author)

---

## 📖 Project Overview

The **Real Estate Investment Advisor** is a full-stack data science project that helps investors make smarter, data-backed decisions when buying property in India.

It solves two core ML problems:

| Task | Type | Target | Best Model |
|------|------|--------|-----------|
| Is this a good investment? | Classification | `Good_Investment` (0 / 1) | XGBoost |
| What will it be worth in 5 years? | Regression | `Future_Price_5Y` (₹ Lakhs) | Linear Regression |

---

## ✨ Features

### 🤖 Machine Learning
- **5 Classification models** compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, LinearSVC
- **5 Regression models** compared: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- Automated best-model selection based on accuracy / R²
- All runs tracked and logged with **MLflow**

### 📊 Streamlit Web App (7 Pages)
- **🏠 Home** — Project overview, dataset stats, objectives
- **🗂️ View & Filter Data** — Browse full raw dataset with 20+ dropdown filters
- **🔍 Predict Investment** — AI-powered prediction form with instant results
- **📊 Data Insights** — Interactive charts with multi-filter support
- **🔬 EDA Visualizations** — All 20 EDA questions with code + chart per question
- **🤖 Model Performance** — Side-by-side comparison of all 10 models
- **👩‍💻 About Creator** — Creator bio, skills, contact

### 🗂️ Data Filtering
- Show full raw table first, then filter
- Categorical filters: dropdown per column (State, City, Property Type, etc.)
- Numeric filters: Min ≥ / Max ≤ dropdown buckets for every numeric column
- Binary filters: Yes / No dropdowns (Parking, Security)
- Sort by any column · Choose rows to display
- Export filtered results as **CSV** or **Excel**

---

## 🛠 Tech Stack

| Category | Libraries / Tools |
|----------|------------------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow |
| **Web App** | Streamlit |
| **Model Persistence** | Joblib |
| **Export** | openpyxl |
| **IDE** | Jupyter Notebook, VS Code |

---

## 📁 Project Structure

```
real_estate_advisor/
│
├── data/
│   ├── india_housing_prices.csv    ← Raw dataset (2,50,000 rows)
│   └── cleaned_data.csv            ← Preprocessed dataset
│
├── models/
│   ├── best_classifier.pkl         ← XGBoost classifier
│   ├── best_regressor.pkl          ← Linear Regression model
│   ├── label_encoders.pkl          ← LabelEncoders for categorical columns
│   ├── scaler.pkl                  ← MinMaxScaler
│   ├── feature_names.pkl           ← List of 25 feature names
│   └── model_info.pkl              ← All model metrics & results
│
├── notebooks/
│   ├── preprocess.ipynb            ← Step 1: Data Preprocessing
│   ├── eda.ipynb                   ← Step 2: Exploratory Data Analysis
│   └── train.ipynb                 ← Step 3: Model Training & MLflow
│
├── mlruns/                         ← MLflow experiment tracking data
│
├── app.py                          ← Streamlit application (main entry point)
├── kavya_photo.jpeg                ← Creator profile photo
└── requirements.txt                ← Python dependencies
```

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **File** | `india_housing_prices.csv` |
| **Rows** | 2,50,000 |
| **Columns** | 23 original + 6 engineered = 29 total |
| **States** | 20 Indian states |
| **Cities** | 42 cities across India |
| **Property Types** | Apartment, Independent House, Villa |

### Original Columns

```
ID, State, City, Locality, Property_Type, BHK, Size_in_SqFt,
Price_in_Lakhs, Price_per_SqFt, Year_Built, Furnished_Status,
Floor_No, Total_Floors, Age_of_Property, Nearby_Schools,
Nearby_Hospitals, Public_Transport_Accessibility, Parking_Space,
Security, Amenities, Facing, Owner_Type, Availability_Status
```

### Engineered Features

| Feature | Description |
|---------|-------------|
| `Amenity_Count` | Number of amenities parsed from string |
| `Price_per_SqFt` | `Price_in_Lakhs / Size_in_SqFt` (refreshed) |
| `Floor_Ratio` | `Floor_No / (Total_Floors + 1)` |
| `Amenity_Density_Score` | `Amenity_Count / (Size_in_SqFt / 1000)` |
| `Is_Ready_to_Move` | 1 if `Availability_Status == Ready_to_Move` |
| `Has_Parking` | Binary encoding of `Parking_Space` |
| `Has_Security` | Binary encoding of `Security` |

### Target Variables

| Target | Formula / Logic |
|--------|----------------|
| `Future_Price_5Y` | `Price_in_Lakhs × (1.08)^5` — assumes 8% annual appreciation |
| `Good_Investment` | 1 if investment score ≥ 3 out of 5 factors (below-median price, below-median price/sqft, BHK≥3, ready to move, has parking) |

---

## 🤖 ML Models

### Classification Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9116 | 0.9117 | 0.9116 | 0.9169 | 0.9764 |
| Decision Tree | 0.9949 | 0.9950 | 0.9949 | 0.9952 | 0.9975 |
| Random Forest | 0.9959 | 0.9960 | 0.9959 | 0.9962 | 0.9999 |
| ⭐ **XGBoost** | **0.9981** | **0.9982** | **0.9981** | **0.9982** | **1.0000** |
| LinearSVC | 0.9117 | 0.9118 | 0.9117 | 0.9171 | 0.9764 |

### Regression Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| ⭐ **Linear Regression** | **0.0000** | **0.0000** | **1.0000** |
| Ridge | 0.0283 | 0.0219 | 1.0000 |
| Lasso | 0.3514 | 0.3034 | 1.0000 |
| Random Forest | 0.0038 | 0.0024 | 1.0000 |
| XGBoost | 0.8256 | 0.7094 | 1.0000 |

> **Note:** R² = 1.0 for all regressors because `Future_Price_5Y` is a deterministic function of `Price_in_Lakhs`. This is expected and by design.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or above
- pip
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Kavya1245/real-estate-investment-advisor.git
cd real-estate-investment-advisor
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
mlflow
streamlit
joblib
openpyxl
pillow
```

---

## ▶️ How to Run

### Step 1 — Run Preprocessing

```bash
jupyter notebook notebooks/preprocess.ipynb
```
> Cleans data, engineers features, creates targets, saves `cleaned_data.csv` and model artifacts.

### Step 2 — Run EDA (Optional)

```bash
jupyter notebook notebooks/eda.ipynb
```
> Generates 20 exploratory visualizations across 4 analysis groups.

### Step 3 — Train Models & Log to MLflow

```bash
jupyter notebook notebooks/train.ipynb
```
> Trains 10 models, evaluates all, saves best models, logs everything to MLflow.

### Step 4 — Launch MLflow UI (Optional)

Open a **separate terminal**:

```bash
mlflow ui
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Step 5 — Launch Streamlit App

```bash
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 📱 App Pages

### 🏠 Home
- Project overview and purpose
- Key metrics: 2,50,000 rows · 99.81% accuracy · R²=1.0 · 10 models
- Dataset info table and step-by-step usage guide

### 🗂️ View & Filter Data
- Full raw `india_housing_prices.csv` table displayed first
- 20+ dropdown filters covering every column
- Categorical dropdowns: State, City, Locality, Property Type, Furnished Status, Facing, Owner Type, Availability, Public Transport, BHK, Parking, Security
- Numeric range dropdowns: Price, Size, Price/SqFt, Year Built, Floor No, Total Floors, Age, Schools, Hospitals
- Sort · Rows per page · Export as CSV or Excel

### 🔍 Predict Investment
- Input form for all 25 features
- Auto-calculates derived features (Price/SqFt, Age, Floor Ratio, etc.)
- Outputs:
  - ✅ Good Investment / ❌ Not Good Investment verdict with confidence %
  - 📈 Predicted price after 5 years with growth % 
  - Confidence bar chart
  - Year-by-year price growth line chart

### 📊 Data Insights
- 4 filters: City, Property Type, BHK, Price Range
- 6 interactive charts: price histogram, avg price by city, property type pie, BHK boxplot, good investment % by city, correlation heatmap

### 🔬 EDA Visualizations
- **Dropdown 1:** Select group (Q1–5, Q6–10, Q11–15, Q16–20)
- **Dropdown 2:** Select individual question
- Shows full Python code block for that chart
- Renders the live visualization below the code

### 🤖 Model Performance
- Best model banners (Classifier + Regressor)
- Full metrics tables for all 10 models
- Grouped bar charts comparing all classifiers
- Individual bar charts for RMSE, MAE, R²

### 👩‍💻 About Creator
- Profile photo, bio, education details
- Technical skills badges
- Contact links: Email, LinkedIn, GitHub

---

## 📊 MLflow Tracking

All 10 training runs are logged to MLflow with:

| Logged Item | Details |
|-------------|---------|
| **Parameters** | `model_name`, `task` (classification / regression) |
| **Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC (CLF) · RMSE, MAE, R² (REG) |
| **Artifacts** | Serialized model `.pkl` files |

**To open MLflow UI:**

```bash
# In a separate terminal (with venv activated)
mlflow ui
# → http://127.0.0.1:5000
```

**Runs logged:**

```
Real_Estate_Investment_Advisor/
├── CLF_Logistic_Regression
├── CLF_Decision_Tree
├── CLF_Random_Forest
├── CLF_XGBoost           ← Best Classifier
├── CLF_LinearSVC
├── REG_Linear_Regression  ← Best Regressor
├── REG_Ridge
├── REG_Lasso
├── REG_Random_Forest
└── REG_XGBoost_Reg
```

---

## 📈 Results Summary

```
✅ Best Classifier  :  XGBoost
   Accuracy         :  99.81%
   F1 Score         :  0.9982
   ROC-AUC          :  1.0000

✅ Best Regressor   :  Linear Regression
   RMSE             :  0.0000
   MAE              :  0.0000
   R²               :  1.0000

📊 Dataset          :  2,50,000 rows × 23 columns
🏙️ Coverage         :  42 cities · 20 states · India
🔧 Features used    :  25 (19 original + 6 engineered)
🤖 Models trained   :  10 (5 CLF + 5 REG)
```

---

## 👩‍💻 Author

**KAVYA S**

B.E Biomedical Engineering · Minor in Artificial Intelligence & Data Science

| Platform | Link |
|----------|------|
| 📧 Email | kavya22s145@gmail.com |
| 💼 LinkedIn | [linkedin.com/in/kavya-s1245](https://www.linkedin.com/in/kavya-s1245/) |
| 🐙 GitHub | [github.com/Kavya1245](https://github.com/Kavya1245) |

---

## 📜 License

This project is created for academic and learning purposes.

---

