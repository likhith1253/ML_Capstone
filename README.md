# ML Capstone

This repository delivers the Machine Learning Capstone project for course 23CSE301, covering end-to-end Machine Learning pipelines spanning Regression, Classification, and Clustering tracks. The team developed, preprocessed, engineered domain features, and evaluated machine learning models across two real-world datasets: the RT-IoT2022 Network Traffic Dataset for continuous network flow duration prediction and the UCI Dry Bean Dataset for multiclass seed cultivar classification. Every pipeline enforces strict data leakage prevention, cross-validated hyperparameter optimization, and isolated held-out test set benchmark evaluations.

## Project Structure

```
ML_Capstone/
├── classification/
│   ├── classification_preprocessing.ipynb
│   └── classification.ipynb
├── data/
│   ├── Dry_Bean_Dataset.csv
│   ├── RT_IOT2022
│   └── RT_IOT2022_cleaned.csv
├── regression/
│   ├── regression_preprocessing.ipynb
│   ├── regression.ipynb
│   └── regression_split_indices.json
├── README.md
└── requirements.txt
```

## Datasets

* **Regression Dataset**: `RT_IOT2022_cleaned.csv` (RT-IoT2022 network traffic dataset), containing 117,922 cleaned records and 83 network attributes derived from real-world IoT device traffic.
* **Classification Dataset**: `Dry_Bean_Dataset.csv` (UCI Dry Bean Dataset), containing 13,543 cleaned records and 17 morphological features extracted from high-resolution seed images across 7 dry bean varieties.

## Regression Track

The regression track models continuous network flow duration (`flow_duration` in seconds) using the RT-IoT2022 dataset. Raw network traffic consisting of 123,117 rows and 85 columns was cleaned to 117,922 rows. To eliminate data leakage, 30 temporal, inter-arrival time (IAT), active/idle duration, and packet rate features were removed. One engineered feature, `total_packets` (`fwd_pkts_tot + bwd_pkts_tot`), was introduced. The dataset was partitioned into an 80% training set (94,337 rows) and a 20% held-out test set (23,585 rows), with categorical variables (`proto`, `service`) one-hot encoded to 64 total feature predictors.

The Decision Tree Regressor achieved the best overall performance on the held-out test set, reaching an **$R^2$ of 0.6210**, **RMSE of 36.12 s**, and **MAE of 1.32 s**.

### Implemented Regression Algorithms & Results

| Model | Test $R^2$ | Test RMSE (s) | Test MAE (s) | Fit Time (s) | Status |
|---|---|---|---|---|---|
| Decision Tree | 0.6210 | 36.12 | 1.32 | 1.82 | Baseline |
| Random Forest (Tuned) | 0.3740 | 46.42 | 1.47 | 22.82 | Tuned |
| KNN ($k=5$, scaled) | -0.0176 | 59.18 | 1.60 | 0.13 | Baseline |
| Gradient Boosting (Tuned) | -0.1466 | 62.82 | 1.98 | 23.43 | Tuned |
| ElasticNet | -2.0803 | 102.97 | 3.25 | 0.65 | Baseline |
| Ridge (Tuned) | -4.3380 | 135.55 | 4.66 | 0.31 | Tuned |
| Linear Regression (OLS) | -4.4362 | 136.79 | 4.66 | 0.46 | Baseline |
| SVR (linear) | -6.5904 | 161.64 | 64.72 | 13.10 | Baseline |
| Lasso ($\alpha=1.0$) | -63.8454 | 472.45 | 7.04 | 6.43 | Baseline |
| Polynomial Regression (deg=2) | -3020.8537 | 3225.16 | 23.54 | 0.12 | Baseline |

## Classification Track

The classification track categorizes 7 registered dry bean cultivars using high-resolution image morphometry from `Dry_Bean_Dataset.csv`. Starting with 13,611 raw records, data cleaning removed 68 exact duplicate rows (0.50%), retaining 13,543 cleaned samples across 17 features. Biological outlier analysis preserved large-seeded varieties (such as `BOMBAY`). An engineered geometric descriptor, `Area_Perimeter_Ratio` (`Area / Perimeter`), was incorporated. A stratified 80/20 train/test split yielded 10,834 training samples and 2,709 held-out test samples, with `StandardScaler` fitted strictly on `X_train`.

### Final Class Breakdown (13,543 Cleaned Records)

* **DERMASON**: 3,546 samples (26.18%)
* **SIRA**: 2,636 samples (19.46%)
* **SEKER**: 2,027 samples (14.97%)
* **HOROZ**: 1,860 samples (13.73%)
* **CALI**: 1,630 samples (12.04%)
* **BARBUNYA**: 1,322 samples (9.76%)
* **BOMBAY**: 522 samples (3.85%)

The Support Vector Machine (SVC) with an RBF kernel ($C=10$) delivered the highest accuracy on the held-out test set, attaining **92.36% Test Accuracy**, **0.9236 Weighted F1**, **0.9345 Macro F1**, and **0.9931 Multiclass ROC-AUC**.

### Implemented Classification Algorithms & Results

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | ROC-AUC | Training CV | Best Parameters |
|---|---|---|---|---|---|---|---|---|
| Support Vector Machine (SVC) | 0.9236 | 0.9237 | 0.9236 | 0.9236 | 0.9345 | 0.9931 | 0.9322 | `{'C': 10, 'kernel': 'rbf'}` |
| Logistic Regression | 0.9206 | 0.9213 | 0.9206 | 0.9208 | 0.9311 | 0.9936 | 0.9244 | `{'C': 10.0}` |
| Random Forest Classifier | 0.9177 | 0.9177 | 0.9177 | 0.9176 | 0.9296 | 0.9921 | 0.9234 | `{'max_depth': 15, 'n_estimators': 100}` |
| K-Nearest Neighbors (KNN) | 0.9169 | 0.9177 | 0.9169 | 0.9171 | 0.9293 | 0.9880 | 0.9232 | `{'n_neighbors': 13}` |
| Decision Tree Classifier | 0.9051 | 0.9051 | 0.9051 | 0.9049 | 0.9187 | 0.9629 | 0.9055 | `{'max_depth': 12, 'min_samples_split': 10}` |

## Clustering Track

The unsupervised clustering track is planned for the next milestone (Review 2).

## Setup & How to Run

Follow these steps to set up the environment and execute the project notebooks.

### 1. Clone the Repository

```bash
git clone https://github.com/likhith1253/ML_Capstone.git
cd ML_Capstone
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Execution Order

Run the notebooks in the following order to ensure data preprocessing pipelines execute before model training and evaluation:

1. **Regression Track**:
   * Run `regression/regression_preprocessing.ipynb` to clean data, engineer `total_packets`, exclude target-leaking features, and generate `regression_split_indices.json`.
   * Run `regression/regression.ipynb` to load stored split indices, execute baseline model training, run hyperparameter grid searches, and compute test benchmarks across all 10 algorithms.
2. **Classification Track**:
   * Run `classification/classification_preprocessing.ipynb` for exploratory data analysis, class-grouped quantile inspection, duplicate removal, and scaling validation.
   * Run `classification/classification.ipynb` to load `Dry_Bean_Dataset.csv`, execute 80/20 stratified splitting, run cross-validated hyperparameter grid searches, and generate confusion matrices and multiclass ROC curves.

## Requirements

Project dependencies are specified in `requirements.txt`. Key packages include:

* `numpy` (>= 1.24.0)
* `pandas` (>= 2.0.0)
* `scikit-learn` (>= 1.2.0)
* `matplotlib` (>= 3.7.0)
* `seaborn` (>= 0.12.0)
* `ipympl` (>= 0.9.0)
* `ipython` (>= 8.0.0)

## Team

* **Likhith Ponnada**
* **Prudhvi Manvith**
* **Semanth Parvataneni**

## Highlights

* **Leakage-Free Feature Selection**: Removed 30 temporal, inter-arrival time (`*iat*`), active/idle duration, and packet rate features from the RT-IoT2022 dataset to prevent data leakage prior to continuous duration modeling.
* **Continuous Quantile Stratification**: Implemented quantile target binning (`pd.qcut`) on `y_train` to enable 5-fold `StratifiedKFold` cross-validation for an extraordinarily right-skewed regression target ($121.03$ skewness).
* **Biological Cultivar Preservation**: Retained authentic large-seeded bean cultivars (such as `BOMBAY`) rather than truncating legitimate specimens with global IQR filters, combined with isolated `StandardScaler` fitting on training splits.
* **Domain Feature Engineering**: Formulated `total_packets` (`fwd_pkts_tot + bwd_pkts_tot`) for IoT flow duration and `Area_Perimeter_Ratio` (`Area / Perimeter`) for seed morphometry to enhance predictive feature representation.

## Acknowledgements

AI coding assistance (Antigravity / LLM tools) was utilized during development for code structure optimization, plotting routines, and formatting notebook outputs.
