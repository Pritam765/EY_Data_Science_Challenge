# EY Data Science Challenge — Classification Dashboard

## 1. Project Overview

This project develops an end-to-end machine learning classification solution using a structured dataset containing seven input features (`Col1` to `Col7`) and one target variable (`ClassificationLabel`).

The objective is to explore and preprocess the data, train and evaluate multiple classification models, select a suitable model, and deploy the prediction pipeline through an interactive Streamlit dashboard.

The dashboard is designed to help users understand the dataset, review model performance, and generate predictions for new records uploaded through a CSV file.

## 2. Project Objectives

- Perform exploratory data analysis (EDA) to understand the dataset and target distribution.
- Identify missing values, duplicate records, inconsistent labels, and data-quality issues.
- Build a reusable preprocessing pipeline for text, numerical, date, and categorical features.
- Compare multiple supervised classification algorithms.
- Evaluate models using accuracy, balanced accuracy, and macro F1-score.
- Deploy the selected model through an interactive Streamlit application.
- Support batch predictions for uploaded CSV files and downloadable prediction results.

## 3. Dataset Description

The dataset contains **5,899 records**, seven input features, and one classification target.

| Component | Description |
|---|---|
| Input features | `Col1` to `Col7` |
| Target variable | `ClassificationLabel` |
| Feature types | Text, numerical, date, and categorical |
| Prediction task | Multiclass classification |
| Original records | 5,899 |

The target contains six classes: `Category_1` through `Category_6`. The class distribution is highly imbalanced, with Category_1 representing the majority of records.

The original challenge dataset is not included in this repository. Place the authorized dataset at `data/raw/Challenge_Data.csv` before running the application.

## 4. Exploratory Data Analysis and Preprocessing

The EDA and preprocessing notebooks cover the following steps:

- Inspect dataset dimensions, column types, and sample records.
- Analyze missing values and duplicate records.
- Standardize inconsistent target-label spellings.
- Convert numerical values and parse dates.
- Extract year and month from date information.
- Engineer structural features from `Col2`, including string length, letter presence, and dash presence.
- Create a missing-value indicator for `Col4`.
- Represent text features using TF-IDF.
- Encode categorical variables using One-Hot Encoding.
- Impute missing values and scale numerical features where appropriate.
- Identify conflicting labels for identical input combinations and exclude those conflicting records from the modeling dataset.

A group-aware train/holdout split is used to reduce the risk of identical input combinations appearing in both sets.

## 5. Model Development and Evaluation

Multiple classification approaches were evaluated:

- Logistic Regression baseline
- Tuned Logistic Regression with class weighting
- Linear Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Logistic Regression with controlled undersampling

### Model comparison

The following results were obtained during the current experimentation:

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Tuned Logistic Regression | 0.9541 | 0.8744 | 0.7439 |
| Random Forest | 0.9492 | 0.8263 | 0.6973 |
| Linear SVM | 0.9557 | 0.7202 | 0.6460 |
| XGBoost | 0.9344 | 0.7729 | 0.6418 |
| Controlled Undersampling | 0.9393 | 0.7218 | 0.6361 |

### Selected model

**Tuned Logistic Regression** was selected as the current candidate because it achieved the highest balanced accuracy and macro F1-score among the evaluated models.

Accuracy alone is not sufficient for this problem because the target classes are highly imbalanced. Balanced accuracy and macro F1 provide additional insight into performance across classes.

**Evaluation limitation:** Some minority classes have very few examples in the holdout set, and Category_5 is absent from the current holdout. The results should therefore be treated as exploratory rather than definitive evidence of performance for every class.

## 6. Streamlit Dashboard

The application is built with Streamlit and provides two main tabs.

### Tab 1: EDA & Model Results

- Dataset overview and summary metrics
- Target-class distribution
- Data-quality information
- Model comparison table

### Tab 2: Prediction & Explain

- Upload a CSV containing the required input features.
- Preview uploaded records.
- Generate batch predictions using the saved machine learning pipeline.
- Review predictions in pages of approximately 25 records.
- Download the prediction results as a CSV file.

The model pipeline includes feature preparation, preprocessing, and classification.

## 7. Project Structure

```text
EY_Data_Science_Challenge/
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Preprocessing.ipynb
│
├── src/
│   ├── __init__.py
│   └── feature_preparation.py
│
├── data/
│   ├── raw/          # Local input dataset; not committed
│   └── processed/    # Generated files; not committed
│
├── models/           # Saved model pipeline; not committed
├── .gitignore
├── requirements.txt
└── README.md
```

## 8. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/Pritam765/EY_Data_Science_Challenge.git
cd EY_Data_Science_Challenge
```

### Step 2: Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Add the authorized dataset

Place the challenge dataset at:

```text
data/raw/Challenge_Data.csv
```

The dataset is excluded from version control and must be obtained separately.

### Step 5: Prepare the saved model

Ensure the trained model pipeline exists at:

```text
models/ey_classification_pipeline.joblib
```

If the model is not available, run the relevant preprocessing and model-training notebook cells to generate it. The dashboard's prediction workflow requires this file.

### Step 6: Run the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Streamlit will provide a local URL that you can open in your browser.

## 9. Input Format for Predictions

The uploaded CSV must contain the following seven columns:

```text
Col1, Col2, Col3, Col4, Col5, Col6, Col7
```

The target column `ClassificationLabel` is not required for prediction. Input values should follow the expected formats used by the preprocessing pipeline.

## 10. Limitations and Future Improvements

- Improve performance on underrepresented target classes by collecting more representative examples where possible.
- Evaluate the final model on a fresh, untouched holdout set.
- Add class-wise precision, recall, F1-scores, and a confusion matrix to the dashboard.
- Add row-level prediction explanations using model-based feature information or a grounded AI explanation workflow.
- Improve input validation and error handling for uploaded files.
- Add automated tests for preprocessing, model loading, and prediction.
- Improve dashboard usability with filters, clearer visualizations, and additional evaluation summaries.

## 11. Technologies Used

- **Programming:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Text processing:** TF-IDF
- **Visualization:** Matplotlib, Seaborn, Streamlit
- **Model persistence:** Joblib
- **Development:** Jupyter Notebook, VS Code, Git, GitHub

## 12. Author

**Pritam Kumar Tripathy**

GitHub: [Pritam765](https://github.com/Pritam765)

---

*This project demonstrates an end-to-end classification workflow, from data exploration and preprocessing to model evaluation and interactive deployment.*