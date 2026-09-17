# Credit Card Fraud Detection using XGBoost and SMOTE

## 📌 Project Overview

Credit card fraud detection is a classification problem where the main challenge is that fraudulent transactions are much fewer than normal transactions.

In this project, **XGBoost** is used to detect fraudulent transactions. Since the dataset is highly imbalanced, **SMOTE (Synthetic Minority Oversampling Technique)** is applied to the training data to improve the detection of the minority fraud class.

The model's **decision threshold is also tuned** to find a suitable balance between precision and recall. Finally, **feature importance** is used to understand which features contribute most to the model's predictions.

---

## 🎯 Objectives

* Detect fraudulent credit card transactions.
* Handle class imbalance using **SMOTE**.
* Train an **XGBoost classification model**.
* Tune the model's **decision threshold**.
* Evaluate the model using:

  * Precision
  * Recall
  * F1 Score
  * ROC-AUC
  * Confusion Matrix
* Identify important features using **XGBoost feature importance**.

---

## 📊 Dataset

The project uses the **IEEE-CIS Fraud Detection dataset** from Kaggle.

The dataset contains transaction-related information and the target column:

```text
isFraud
```

where:

* `0` → Normal transaction
* `1` → Fraudulent transaction

The dataset is highly imbalanced, with significantly fewer fraudulent transactions compared to normal transactions.

> **Note:** The dataset files are not included in this repository because of their large size. They can be downloaded separately from Kaggle.

---

## 🛠️ Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Matplotlib

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Check Missing Values
   ↓
Handle Missing Values
   ↓
Select Numerical Features
   ↓
Train-Test Split
   ↓
SMOTE on Training Data
   ↓
Train XGBoost Model
   ↓
Predict Fraud Probabilities
   ↓
Tune Decision Threshold
   ↓
Model Evaluation
   ↓
Feature Importance
```

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the transaction dataset.
2. Checked for missing/null values.
3. Selected numerical features.
4. Replaced missing numerical values with their median.
5. Divided the dataset into training and testing sets.
6. Used `stratify=y` during splitting to preserve the class distribution.

---

## ⚖️ Handling Class Imbalance with SMOTE

Because fraudulent transactions represent a small portion of the dataset, directly training the model can make it biased toward the majority class.

**SMOTE** was applied only to the training data.

It generates synthetic examples of the minority class instead of simply duplicating existing fraud transactions.

```python
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
```

The test set was not oversampled so that it continues to represent unseen data.

---

## 🤖 XGBoost Model

XGBoost (Extreme Gradient Boosting) was used as the classification algorithm.

```python
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_smote, y_train_smote)
```

The model produces a probability for each transaction indicating how likely it is to be fraudulent.

---

## 🎚️ Decision Threshold Tuning

Instead of always using the default threshold of `0.5`, different thresholds were tested.

For example:

```text
0.3
0.4
0.5
0.6
0.7
```

A lower threshold can classify more transactions as fraud, which may increase **recall** but can also increase false positives.

The threshold giving the highest F1 score was selected for the final evaluation.

```python
y_pred = (y_prob >= best_threshold).astype(int)
```

---

## 📈 Evaluation Metrics

The model is evaluated using:

### Precision

Measures how many transactions predicted as fraud were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures how well the model separates fraudulent and normal transactions across different thresholds.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

For fraud detection, reducing **false negatives** is particularly important because they represent fraudulent transactions that were missed.

---

## 🔍 Feature Importance

XGBoost provides feature importance scores that can be used to identify the features that contributed most to the model's predictions.

The project visualizes the **Top 15 important features** using a bar chart.

```python
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
})
```

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── README.md
└── .gitignore
```

The dataset files are intentionally not included in the repository.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-link>
```

### 2. Open the notebook

Open:

```text
Credit_Card_Fraud_Detection.ipynb
```

using **Google Colab** or Jupyter Notebook.

### 3. Download the dataset

Download the IEEE-CIS Fraud Detection dataset from Kaggle and upload/extract it in the Colab environment.

### 4. Run the notebook

Run the cells in order to perform:

* Data preprocessing
* SMOTE
* XGBoost training
* Threshold tuning
* Model evaluation
* Feature importance analysis

---

## 📌 Key Takeaways

* Credit card fraud detection is a highly **imbalanced classification problem**.
* SMOTE helps address the imbalance during model training.
* XGBoost can capture complex relationships between transaction features.
* Decision threshold tuning allows the model to be adjusted based on the desired precision-recall trade-off.
* Feature importance provides an interpretable view of which features are most useful to the model.

---

## 👩‍💻 Author

**Your Name**

This project was developed as part of a machine learning case study on credit card fraud detection.

