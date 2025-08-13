# Titanic Survival Prediction - Final Report

## 📌 Project Overview
This notebook predicts whether a passenger survived the Titanic disaster based on historical data.  
It uses supervised machine learning (Logistic Regression) to classify survival outcomes.

---

## **1. Understanding the Problem**

**Problem Statement:**  
Predict passenger survival using the dataset `train.csv` and test the model on `test.csv`.

**Type:**  
- Supervised Learning  
- Classification problem (binary output: 0 or 1)

**Target Column:**  
`Survived` — 0 = dead, 1 = alive

**Features (Variables):**  
Passenger ID, Age, Gender, Fare, etc.

---

## **2. Loading & Exploring Data**

Steps:  
1. Loaded data using `pandas.read_csv()`  
2. Viewed dataset information and structure using:
    ```python
    data.info()
    data.describe()
    data.head()
    data.isnull().sum()
    ```

Purpose: To understand data types, missing values, and overall structure.

---

## **3. Data Cleaning & Preprocessing**

Tasks performed:  
- Dropped `Cabin` (too many missing values)  
- Converted categorical values (`male` → 1, `female` → 0)  
- Filled or removed missing values  
- Created a reusable data cleaning function  

Outcome: Dataset was clean, numeric, and ready for model training.

---

## **4. Model Training & Evaluation**

Model used: **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)  

Why Logistic Regression?  
- Effective for binary classification problems  
- Easy to implement and interpret  

Training Example:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

**Accuracy Achieved:** ~81% on validation data

---

## **5. Testing with Unseen Data**

- Used the trained model to predict survival on `test.csv`  
- Generated predictions for submission

---

## **6. Kaggle Submission Results**

- Submitted predictions to Kaggle  
- **Public Leaderboard Score:** 0.76

---

## **✅ Conclusion & Learnings**

- Learned complete ML workflow: loading, exploring, cleaning, training, testing, and submission  
- Reinforced importance of data preprocessing for better accuracy  
- Built confidence to move towards more complex ML problems

**Next Step:** Apply the same structured workflow to other datasets and improve feature engineering techniques.
