# Titanic Survival Prediction

## Project Overview
This project predicts whether a passenger survived the Titanic disaster based on historical data. It uses machine learning to classify passengers as `Survived` (1) or `Not Survived` (0) based on given features.

---

## 1. Understanding the Problem

**Problem Statement:**  
Determine whether a passenger survived based on the data in `train.csv`, and test the model on `test.csv`.

**Type:**  
Supervised Learning — Classification problem (binary output: 0 or 1).

**Target Column:**  
`Survived` (0 = dead, 1 = alive)

**Features (Variables):**  
Passenger ID, Age, Gender, Fare, etc.

---

## 2. Loading & Exploring the Data

Loaded the dataset into Python using Pandas:
```python
import pandas as pd
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
```

Explored with:
```python
train_data.info()
train_data.describe()
train_data.head()
train_data.isnull().sum()
```

---

## 3. Data Cleaning & Preprocessing

- Dropped columns with excessive missing values (e.g., `Cabin`).
- Converted categorical data to numeric (`male` = 1, `female` = 0).
- Removed or imputed missing values.
- Created a separate function for cleaning to ensure reusability.

---

## 4. Model Training & Evaluation

Model Used: **Logistic Regression** (from `sklearn`).

Why Logistic Regression?  
It is effective for binary classification problems and provides good interpretability.

Training Example:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

**Final Accuracy:** 81% on training/validation data.

---

## 5. Testing with Unseen Data

The model was tested on `test.csv` to predict survival outcomes.

---

## 6. Kaggle Submission

Predictions were submitted to Kaggle: **Public Score: 0.76**.

---

## Conclusion

This project was an exciting opportunity to apply data cleaning, preprocessing, and model training from scratch. Although I have done similar projects before, doing it independently was a valuable learning experience.

**Next Steps:** Move on to more complex machine learning problems.
