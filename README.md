## Predict whether students pass or fail a math course based on demographics, academics, and behavior using the Student Performance dataset (395 records, 33 features).

## ✅ Solution
Transform the dataset and build machine learning models to classify student outcomes (Pass/Fail based on G3 ≥ 10).

## ⚙️ Approach

EDA: Cleaned data, visualized trends, analyzed feature correlations.
Preprocessing: One-hot encoding, z-score scaling, binary target transformation.
Feature Selection: Used RFECV to select top 6 features.
Models Used:
Logistic Regression (Best: 94% accuracy)
SVM (RBF) (91% accuracy)
Random Forest (88% accuracy)
Evaluation: Confusion matrix, McNemar’s test & Z-test — no significant performance difference.
