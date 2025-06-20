#Project Advance Data Analytics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import classification_report
from statsmodels.stats.proportion import proportions_ztest


# Read the Data
df = pd.read_csv('student-mat.csv', sep=';')

# check if there are any missing values
missing_values = df.isnull().sum()
total_missing_rows = df.isnull().any(axis=1).sum()
encoded_needed= df.select_dtypes(include=['object']).columns.tolist()

df_info = {
    "Data Shape": df.shape,
    "Missing Values per Column": missing_values,
    "Total Rows with Missing Values": total_missing_rows,
    "Columns Needing Encoding": encoded_needed
}

print(df.head(), df_info)


# create binary target variable for classification
df['pass'] = (df['G3'] >= 10).astype(int)
df['pass_label'] = df['pass'].map({0: 'Fail', 1: 'Pass'})

# Histograms, heatmaps, and boxplots to understand feature distribution and correlation with the target
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], bins=20, kde=True, color='skyblue')
plt.axvline(x=10, color='red', linestyle='--')
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("G3")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots for categorical analysis
for col in ['failures', 'schoolsup', 'higher', 'romantic']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=col, y='G3', data=df)
    plt.title(f"G3 by {col}")
    plt.tight_layout()
    plt.show()

# Countplots of pass/fail
for col in ['failures', 'schoolsup', 'higher', 'romantic']:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='pass_label', data=df)
    plt.title(f"Pass/Fail Distribution by {col}")
    plt.tight_layout()
    plt.show()

# Preprocess Features, Separate numeric and categorical variables, encoding
X_raw = df.drop(['G3', 'pass', 'pass_label'], axis=1)
y = df['pass']
numeric_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_raw.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_processed = preprocessor.fit_transform(X_raw)

# Feature Selection using RFECV
# Use recursive feature elimination with cross-validation to find best features
lr_selector = LogisticRegression(solver='liblinear')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
selector = RFECV(estimator=lr_selector, step=1, cv=cv, scoring='accuracy')
selector.fit(X_processed, y)
X_selected = selector.transform(X_processed)

feature_names = preprocessor.get_feature_names_out()
selected_features = feature_names[selector.support_]
print("Selected features by RFECV:", selected_features)
print("Number of features selected by RFECV:", selector.support_.sum())

# STrain/Test Split
# Split dataset for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2,
                                                    stratify=y, random_state=15)

# Test SVM kernels 
# Test difference C and kernel values to find the best SVM
C_values = [0.15, 0.5, 1, 2]
kernel_values = ['linear', 'rbf', 'sigmoid']
results_list = []

for C in C_values:
    for k in kernel_values:
        clf = SVC(kernel=k, C=C)
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
        temp = {'Kernel': k, 'C': C, 'Avg_Accuracy': scores.mean(), 'Std_Dev': scores.std()}
        results_list.append(temp)

results_df = pd.DataFrame(results_list)
print("== all results")
print(results_df)

# Train final SVM with kernel = 'Linear' and 'C = 1.0'
final_clf = SVC(kernel='linear', C=1.0)
run = final_clf.fit(X_train, y_train)
print("Number of Support Vectors for Each Class:", final_clf.n_support_)
print("Support Vectors Shape:", final_clf.support_vectors_.shape)

# Predict and Evaluate
# Make predictions and evaluate SVM performance using accuracy and confusion matrix
prediction = final_clf.predict(X_test)
accuracy = sum(y_test == prediction)/len(y_test)
results_matrix = pd.crosstab(y_test, prediction, rownames=['actual'], colnames=['prediction'])
print('prediction:', prediction)
print('accuracy:', accuracy)
print(results_matrix)

# Train Logistic Regression and Random Forest
# This is the baseline using same features to compare with SVM 
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=15)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"Accuracy - SVM: {accuracy:.4f}, Logistic Regression: {acc_lr:.4f}, Random Forest: {acc_rf:.4f}")

# STEP 9: McNemar and Z-Test for SVM vs Logistic Regression, SVM vs Random Forest
svm_correct = (prediction == y_test).to_numpy()
lr_correct = (y_pred_lr == y_test).to_numpy()
rf_correct = (y_pred_rf == y_test).to_numpy()

# SVM vs Logistic Regression
both_correct_lr = np.sum(np.logical_and(svm_correct == True, lr_correct == True))
both_wrong_lr = np.sum(np.logical_and(svm_correct == False, lr_correct == False))
svm_only_correct_lr = np.sum(np.logical_and(svm_correct == True, lr_correct == False))
lr_only_correct = np.sum(np.logical_and(svm_correct == False, lr_correct == True))

contingency_lr = pd.DataFrame(
    [[both_correct_lr, svm_only_correct_lr], [lr_only_correct, both_wrong_lr]],
    index=["Correct (LR)", "Incorrect (LR)"],
    columns=["Correct (SVM)", "Incorrect (SVM)"]
)
print("Contingency Table (SVM vs LR):", contingency_lr)
mc_result_lr = mcnemar(contingency_lr.to_numpy(), exact=False, correction=True)
print("McNemar Test (SVM vs LR):")
print("Statistic:", mc_result_lr.statistic)
print("P-value:", mc_result_lr.pvalue)

# SVM vs Random Forest
both_correct_rf = np.sum(np.logical_and(svm_correct == True, rf_correct == True))
both_wrong_rf = np.sum(np.logical_and(svm_correct == False, rf_correct == False))
svm_only_correct_rf = np.sum(np.logical_and(svm_correct == True, rf_correct == False))
rf_only_correct = np.sum(np.logical_and(svm_correct == False, rf_correct == True))

contingency_rf = pd.DataFrame(
    [[both_correct_rf, svm_only_correct_rf], [rf_only_correct, both_wrong_rf]],
    index=["Correct (RF)", "Incorrect (RF)"],
    columns=["Correct (SVM)", "Incorrect (SVM)"]
)
print("Contingency Table (SVM vs RF):", contingency_rf)
mc_result_rf = mcnemar(contingency_rf.to_numpy(), exact=False, correction=True)
print("McNemar Test (SVM vs RF):")
print("Statistic:", mc_result_rf.statistic)
print("P-value:", mc_result_rf.pvalue)

# Z-Test SVM vs LR and SVM vs RF
count_lr = [svm_correct.sum(), lr_correct.sum()]
nobs_lr = [len(svm_correct), len(lr_correct)]
z_stat_lr, pval_lr = proportions_ztest(count_lr, nobs_lr)
print("Z-statistic (SVM vs LR):", z_stat_lr)
print("p-value:", pval_lr)

count_rf = [svm_correct.sum(), rf_correct.sum()]
nobs_rf = [len(svm_correct), len(rf_correct)]
z_stat_rf, pval_rf = proportions_ztest(count_rf, nobs_rf)
print("Z-statistic (SVM vs RF):", z_stat_rf)
print("p-value:", pval_rf)
