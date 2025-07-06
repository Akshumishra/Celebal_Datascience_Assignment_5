# -*- coding: utf-8 -*-
"""
Cleaned House Price Prediction using Ridge and Lasso
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 1️⃣ Load training data
train_df = pd.read_csv('train.csv')
print(f"Train shape: {train_df.shape}")
print(train_df.head())

# 2️⃣ Visualize SalePrice distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_df['SalePrice'], kde=True, color='skyblue')
plt.title('SalePrice Distribution')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# 3️⃣ Correlation analysis
numeric_df = train_df.select_dtypes(include=[np.number])
corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
print("\nTop 10 positively correlated features:\n", corr.head(10))
print("\nTop 10 negatively correlated features:\n", corr.tail(10))

# 4️⃣ Handling missing values
# Fill categorical NaNs with 'None'
for col in train_df.select_dtypes(include='object').columns:
    train_df[col] = train_df[col].fillna('None')

# Fill numeric NaNs with median
for col in train_df.select_dtypes(include=['int64', 'float64']).columns:
    train_df[col] = train_df[col].fillna(train_df[col].median())

# 5️⃣ Feature Engineering
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
train_df['TotalBath'] = (train_df['FullBath'] + 0.5 * train_df['HalfBath'] +
                         train_df['BsmtFullBath'] + 0.5 * train_df['BsmtHalfBath'])
train_df['HouseAge'] = train_df['YrSold'] - train_df['YearBuilt']
train_df['RemodAge'] = train_df['YrSold'] - train_df['YearRemodAdd']
train_df['IsRemodeled'] = (train_df['YearBuilt'] != train_df['YearRemodAdd']).astype(int)

train_df['MSSubClass'] = train_df['MSSubClass'].astype(str)

# 6️⃣ One-hot encoding
train_df = pd.get_dummies(train_df)
print(f"Shape after get_dummies: {train_df.shape}")

# 7️⃣ Standardization
scaler = StandardScaler()
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])

# 8️⃣ Prepare training and validation data
X = train_df.drop(['Id', 'SalePrice'], axis=1)
y = train_df['SalePrice']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 9️⃣ Train Ridge and Lasso models
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
preds_ridge = ridge.predict(X_valid)

lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)
preds_lasso = lasso.predict(X_valid)

# 10️⃣ Evaluation
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Ridge RMSE: {rmse(y_valid, preds_ridge):.4f}")
print(f"Lasso RMSE: {rmse(y_valid, preds_lasso):.4f}")

# 11️⃣ Submission using Ridge on full training data
final_preds_log = ridge.predict(X)
final_preds = np.expm1(final_preds_log)

submission_train = pd.DataFrame({
    "Id": train_df["Id"],
    "SalePrice": final_preds
})
submission_train.to_csv("ridge_submission_train.csv", index=False)
print("Train prediction submission 'ridge_submission_train.csv' created successfully.")

# 12️⃣ Load and prepare test data
test_df = pd.read_csv('test.csv')
print(f"Test shape: {test_df.shape}")
print(test_df.head())

# Handle missing values in test data
for col in test_df.select_dtypes(include='object').columns:
    test_df[col] = test_df[col].fillna('None')
for col in test_df.select_dtypes(include=['int64', 'float64']).columns:
    test_df[col] = test_df[col].fillna(test_df[col].median())

# Feature Engineering on test data
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
test_df['TotalBath'] = (test_df['FullBath'] + 0.5 * test_df['HalfBath'] +
                        test_df['BsmtFullBath'] + 0.5 * test_df['BsmtHalfBath'])
test_df['HouseAge'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['RemodAge'] = test_df['YrSold'] - test_df['YearRemodAdd']
test_df['IsRemodeled'] = (test_df['YearBuilt'] != test_df['YearRemodAdd']).astype(int)
test_df['MSSubClass'] = test_df['MSSubClass'].astype(str)

# One-hot encoding
test_df = pd.get_dummies(test_df)

# Align test and train data
X_aligned, test_aligned = X.align(test_df, join='left', axis=1, fill_value=0)

# Scale test data
test_scaled = scaler.transform(test_aligned)

# Predict using Ridge
test_preds_log = ridge.predict(test_scaled)
test_preds = np.expm1(test_preds_log)

# Create test submission
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_preds
})
submission.to_csv("ridge_submission_test.csv", index=False)
print("Test submission 'ridge_submission_test.csv' created successfully.")
