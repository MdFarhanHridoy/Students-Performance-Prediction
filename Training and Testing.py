# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bpaw6yKkAr7FH3h0CcgA_dCdpZnP8Wy0
"""

from google.colab import drive
drive.mount('/content/drive')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import math

data=pd.read_csv("/content/drive/MyDrive/StudentsPerformance.csv")

data.head()

data = pd.get_dummies(data, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'], drop_first=True)

from sklearn.model_selection import train_test_split
X = data.drop('math score', axis=1)
y = data['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
r2_tree = r2_score(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)

forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)


y_pred_forest = forest_model.predict(X_test)

r2_forest = r2_score(y_test, y_pred_forest)
mse_forest = mean_squared_error(y_test, y_pred_forest)

models = ['Linear Regression', 'Decision Tree', 'Random Forest']
r2_values = [r2_linear, r2_tree, r2_forest]
mse_values = [mse_linear, mse_tree, mse_forest]

x = range(len(models))

plt.figure(figsize=(14, 7))


plt.subplot(1, 2, 1)
plt.bar(x, r2_values, color='b', width=0.4)
plt.xlabel('Models')
plt.ylabel('R-squared')
plt.xticks(x, models)
plt.title('R-squared for Different Models')

plt.subplot(1, 2, 2)
plt.bar(x, mse_values, color='g', width=0.4)
plt.xlabel('Models')
plt.ylabel('MSE')
plt.xticks(x, models)
plt.title('MSE for Different Models')

plt.tight_layout()
plt.show()

r2_values, mse_values