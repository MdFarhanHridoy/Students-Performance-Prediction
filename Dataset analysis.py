#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("StudentsPerformance.csv")


# In[3]:


data.head


# In[4]:


data.info()


# In[10]:


df = pd.read_csv('StudentsPerformance.csv')
feature_names = df.columns.tolist()
feature_names


# In[11]:


df.shape


# In[14]:


num_data_points = len(data)
print(f'The dataset contains {num_data_points} data points.')


# In[16]:


quantitative_features = data.select_dtypes(include=['int64', 'float64'])
print("Quantitative Features:")
print(quantitative_features)

categorical_features = data.select_dtypes(include=['object'])
print("\nCategorical Features:")
print(categorical_features)


# In[19]:


# Assuming you've already imported the dataset
data = pd.read_csv("StudentsPerformance.csv")

# Calculate correlation matrix
correlation_matrix = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Display the plot
plt.show()


# In[22]:


data_types = data.dtypes

target_variable = "math score"

if data_types[target_variable] == 'object' or data_types[target_variable] == 'category':
    problem_type = 'Classification'
else:
    problem_type = 'Regression'

print(f'The machine learning problem is a {problem_type} problem.')


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have already imported the dataset
data = pd.read_csv("StudentsPerformance.csv")

# Assuming your dataset has a column named 'Class' representing the class information
class_counts = data['parental level of education'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Students in Each Class')
plt.xlabel('Class')
plt.ylabel('Number of Students')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# In[5]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
data = pd.read_csv("StudentsPerformance.csv")

# Display information about missing values
print("Missing values before handling:")
print(data.isnull().sum())

# Assuming 'gender', 'race/ethnicity', 'parental level of education', 'lunch', and 'test preparation course' are categorical columns

# Handle missing/faulty categorical values
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# Label encoding for ordinal variables
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = data[column].fillna("Unknown")  # Replace missing values with a placeholder
    data[column] = label_encoder.fit_transform(data[column])

# One-hot encoding for nominal variables
data = pd.get_dummies(data, columns=['race/ethnicity'], prefix='ethnicity', drop_first=True)

# Display information after handling missing values
print("\nMissing values after handling:")
print(data.isnull().sum())

# Now, you can use the encoded data for further analysis or machine learning modeling

