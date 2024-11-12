# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Step 1: Data Collection
print("Data Collection:\n", df.head())

# Step 2: Data Cleaning
# Inspecting for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Handling missing values
# Drop columns with too many missing values or impute where appropriate
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.drop(columns=['deck', 'embark_town', 'alive'], inplace=True)

# Step 3: Handling Outliers
# Using box plots to detect outliers in 'age' and 'fare'
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='age')
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='fare')
plt.show()

# Removing outliers by capping based on quantiles
Q1 = df[['age', 'fare']].quantile(0.25)
Q3 = df[['age', 'fare']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['age', 'fare']] < (Q1 - 1.5 * IQR)) |(df[['age', 'fare']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 4: Data Normalization
# Applying Min-Max scaling to 'age' and 'fare'
scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# Step 5: Feature Engineering
# Creating 'family_size' and 'title' features
df['family_size'] = df['sibsp'] + df['parch']
df['title'] = df['who'].map({'man': 'Mr', 'woman': 'Mrs', 'child': 'Master'})

# Dropping irrelevant columns
df.drop(columns=['sibsp', 'parch', 'who', 'adult_male', 'class'], inplace=True)

# Step 6: Feature Selection
# Checking correlation matrix for feature selection
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Step 7: Model Building
# Encoding categorical variables and splitting data
df = pd.get_dummies(df, columns=['sex', 'embarked', 'title'], drop_first=True)
X = df.drop('survived', axis=1)
y = df['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training and evaluation
# Logistic Regression
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
print("Logistic Regression Accuracy:", logreg.score(X_test, y_test))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", rf.score(X_test, y_test))
