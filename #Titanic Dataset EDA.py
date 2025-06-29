#Titanic Dataset EDA
# Author: Your Name

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
titanic = sns.load_dataset('titanic')

# Show basic info
print("Shape of dataset:", titanic.shape)
print("\nFirst 5 rows:\n", titanic.head())
print("\nData Types:\n")
print(titanic.dtypes)
print("\nMissing values:\n")
print(titanic.isnull().sum())

# Clean deck and embark_town - too many missing
titanic = titanic.drop(columns=['deck', 'embark_town'])

# Histogram - Age
plt.figure(figsize=(8,4))
sns.histplot(titanic['age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Boxplot - Fare
plt.figure(figsize=(8,4))
sns.boxplot(x=titanic['fare'])
plt.title('Boxplot of Fare')
plt.show()

# Countplot - Survival by Sex
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title('Survival by Sex')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Heatmap of numeric correlations
plt.figure(figsize=(8,6))
corr = titanic.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print("EDA is running...")
plt.show(block=True)
input("Press Enter to exit...")


