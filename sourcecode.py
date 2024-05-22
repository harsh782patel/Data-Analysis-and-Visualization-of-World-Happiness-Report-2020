import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('WHR20_DataForFigure2.1.csv')
#Exploring the data by checking the shape of the dataframe, column names, and data types of each column.
print(df.shape)
print(df.columns)
print(df.dtypes)
#Checking for missing values.
print(df.isnull().sum())
#Explore the data by calculating summary statistics of numeric columns.
print(df.describe())
#Visualizing the data using histograms.
df.hist(figsize=(10,10))
plt.show()
#Calculating correlations between variables and create a correlation heatmap.
corr = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
