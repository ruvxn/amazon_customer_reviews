import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
df_alexa = pd.read_csv('dataset/amazon_alexa.tsv', sep='\t')

# Display first few rows
print(df_alexa.head())

# Split into positive and negative feedback
positive = df_alexa[df_alexa['feedback'] == 1]
negative = df_alexa[df_alexa['feedback'] == 0]

# Plot feedback distribution
sns.countplot(x=df_alexa['feedback'], label="Count")

# Drop unnecessary columns
df_alexa = df_alexa.drop(['rating', 'date'], axis=1)

# One-hot encoding for 'variation' column
variation_dummies = pd.get_dummies(df_alexa['variation'], drop_first=True)
print(variation_dummies.head())

# Replace variation column with one-hot encoded columns
df_alexa.drop(['variation'], axis=1, inplace=True)
df_alexa = pd.concat([df_alexa, variation_dummies], axis=1)

print(df_alexa.head())

# Handle missing values in verified_reviews
print("Missing values before processing:", df_alexa['verified_reviews'].isnull().sum())
df_alexa.loc[:, 'verified_reviews'] = df_alexa['verified_reviews'].fillna("")
print("Missing values after processing:", df_alexa['verified_reviews'].isnull().sum())

# Vectorizer object
vectorizer = CountVectorizer()
alexa_countvectorizer = vectorizer.fit_transform(df_alexa['verified_reviews'])

# Print vocabulary and encoded matrix
print(vectorizer.get_feature_names_out())  
print(alexa_countvectorizer.toarray())

# Replace verified_reviews column with the transformed numerical representation
df_alexa.drop(['verified_reviews'], axis=1, inplace=True)
encoded_reviews = pd.DataFrame(alexa_countvectorizer.toarray())

# Merge the transformed reviews back into the dataset
df_alexa = pd.concat([df_alexa, encoded_reviews], axis=1)

# Save processed data
df_alexa.to_csv('dataset/amazon_alexa_cleaned.csv', index=False)
print('Data saved to amazon_alexa_cleaned.csv')
