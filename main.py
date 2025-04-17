import pandas as pd
from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.kmeans_model import apply_kmeans
from src.visualizer import plot_clusters
import matplotlib.pyplot as plt
import seaborn as sns


df = load_data('data/mall_customers.csv')
print("🔹 First 5 rows of the dataset:")
print(df.head())

print("\n🔍 DataFrame Info:")
print(df.info())

print("\n🧩 Missing Values Count:")
print(df.isnull().sum())

df_cleaned = df.dropna()


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df_cleaned = remove_outliers_iqr(df, "Annual Income (k$)")
df_cleaned = remove_outliers_iqr(df_cleaned, "Spending Score (1-100)")

print("\n✅ Cleaned Data Info:")
print(df_cleaned.info())

print("\n📈 Cleaned Data Summary:")
print(df_cleaned.describe())



df["Annual Income (k$)"] = df["Annual Income (k$)"].fillna(df["Annual Income (k$)"].mean())
df["Spending Score (1-100)"] = df["Spending Score (1-100)"].fillna(df["Spending Score (1-100)"].median())



print("\n🔹 Summary statistics:")
print(df.describe())

plt.figure(figsize=(10, 4))
sns.histplot(df["Age"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(df["Annual Income (k$)"], bins=20, kde=True, color="green")
plt.title("Distribution of Annual Income (k$)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True, color="coral")
plt.title("Distribution of Spending Score (1-100)")
plt.xlabel("Spending Score")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
