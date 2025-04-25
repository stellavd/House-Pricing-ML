# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def ensure_dirs():
    os.makedirs("plots/distributions", exist_ok=True)
    os.makedirs("plots/boxplots", exist_ok=True)
    os.makedirs("plots/correlations", exist_ok=True)

def load_data(path="data/house_price_regression_dataset.csv"):
    return pd.read_csv(path)

def basic_info(df):
    print("\n🧾 Basic Info:")
    print(df.info())
    print("\n📊 Descriptive Stats:")
    print(df.describe())
    print("\n🔍 Missing Values:")
    print(df.isnull().sum())

def plot_distributions(df, target_column):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.savefig(f"plots/distributions/{col}_distribution.png")
        plt.close()

def plot_boxplots(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(f"plots/boxplots/{col}_boxplot.png")
        plt.close()

def correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlations/correlation_heatmap.png")
    plt.close()

def main():
    ensure_dirs()
    df = load_data()
    target_column = "House_Price"  # FIXED HERE

    basic_info(df)
    plot_distributions(df, target_column)
    plot_boxplots(df)
    correlation_heatmap(df)

    print("\n✅ EDA completed. Check the 'plots/' folder for visualizations.")

if __name__ == "__main__":
    main()
