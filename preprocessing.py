# preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def load_data(path="data/house_price_regression_dataset.csv"):
    return pd.read_csv(path)

def split_features_target(df, target_col="House_Price"):  # FIXED HERE
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def identify_columns(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return numerical_cols, categorical_cols

def build_preprocessor(numerical_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor

def preprocess_and_split(df, target_col="House_Price", test_size=0.2, random_state=42):  # FIXED HERE
    X, y = split_features_target(df, target_col)
    numerical_cols, categorical_cols = identify_columns(X)
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

    return X_train_transformed, X_test_transformed, y_train, y_test

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    print(f"\n✅ Preprocessing complete.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

if __name__ == "__main__":
    main()
