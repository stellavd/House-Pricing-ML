# modeling.py

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from preprocessing import load_data, preprocess_and_split

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n📌 {name} Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")

    return rmse, model

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    }

    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        rmse, trained_model = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = trained_model
            best_name = name

    joblib.dump(best_model, f"artifacts/best_model.pkl")
    print(f"\n✅ Best model saved: {best_name} with RMSE = {best_rmse:.3f}")

if __name__ == "__main__":
    main()
