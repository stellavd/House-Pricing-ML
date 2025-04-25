# infer.py

import pandas as pd
import joblib
import numpy as np

def load_new_data(path="data/new_house_data.csv"):
    return pd.read_csv(path)

def predict_new_data(new_data, model_path="artifacts/best_model.pkl", preprocessor_path="artifacts/preprocessor.pkl"):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    transformed_data = preprocessor.transform(new_data)
    predictions = model.predict(transformed_data)

    return predictions

def save_predictions(predictions, output_path="predictions.csv"):
    df_pred = pd.DataFrame({'Predicted Price': predictions})
    df_pred.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to {output_path}")

def main():
    new_data = load_new_data()  # Update path if needed
    print(f"📂 Loaded {new_data.shape[0]} rows of new input data.")

    predictions = predict_new_data(new_data)
    save_predictions(predictions)

if __name__ == "__main__":
    main()
