# main.py

import os
import sys
import eda
import preprocessing
import modeling
import infer

def run_eda():
    print("\n🔍 Running EDA...")
    eda.main()

def run_preprocessing():
    print("\n🧼 Running Preprocessing...")
    df = preprocessing.load_data()
    X_train, X_test, y_train, y_test = preprocessing.preprocess_and_split(df)
    return X_train, X_test, y_train, y_test

def run_modeling():
    print("\n🏗️ Running Modeling...")
    modeling.main()

def run_inference(optional=False):
    if optional and not os.path.exists("data/new_house_data.csv"):
        print("\nℹ️ No new data found. Skipping inference.")
        return
    print("\n🔮 Running Inference...")
    infer.main()

def main():
    print("\n🚀 Starting House Price Prediction Pipeline")

    run_eda()
    run_modeling()
    run_inference(optional=True)

    print("\n✅ Pipeline complete. Check the 'artifacts/' and 'plots/' folders for results.")

if __name__ == "__main__":
    main()
