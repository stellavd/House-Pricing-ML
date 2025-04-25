title: "House Price Prediction Pipeline"
description: |
  A modular machine learning pipeline to predict house prices based on physical features of residential properties. This project is structured for clarity, scalability, and reproducibility — inspired by the `Thesis` architecture.

dataset:
  title: " Dataset"
  description: |
    This project uses the **Home Value Insights** dataset from Kaggle:  
    🔗 [Home Value Insights Dataset on Kaggle](https://www.kaggle.com/datasets/prokshitha/home-value-insights?resource=download)

    Place the dataset file inside the `data/` directory with the following name:

    ```
    data/house_price_regression_dataset.csv
    ```

    Ensure the target column is named `House_Price` (already handled in the pipeline).

usage:
  title: "How to Run"
  steps:
    - title: "Install Requirements"
      description: |
        Create a virtual environment (optional but recommended) and run:

        ```bash
        pip install -r requirements.txt
        ```

        Dependencies include:
        - pandas
        - scikit-learn
        - matplotlib
        - seaborn
        - xgboost
        - joblib

    - title: " Run the Pipeline"
      description: |
        ```bash
        python main.py
        ```

        This will:
        - Perform data exploration (EDA)
        - Preprocess the dataset
        - Train 3 regressors: Linear Regression, Random Forest, XGBoost
        - Save the best model based on RMSE
        - Run inference (if `new_house_data.csv` exists)

inference:
  title: "Inference on New Data"
  description: |
    To predict prices for unseen properties:

    1. Create a CSV named `new_house_data.csv` inside the `data/` folder.
    2. Make sure it includes all required features (no `House_Price` column needed).
    3. Run:

    ```bash
    python infer.py
    ```

    It will save `predictions.csv` in your working directory.

outputs:
  title: "Output Files"
  table: |
    | Folder / File          | Description                                 |
    |------------------------|---------------------------------------------|
    | `plots/`               | Distribution, boxplots, and heatmaps        |
    | `artifacts/`           | Saved model (`best_model.pkl`) and preprocessor |
    | `predictions.csv`      | Output predictions on new data              |

customization:
  title: "🛠️ Customization Options"
  description: |
    - **Change Target Column**: Default is `"House_Price"` — can be updated in `preprocessing.py`
    - **Add More Models**: Modify `modeling.py` to include LightGBM, CatBoost, etc.
    - **Advanced Tuning**: For later improvements, integrate Optuna or GridSearchCV
    - **Experiment with Features**: Add interaction terms, log transforms, or polynomial features

author:
  title: "Author"
  description: |
    Developed by Styliani Vasileiadou

license:
  title: "License & Attribution"
  description: |
    This project is for educational purposes only.  
    Dataset from: [Kaggle - Home Value Insights](https://www.kaggle.com/datasets/prokshitha/home-value-insights)
