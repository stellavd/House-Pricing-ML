# House Price Prediction Pipeline

A modular machine learning pipeline to predict house prices using structured property data.  
Inspired by the structure of the `Thesis` project.

---

## Dataset

This project uses the **[Home Value Insights](https://www.kaggle.com/datasets/prokshitha/home-value-insights?resource=download)** dataset from Kaggle.

Place the file in:
data/house_price_regression_dataset.csv

## Inference (Optional)
To predict new house prices:

Add data/new_house_data.csv (without the House_Price column)

Run:

bash
Copy
Edit
python infer.py
Output saved to: predictions.csv

## Output Files
1. EDA Plots → plots/

2. Trained Model → artifacts/best_model.pkl

3. Preprocessor → artifacts/preprocessor.pkl

4. Predictions → predictions.csv

## Author
Developed by Styliani Vasileiadou
Built using a modular structure inspired by the Thesis project.



## License & Attribution
This project is for educational use only.
Dataset source: **[Kaggle - Home Value Insights](https://www.kaggle.com/datasets/prokshitha/home-value-insights?resource=download)** dataset from Kaggle
