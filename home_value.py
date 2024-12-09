import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'add/your/filepath/here/house_price_regression_dataset.csv'
dataset = pd.read_csv(file_path)

#  Separate features (X) and target variable (y)
X = dataset[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 
             'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
y = dataset['House_Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualize actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
         color='red', linestyle='--', label='Ideal Fit (y=x)')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Home Prices")
plt.grid(True)
plt.legend()
plt.show()

# Output results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
