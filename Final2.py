import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

# Load your dataset
data = pd.read_csv('Chennai houseing sale_2.csv')

# Preprocess data
data.columns = data.columns.str.lower()
data.area = data.area.str.lower()
data.area = data.area.map({'karapakkam': 1, 'adyar': 2, 'chrompet': 3, 'velachery': 4,
                           'kk nagar': 5, 'anna nagar': 6, 't nagar': 7})
data.park_facil = data.park_facil.map({'yes': 1, 'no': 0})

# Define features and target variables
X = data.drop(columns=['commis', 'sales_price', 'reg_fee'], axis=1)
y = data['sales_price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the imputer with strategy='mean' to impute missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on the training data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data using the fitted imputer
X_test_imputed = imputer.transform(X_test)

# Data preprocessing: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize Decision Tree Regression model
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train_scaled, y_train)

def sam(area, sqft, dist_main, bedrooms, bathrooms, room, park):
    # Example prediction for a new house using Decision Tree Regression
    new_house_features = np.array([[area, sqft, dist_main, bedrooms, bathrooms, room, park]])
    new_house_features_scaled = scaler.transform(new_house_features)
    dt_predicted_price = dt_regressor.predict(new_house_features_scaled)[0]
    return dt_predicted_price

if __name__ == "__main__":
    # Manually input features for a new house
    area = int(input("Area (zipcode, district, etc.): "))
    sqft = int(input("Square footage: "))
    dist_main = int(input("Distance to main area: "))
    bedrooms = int(input("Number of bedrooms: "))
    bathrooms = int(input("Number of bathrooms: "))
    room = int(input("Number of rooms:"))
    park = int(input("Nearby park (1 for yes, 0 for no): "))

    # Predict house price
    predicted_price = sam(area, sqft, dist_main, bedrooms, bathrooms, room, park)
    print("Predicted Price for the New House (Decision Tree Regression):", predicted_price)
