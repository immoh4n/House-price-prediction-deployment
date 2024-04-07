import os
import time
import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Final2 import sam, sam1, sam2
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration
st.set_page_config(page_title="House price prediction",
                   layout="wide",
                   page_icon="🏘️")

# Define the area options and their corresponding integer representations
area_mapping = {
    'karapakkam': 1,
    'adyar': 2,
    'chrompet': 3,
    'velachery': 4,
    'kk_nagar': 5,
    'anna_nagar': 6,
    't_nagar': 7
}
park_mapping = {
    'yes': 1,
    'no': 0
}

# Introduction and Project Description
st.title('House Price Prediction')
st.write('Welcome to the House Price Prediction app! This project uses machine learning algorithms to predict house prices based on various features such as location, square footage, distance to the main road, number of bedrooms and bathrooms, number of rooms, and parking availability.')

st.write('Please enter the details in the sidebar and click the "Predict House Price" button to see the predicted prices.')

# Sidebar for user input
with st.sidebar:
    st.write('### Input Details')

    area = st.selectbox('Area', list(area_mapping.keys()))
    sqft = st.text_input('Square Footage')
    dist_main = st.text_input('Distance to main road')
    bedrooms = st.text_input('Number of Bedrooms')
    bathrooms = st.text_input('Number of Bathrooms')
    rooms = st.text_input('Number of Rooms')
    park = st.selectbox('Parking', list(park_mapping.keys()))

# Convert input to appropriate data types
def validate_input(area, sqft, dist_main, bedrooms, bathrooms, rooms, park):
    try:
        area = area_mapping[area]
        sqft = float(sqft)
        dist_main = float(dist_main)
        bedrooms = int(bedrooms)
        bathrooms = int(bathrooms)
        rooms = int(rooms)
        
        # Check if park value exists in park_mapping
        if park in park_mapping:
            park = park_mapping[park]
        else:
            raise ValueError("Invalid value for 'park'")

        return True, area, sqft, dist_main, bedrooms, bathrooms, rooms, park
    except ValueError:
        return False, None, None, None, None, None, None, None

# Button for prediction
if st.button('Predict House Price'):
    is_valid, area, sqft, dist_main, bedrooms, bathrooms, rooms, park = validate_input(area, sqft, dist_main, bedrooms, bathrooms, rooms, park)
    
    if not is_valid:
        st.error('Please enter valid numerical values for input fields')
    else:
        prediction1 = sam(area, sqft, dist_main, bedrooms, bathrooms, rooms, park)
        prediction2 = sam1(area, sqft, dist_main, bedrooms, bathrooms, rooms, park)
        prediction3 = sam2(area, sqft, dist_main, bedrooms, bathrooms, rooms, park)
        
        st.write('Predicted House Price (DT):', prediction1)
        st.write('Predicted House Price (KNN):', prediction2)
        st.write('Predicted House Price (LR):', prediction3)
        
        # Plotting the forecasted prices
        forecasted_prices = [prediction1, prediction2, prediction3]
        models = ['Decision Tree', 'KNN', 'Linear Regression']

        plt.figure(figsize=(10, 6))
        plt.bar(models, forecasted_prices, color=['blue', 'green', 'orange'])
        plt.title('Predicted House Prices')
        plt.xlabel('Regression Model')
        plt.ylabel('Predicted Price')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Display the plot in Streamlit
        st.pyplot(plt)

# Comparative Analysis (2)
with st.sidebar:
    st.write('### Comparative Analysis')
    area1 = st.selectbox('Area 1', list(area_mapping.keys()))
    area2 = st.selectbox('Area 2', list(area_mapping.keys()))

if st.button('Compare House Prices'):
    is_valid_area1, area1, sqft, dist_main, bedrooms, bathrooms, rooms, park = validate_input(area1, sqft, dist_main, bedrooms, bathrooms, rooms, park)
    is_valid_area2, area2, sqft, dist_main, bedrooms, bathrooms, rooms, park = validate_input(area2, sqft, dist_main, bedrooms, bathrooms, rooms, park)

    if not is_valid_area1 or not is_valid_area2:
        st.error('Please enter valid numerical values for input fields')
    else:
        prediction1_area1 = sam(area1, sqft, dist_main, bedrooms, bathrooms, rooms, park)
        prediction1_area2 = sam(area2, sqft, dist_main, bedrooms, bathrooms, rooms, park)
        
        st.write('Predicted House Price for Area 1 (DT):', prediction1_area1)
        st.write('Predicted House Price for Area 2 (DT):', prediction1_area2)

# Forecasting (4)
if st.button('Forecast House Prices'):
    is_valid, area, sqft, dist_main, bedrooms, bathrooms, rooms, park = validate_input(area, sqft, dist_main, bedrooms, bathrooms, rooms, park)
    
    if not is_valid:
        st.error('Please enter valid numerical values for input fields')
    else:
        # Fit ARIMA model
        model = ARIMA(time_series_data, order=(5,1,0))
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=12)  # Forecasting 12 months ahead

        # Plotting the forecasted prices
        forecasted_months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6', 'Month 7', 'Month 8', 'Month 9', 'Month 10', 'Month 11', 'Month 12']

        plt.figure(figsize=(10, 6))
        plt.plot(forecasted_months, forecast, marker='o', linestyle='-')
        plt.title('Forecasted House Prices for Next 12 Months')
        plt.xlabel('Month')
        plt.ylabel('Predicted Price')
        plt.grid(True)
        
        # Display the plot in Streamlit
        st.pyplot(plt)

# Heatmaps (6)
if st.button('Show Heatmap'):
    # Assuming you have a DataFrame with house price data for different regions
    # Pivot the data for heatmap visualization
    pivot_data = pd.pivot_table(data=df, values='house_price', index='region', columns='month')

    # Plotting the heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, cmap='coolwarm', annot=True, fmt=".0f")
    plt.title('House Prices Across Regions')
    plt.xlabel('Month')
    plt.ylabel('Region')
    plt.show()
