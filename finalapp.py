import os
import time
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from Final2 import sam
from Final2 import sam1
from Final2 import sam2

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
    'yes':1,
    'no':0
}

# Sidebar for user input
with st.sidebar:
    st.title('House Price Prediction')
    st.write('Please enter details')

    col1, = st.columns(1)
    with col1:
        area = st.selectbox('Area',list(area_mapping.keys()))
        sqft = st.text_input('Square Footage')
        dist_main = st.text_input('Distance to main road')
        bedrooms = st.text_input('Number of Bedrooms')
        bathrooms = st.text_input('Number of Bathrooms')
        rooms = st.text_input('Number of Rooms')
        park = st.selectbox('Parking',list(park_mapping.keys()))

# Convert input to appropriate data types
try:
    area = area_mapping[area]
    sqft = float(sqft)
    dist_main = float(dist_main)
    bedrooms = int(bedrooms)
    bathrooms = int(bathrooms)
    rooms = int(rooms)
    park = park_mapping[park]
except ValueError:
    st.error('Please enter valid numerical values for input fields')
    st.stop()

# Button for prediction
if st.button('Predict House Price'):
    prediction1 = sam(area, sqft, dist_main, bedrooms, bathrooms, rooms, park)
    prediction2 = sam1(area, sqft, dist_main, bedrooms, bathrooms, rooms, park)
    prediction3 = sam2(area, sqft, dist_main, bedrooms, bathrooms, rooms, park)
    st.write('Predicted House Price(DT):', prediction1)
    st.write('Predicted House Price(KNN):', prediction2)
    st.write('Predicted House Price(LR):', prediction3)

    # Plotting the predicted prices
    labels = ['Decision Tree', 'KNN', 'Linear Regression']
    predicted_prices = [prediction1, prediction2, prediction3]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, predicted_prices, color=['blue', 'green', 'orange'])
    plt.title('Predicted House Prices')
    plt.xlabel('Regression Model')
    plt.ylabel('Predicted Price')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot in Streamlit
    st.pyplot(plt)

# Add background image from the same repository
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://s1.dmcdn.net/v/T8tfQ1ZkBZd0i6gpw/x720") no-repeat center center fixed;
        background-size: cover;
        filter: blur(5px);
    }
    </style>
    """,
    unsafe_allow_html=True
)
