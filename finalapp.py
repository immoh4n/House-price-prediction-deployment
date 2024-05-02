import os
import time
import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Final2 import sam, sam1, sam2

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

# Add CSS for background image with blur effect
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('https://image.freepik.com/free-photo/house-blur-background_93675-6645.jpg');
        background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Introduction and Project Description
st.title('House Price Prediction')
st.write('Welcome to the House Price Prediction app! This project uses machine learning algorithms to predict house prices based on various features such as location, square footage, distance to the main road, number of bedrooms and bathrooms, number of rooms, and parking availability.')

st.write('Please enter the details in the sidebar and click the "Predict House Price" button to see the predicted prices.')

# Sidebar for user input
with st.sidebar:
    st.write('### Input Details')

    area = st.selectbox('Area', list(area_mapping.keys()))
    sqft = st.text_input('Square Feet')
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
        park = park_mapping[park]
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

# Display the footer
st.markdown('---')
st.markdown('This website is available for both mobile and desktop.')



