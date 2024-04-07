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
import folium

# Set page configuration
st.set_page_config(page_title="House price prediction",
                   layout="wide",
                   page_icon="🏘️")

# Define the area options and their corresponding integer representations
area_mapping = {
    'karapakkam': {'location': (12.893, 80.201), 'color': 'blue'},
    'adyar': {'location': (13.005, 80.251), 'color': 'green'},
    'chrompet': {'location': (12.949, 80.142), 'color': 'red'},
    'velachery': {'location': (12.978, 80.220), 'color': 'purple'},
    'kk_nagar': {'location': (13.036, 80.213), 'color': 'orange'},
    'anna_nagar': {'location': (13.087, 80.217), 'color': 'yellow'},
    't_nagar': {'location': (13.037, 80.233), 'color': 'pink'}
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
        prediction1 = sam(area['location'][0], sqft, dist_main, bedrooms, bathrooms, rooms, park)
        prediction2 = sam1(area['location'][0], sqft, dist_main, bedrooms, bathrooms, rooms, park)
        prediction3 = sam2(area['location'][0], sqft, dist_main, bedrooms, bathrooms, rooms, park)
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

# Display Map
st.subheader('Location Map')
map_center = area_mapping[area]['location']
m = folium.Map(location=map_center, zoom_start=12)
folium.Marker(location=map_center, popup=area, icon=folium.Icon(color=area_mapping[area]['color'])).add_to(m)
folium_static(m)
