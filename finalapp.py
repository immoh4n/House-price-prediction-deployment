import os
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
from Final2 import sam
# Set page configuration
st.set_page_config(page_title="House price prediction",
                   layout="wide",
                   page_icon="🏘️")


# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for user input
with st.sidebar:
    st.title('House Price Prediction')
    st.write('Please enter details')

    col1, col2 = st.columns(2)
    with col1:
        area = st.text_input('Area (1 to 7)')
        sqft = st.text_input('Square Footage')
        dist_main = st.text_input('Distance to main road')
        bedrooms = st.text_input('Number of Bedrooms')
        
    with col2:
        bathrooms = st.text_input('Number of Bathrooms')
        rooms = st.text_input('Number of Rooms')
        park = st.text_input('Park Facility (yes=1 or no=0)')

# Convert input to appropriate data types
try:
    area = float(area)
    sqft = float(sqft)
    dist_main = float(dist_main)
    bedrooms = int(bedrooms)
    bathrooms = int(bathrooms)
    rooms = int(rooms)
    park = int(park)
except ValueError:
    st.error('Please enter valid numerical values for input fields')
    st.stop()

# # Transform the new input features
# new_house_features = np.array([[area, sqft, dist_main, bedrooms, bathrooms, rooms, park]])
# new_house_features_scaled = scaler.transform(new_house_features)

# Button for prediction
if st.button('Predict House Price'):
    # Perform prediction using the model
    prediction = sam(area, sqft, dist_main, bedrooms, bathrooms, rooms, park) 

    # Display prediction
    st.write('Predicted House Price:', prediction)
