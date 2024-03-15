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

# # Load the model
# working_dir = os.path.dirname(os.path.abspath(__file__))
# with open('house_price_model.sav', 'rb') as file:
#     house_price_model = pickle.load(file)
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
park_mapping{
  'yes':1.
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
    park=park_mapping[park]
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
