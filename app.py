import streamlit as st
import pandas as pd
import pickle

# Load the model from file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title for the app
st.title("Movie Success Prediction")

# Input fields for user data
director_name = st.text_input("Director Name")
duration = st.number_input("Duration (in minutes)", min_value=0)
actor_1_name = st.text_input("Main Actor Name")
budget = st.number_input("Budget (in dollars)", min_value=0)
genres = st.text_input("Genres (e.g., Action|Adventure)")
title_year = st.number_input("Year of Release", min_value=1900, max_value=2025)

# Predict button
if st.button("Predict"):
    # Prepare input data in DataFrame format
    input_data = pd.DataFrame({
        'director_name': [director_name],
        'duration': [duration],
        'actor_1_name': [actor_1_name],
        'budget': [budget],
        'genres': [genres],
        'title_year': [title_year]
    })

    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted Gross Revenue: ${prediction[0]:,.2f}")
