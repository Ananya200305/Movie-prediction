import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and expected features
with open('/Users/ananyaaggarwal/Desktop/Movie Prediction Project/models/sequel_model_60_40.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
feature_cols = model_package['features']

st.title('🎬 Movie Sequel Success Predictor')

# Input fields
budget = st.number_input('Budget')
revenue = st.number_input('Revenue')
success_ratio = revenue / budget if budget > 0 else 0.0

director_avg_revenue = st.number_input("Director's Avg Revenue")
cast_avg_revenue = st.number_input("Cast's Avg Revenue")

genre = st.selectbox(
    'Genre',
    ['Action', 'Adventure', 'Sci-Fi', 'Drama', 'Comedy', 'Horror', 'Romance',
     'Thriller', 'Fantasy', 'Animation', 'Documentary', 'Family',
     'Mystery', 'Western', 'War', 'Musical', 'History']
)

# Build input data
if st.button('Predict'):
    # Start with base inputs
    input_dict = {
        'budget': budget,
        'revenue': revenue,
        'success_ratio': success_ratio,
        'director_avg_revenue': director_avg_revenue,
        'cast_avg_revenue': cast_avg_revenue
    }

    # Add genre one-hot encoding
    for g in [col for col in feature_cols if col.startswith('genres_')]:
        input_dict[g] = 1 if g == f'genres_{genre}' else 0

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])[feature_cols]  # Reorder columns to match training
    
    # Predict
    prediction = model.predict(input_df)[0]
    st.write('🎯 Prediction:', '**Success**' if prediction == 1 else '**Failure**')
