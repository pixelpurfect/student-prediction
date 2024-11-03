import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_and_preprocess_data, preprocess_input

# Load and preprocess the data
X_train, X_test, y_train, y_test, scaler, label_encoders = load_and_preprocess_data("Real Time Dataset - Form responses 1.csv")

# Train a model (you can choose any model)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Streamlit UI
st.title("CGPA Predictor")

# Create input fields for each feature
input_data = {}
for column in label_encoders.keys():
    if column in ['What is your gender?  ', 'Do you participate in extracurricular activities?']:
        input_data[column] = st.selectbox(column, options=list(label_encoders[column].classes_))
    else:
        input_data[column] = st.text_input(column)

# Button to predict
if st.button("Predict CGPA"):
    try:
        # Preprocess input data
        processed_data = preprocess_input(input_data, label_encoders, scaler)

        # Make prediction
        prediction = model.predict(processed_data)
        st.success(f"Predicted CGPA: {prediction[0]:.2f}")

    except ValueError as e:
        st.error(f"Error in processing input: {str(e)}")
