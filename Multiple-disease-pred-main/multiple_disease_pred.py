# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 19:49:06 2024

@author: Mohamed MOUKHLISSI
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# App title and logo
st.title("🧑‍⚕️ Health Assistant - Multiple Disease Prediction")
st.write("Welcome to the Health Assistant app, your one-stop solution for predicting multiple diseases using machine learning models.")

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Disease Prediction Menu',
                           ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
                           menu_icon='hospital',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Define file paths for saved models
diabetes_model_path = 'diabetes_model.sav'
heart_disease_model_path = 'heart_disease_model.sav'
parkinsons_model_path = 'parkinsons_model.sav'

# Load models
with open(diabetes_model_path, 'rb') as file:
    diabetes_model = pickle.load(file)

with open(heart_disease_model_path, 'rb') as file:
    heart_disease_model = pickle.load(file)

with open(parkinsons_model_path, 'rb') as file:
    parkinsons_model = pickle.load(file)


# Helper function for displaying prediction results
def display_prediction(result, positive_msg, negative_msg):
    if result == 1:
        st.success(positive_msg)
    else:
        st.warning(negative_msg)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    st.header('Diabetes Prediction')
    st.image('https://images.unsplash.com/photo-1550571226-4b3b3e8a4c8b', use_column_width=True)

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0.0, max_value=300.0, value=0.0)
    with col3:
        BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0)

    with col1:
        SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0.0, max_value=900.0, value=0.0)
    with col3:
        BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
    with col2:
        Age = st.number_input('Age', min_value=0, max_value=120, value=0)

    # Prediction button
    if st.button('Predict Diabetes'):
        with st.spinner('Predicting...'):
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            diab_prediction = diabetes_model.predict([user_input])
            display_prediction(diab_prediction[0], 'The person is diabetic.', 'The person is not diabetic.')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    st.header('Heart Disease Prediction')
    st.image('https://images.unsplash.com/photo-1513628253939-010e64ac66cd', use_column_width=True)

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=0)
    with col2:
        sex = st.selectbox('Sex', options=['Male', 'Female'])
    with col3:
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])

    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=0)
    with col2:
        chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=0.0, max_value=600.0, value=0.0)
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])

    with col1:
        restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2])
    with col2:
        thalach = st.number_input('Max Heart Rate Achieved', min_value=0, max_value=220, value=0)
    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1])

    with col1:
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=0.0)
    with col2:
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
    with col3:
        ca = st.selectbox('Major Vessels Colored by Flourosopy', options=[0, 1, 2, 3, 4])

    with col1:
        thal = st.selectbox('Thalassemia', options=[0, 1, 2])

    # Prediction button
    if st.button('Predict Heart Disease'):
        with st.spinner('Predicting...'):
            user_input = [age, int(sex == 'Male'), cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            heart_prediction = heart_disease_model.predict([user_input])
            display_prediction(heart_prediction[0], 'The person has heart disease.', 'The person does not have heart disease.')

# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":

    st.header("Parkinson's Disease Prediction")
    st.image('https://images.unsplash.com/photo-1594404682762-1e8a2a3803f5', use_column_width=True)

    # Input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=300.0, value=0.0)
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=400.0, value=0.0)
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=300.0, value=0.0)
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=0.1, value=0.0)
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=0.01, value=0.0)

    with col1:
        RAP = st.number_input('MDVP:RAP', min_value=0.0, max_value=0.1, value=0.0)
    with col2:
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, max_value=0.1, value=0.0)
    with col3:
        DDP = st.number_input('Jitter:DDP', min_value=0.0, max_value=0.3, value=0.0)
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.0)
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=3.0, value=0.0)

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=0.5, value=0.0)
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=0.5, value=0.0)
    with col3:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, max_value=0.5, value=0.0)
    with col4:
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, max_value=1.0, value=0.0)
    with col5:
        NHR = st.number_input('NHR', min_value=0.0, max_value=1.0, value=0.0)

    with col1:
        HNR = st.number_input('HNR', min_value=0.0, max_value=50.0, value=0.0)
    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.0)
    with col3:
        DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.0)
    with col4:
        spread1 = st.number_input('spread1', min_value=-10.0, max_value=1.0, value=0.0)
    with col5:
        spread2 = st.number_input('spread2', min_value=0.0, max_value=1.0, value=0.0)

    with col1:
        D2 = st.number_input('D2', min_value=0.0, max_value=5.0, value=0.0)
    with col2:
        PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.0)

    # Prediction button
    if st.button("Predict Parkinson's"):
        with st.spinner('Predicting...'):
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            display_prediction(parkinsons_prediction[0], "The person has Parkinson's disease.", "The person does not have Parkinson's disease.")
