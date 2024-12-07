
import streamlit as st
import pickle
from datetime import datetime

#Page configuration
st.set_page_config(page_title='CCP-I App', page_icon="📈", layout="wide", initial_sidebar_state="auto", menu_items=None)

# Title of the app
st.title(':blue[Customer Conversion Prediction - Insurance]📈')

col1, col2 =st.columns(2)
with col1:

    Job_dict = {'Blue collar': 1, 'Management': 2, 'Technician': 3, 'Admin.': 4, 'Services': 5, 
        'Retired': 6, 'Self-employed': 7, 'Entrepreneur': 8, 'Unemployed': 9, 'Housemaid': 10, 'Student': 11}
    Job = st.selectbox('Job', list(Job_dict.keys()))
    Job_val = Job_dict[Job]

    Marital_dict = {'Married': 1, 'Single': 2, 'Divorced': 3}
    Marital = st.selectbox('Marital Status', list(Marital_dict.keys()))
    Marital_val = Marital_dict[Marital]

    Education_dict = {'Secondary': 1, 'Tertiary': 2, 'Primary': 3}
    Education_Qualification = st.selectbox('Education Qualification', list(Education_dict.keys()))
    Education_val = Education_dict[Education_Qualification]

    # Date input for day and month
    date_input = st.date_input("Day of the month when the call was made.", datetime.now())
    Day = date_input.day
    Month = date_input.month

    

with col2:
    # Input parameters for user
    Age = st.slider('Age', 18, 70)

    Duration = st.slider('Duration of the call (in seconds)', 1, 640)

    Number_of_calls = st.slider('Number of calls made to the customer before this interaction.', 1, 6)

    

# Predict button
if st.button('**Predict**',use_container_width=True):
  try:
    # Backend: Loading the model and scaler and making the prediction
    with open('Insurance_pridiction_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open('Insurance_pridiction_rfc_model.pkl', 'rb') as model_file:
        clf_optimal = pickle.load(model_file)

    # Prepare the input data for prediction
    parameter = [[Age, Job_val, Marital_val, Education_val, Day, Month, Duration, Number_of_calls]]
    
    # Scale the input data
    parameter_scaled = scaler.transform(parameter)
    
    # Predict the result
    y_pred = clf_optimal.predict(parameter_scaled)

    # Display the result
    if y_pred[0] == 1:
        st.success("The respective customer will possibly subscribe to an insurance policy 👍")
        st.stop()
    else:
        st.error("The respective customer will not possibly subscribe to an insurance policy 👎")
        st.stop()

  except Exception as e:
    st.error(f"Error loading the model and scaler: {e}")
    st.warning("Please make sure the model and scaler files are available with this names(model->decision_tree_model.pkl, scaler->scaler.pkl) .")
    st.stop()  # Stop the app execution