import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.markdown("""
    <style>
        body {
            background-image: url("https://kineticom.com/wp-content/uploads/posts/telecom-recruiting-telecommunications/telecom-recruiting-companies-1280x627.png");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)










# Load the trained model from the pickle file
with open('randomf_model.pickle', 'rb') as f:
    model = pickle.load(f)

st.write('### TELECO CUTOMER CHURN PREDICTION')
internet_service = st.selectbox('Internet Service', ['Yes', 'No'])
contract = st.selectbox('Contract', ['Month-to-Month', 'One Year', 'Two Year'])
married = st.selectbox('Marital Status', ['Yes', 'No'])
dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, step=1)
tenure = st.number_input('Tenure in Months', min_value=0, max_value=72, step=1)

# Wrap the input data in a DataFrame
input_data = pd.DataFrame({
    'internetservice': [internet_service],
    'contract': [contract],
    'married': [married],
    'numberofdependents': [dependents],
    'tenureinmonths': [tenure]
})

if st.button('Predict'):
    # Make a prediction using the model
    y_pred = model.predict(input_data)

    if y_pred[0] == 1:
       st.markdown(f'<p style="font-size:50px; font-weight:bold; color:#333;">This customer is likely to churn.</p>',unsafe_allow_html=True)

    else:
       st.markdown(f'<p style="font-size:50px; font-weight:bold; color:#006400;">This customer is unlikely to churn.</p>',unsafe_allow_html=True)

