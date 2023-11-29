import streamlit as st
import pandas as pd
import pickle

st.sidebar.title("Car Price Predictor")
st.title("Welcome to Car Price Predictor")

car = pd.read_csv('cleaned_car.csv')



fuel_type = car['fuel_type'].unique()

companies = sorted(car['company'].unique())
selected_company = st.selectbox("Select the Company: ",companies)

selected_company_df = car[car['company'] == selected_company]
car_models = sorted(selected_company_df['name'].unique())
selected_name = st.selectbox('Select the Model: ',car_models)

year = sorted(selected_company_df['year'].unique(),reverse=True)
selected_year = st.selectbox("Select Year of Purchase: ",year)


selected_fuel_type = st.selectbox("Select the Fuel Type: ",fuel_type)
selected_kms = st.number_input("Enter the Number of Kilometres that the car has travelled: ")

def predict():
    file = open("linearRegressionModel.pkl", 'rb')
    pipe = pickle.load(file)
    pred = pipe.predict(pd.DataFrame([[selected_name, selected_company, int(selected_year), selected_kms, selected_fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return round(pred[0])

if st.button("Predict"):
    file = open("linearRegressionModel.pkl", 'rb')
    pipe = pickle.load(file)
    pred = pipe.predict(
        pd.DataFrame([[selected_name, selected_company, int(selected_year), selected_kms, selected_fuel_type]],
                     columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    st.success("Prediction: ₹{}".format(round(pred[0])))

