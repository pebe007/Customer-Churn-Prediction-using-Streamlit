# write app to sandbox
import streamlit as st
import numpy as np
import pickle
from xgboost import XGBClassifier
from models import MyXGBoostClassifier
from models import RandomForest
# Mapping dictionaries for Geography and Gender
geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
gender_mapping = {'Male': 0, 'Female': 1}

# define your app content
def main():
    st.info("Credit Score")
    credit_score = st.number_input("Enter Credit Score", value=0)

    st.info("Geography")
    selected_geography = st.selectbox("Select Geography", ['France', 'Germany', 'Spain'])
    geography_encoded = geography_mapping[selected_geography]

    st.info("Gender")
    selected_gender = st.selectbox("Select Gender", ['Male', 'Female'])
    gender_encoded = gender_mapping[selected_gender]

    st.info("Age")
    age = st.number_input("Enter Age", value=0)

    st.info("Tenure")
    tenure = st.number_input("Enter Tenure", value=0)

    st.info("Balance")
    balance = st.number_input("Enter Balance", value=0.0)

    st.info("Number of Products")
    num_products = st.number_input("Enter Number of Products", value=0)

    st.info("Has Credit Card")
    has_credit_card = st.selectbox("Select Has Credit Card (False= 0, True = 1)", [0, 1])

    st.info("Is Active Member")
    is_active_member = st.selectbox("Select Is Active Member (False= 0, True = 1)", [0, 1])

    st.info("Estimated Salary")
    estimated_salary = st.number_input("Enter Estimated Salary (False= 0, True = 1)", value=0)

    input_data = np.array([[credit_score, geography_encoded, gender_encoded, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary]]).astype(np.float64)
    
    pickle_in = open('model.pkl', 'rb')
    pickle_model = pickle.load(pickle_in)
    prediction = pickle_model.predict(input_data)

    if st.button("Make Prediction"):
        prediction = pickle_model.predict(input_data)
        st.info("Customer Churn")
        st.write(prediction)

# execute the main function  	
if __name__ == '__main__':
    main()
