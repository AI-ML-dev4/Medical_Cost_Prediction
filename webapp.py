import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.title("💊Medical Insurance Cost Predictor")

df = pd.read_csv("insurance.csv")

le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", le_sex.classes_)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", le_smoker.classes_)
region = st.selectbox("Region", le_region.classes_)

if st.button("Predict Insurance Cost"):
    sex_encoded = le_sex.transform([sex])[0]
    smoker_encoded = le_smoker.transform([smoker])[0]
    region_encoded = le_region.transform([region])[0]

    user_input = [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]]
    prediction = model.predict(user_input)[0]

    st.success(f"💰 Estimated Insurance Cost: ${prediction:.2f}")
    fig = px.scatter(df, x='bmi', y='charges', color='smoker',
                     title='BMI vs Medical Charges (Colored by Smoker)',
                     labels={"bmi": "BMI", "charges": "Medical Charges"})
    
    fig.add_scatter(x=[bmi], y=[prediction], mode='markers',
                    marker=dict(size=12, color='red'),
                    name='Your Prediction')

    st.plotly_chart(fig, use_container_width=True)