import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
page=st.sidebar.selectbox(
    'select',['🏠 Home','🤖 Insurance Prediction','📉 Model Performance','👨‍💻 About Developer']
)
if page=='🏠 Home':
  st.title('Health Insurance Cost Prediction')
  st.write("""Welcome to the Health Insurance Cost Prediction System.
  This machine learning application predicts a person's medical insurance charges
  based on personal and health-related information such as age, gender, BMI, 
  number of children, smoking status, and region.
  """)

  st.title('🎯 Project Objective')
  st.write("""The goal of this project is to estimate future medical insurance
   costs using machine learning techniques. This helps individuals understand 
   how different factors influence insurance expenses.
  """)
  st.title('📋 Features Used')
  st.write('Age')
  st.write('Sex')
  st.write('BMI (Body Mass Index)')
  st.write('Number of Children')
  st.write('Smoking Status')
  st.write('Region')
  st.title("🤖 Machine Learning Model")
  st.write('''This application uses a trained machine learning model to predict
   insurance charges based on user inputs.
  ''')
  st.title('📊 Application Sections')
  st.write('Home – Project overview')
  st.write('Visualizations – Data analysis and charts')
  st.write('Prediction – Predict insurance charges')
  st.write('Model Performance – Evaluation metrics')
  st.write('About – Project details')
  st.title('🚀 How to Use')
  st.write('Open the Prediction page from the sidebar.')
  st.write('Enter your details.')
  st.write('Click the Predict button.')
  st.write('View the estimated insurance charges.')
elif page =='🤖 Insurance Prediction':
    model=joblib.load('model.pkl')
    Robust=joblib.load('Robust.pkl')
    Standard=joblib.load('Standard.pkl')
    Encoder=joblib.load('Encoder.pkl')
    st.title('Health Insurance')
    Sex = st.selectbox("Sex", ['male', 'female'])
    Smoker = st.selectbox("Smoker", ['yes', 'no'])
    Region = st.selectbox("Region", ['southwest', 'southeast','northwest','northeast'])
    Age = st.number_input("Enter your Age:", min_value=0.0, max_value=100.0,format="%.2f")
    children = st.number_input("Enter your no of children:", min_value=0,max_value=3)
    bmi = st.number_input("Enter your bmi:", min_value=0.0, format="%.2f")
    if st.button("Predict"):

      input_data = pd.DataFrame({
        "age": [Age],
        "sex":[Sex],
        "bmi": [bmi],
        "children": [children],
        
        "smoker":[Smoker],
        "region":[Region],

    })
    input_data[["age"]] = Standard.transform(
    input_data[["age"]])
    input_data[["bmi"]] = Robust.transform(
    input_data[["bmi"]])
   
    input_data[['region']]=input_data[['region']].replace({
        'northwest':0,
     'northeast':0,
     'southeast':1,
     'southwest':1
    })
    input_data['sex'] = input_data['sex'].map({
    'female': 0,
    'male': 1
    })

    input_data['smoker'] = input_data['smoker'].map({
    'no': 0,
    'yes': 1
   })
   

    prediction = model.predict(input_data)
    prediction = np.expm1(prediction)
    st.success(f"Predicted Charge: ₹{prediction[0]:,.2f}")
elif page=='📉 Model Performance':
  st.title('Model performance')
  st.header('Linear Regression') 
  st.subheader('prediction for training data')
  st.write('r2_score:0.776123882239361') 
  st.write('mean_squared_error:0.19390630103852768')
  st.write('mean_absolute_error:0.27372234779261984')
  st.subheader('prediction for testing data')
  st.write('r2 score:0.7206234519855912')
  st.write('mean_squared_error:0.20878807830472793')
  st.write('mean_absolute_error:0.27475276987259506')
  st.header('Random Forest Regressor')
  st.subheader('prediction for training data')
  st.write('r2_score:0.9708025399290172')
  st.write('mean_squared_error:0.025288858582662842')
  st.write('mean_absolute_error:0.0769984865187981')
  st.subheader('prediction for testing data')
  st.write('r2 score:0.7864302611468955')
  st.write('mean_squared_error:0.15960829810554641')
  st.write('mean_absolute_error:0.21067824594547013')
  st.header('Gradient Boosting Regressor')
  st.subheader('prediction for training data')
  st.write('r2_score:0.8895441559382123')
  st.write('mean_squared_error:0.09566935662610156')
  st.write('mean_absolute_error:0.1556262603972643')
  st.subheader('prediction for testing data')
  st.write('r2 score:0.7960101906798884')
  st.write('mean_squared_error:0.15244887441123853')
  st.write('mean_absolute_error:0.20975521272318634')
elif page=='👨‍💻 About Developer':
  st.title('👨‍💻 About Developer')
  st.write("""
    Hi, I'm Ankit Kumar, a B.Tech Computer Science student.

    I am passionate about Data Science, Machine Learning, and Artificial Intelligence.
    I enjoy building data-driven applications and solving real-world problems using machine learning.
    """)
  st.subheader('Skills:')
  st.write('python')
  st.write('Machine Learning')
  st.write('Data Analysis')
  st.write('SQL')
  st.write('Streamlit')
  st.write('NLP')
  st.write('Deep Learning')
  st.subheader('Projects')
  st.write('• Health Insurance Cost Prediction System')
  
  st.subheader('Thank you for visiting this project.')

  

    
