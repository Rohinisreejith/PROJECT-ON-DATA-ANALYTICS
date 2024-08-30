import streamlit as st
import joblib

st.set_page_config(page_title="Laptop price prediction",page_icon=" 💻 ",layout="wide")
st.title(" 💻 Laptop price prediction 💻 ")


model_ridge      = joblib.load('project/rid1.pkl')
model_lasso      = joblib.load('project/lass1.pkl')
model_elasticnet = joblib.load('project/enet1.pkl')

st.header("Prediction")
st.image("laptop.jpeg")
st.write("3.830295706	,  16	, 512,11.18514743  , 2.641094425	, 17395.09306")
n = st.number_input("Enter value for Brand : ")
n1 = st.number_input("Enter value for Processor_Speed : ")
n2 = int(st.number_input("Enter value for RAM_Size: "))
n3 = int(st.number_input("Enter value for Storage_Capacity : "))
n4 = st.number_input("Enter value for Screen_Size : ")
n5 = st.number_input("Enter value for Weight : ")

sample1 = [[n,n1,n2,n3,n4,n5]]

if st.button("Predict the price"):
    t1=model_ridge.predict(sample1)
    t2=model_lasso.predict(sample1)
    t3=model_elasticnet.predict(sample1)
    if (t1):
        st.write("Predicted price for laptop is")
        c1,c2,c3 = st.columns(3)
        c1.subheader("Ridge Regression")
        c2.subheader("LASSO Regression")
        c3.subheader("ElasticNet Regression")
        c1.write(t1)
        c2.write(t2)
        c3.write(t3)
    else:
        st.write("Price cannot be determined")
