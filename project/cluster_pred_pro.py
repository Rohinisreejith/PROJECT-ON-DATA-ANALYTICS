import streamlit as st
import joblib

# Load the trained KMeans model
model1 = joblib.load('project/kmeans_penguins.pkl')

st.header("🐧Penguin Species Prediction🐧")
st.image("project/adelie.jpeg")

# Input fields for penguin features
c1, c2 = st.columns(2)
n1 = float(c1.number_input("Enter bill length (mm)"))
n2 = float(c1.number_input("Enter bill depth (mm)"))
n3 = float(c2.number_input("Enter flipper length (mm)"))
n4 = float(c2.number_input("Enter body mass (g)"))

st.write("Sample values: bill length (mm), bill depth (mm), flipper length (mm), body mass (g)")
st.write("Sample values: Adelie Penguin : 39.1, 18.7, 181, 3750")
st.write("Sample values: Gentoo Penguin : 50.0, 15.0, 220, 5000")
st.write("Sample values: Chinstrap Penguin : 46.0, 17.5, 195, 3650")

# Prepare the sample for prediction
sample = [[n1, n2, n3, n4]]

# Prediction button and result display
if st.button("Predict Penguin Species"):
    t = model1.predict(sample)
    if t == 0:
        st.write("🐧 Adelie Penguin")
        st.image("project/adelie.jpeg")
    elif t == 1:
        st.write("🐧 Chinstrap Penguin")
        st.image("project/chin.jpeg")
    elif t == 2:
        st.write("🐧 Gentoo Penguin")
        st.image("project/gentoo.jpeg")
    else:
        st.write("Penguin species not listed")
        
