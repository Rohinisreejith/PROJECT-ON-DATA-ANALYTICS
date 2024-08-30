import streamlit as st

pg=st.navigation([st.Page("clustering_pro.py",title="Penguins Species analysis"),


st.Page("cluster_pred_pro.py",title="Penguins Species Prediction"),

st.Page("regression_pro_eda.py",title="Laptop price data analysis "),

st.Page("regression_pro_pred.py",title="Laptop price prediction "),

st.Page("class_heart_pro_eda.py",title="Heart disease data analysis "),

st.Page("class_heart_pro_pred.py",title="Heart disease data prediction ")])

pg.run()