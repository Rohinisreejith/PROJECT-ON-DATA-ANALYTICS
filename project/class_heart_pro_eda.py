import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Analysis", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Data Analysis")
st.image("project/heart1.jpeg")
# Loading the heart disease dataset
hdf = pd.read_csv('project/heart.csv')
st.header("Heart Disease Dataset")
st.dataframe(hdf.head())

# Setting numerical labels (if needed)
st.subheader("Converting Data to Numerical Labels (if needed)")
le = LabelEncoder()
for column in hdf.columns:
    if hdf[column].dtype == 'object':
        hdf[column] = le.fit_transform(hdf[column])
st.dataframe(hdf.head())

st.title("Heart Disease Data Analysis")
st.subheader("Heart Disease Dataset")
st.dataframe(hdf)

st.subheader("Summary Statistics")
st.write(hdf.describe())

st.subheader("Pairplot")
pairplot = sns.pairplot(hdf, hue='target')  # Assuming 'target' is the column indicating heart disease presence
st.pyplot(pairplot)

plt.figure(figsize=(10, 6))
st.subheader("Correlation Heatmap")
heatmap = sns.heatmap(hdf.corr(), annot=True, cmap='coolwarm')
st.pyplot(heatmap.figure)
