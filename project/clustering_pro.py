import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics as mat
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Penguins Data Analysis", page_icon="🐧", layout="wide")
st.title("🐧 Penguins Data Analysis 🐧")

# Load the dataset
df = pd.read_csv('penguins.csv')
st.header('🐧 PENGUINS DATA SET 🐧')
st.table(df.head())

# Handle missing values if any
df.dropna(inplace=True)

cl1, cl2 = st.columns(2)
le=LabelEncoder()

df['species']=le.fit_transform(df['species'])

cl1.header("Count of unique species")
cl1.table(df['species'].value_counts())

cl1.header("Count of unique islands")
cl1.table(df['island'].value_counts())

# Feature selection for clustering
x = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df[['species']]

wcss = []
k = []
for i in range(1, 11):
    k.append(i)
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=30, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

st.write("Values of k and wcss")
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(k, wcss, c='g', marker='o', mfc='r')
st.pyplot(fig)

km_final = KMeans(n_clusters=3, init='k-means++', max_iter=30, random_state=0)  # Assuming 3 species
df['new_label'] = km_final.fit_predict(x)

st.header('🐧 PENGUINS DATA SET WITH NEW LABELS 🐧')
st.table(df.head())

st.header("Visualizing the new labels and clusters")

fig2 = px.scatter(df, x='bill_length_mm', y='flipper_length_mm', size='body_mass_g', color='new_label')
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(df, x='bill_length_mm', y='flipper_length_mm', size='body_mass_g', color='species')
st.plotly_chart(fig3, use_container_width=True)

dbs = mat.davies_bouldin_score(x, km_final.labels_)
sil = mat.silhouette_score(x, km_final.labels_)
cal = mat.calinski_harabasz_score(x, km_final.labels_)

ars = mat.adjusted_rand_score(df['species'], km_final.labels_)
mu = mat.mutual_info_score(df['species'], km_final.labels_)

st.header("Evaluation Score")
st.image("gentoo.jpeg")

c1, c2, c3 = st.columns(3)
c4, c5 = st.columns(2)

c1.subheader('Davies-Bouldin Score')
c1.subheader(dbs)
c2.subheader('Silhouette Score')
c2.subheader(sil)
c3.subheader('Calinski-Harabasz Score')
c3.subheader(cal)
c4.subheader('Adjusted Rand Score')
c4.subheader(ars)
c5.subheader('Mutual Information Score')
c5.subheader(mu)

m1 = pickle.dump(km_final, open('kmeans_penguins.pkl', 'wb'))