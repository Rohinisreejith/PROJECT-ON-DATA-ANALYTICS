import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as mat
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split as tts

st.set_page_config(page_title="Heart Disease Classification", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Data Analysis and Classification")

# Load the heart disease dataset
hdf = pd.read_csv('project/heart.csv')
st.header("Heart Disease Dataset")
st.dataframe(hdf.head())

# Encode categorical variables to numerical values (if any)
le = LabelEncoder()
for column in hdf.columns:
    if hdf[column].dtype == 'object':  # Only encode if the column is categorical
        hdf[column] = le.fit_transform(hdf[column])

st.subheader("Encoded Dataset")
st.dataframe(hdf.head())

# Define features (X) and target (y)
x = hdf.drop('target', axis=1)  # Assuming 'target' is the column to predict
y = hdf['target']

# Initialize the Decision Tree Classifier
heart_model = dtc(criterion='entropy', random_state=0)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=42)
heart_model.fit(xtrain, ytrain)

# Predict on the test set
ypred = heart_model.predict(xtest)

# Display the classification report
st.header("Classification Report")
st.table(mat.classification_report(ytest, ypred, output_dict=True))
st.write("58	0	0	100	248	0	0	122	0	1	1	0	2")

# Prediction section
st.header("Prediction")

st.image("project/heart1.jpeg")
n1 = st.number_input("Enter age", min_value=float(x['age'].min()), max_value=float(x['age'].max()))
n2 = st.number_input("Enter sex (0 = Female, 1 = Male)", min_value=float(x['sex'].min()), max_value=float(x['sex'].max()))
n3 = st.number_input("Enter cp (chest pain type)", min_value=float(x['cp'].min()), max_value=float(x['cp'].max()))
n4 = st.number_input("Enter trestbps (resting blood pressure)", min_value=float(x['trestbps'].min()), max_value=float(x['trestbps'].max()))
n5 = st.number_input("Enter chol (serum cholesterol in mg/dl)", min_value=float(x['chol'].min()), max_value=float(x['chol'].max()))
n6 = st.number_input("Enter fbs (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)", min_value=float(x['fbs'].min()), max_value=float(x['fbs'].max()))
n7 = st.number_input("Enter restecg (resting electrocardiographic results)", min_value=float(x['restecg'].min()), max_value=float(x['restecg'].max()))
n8 = st.number_input("Enter thalach (maximum heart rate achieved)", min_value=float(x['thalach'].min()), max_value=float(x['thalach'].max()))
n9 = st.number_input("Enter exang (exercise induced angina) (1 = yes; 0 = no)", min_value=float(x['exang'].min()), max_value=float(x['exang'].max()))
n10 = st.number_input("Enter oldpeak (ST depression induced by exercise relative to rest)", min_value=float(x['oldpeak'].min()), max_value=float(x['oldpeak'].max()))
n11 = st.number_input("Enter slope (the slope of the peak exercise ST segment)", min_value=float(x['slope'].min()), max_value=float(x['slope'].max()))
n12 = st.number_input("Enter ca (number of major vessels colored by fluoroscopy)", min_value=float(x['ca'].min()), max_value=float(x['ca'].max()))
n13 = st.number_input("Enter thal (thalassemia) (3 = normal; 6 = fixed defect; 7 = reversable defect)", min_value=float(x['thal'].min()), max_value=float(x['thal'].max()))

# Prepare the input data as a list of lists (1 sample with all features)
sample1 = [[n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]]

if st.button("Predict Heart Disease"):
    target_sp = heart_model.predict(sample1)
    proba = heart_model.predict_proba(sample1)
    st.write("Probability:", proba)
    st.write("Prediction (0 = No Heart Disease, 1 = Heart Disease):", target_sp)

    if target_sp == 1:
        st.write("This patient is likely to have Heart Disease.")
        st.image("project/heart.jpeg")
    else:
        st.write("This patient is unlikely to have Heart Disease.")
        st.image("project/healthy.jpeg")
