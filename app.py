import streamlit as st
import pandas as pd
import joblib
import os

# Define the directory where artifacts are saved
artifact_dir = "deployment_artifacts"

# Define the file paths for the model and scaler
model_path = os.path.join(artifact_dir, "best_decision_tree_model.joblib")
scaler_path = os.path.join(artifact_dir, "scaler.joblib")

# Load the trained model and scaler
try:
    best_dt_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'deployment_artifacts' directory and its contents are in the same directory as the app.")
    st.stop()

# Get the list of feature names that the model was trained on
# This is important to ensure the order of features is correct for prediction
# In a real app, you might save this list during training or ensure your input
# DataFrame columns match the training data columns. For this example,
# we'll define them based on the features used in the notebook's training step
feature_names = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                 'Income']


st.title("Diabetes Prediction App")
st.write("Enter the individual's health indicators to predict the likelihood of diabetes.")

# Create input fields for each feature
input_data = {}

# Group inputs for better organization in the UI
st.header("Health Indicators")

col1, col2 = st.columns(2)

with col1:
    input_data['HighBP'] = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['HighChol'] = st.selectbox("High Cholesterol", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['CholCheck'] = st.selectbox("Cholesterol Check in last 5 years", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['BMI'] = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    input_data['Smoker'] = st.selectbox("Smoked at least 100 cigarettes in lifetime", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['Stroke'] = st.selectbox("Ever had a stroke", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['HeartDiseaseorAttack'] = st.selectbox("Ever had coronary heart disease or myocardial infarction", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['PhysActivity'] = st.selectbox("Physical activity in past 30 days", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['Fruits'] = st.selectbox("Consume Fruit 1 or more times per day", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['Veggies'] = st.selectbox("Consume Vegetables 1 or more times per day", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['HvyAlcoholConsump'] = st.selectbox("Heavy Alcohol Consumption (adult men >14 drinks per week, adult women >7 drinks per week)", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col2:
    input_data['AnyHealthcare'] = st.selectbox("Have any kind of health care coverage", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['NoDocbcCost'] = st.selectbox("Could not see doctor because of cost in past 12 months", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['GenHlth'] = st.slider("General Health (1-Excellent, 2-Very Good, 3-Good, 4-Fair, 5-Poor)", 1, 5, 3)
    input_data['MentHlth'] = st.slider("Days of poor mental health in past 30 days", 0, 30, 0)
    input_data['PhysHlth'] = st.slider("Days of poor physical health in past 30 days", 0, 30, 0)
    input_data['DiffWalk'] = st.selectbox("Have serious difficulty walking or climbing stairs", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['Sex'] = st.selectbox("Sex", [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    input_data['Age'] = st.slider("Age (1-18-24, ..., 13-80+)", 1, 13, 7) # Assuming age is on a scale 1-13
    input_data['Education'] = st.slider("Education (1-No high school diploma, ..., 6-College graduate)", 1, 6, 5) # Assuming education is on a scale 1-6
    input_data['Income'] = st.slider("Income (1-Less than $10k, ..., 8-$75k or more)", 1, 8, 6) # Assuming income is on a scale 1-8


# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data], columns=feature_names)

# Apply preprocessing (scaling) - ensure numerical features are in the correct columns
numeric_features = ['BMI', 'MentHlth', 'PhysHlth'] # Features that were scaled

# Apply the same scaler used during training
input_df[numeric_features] = scaler.transform(input_df[numeric_features])


# Make prediction when button is clicked
if st.button("Predict Diabetes"):
    prediction = best_dt_model.predict(input_df)

    st.header("Prediction Result")

    if prediction[0] == 1:
        st.error("Based on the provided information, this individual is likely to have diabetes.")
        st.write("Consider consulting a healthcare professional for a proper diagnosis.")
    else:
        st.success("Based on the provided information, this individual is likely not to have diabetes.")
        st.write("Remember that this is a prediction and not a medical diagnosis.")