import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open("deepmammoai_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# Streamlit App UI
st.title("DeepMammoAI: Breast Cancer Diagnosis")
st.write("Input histological features to predict malignancy.")

# Input fields for 10 features (WDBC dataset)
st.sidebar.header("Input Features")
features = []
for i in range(10):
    feature = st.sidebar.number_input(f"Feature {i+1}", value=0.0, step=0.1)
    features.append(feature)

# Predict button
if st.sidebar.button("Predict"):
    # Preprocess input
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Display results
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("Diagnosis: Malignant")
    else:
        st.success("Diagnosis: Benign")
    st.write(f"Probability of Malignancy: {probability:.2%}")


# User Instruction
st.markdown("""
**Note**:  
- Features correspond to the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
**Example Input (Malignant Case):**  
[17.99, 10.38, 122.8, 1001, 0.3001, 0.1471, 25.38, 184.6, 2019, 0.2654]
""")