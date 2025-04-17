import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open("deepmammoai_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/DeepMammoAI_sl_ft.csv')  # Preprocessed dataset
    return model, scaler, df

model, scaler, df = load_model()


# Streamlit App UI
st.title("🎯 DeepMammoAI: Breast Cancer Diagnosis")
st.write("Interact with the app to predict malignancy and explore data!")

# --- BACKGROUND SECTION ---
st.markdown("""
### **Why Breast Cancer Screening Matters**
Breast cancer is the **most common cancer among women globally**, with over **2.3 million new cases** diagnosed annually (WHO, 2023). 
In Africa, it is a **critical public health issue** due to:
- **Late-stage diagnosis**: Limited access to screening leads to higher mortality rates.
- **Resource constraints**: Shortages of oncologists, radiologists, and diagnostic tools.
- **Cultural stigma**: Misconceptions delay medical consultation.
- **Rising incidence**: Urbanization and lifestyle changes are increasing risk factors.

#### **The African Context**
- Breast cancer accounts for **23% of all cancers in African women** (African Cancer Registry, 2022).
- Mortality rates in Africa are **50% higher** than in high-income countries due to delayed diagnosis.
- Only **10-20% of patients** in sub-Saharan Africa receive early-stage diagnosis (Journal of Global Oncology).

DeepMammoAI aims to bridge this gap by providing **AI-driven early diagnosis** to support healthcare workers in resource-limited settings.
""")

# Input fields for 10 features (WDBC dataset)
st.sidebar.header("🎮 Input Features")
st.sidebar.write("Adjust sliders or use sample data:")

# Display dataset preview
with st.expander("🔍 Dataset Preview"):
    st.write("First 5 rows:")
    st.dataframe(df.head())
    st.write("Last 5 rows:")
    st.dataframe(df.tail())

# Input fields for selected features
radius_mean = st.sidebar.slider("Radius (mean)", 0.0, 30.0, 14.0)
texture_mean = st.sidebar.slider("Texture (mean)", 4.0, 40.0, 20.0)
perimeter_mean = st.sidebar.slider("Perimeter (mean)", 40.0, 200.0, 90.0)
area_mean = st.sidebar.slider("Area (mean)", 100.0, 2500.0, 600.0)
concavity_mean = st.sidebar.slider("Concavity (mean)", 0.0, 0.5, 0.1)
concave_points_mean = st.sidebar.slider("Concave points (mean)", 0.0, 0.3, 0.1)
radius_worst = st.sidebar.slider("Radius (worst)", 5.0, 37.0, 17.0)
perimeter_worst = st.sidebar.slider("Perimeter (worst)", 50.0, 250.0, 100.0)
area_worst = st.sidebar.slider("Area (worst)", 100.0, 4250.0, 700.0)
concave_points_worst = st.sidebar.slider("Concave points (worst)", 0.0, 0.3, 0.1)

# # Define selected features (same as in training)
# selected_features = [
#     "radius_mean",
#     "texture_mean",
#     "perimeter_mean",
#     "area_mean",
#     "concavity_mean",
#     "concave_points_mean",
#     "radius_worst",
#     "perimeter_worst",
#     "area_worst",
#     "concave_points_worst"
# ]

# # Create input fields
# features = []
# for feature in selected_features:
#     value = st.sidebar.number_input(f"{feature}:", step=0.1)
#     features.append(value)

# Gamified button
if st.sidebar.button("🚀 Predict & Play"):
    # Collect inputs
    features = np.array([
        radius_mean, texture_mean, perimeter_mean, area_mean, concavity_mean,
        concave_points_mean, radius_worst, perimeter_worst,
        area_worst, concave_points_worst
    ]).reshape(1, -1)


# Predict button
    # Preprocess input
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Display results
    st.subheader("Prediction Result 📊")
    if prediction[0] == 1:
        st.error("🚨 Diagnosis: Malignant")
    else:
        st.success("🎉 Diagnosis: Benign!")
        st.balloons()  # Celebrate benign results
    st.write(f"Probability of Malignancy: {probability:.2%}")
    

    # # Visualize feature importance
    # st.write("### 📈 Feature Impact")
    # fig, ax = plt.subplots()
    # sns.barplot(x=model.feature_importances_, y=df.columns[:-1])
    # plt.title("Feature Importance")
    # st.pyplot(fig)

# Gamification: Add a "Play Again" button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Inputs"):
    st.experimental_rerun()


# User Instruction
st.markdown("""
**Note**:  
- Features correspond to the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
**Example Input (Malignant Case):**  
[17.99, 10.38, 122.8, 1001, 0.3001, 0.1471, 25.38, 184.6, 2019, 0.2654]
""")

# Add a fun footer
st.markdown("""
<footer>
    🎮 Made with Streamlit & ❤️ by Y.O. Ibrahim
</footer>
""", unsafe_allow_html=True)