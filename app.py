#🧠 STEP 1: Define feature lists
parkinsons_features = ["name",
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

kidney_features = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba',
    'bgr','bu','sc','sod','pot','hemo','pcv',
    'wc','rc','htn','dm','cad','appet','pe','ane'
]

liver_features = [
    "Age", "Gender", "Total_Bilirubin",
    "Direct_Bilirubin", "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens", "Albumin",
    "Albumin_and_Globulin_Ratio"
]

#🧩 STEP 2: Load models (pickle)
import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

parkinsons_model = load_model("parkinsons_model.pkl")
kidney_model = load_model("kidney_model.pkl")
liver_model = load_model("liver_model.pkl")

#🎨 STEP 3: Streamlit Sidebar
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

st.sidebar.title("🧠 Multiple Disease Prediction System")

page = st.sidebar.radio(
    "Select Disease",
    ["Parkinson's Prediction", "Kidney Prediction", "Liver Prediction"]
)

#🧪 STEP 4: Parkinson’s Prediction Page
if page == "Parkinson's Prediction":
    st.title("🧪Parkinson's Disease Prediction using ML")

    cols = st.columns(5)
    user_input = {}

    for i, feature in enumerate(parkinsons_features):
        user_input[feature] = cols[i % 5].number_input(feature, format="%.5f")

    if st.button("Parkinson's Test Result"):
        input_df = pd.DataFrame([user_input])
        prediction = parkinsons_model.predict(input_df)[0]
        prob = parkinsons_model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Parkinson's Detected (Risk: {prob:.2%})")
        else:
            st.success(f"✅ No Parkinson's Detected (Risk: {prob:.2%})")

#🩺 STEP 5: Kidney Prediction Page
elif page == "Kidney Prediction":
    st.title("🩺Kidney Disease Prediction using ML")

    cols = st.columns(4)
    user_input = {}

    for i, feature in enumerate(kidney_features):
        user_input[feature] = cols[i % 4].number_input(feature)

    if st.button("Kidney Test Result"):
        input_df = pd.DataFrame([user_input])
        prediction = kidney_model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ Kidney Disease Detected")
        else:
            st.success("✅ No Kidney Disease Detected")

#🍺 STEP 6: Liver Prediction Page
elif page == "Liver Prediction":
    st.title("🍺Liver Disease Prediction using ML")

    cols = st.columns(3)
    user_input = {}

    for i, feature in enumerate(liver_features):
        #print gender as radio button in streamlit
        if feature == "Gender":
            user_input[feature] = st.radio(feature, ("Male", "Female"))
        user_input[feature] = cols[i % 3].number_input(feature)
        
    if st.button("Liver Test Result"):

        input_df = pd.DataFrame([user_input])
        prediction = liver_model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ Liver Disease Detected")
        else:
            st.success("✅ No Liver Disease Detected")
