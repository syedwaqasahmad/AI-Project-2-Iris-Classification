import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris KNN Classifier", page_icon="🌼")
st.title("Iris Species Classifier")
st.write("Enter the flower measurements to predict its species using KNN.")

model_path = "models/classification_model.pkl"
if not os.path.exists(model_path):
    st.error("Trained model not found. From the project root run: python -m src.train_model")
    st.stop()

bundle = joblib.load(model_path)
c1, c2 = st.columns(2)
with c1:
    sepal_length = st.number_input("Sepal length (cm)", 0.1, 10.0, 5.1, 0.1)
    petal_length = st.number_input("Petal length (cm)", 0.1, 10.0, 1.4, 0.1)
with c2:
    sepal_width = st.number_input("Sepal width (cm)", 0.1, 10.0, 3.5, 0.1)
    petal_width = st.number_input("Petal width (cm)", 0.1, 10.0, 0.2, 0.1)

if st.button("Predict species"):
    sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=bundle["features"])
    pred = bundle["model"].predict(bundle["scaler"].transform(sample))[0]
    st.success(f"Predicted species: **{pred.title()}**")
    if hasattr(bundle["model"], "predict_proba"):
        probs = bundle["model"].predict_proba(bundle["scaler"].transform(sample))[0]
        st.bar_chart(pd.DataFrame({"Probability": probs}, index=bundle["model"].classes_))
