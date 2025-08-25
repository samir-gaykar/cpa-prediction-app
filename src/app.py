import streamlit as st
import requests

st.set_page_config(page_title="CPA Prediction", layout="centered")

st.title("📊 CPA Prediction App")
st.write("Enter details below to get a CPA prediction:")

with st.form("prediction_form"):
    date = st.text_input("Date (YYYY-MM-DD)")
    category_id = st.text_input("Category ID")
    industry = st.text_input("Industry")
    publisher = st.text_input("Publisher")
    market_id = st.text_input("Market ID")

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "date": date,
        "category_id": category_id,
        "industry": industry,
        "publisher": publisher,
        "market_id": market_id,
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted CPA: {result}")
        else:
            st.error("Error from API")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")
