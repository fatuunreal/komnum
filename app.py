import streamlit as st
import pandas as pd
import pickle

# Load model dan scaler
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.set_page_config(page_title="Prediksi Stunting", layout="centered")
st.title("🧒 Prediksi Stunting pada Anak")
st.write("Masukkan data anak untuk memprediksi apakah termasuk kategori stunting atau tidak.")

# Form input pengguna
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age = st.number_input("Usia (bulan)", min_value=0.0)
birth_weight = st.number_input("Berat Lahir (kg)", min_value=0.0)
birth_length = st.number_input("Panjang Lahir (cm)", min_value=0.0)
body_weight = st.number_input("Berat Badan Saat Ini (kg)", min_value=0.0)
body_length = st.number_input("Panjang Badan Saat Ini (cm)", min_value=0.0)
breastfeeding = st.selectbox("ASI Eksklusif?", ["Ya", "Tidak"])

# Konversi input ke DataFrame (dengan urutan yang BENAR!)
input_df = pd.DataFrame([{
    "Gender": 1 if gender == "Laki-laki" else 0,
    "Age": age,
    "Birth Weight": birth_weight,
    "Birth Length": birth_length,
    "Body Weight": body_weight,
    "Body Length": body_length,
    "Breastfeeding": 1 if breastfeeding == "Ya" else 0
}])

# Scaling
input_scaled = scaler.transform(input_df)

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_scaled)
    label = "Stunting" if prediction[0] == 1 else "Normal"
    st.success(f"Hasil Prediksi: **{label}**")
