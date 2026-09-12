import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Risiko Jantung", page_icon="❤️", layout="centered")

# Load model
@st.cache_resource
def load_model():
    payload = joblib.load("model.joblib")
    return payload["pipeline"], payload["features"]


pipeline, features = load_model()

st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.write(
    "Masukkan data medis pasien di bawah ini, lalu klik **Prediksi** untuk "
    "mengetahui apakah pasien berisiko terkena penyakit jantung."
)

st.divider()

# Form input
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Jenis Kelamin", options=[("Laki-laki", 1), ("Perempuan", 0)], format_func=lambda x: x[0])
        cp = st.selectbox(
            "Tipe Nyeri Dada (cp)",
            options=[
                ("Typical angina", 0),
                ("Atypical angina", 1),
                ("Non-anginal pain", 2),
                ("Asymptomatic", 3),
            ],
            format_func=lambda x: x[0],
        )
        trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Kolesterol Serum (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", options=[("Ya", 1), ("Tidak", 0)], format_func=lambda x: x[0])
        restecg = st.selectbox(
            "Hasil EKG Istirahat (restecg)",
            options=[("Normal", 0), ("ST-T abnormal", 1), ("LV hypertrophy", 2)],
            format_func=lambda x: x[0],
        )

    with col2:
        thalach = st.number_input("Detak Jantung Maksimum", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Nyeri Dada Saat Olahraga?", options=[("Ya", 1), ("Tidak", 0)], format_func=lambda x: x[0])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope ST Segment",
            options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
            format_func=lambda x: x[0],
        )
        ca = st.selectbox("Jumlah Pembuluh Darah Utama (ca)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)],
            format_func=lambda x: x[0],
        )

    submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)

# Prediction
if submitted:
    input_dict = {
        "age": age,
        "sex": sex[1],
        "cp": cp[1],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs[1],
        "restecg": restecg[1],
        "thalach": thalach,
        "exang": exang[1],
        "oldpeak": oldpeak,
        "slope": slope[1],
        "ca": ca,
        "thal": thal[1],
    }

    # Susun input sesuai urutan fitur yang dipakai saat training
    input_df = pd.DataFrame([[input_dict[feat] for feat in features]], columns=features)

    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ Pasien **berisiko** terkena penyakit jantung.")
    else:
        st.success(f"✅ Pasien **tidak berisiko** / cenderung sehat.")

    st.metric("Probabilitas Risiko", f"{proba * 100:.1f}%")
    st.progress(min(max(proba, 0.0), 1.0))

    with st.expander("Lihat data input yang digunakan"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Model: Random Forest Classifier (scikit-learn). "
    "Hasil prediksi ini bersifat indikatif dan bukan pengganti diagnosis medis profesional."
)
