import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="HeartPredict",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM STYLE
# ==========================================================

st.markdown("""
<style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    /* Section title */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    /* Information box */
    .info-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f1f5f9;
        border-left: 5px solid #ef4444;
        margin-bottom: 20px;
    }

    /* Result */
    .result-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .risk-high {
        padding: 20px;
        border-radius: 14px;
        background-color: #fee2e2;
        border-left: 6px solid #dc2626;
        margin: 15px 0;
    }

    .risk-low {
        padding: 20px;
        border-radius: 14px;
        background-color: #dcfce7;
        border-left: 6px solid #16a34a;
        margin: 15px 0;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 25px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    payload = joblib.load("model.joblib")
    return payload["pipeline"], payload["features"]


pipeline, features = load_model()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## ❤️ HeartPredict")

    st.caption(
        "Sistem Prediksi Risiko Penyakit Jantung"
    )

    st.divider()

    st.markdown("### 📌 Tentang Aplikasi")

    st.write(
        "HeartPredict merupakan aplikasi berbasis Machine Learning "
        "yang digunakan untuk memperkirakan risiko penyakit jantung "
        "berdasarkan beberapa parameter kesehatan pasien."
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.info(
        "Random Forest Classifier"
    )

    st.write(
        f"Jumlah fitur yang digunakan: **{len(features)}**"
    )

    st.divider()

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    st.markdown("### 📊 Feature Importance")

    try:

        classifier = pipeline.named_steps["classifier"]

        importance = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        chart_data = importance_df.set_index("Feature")

        st.bar_chart(
            chart_data,
            horizontal=True
        )

        most_important = importance_df.iloc[0]

        st.caption(
            f"Fitur paling berpengaruh adalah "
            f"**{most_important['Feature']}** "
            f"dengan importance "
            f"**{most_important['Importance'] * 100:.1f}%**."
        )

    except Exception:

        st.warning(
            "Feature importance tidak tersedia."
        )

    st.divider()

    st.caption(
        "Hasil prediksi bersifat indikatif dan bukan "
        "pengganti diagnosis medis profesional."
    )


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">❤️ HeartPredict</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Prediksi Risiko Penyakit Jantung Menggunakan Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


st.markdown("""
<div class="info-box">

<b>🩺 Cara menggunakan aplikasi</b><br><br>

Masukkan informasi kesehatan pasien pada form di bawah.
Pastikan seluruh data telah diisi dengan benar, kemudian
tekan tombol <b>Prediksi Risiko</b> untuk mendapatkan hasil.

</div>
""", unsafe_allow_html=True)


# ==========================================================
# DATA PASIEN
# ==========================================================

st.markdown(
    '<div class="section-title">👤 Data Pasien</div>',
    unsafe_allow_html=True
)

st.caption(
    "Masukkan 13 parameter kesehatan yang dibutuhkan oleh model."
)


# ==========================================================
# FORM
# ==========================================================

with st.form("input_form"):

    col1, col2 = st.columns(2, gap="large")


    # ======================================================
    # LEFT COLUMN
    # ======================================================

    with col1:

        st.markdown("#### 👤 Informasi Dasar")

        age = st.number_input(
            "Usia (tahun)",
            min_value=1,
            max_value=120,
            value=50,
            help="Usia pasien dalam tahun."
        )

        sex = st.selectbox(
            "Jenis Kelamin",
            options=[
                ("Laki-laki", 1),
                ("Perempuan", 0)
            ],
            format_func=lambda x: x[0]
        )

        st.markdown("#### 🫀 Pemeriksaan Jantung")

        cp = st.selectbox(
            "Tipe Nyeri Dada",
            options=[
                ("Typical angina", 0),
                ("Atypical angina", 1),
                ("Non-anginal pain", 2),
                ("Asymptomatic", 3)
            ],
            format_func=lambda x: x[0],
            help="Jenis nyeri dada yang dialami pasien."
        )

        trestbps = st.number_input(
            "Tekanan Darah Istirahat (mm Hg)",
            min_value=50,
            max_value=250,
            value=120
        )

        chol = st.number_input(
            "Kolesterol Serum (mg/dl)",
            min_value=100,
            max_value=600,
            value=200
        )

        fbs = st.selectbox(
            "Gula Darah Puasa > 120 mg/dl",
            options=[
                ("Ya", 1),
                ("Tidak", 0)
            ],
            format_func=lambda x: x[0]
        )

        restecg = st.selectbox(
            "Hasil EKG Istirahat",
            options=[
                ("Normal", 0),
                ("ST-T abnormal", 1),
                ("LV hypertrophy", 2)
            ],
            format_func=lambda x: x[0]
        )


    # ======================================================
    # RIGHT COLUMN
    # ======================================================

    with col2:

        st.markdown("#### 🏃 Pemeriksaan Aktivitas")

        thalach = st.number_input(
            "Detak Jantung Maksimum",
            min_value=50,
            max_value=250,
            value=150,
            help="Maximum heart rate achieved."
        )

        exang = st.selectbox(
            "Nyeri Dada Saat Olahraga?",
            options=[
                ("Ya", 1),
                ("Tidak", 0)
            ],
            format_func=lambda x: x[0]
        )

        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope ST Segment",
            options=[
                ("Upsloping", 0),
                ("Flat", 1),
                ("Downsloping", 2)
            ],
            format_func=lambda x: x[0]
        )

        st.markdown("#### 🔬 Pemeriksaan Pembuluh Darah")

        ca = st.selectbox(
            "Jumlah Pembuluh Darah Utama (ca)",
            options=[0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[
                ("Normal", 1),
                ("Fixed defect", 2),
                ("Reversible defect", 3)
            ],
            format_func=lambda x: x[0]
        )


    # ======================================================
    # PREDICTION BUTTON
    # ======================================================

    st.write("")

    submitted = st.form_submit_button(
        "🔍  PREDIKSI RISIKO",
        use_container_width=True
    )


# ==========================================================
# PREDICTION
# ==========================================================

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

        "thal": thal[1]
    }


    # ======================================================
    # DATAFRAME
    # ======================================================

    input_df = pd.DataFrame(
        [[input_dict[feat] for feat in features]],
        columns=features
    )


    # ======================================================
    # MODEL PREDICTION
    # ======================================================

    prediction = pipeline.predict(input_df)[0]

    proba = pipeline.predict_proba(input_df)[0][1]


    # ======================================================
    # RESULT
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="result-title">📊 Hasil Prediksi</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.markdown(
            """
            <div class="risk-high">

            <h3>⚠️ Risiko Terdeteksi</h3>

            <p>
            Berdasarkan pola data yang dipelajari model,
            pasien diprediksi memiliki risiko penyakit jantung.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="risk-low">

            <h3>✅ Risiko Tidak Terdeteksi</h3>

            <p>
            Berdasarkan pola data yang dipelajari model,
            pasien diprediksi memiliki risiko yang lebih rendah.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ======================================================
    # PROBABILITY
    # ======================================================

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "🎯 Probabilitas Risiko",
            f"{proba * 100:.1f}%"
        )


    with result_col2:

        if proba < 0.3:

            risk_level = "Rendah"

        elif proba < 0.7:

            risk_level = "Sedang"

        else:

            risk_level = "Tinggi"


        st.metric(
            "📌 Tingkat Risiko",
            risk_level
        )


    st.write("")

    st.progress(
        min(max(float(proba), 0.0), 1.0)
    )


    # ======================================================
    # INPUT DETAIL
    # ======================================================

    with st.expander("📋 Lihat Data Input yang Digunakan"):

        st.dataframe(
            input_df,
            use_container_width=True
        )


# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

st.divider()

st.markdown(
    '<div class="section-title">ℹ️ Informasi Sistem</div>',
    unsafe_allow_html=True
)


info1, info2, info3 = st.columns(3)


with info1:

    st.metric(
        "Jumlah Fitur",
        "13"
    )


with info2:

    st.metric(
        "Model",
        "Random Forest"
    )


with info3:

    st.metric(
        "Jenis Output",
        "Risiko Jantung"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">

    ❤️ <b>HeartPredict</b><br>
    Machine Learning - Prediksi Risiko Penyakit Jantung<br><br>

    Hasil prediksi bersifat indikatif dan bukan pengganti
    diagnosis medis profesional.

    </div>
    """,
    unsafe_allow_html=True
)