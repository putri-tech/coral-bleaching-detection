import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Deteksi Coral Bleaching | Kelompok 5",
    page_icon="🪸",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = None

if os.path.exists("coral_model.keras"):
    model = tf.keras.models.load_model(
        "coral_model.keras",
        compile=False
    )

# ==========================================
# PREPROCESSING
# ==========================================

IMG_SIZE = 224

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ==========================================
# HEADER
# ==========================================

st.title("🪸 Aplikasi Deteksi Dini Pemutihan Terumbu Karang")

st.subheader(
    "Metode Convolutional Neural Network Berbasis Web"
)

st.caption(
    "Proyek Tugas Besar Mata Kuliah Pengolahan Citra Digital — Teknik Informatika UMRAH"
)

st.markdown("---")

# ==========================================
# INPUT GAMBAR
# ==========================================

st.markdown("### 📸 Pilih Metode Input Citra")

tab1, tab2 = st.tabs(
    [
        "📁 Unggah Berkas Gambar",
        "📷 Ambil Foto via Kamera"
    ]
)

uploaded_file = None

with tab1:

    file_input = st.file_uploader(
        "Pilih file gambar terumbu karang",
        type=["jpg", "jpeg", "png"]
    )

    if file_input is not None:
        uploaded_file = file_input

with tab2:

    camera_input = st.camera_input(
        "Posisikan objek tepat di depan kamera"
    )

    if camera_input is not None:
        uploaded_file = camera_input

# ==========================================
# PREDIKSI
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Citra Terumbu Karang",
        use_container_width=True
    )

    st.success("✔ Berkas citra berhasil dimuat")

    if st.button(
        "Jalankan Klasifikasi Citra",
        type="primary"
    ):

        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis")

        if model is None:

            st.error(
                "File coral_model.keras atau coral_model.h5 tidak ditemukan."
            )

        else:

            with st.spinner("Menganalisis gambar..."):

                img_tensor = preprocess_image(image)

                prediction = model.predict(
                    img_tensor,
                    verbose=0
                )

                raw_score = float(prediction[0][0])

                st.info(
                    f"Raw Score Model : {raw_score:.4f}"
                )

                if raw_score > 0.5:

                    hasil = "Healthy Coral (Sehat)"
                    confidence = raw_score * 100

                    st.success(
                        f"### KONDISI AMAN: {hasil}"
                    )

                else:

                    hasil = "Bleached Coral (Memutih/Sakit)"
                    confidence = (1 - raw_score) * 100

                    st.error(
                        f"### KONDISI KRITIS: {hasil}"
                    )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Status Klasifikasi",
                        "Selesai ✔"
                    )

                with col2:
                    st.metric(
                        "Confidence Score",
                        f"{confidence:.2f}%"
                    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#888888;font-size:0.85em;">
        <strong>Dibuat oleh Kelompok 5 - Teknik Informatika UMRAH</strong><br>
        Syawal Rizal Utama | Meyza Zaharanie |
        Putri Ramadhanti | Zony Fatma Mulia |
        Tommy Susanto | Rusydi Ardani |
        Rani Nadia Sihombing | Luvita Septiana Putri
    </div>
    """,
    unsafe_allow_html=True
)