# File: project/app.py
# Streamlit App untuk klasifikasi gambar uang

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# ================================
# Load model
# ================================
MODEL_PATH = 'model/model.h5'
model = load_model(MODEL_PATH)

# Kelas yang sesuai dengan folder dataset
class_names = os.listdir('data/Uang Baru')
class_names.sort()  # urutkan untuk konsistensi

# ================================
# Fungsi Prediksi
# ================================
def predict(image: Image.Image):
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# ================================
# UI Streamlit
# ================================
st.title("Klasifikasi Gambar Uang")
st.write("Upload gambar uang untuk diklasifikasikan berdasarkan model CNN.")

uploaded_file = st.file_uploader("Pilih gambar uang", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diupload', use_column_width=True)
    
    predicted_class, confidence = predict(image)
    st.markdown(f"### Prediksi: `{predicted_class}`")
    st.markdown(f"**Tingkat Keyakinan:** {confidence * 100:.2f}%")
