# # File: project/app.py
# # Streamlit App untuk klasifikasi gambar uang

# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
# import os

# # ================================
# # Load model
# # ================================
# MODEL_PATH = 'model/model.h5'
# model = load_model(MODEL_PATH)

# # ================================
# # Load class names dari ImageDataGenerator
# # ================================
# temp_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# temp_generator = temp_datagen.flow_from_directory(
#     'dataset/Uang Baru',
#     target_size=(224, 224),
#     batch_size=1,
#     class_mode='categorical',
#     subset='training',
#     shuffle=False
# )
# class_indices = temp_generator.class_indices
# class_names = list(class_indices.keys())

# # ================================
# # Fungsi Prediksi
# # ================================
# def predict(image: Image.Image):
#     img = image.resize((224, 224))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = np.max(prediction)
#     return predicted_class, confidence

# # ================================
# # UI Streamlit
# # ================================
# st.title("Klasifikasi Gambar Uang")
# st.write("Upload gambar uang untuk diklasifikasikan berdasarkan model CNN.")

# uploaded_file = st.file_uploader("Pilih gambar uang", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Gambar yang diupload', use_container_width=True)

#     predicted_class, confidence = predict(image)
#     st.markdown(f"### Prediksi: `{predicted_class}`")
#     st.markdown(f"**Tingkat Keyakinan:** {confidence * 100:.2f}%")

# File: app.py
# Streamlit App untuk klasifikasi gambar uang dari webcam

# File: app.py
# Streamlit App untuk klasifikasi gambar uang (upload atau webcam)

# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

# # ================================
# # Load model
# # ================================
# MODEL_PATH = 'model/model.h5'
# model = load_model(MODEL_PATH)

# # ================================
# # Load class names dari dataset
# # ================================
# temp_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# temp_generator = temp_datagen.flow_from_directory(
#     'dataset/Uang Baru',
#     target_size=(224, 224),
#     batch_size=1,
#     class_mode='categorical',
#     subset='training',
#     shuffle=False
# )
# class_indices = temp_generator.class_indices
# class_names = list(class_indices.keys())

# # ================================
# # Fungsi Prediksi
# # ================================
# def predict(image: Image.Image):
#     img = image.resize((224, 224))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = np.max(prediction)
#     return predicted_class, confidence

# # ================================
# # UI Streamlit
# # ================================
# st.title("Klasifikasi Gambar Uang")
# st.write("Upload gambar uang atau ambil foto menggunakan webcam.")

# # Pilihan input: file atau webcam
# input_option = st.radio("Pilih metode input gambar:", ["Upload File", "Gunakan Webcam"])

# image = None

# if input_option == "Upload File":
#     uploaded_file = st.file_uploader("Upload gambar uang", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert('RGB')

# elif input_option == "Gunakan Webcam":
#     camera_image = st.camera_input("Ambil gambar menggunakan webcam")
#     if camera_image is not None:
#         image = Image.open(camera_image).convert('RGB')

# # Prediksi jika ada gambar
# if image is not None:
#     st.image(image, caption='Gambar yang digunakan', use_container_width=True)
#     with st.spinner("Melakukan prediksi..."):
#         predicted_class, confidence = predict(image)
#     st.markdown(f"### Prediksi: `{predicted_class}`")
#     st.markdown(f"**Tingkat Keyakinan:** {confidence * 100:.2f}%")

# File: app.py
# Streamlit App untuk klasifikasi gambar uang dengan input file/webcam + output suara

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import pyttsx3

# ================================
# Fungsi Suara
# ================================
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ================================
# Load model
# ================================
MODEL_PATH = 'model/model.h5'
model = load_model(MODEL_PATH)

# ================================
# Load class names dari dataset
# ================================
temp_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
temp_generator = temp_datagen.flow_from_directory(
    'dataset/Uang Baru',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    subset='training',
    shuffle=False
)
class_indices = temp_generator.class_indices
class_names = list(class_indices.keys())

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
st.title("🪙 KLASIFIKASI GAMBAR MATA UANG")
st.write("Website ini digunakan untuk membantu disabilitas penyandang low vision / gangguan pada mata. Penyandang disabilitas low vision dapat terbantu jika mereka membuka suatu usaha untuk menyambung hidup mereka.")
st.write("CARA PENGGUNAAN: Upload gambar uang atau ambil foto menggunakan webcam. Sistem akan menebak nominalnya, dan menyebutkannya.")

# Pilihan input
input_option = st.radio("Pilih metode:", ["Upload File", "Gunakan Webcam"])

image = None

if input_option == "Upload File":
    uploaded_file = st.file_uploader("Upload gambar uang", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

elif input_option == "Gunakan Webcam":
    camera_image = st.camera_input("Ambil gambar menggunakan webcam")
    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')

# Prediksi jika ada gambar
if image is not None:
    st.image(image, caption='Gambar yang digunakan', use_container_width=True)
    with st.spinner("Melakukan prediksi..."):
        predicted_class, confidence = predict(image)

    st.markdown(f"### Prediksi: `{predicted_class}`")
    st.markdown(f"**Tingkat Keyakinan:** {confidence * 100:.2f}%")

    # Tombol suara
    if st.button("🔊 Sebutkan Nominal"):
        speak_text(f"Ini adalah uang {predicted_class.replace('_', ' ')}")

