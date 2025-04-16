# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo entrenado
model = load_model('modelo_neumonia.h5')

# Configurar la interfaz
st.title("🔍 Detector de Neumonía en Rayos X")
st.markdown("Sube una radiografía de tórax para analizar si hay neumonía.")

# Widget para subir la imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, caption="Imagen subida", width=300)
    
    # Preprocesar la imagen
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Hacer la predicción
    prediction = model.predict(img_array)[0][0]
    
    # Mostrar resultados
    st.subheader("Resultado:")
    if prediction > 0.5:
        st.error("⚠️ **Posible neumonía detectada** (probabilidad: {:.2f}%)".format(prediction * 100))
    else:
        st.success("✅ **No se detectó neumonía** (probabilidad: {:.2f}%)".format((1 - prediction) * 100))