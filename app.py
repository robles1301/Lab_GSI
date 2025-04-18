import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuración
st.set_page_config(page_title="Detector de Neumonía", layout="wide")
model = load_model('modelo_neumonia.h5')

# Interfaz
st.title("🩺 Detector de Neumonía en Radiografías")
uploaded_file = st.file_uploader("Sube una radiografía...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Preprocesamiento
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predicción
        prob_neumonia = model.predict(img_array, verbose=0)[0][0]
        prob_normal = 1 - prob_neumonia
        
        # Resultados
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Imagen analizada", width=300)
            if prob_neumonia > 0.5:
                st.error(f"🚨 **Neumonía detectada** (Probabilidad: {prob_neumonia:.2%})")
            else:
                st.success(f"✅ **Sin neumonía** (Probabilidad: {prob_normal:.2%})")
        
        with col2:
            # Gráfico de probabilidades
            fig, ax = plt.subplots()
            ax.barh(['Normal', 'Neumonía'], [prob_normal, prob_neumonia], color=['green', 'red'])
            ax.set_xlim(0, 1)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---\n**Nota:** Consulte a un profesional médico para validar los resultados.")