import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "tomato_mosaic_classifier.keras",
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v3.preprocess_input},
        safe_mode=False
    )

def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "Tomato Mosaic Virus" if prob >= 0.5 else "Healthy"
    return label, (1-prob)*100, prob*100

st.title("🍅 Tomato Leaf Health Classifier")
st.write("Upload a tomato leaf photo to check for Mosaic Virus.")

model = load_model()
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)
    label, healthy_pct, virus_pct = predict(model, img)
    st.write(f"**Prediction:** {label}")
    st.progress(int(healthy_pct), text=f"Healthy: {healthy_pct:.1f}%")
    st.progress(int(virus_pct), text=f"Mosaic Virus: {virus_pct:.1f}%")
