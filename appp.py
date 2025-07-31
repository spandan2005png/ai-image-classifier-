import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Set up page title and header
st.set_page_config(page_title="AI Image Classifier", layout="centered")
st.title("🔍 AI Image Classifier by Spandan Chakraborty")
st.markdown("Upload an image (JPG/PNG) and let AI predict its category using a deep learning model.")

# Load model
@st.cache_resource
def load_classifier():
    return load_model("sample_model.h5")

model = load_classifier()

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    image = image.resize((32, 32))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    with st.spinner('Classifying...'):
        predictions = model.predict(image_array)[0]
        top_indices = predictions.argsort()[-3:][::-1]

    st.subheader("🔎 Top Predictions:")
    for i in top_indices:
        st.write(f"**{class_names[i]}** — {predictions[i]*100:.2f}%")
