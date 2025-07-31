
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load pre-trained model
model = load_model("model.h5")

# Class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

st.title("📱 AI Image Classifier (Mobile-Friendly)")
st.write("Upload a 32x32 image and get prediction")

uploaded = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded:
    image = Image.open(uploaded).resize((32, 32))
    st.image(image, caption="Uploaded Image", width=150)

    img = np.array(image) / 255.0
    img = img.reshape(1, 32, 32, 3)

    prediction = model.predict(img)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {pred_class} ({confidence:.2f}%)")
