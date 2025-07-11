import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

#  Custom CSS for Black Background
st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: white;
    }
    .stApp {
        background-color: #000000;
    }
    h1, h2, h3, h4, h5, h6, .stTextInput label, .stFileUploader label {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

#  Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Weather_model.h5')
    return model

model = load_model()

#  App Title and Instructions
st.title(" Multiclass Weather Prediction (CNN)")
st.write("Upload a weather image and the model will classify it as **Cloudy**, **Rain**, **Shine**, or **Sunrise**.")

#  Upload image
file = st.file_uploader(" Choose a weather image", type=["jpg", "jpeg", "png"])

#  Prediction function
def import_and_predict(image_data, model):
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

#  Display result
if file is None:
    st.info(" Upload an image to start prediction.")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    prediction = import_and_predict(image, model)
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f" **Prediction:** {predicted_class}")
    st.info(f" **Model Confidence:** {confidence:.2f}%")
