import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("cat_dog_model.h5", compile=False)

st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    img = img.resize((150,150))
    img_array = np.array(img)/255
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.8:
        st.write("Dog 🐶")
    elif pred < 0.2:
        st.write("Cat 🐱")
    else:
        st.write("Other ❓")
