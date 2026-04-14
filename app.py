import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# UI
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

def predict(img):
    img = img.resize((150,150))
    img_array = np.array(img)/255
    img_array = np.expand_dims(img_array, axis=0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if pred > 0.8:
        return "Dog 🐶"
    elif pred < 0.2:
        return "Cat 🐱"
    else:
        return "Other ❓"

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    result = predict(img)
    st.success(f"Prediction: {result}")
