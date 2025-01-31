import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile

# Load the trained model
model = tf.keras.models.load_model("saved_models/resnet_model.keras")  # Updated format

# Define class labels
class_labels = ["Cloudy", "Rain", "Shine", "Sunrise"]

# Streamlit UI
st.title("🌦 Multi-Class Weather Classification")
st.write("Upload an image to classify its weather condition!")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Read and preprocess the image
    image = cv2.imread(temp_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256)) / 255.0  # Resize and normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display results
    st.image(uploaded_file, caption=f"Predicted: {predicted_class}", use_column_width=True)
    st.write(f"### 🔍 Prediction: **{predicted_class}**")
