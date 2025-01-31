import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import os

# Google Drive File ID of Custom CNN Model
file_id = "1U1vBcEG9h8BI59tN_JymOXHRxHsBaLMS"
output_path = "custom_cnn_model.keras"

# Download the Custom CNN model from Google Drive if it doesn't exist
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Load the Custom CNN model
model = tf.keras.models.load_model(output_path)

# Define class labels
class_labels = ["Cloudy", "Rain", "Shine", "Sunrise"]

# Streamlit UI
st.title("🌦 Multi-Class Weather Classification - Custom CNN")
st.write("Upload an image to classify its weather condition!")

# Description of classes
st.markdown("""
### 🌍 **Classes in the Model**
- ☁ **Cloudy**: Overcast sky with dense clouds.  
- 🌧 **Rain**: Rainy conditions with visible precipitation.  
- ☀ **Shine**: Clear sky with bright sunlight.  
- 🌅 **Sunrise**: Early morning sky with warm sunrise hues.  
""")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    # Read and preprocess the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256)) / 255.0  # Resize and normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display results
    st.image(image, caption=f"Predicted: {predicted_class}", use_column_width=True)
    st.write(f"### 🔍 Prediction: **{predicted_class}**")

