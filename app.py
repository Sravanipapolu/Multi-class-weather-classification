import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import os
import warnings



# Hide all warnings
warnings.filterwarnings("ignore")
st.image("app_preview.png", caption="🔍 App Preview", use_container_width=True)


# Google Drive File ID of Custom CNN Model
file_id = "1U1vBcEG9h8BI59tN_JymOXHRxHsBaLMS"  # Replace with your actual file ID
output_path = "custom_cnn_model.keras"

# Download the Custom CNN model from Google Drive if it doesn't exist
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Load the Custom CNN model
model = tf.keras.models.load_model(output_path)

# Define class labels
class_labels = ["Cloudy", "Rain", "Shine", "Sunrise"]

# Define the expected input shape of the model
EXPECTED_SHAPE = (128, 128, 3)  # Update this based on model.input_shape

# 🎯 Add Title & Your Name
st.title("🌦 Multi-Class Weather Classification - Custom CNN")
#st.markdown("### Developed by **Sravani Papolu**")

# 📌 Classes in the Model
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
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read image using OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Display the uploaded image (only once)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)  # FIXED

    # Resize and preprocess the image for the model
    image_resized = cv2.resize(image, (EXPECTED_SHAPE[0], EXPECTED_SHAPE[1]))  # Resize to 128x128
    image_resized = image_resized / 255.0  # Normalize pixel values (0 to 1)
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension to match model input

    # Debugging: Print Image Shape (Hidden)
    # st.write(f"🖼️ **Processed Image Shape:** {image_resized.shape}")

    # Ensure correct shape before prediction
    if image_resized.shape[1:] != EXPECTED_SHAPE:
        st.error(f"❌ Error: Image shape {image_resized.shape[1:]} does not match expected shape {EXPECTED_SHAPE}. Please upload a valid image.")
    else:
        try:
            # Make prediction
            prediction = model.predict(image_resized)
            predicted_class = class_labels[np.argmax(prediction)]

            # Display Prediction
            st.markdown(f"### 🔍 Prediction: **{predicted_class}**")

        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")
