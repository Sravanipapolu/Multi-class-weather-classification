import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import os
import warnings
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Weather Classification",
    page_icon="🌦",
    layout="wide"
)

# Custom CSS with nice background and enhanced styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        margin: 2rem 0;
        backdrop-filter: blur(4px);
    }
    .upload-section {
        padding: 2rem;
        border: 2px dashed #dee2e6;
        border-radius: 15px;
        margin: 2rem 0;
        background-color: rgba(255, 255, 255, 0.8);
    }
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }
    .step-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Hide warnings
warnings.filterwarnings("ignore")

# Load model
@st.cache_resource
def load_model():
    file_id = "1U1vBcEG9h8BI59tN_JymOXHRxHsBaLMS"
    output_path = "custom_cnn_model.keras"
    if not os.path.exists(output_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    return tf.keras.models.load_model(output_path)

model = load_model()
class_labels = ["Cloudy", "Rain", "Shine", "Sunrise"]
EXPECTED_SHAPE = (128, 128, 3)

# Sidebar with steps and information
with st.sidebar:
    st.header("📝 Process Steps")
    
    # Step 1
    st.markdown("""
    <div class='step-box'>
        <h3>Step 1: Upload Image</h3>
        <p>Upload a weather image in JPEG format only.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 2
    st.markdown("""
    <div class='step-box'>
        <h3>Step 2: Image Processing</h3>
        <p>The system will automatically process and analyze your image.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 3
    st.markdown("""
    <div class='step-box'>
        <h3>Step 3: Classification</h3>
        <p>The model will classify your image into one of these categories:</p>
        <ul>
            <li>☁ <strong>Cloudy</strong>: Overcast sky with dense clouds</li>
            <li>🌧 <strong>Rain</strong>: Rainy conditions with visible precipitation</li>
            <li>☀ <strong>Shine</strong>: Clear sky with bright sunlight</li>
            <li>🌅 <strong>Sunrise</strong>: Early morning sky with warm sunrise hues</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.title("🌦 Weather Classification")
st.markdown("#### Intelligent Weather Scene Analysis Using Custom CNN")

# File upload section
st.markdown("### 📤 Upload Your Weather Image")
with st.container():
    uploaded_file = st.file_uploader(
        "Choose a weather image to classify (JPEG format only)", 
        type=["jpeg"],
        help="Please upload a JPEG image of a weather scene"
    )

# Display and prediction
if uploaded_file:
    # Process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a container for the image and prediction
    with st.container():
        # Display original image with a nice caption
        st.markdown("### 📷 Your Weather Image")
        st.image(image, use_container_width=True, caption="Uploaded Image")
        
        # Make prediction
        with st.spinner("🔄 Analyzing weather conditions..."):
            # Preprocess image
            image_resized = cv2.resize(image, (EXPECTED_SHAPE[0], EXPECTED_SHAPE[1]))
            image_resized = image_resized / 255.0
            image_resized = np.expand_dims(image_resized, axis=0)
            
            try:
                # Get prediction and confidence
                prediction = model.predict(image_resized)
                predicted_class = class_labels[np.argmax(prediction)]
                confidence = float(np.max(prediction)) * 100
                
                # Display results in an attractive box
                st.markdown(
                    f"""
                    <div class='prediction-box'>
                        <h2 style='color: #1a73e8; margin-bottom: 1rem;'>Weather Classification Result</h2>
                        <div style='font-size: 28px; font-weight: bold; color: #2c3e50; margin-bottom: 1rem;'>
                            {predicted_class}
                        </div>
                        <div style='font-size: 20px; color: #1a73e8; margin-bottom: 1rem;'>
                            Confidence: {confidence:.1f}%
                        </div>
                        <div style='color: #666; font-size: 14px;'>
                            Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"⚠️ An error occurred during analysis: {str(e)}")


