# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide"
)

def main():
    st.title("🔬 Skin Lesion Classification System")
    st.markdown("### Deep Learning-based Dermoscopic Image Analysis")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Upload & Predict", "Model Comparison", 
         "Performance Metrics", "About"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Upload & Predict":
        show_prediction_page()
    elif page == "Model Comparison":
        show_comparison_page()
    elif page == "Performance Metrics":
        show_metrics_page()
    elif page == "About":
        show_about_page()

def show_prediction_page():
    """
    Main prediction interface
    """
    st.header("Upload Dermoscopic Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Model selection
            model_choice = st.selectbox(
                "Select Model",
                ["Basic ResNet50", "Fine-tuned ResNet50", 
                 "Basic EfficientNet", "Fine-tuned EfficientNet"]
            )
            
            if st.button("Classify"):
                with st.spinner("Analyzing..."):
                    # Preprocess image
                    processed_img = preprocess_image(image)
                    
                    # Load selected model
                    model = load_model(model_choice)
                    
                    # Make prediction
                    predictions = model.predict(processed_img)
                    
                    # Display results
                    display_results(predictions)
                    
                    # Show Grad-CAM
                    st.subheader("Attention Heatmap")
                    heatmap = generate_gradcam(model, processed_img)
                    st.image(heatmap, use_column_width=True)