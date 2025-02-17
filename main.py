import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import plotly.graph_objects as go
import requests
from io import BytesIO
import pyperclip
from io import BytesIO
import base64

# Must be the first Streamlit command
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tumor information dictionary
TUMOR_INFO = {
    "Glioma": {
        "title": "Glioma Tumor Detected",
        "description": """
        ### What is a Glioma?
        Gliomas are tumors that start in the glial cells of the brain or spine. They're one of the most common types of primary brain tumors.

        #### Key Points:
        - Usually begins in the brain or spinal cord
        - Can affect different types of glial cells
        - May be slow-growing or aggressive
        - Common symptoms include headaches, seizures, and vision problems

        #### Treatment Options:
        1. Surgery
        2. Radiation therapy
        3. Chemotherapy
        4. Targeted drug therapy
        """,
        "color": "#FF6B6B"
    },
    "Meningioma": {
        "title": "Meningioma Tumor Detected",
        "description": """
        ### What is a Meningioma?
        Meningiomas are tumors that arise from the meninges - the membranes that surround your brain and spinal cord.

        #### Key Points:
        - Most are benign (not cancerous)
        - Slow-growing in most cases
        - More common in women
        - Often discovered incidentally

        #### Treatment Options:
        1. Observation (for small, slow-growing tumors)
        2. Surgery
        3. Radiation therapy
        4. Regular monitoring
        """,
        "color": "#4ECDC4"
    },
    "Pituitary": {
        "title": "Pituitary Tumorstream Detected",
        "description": """
        ### What is a Pituitary Tumor?
        A pituitary tumor is a non-cancerous growth in the pituitary gland, a small gland at the base of the brain that regulates hormone production.

        These tumors can cause hormone imbalances, leading to various symptoms. Additionally, they may press on the optic nerve, affecting vision.

        #### Key Points To Note:
        - Can affect hormone production
        - Usually benign
        - May cause vision problems
        - Can affect metabolism and growth

        #### Treatment Options: 
        1. Medication to control hormone production
        2. Surgery
        3. Radiation therapy
        4. Regular hormone level monitoring
        """,
        "color": "#FFD93D"
    },
    "No-tumor": {
        "title": "No Tumor Detected! 😊",
        "description": """
        ### Great News! 
        🎉 Your scan appears to be clear of any tumors! 

        #### Maintaining Brain Health:
        - Regular exercise
        - Healthy diet
        - Adequate sleep
        - Mental stimulation
        - Regular check-ups

        Remember to maintain a healthy lifestyle and consult with healthcare professionals for regular check-ups! 
        Stay healthy! 🌟
        """,
        "color": "#95D5B2"
    }
}


def validate_mri_image(image):
    """
    Validate if the image appears to be an MRI scan
    Returns: (bool, str) - (is_valid, message)
    """
    try:
        # Convert to grayscale for analysis
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Check image characteristics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # MRI scans typically have:
        # - High contrast (high standard deviation)
        # - Specific intensity range
        # - Mostly black background with bright regions

        if std_intensity < 30:  # Low contrast
            return False, "This image appears to have low contrast. MRI scans typically have high contrast between tissues."

        if mean_intensity > 200 or mean_intensity < 20:  # Too bright or too dark
            return False, "This image is too bright or too dark to be a typical MRI scan."

        # Check for circular/oval shape typical in brain MRIs
        height, width = gray.shape
        if abs(height / width - 1) > 0.5:  # Not roughly square
            return False, "This image's dimensions don't match typical MRI scan proportions."

        return True, "Image appears to be a valid MRI scan."

    except Exception as e:
        return False, f"Error validating image: {str(e)}"


def preprocess_image(image):
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Ensure image is RGB
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Resize image to 150x150 to match the training size
        img_resized = cv2.resize(img_array, (150, 150))

        # Normalize pixel values
        img_normalized = img_resized / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch

    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


@st.cache_resource
def load_model_file():
    try:
        model = load_model('brain_tumor_detection_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Main layout
with st.sidebar:
    st.image(
        "https://cdn.pixabay.com/animation/2024/02/08/10/06/10-06-13-48_512.gif")
    st.title("🧠 Brain Tumor Classification AI")
    st.markdown("""
    ### How to use:
    1. Upload an MRI scan image
    2. Paste from clipboard
    3. Get instant analysis

    ### About
    This AI-powered tool helps detect:
    - Glioma tumors
    - Meningioma tumors
    - Pituitary tumors


    """)

# Main content
st.markdown("""
    <div class="main">
        <h1 style="text-align: center;"> Neuro-Check </h1>
        <p style="text-align: center; font-size: 1.2em;">Upload or paste your brain MRI scan for instant analysis</p>
    </div>
""", unsafe_allow_html=True)

# Create columns for upload methods
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="uploadZone">
            <h3>📤 Upload Image</h3>
            <p>Drag and drop or click to upload</p>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown("""
        <div class="uploadZone">
            <h3>📋 Paste from Clipboard</h3>
            <p>Use Ctrl+V or Cmd+V to paste</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Paste from Clipboard"):
        image = get_image_from_clipboard()
        if image:
            st.session_state.image = image
            st.success("Image pasted successfully!")
        else:
            st.error("No image found in clipboard!")

# Process uploaded file
if uploaded_file is not None:
    st.session_state.image = Image.open(uploaded_file)

# Display and analyze image
if hasattr(st.session_state, 'image'):
    st.image(st.session_state.image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image", key="analyze"):
        with st.spinner("Analyzing image..."):
            try:
                model = load_model_file()
                processed_image = preprocess_image(st.session_state.image)

                if processed_image is not None and model is not None:
                    predictions = model.predict(processed_image)
                    classes = ["Glioma", "Meningioma", "No-tumor", "Pituitary"]
                    predicted_class = classes[np.argmax(predictions[0])]
                    confidence = float(np.max(predictions[0]) * 100)

                    tumor_info = TUMOR_INFO[predicted_class]

                    st.markdown(f"""
                        <div class="results-container">
                            <h2 style="color: {tumor_info['color']};">{tumor_info['title']}</h2>
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(tumor_info["description"])
                        st.metric("Confidence", f"{confidence:.1f}%")

                    with col2:
                        fig = go.Figure(data=[go.Bar(
                            x=[p * 100 for p in predictions[0]],
                            y=classes,
                            orientation='h',
                            marker_color=[tumor_info["color"] if i == predicted_class else '#E5E7EB'
                                          for i in classes]
                        )])

                        fig.update_layout(
                            title="Prediction Probabilities",
                            xaxis_title="Probability (%)",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=16)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    if confidence < 70:
                        st.warning("""
                            ⚠ Low confidence prediction. Consider:
                            - Using a clearer MRI scan
                            - Ensuring proper image orientation
                            - Consulting a medical professional
                        """)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

st.markdown("---")
st.markdown("### Important Notes")
st.write("""
⚠ This model is designed specifically for brain MRI scans and will not work correctly with:
- Regular photographs
- Other medical images
- Screenshots or diagrams
- Modified or edited images

For accurate results, please ensure you upload:
- Actual brain MRI scan images
- Clear, medical-quality scans
- Unmodified original images
- Images similar to the examples shown above
""")

st.markdown("### About")
st.write("""
This application uses a Convolutional Neural Network (CNN) trained specifically on brain MRI scans 
to detect and classify brain tumors. The model can identify four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary
""")
