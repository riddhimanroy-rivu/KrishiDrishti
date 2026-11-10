"""
KrishiDrishti - Disease Detection Page
Primary Interface for Leaf Image Upload, Real-Time Inference, and Agronomic Guidance.
"""

import sys
from pathlib import Path
import io
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROJECT_NAME,
    TAGLINE,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    ALLOWED_EXTENSIONS,
    PROCESSED_DATA_DIR,
)
from src.preprocessing import validate_image_file, ImageValidationError
from src.predict import (
    predict_crop_disease,
    is_model_available,
    load_prediction_model,
    load_class_names,
)
from src.disease_info import AGRICULTURAL_DISCLAIMER

# Page Setup
st.set_page_config(
    page_title=f"Disease Detection | {PROJECT_NAME}",
    page_icon="🌱",
    layout="wide",
)

# Load CSS
css_file = PROJECT_ROOT / "assets" / "css" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.title("🌱 Real-Time Crop Disease Detection")
st.caption("Upload a high-resolution photograph of a tomato crop leaf for convolutional neural network screening.")

# Model status check
model_ready = is_model_available()
if not model_ready:
    st.error(
        """
        ⚠️ **Trained Model Unavailable**: No deep learning weights found at `models/best_model.keras`.
        
        To train the MobileNetV2 model on the dataset, run:
        ```bash
        python -m src.train
        ```
        The application strictly avoids fabricated predictions and requires genuine trained weights to operate.
        """,
        icon="🚨",
    )
    st.stop()

# Streamlit Caching for model loading
@st.cache_resource(show_spinner="Loading MobileNetV2 Neural Network into memory...")
def get_cached_model():
    return load_prediction_model()

@st.cache_data
def get_cached_classes():
    return load_class_names()

try:
    model = get_cached_model()
    class_names = get_cached_classes()
except Exception as e:
    st.error(f"Error initializing deep learning model: {str(e)}")
    st.stop()

# Main Interface: Upload and Preview
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload Leaf Image")

    # Option to upload or select sample image
    upload_tab, sample_tab = st.tabs(["📁 Upload Your Own Image", "🖼️ Choose Sample Leaf Image"])

    selected_image = None
    image_name = None

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a crop leaf photograph (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="For optimal classification accuracy, upload a well-lit photograph focusing on a single leaf against a neutral background.",
        )
        if uploaded_file is not None:
            selected_image = uploaded_file
            image_name = uploaded_file.name

    with sample_tab:
        # Collect sample images if available in test/train directories
        sample_images = []
        if PROCESSED_DATA_DIR.exists():
            for sample_path in PROCESSED_DATA_DIR.glob("*/*/*.jpg"):
                sample_images.append(sample_path)
                if len(sample_images) >= 12:
                    break

        if sample_images:
            sample_options = {p.name + f" ({p.parent.name.replace('Tomato___', '')})": p for p in sample_images}
            chosen_sample = st.selectbox("Select a verification sample:", list(sample_options.keys()))
            if chosen_sample:
                selected_sample_path = sample_options[chosen_sample]
                if uploaded_file is None:
                    selected_image = selected_sample_path
                    image_name = selected_sample_path.name
        else:
            st.info("No pre-packaged verification samples found. Please upload an image directly.")

    # Image Preview
    if selected_image is not None:
        try:
            if isinstance(selected_image, Path):
                pil_image = Image.open(selected_image)
            else:
                bytes_data = selected_image.getvalue()
                # Validate image file integrity
                is_valid, err_msg = validate_image_file(bytes_data)
                if not is_valid:
                    st.error(f"❌ Invalid image file: {err_msg}")
                    selected_image = None
                    st.stop()
                pil_image = Image.open(io.BytesIO(bytes_data))

            st.image(pil_image, caption=f"Uploaded: {image_name} ({pil_image.size[0]}x{pil_image.size[1]} px)", use_container_width=True)

            detect_btn = st.button("🔍 Detect Crop Disease", type="primary", use_container_width=True)
        except Exception as exc:
            st.error(f"Error reading image: {str(exc)}")
            selected_image = None
            detect_btn = False
    else:
        detect_btn = False
        st.info("Please select or upload a leaf photograph to activate disease detection.")

with col_right:
    st.markdown("### 📊 Diagnosis & Agronomic Report")

    if selected_image is not None and detect_btn:
        with st.spinner("Analyzing leaf pathology with MobileNetV2..."):
            try:
                # Run actual model inference
                if isinstance(selected_image, Path):
                    prediction = predict_crop_disease(
                        selected_image,
                        model=model,
                        class_names=class_names,
                        top_k=3,
                    )
                else:
                    prediction = predict_crop_disease(
                        selected_image.getvalue(),
                        model=model,
                        class_names=class_names,
                        top_k=3,
                    )

                # Status Badge
                is_healthy = prediction["status"] == "Healthy"
                badge_class = "kd-status-healthy" if is_healthy else "kd-status-diseased"
                badge_icon = "🌿" if is_healthy else "⚠️"

                st.markdown(
                    f"""
                    <div style="margin-bottom: 16px;">
                        <span class="{badge_class}">
                            {badge_icon} Status: {prediction['status']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Primary Prediction
                st.markdown(f"## **{prediction['display_name']}**")

                col_conf1, col_conf2 = st.columns([1, 2])
                with col_conf1:
                    st.metric(
                        label="Model Confidence",
                        value=prediction["confidence_percent"],
                    )
                with col_conf2:
                    st.write("**Classification Confidence Level:**")
                    st.progress(float(prediction["confidence"]))

                st.markdown("---")

                # Top-3 Alternative Predictions
                st.markdown("##### 🎯 Top Predictions Probability Distribution")
                for top_item in prediction["top_predictions"]:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(f"• **{top_item['display_name']}**")
                    with c2:
                        st.write(f"`{top_item['confidence_percent']}`")
                    st.progress(top_item["confidence"])

                st.markdown("---")

                # Educational Disease Information
                st.markdown("##### 🔬 Pathological & Agronomic Profile")

                sev = prediction.get("severity", "N/A")
                sev_tag_class = (
                    "kd-tag-critical" if sev in ["High", "Critical / High"]
                    else "kd-tag-moderate" if "Moderate" in sev
                    else "kd-tag-low"
                )

                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-bottom: 14px;">
                        <p style="margin: 0 0 6px 0;"><strong>Causal Agent:</strong> <em>{prediction['causal_agent']}</em></p>
                        <p style="margin: 0;"><strong>Severity Level:</strong> <span class="{sev_tag_class}">{sev}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("📖 Disease Description", expanded=True):
                    st.write(prediction["description"])

                with st.expander("🔍 Visible Symptoms", expanded=False):
                    for sym in prediction["symptoms"]:
                        st.write(f"• {sym}")

                with st.expander("🛡️ Preventive Cultural Practices", expanded=False):
                    for prev in prediction["preventive_practices"]:
                        st.write(f"• {prev}")

                with st.expander("💡 Agricultural Recommendations", expanded=False):
                    for rec in prediction["recommendations"]:
                        st.write(f"• {rec}")

            except ImageValidationError as ve:
                st.error(f"Image Validation Error: {str(ve)}")
            except Exception as exc:
                st.error(f"Inference Failure: {str(exc)}")
    else:
        st.markdown(
            """
            <div style="background: #f9fafb; border: 2px dashed #d1d5db; border-radius: 12px; padding: 48px 24px; text-align: center; color: #6b7280;">
                <div style="font-size: 2.8rem; margin-bottom: 12px;">🌱</div>
                <h4 style="color: #374151; margin-bottom: 6px;">Awaiting Image Input</h4>
                <p style="font-size: 0.9rem; max-width: 360px; margin: 0 auto;">
                    Select or upload a crop leaf image on the left and click <strong>Detect Crop Disease</strong> to trigger real-time AI classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Agricultural Disclaimer Banner
st.markdown("---")
st.markdown(
    f"""
    <div class="kd-disclaimer">
        <strong>Agronomic Disclaimer:</strong> {AGRICULTURAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True,
)
