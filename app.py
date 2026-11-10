"""
KrishiDrishti - AI-Powered Crop Disease Detection and Classification System
Main Entry Point and Home Dashboard Page
Subject: Neural Network & Deep Learning (PCCCS582)
Alignment: UN SDG 2 - Zero Hunger
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROJECT_NAME,
    TAGLINE,
    ACADEMIC_SUBJECT,
    UN_SDG_TARGET,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from src.disease_info import AGRICULTURAL_DISCLAIMER

# Page Configuration
st.set_page_config(
    page_title=f"{PROJECT_NAME} | AI Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
css_file = PROJECT_ROOT / "assets" / "css" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar Navigation and Branding
with st.sidebar:
    logo_path = PROJECT_ROOT / "assets" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=180)

    st.markdown(f"### **{PROJECT_NAME}**")
    st.caption(TAGLINE)
    st.markdown("---")
    st.markdown(f"**Course:** {ACADEMIC_SUBJECT}")
    st.markdown(f"**Alignment:** UN SDG 2")

    # Quick System Status Indicator
    st.markdown("---")
    st.markdown("#### **System Diagnostics**")
    model_ready = BEST_MODEL_PATH.exists()
    classes_ready = CLASS_NAMES_PATH.exists()

    if model_ready:
        st.success("Deep Learning Model: **Online**", icon="✅")
    else:
        st.warning("Deep Learning Model: **Not Trained Yet**", icon="⚠️")

    if classes_ready:
        st.info("Class Registry: **Detected**", icon="📋")
    else:
        st.info("Class Registry: **Awaiting Training**", icon="ℹ️")

    st.markdown("---")
    st.caption("College Academic Practical (3rd Year CSE/AI)")

# Hero Section
st.markdown(
    f"""
    <div class="kd-hero">
        <span class="kd-badge-sdg">🌍 Aligned with {UN_SDG_TARGET}</span>
        <h1 style="margin-top: 14px;">{PROJECT_NAME}</h1>
        <p class="tagline">{TAGLINE}</p>
        <p style="max-width: 820px; font-size: 1rem; line-height: 1.6; opacity: 0.95; margin: 0;">
            An intelligent, deep-learning-driven agricultural screening platform designed to assist smallholder farmers,
            extension workers, and researchers in early identification and management of crop foliar pathologies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Action Banner / Navigation Helper
col_act1, col_act2, col_act3, col_act4 = st.columns(4)

with col_act1:
    st.markdown(
        """
        <div class="kd-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🌱</div>
            <h4 style="margin: 0 0 6px 0; color: #1b5e20;">Disease Detection</h4>
            <p style="font-size: 0.85rem; color: #6b7280; margin: 0 0 12px 0;">Upload a tomato leaf photo for real-time AI classification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Detection", key="btn_nav_detect", use_container_width=True):
        st.switch_page("pages/1_🌱_Disease_Detection.py")

with col_act2:
    st.markdown(
        """
        <div class="kd-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">📖</div>
            <h4 style="margin: 0 0 6px 0; color: #1b5e20;">Disease Library</h4>
            <p style="font-size: 0.85rem; color: #6b7280; margin: 0 0 12px 0;">Explore symptoms, causes, and sustainable organic remedies.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Library", key="btn_nav_info", use_container_width=True):
        st.switch_page("pages/2_📖_Disease_Information.py")

with col_act3:
    st.markdown(
        """
        <div class="kd-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">📊</div>
            <h4 style="margin: 0 0 6px 0; color: #1b5e20;">Model Metrics</h4>
            <p style="font-size: 0.85rem; color: #6b7280; margin: 0 0 12px 0;">Review test accuracy, confusion matrix, and training curves.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Performance", key="btn_nav_perf", use_container_width=True):
        st.switch_page("pages/3_📊_Model_Performance.py")

with col_act4:
    st.markdown(
        """
        <div class="kd-card" style="text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">ℹ️</div>
            <h4 style="margin: 0 0 6px 0; color: #1b5e20;">About Project</h4>
            <p style="font-size: 0.85rem; color: #6b7280; margin: 0 0 12px 0;">Academic report, architecture details, and SDG 2 scope.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Read About", key="btn_nav_about", use_container_width=True):
        st.switch_page("pages/4_ℹ️_About_Project.py")

st.markdown("---")

# Main Content: How AI Works & SDG 2
col_main1, col_main2 = st.columns([3, 2], gap="large")

with col_main1:
    st.markdown("### 🔬 How KrishiDrishti Works")
    st.write(
        """
        KrishiDrishti leverages **Transfer Learning** using a **MobileNetV2** Convolutional Neural Network 
        pretrained on ImageNet. It processes input leaf photographs through depthwise separable convolutions 
        to detect micro-features such as chlorosis rings, bacterial lesions, fungal mold, and viral stunting.
        """
    )

    steps = [
        ("1. Image Capture & Upload", "Farmers or students capture a close-up photograph of a suspect leaf under natural lighting."),
        ("2. Normalization & Preprocessing", "The image is resized to 224x224 pixels and normalized to [-1.0, 1.0] matching MobileNetV2 parameters."),
        ("3. Neural Feature Extraction", "Pretrained convolutional filters extract high-level spatial visual representations of foliage patterns."),
        ("4. Softmax Classification & Advisory", "A customized Dense classification head computes class probabilities, identifying disease condition and immediate advisory."),
    ]

    for title, desc in steps:
        st.markdown(
            f"""
            <div style="display: flex; gap: 14px; margin-bottom: 14px; background: #ffffff; padding: 12px 16px; border-radius: 10px; border: 1px solid #e5e7eb;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #2e7d32; min-width: 140px;">{title.split('.')[0]}.</div>
                <div>
                    <strong style="color: #1f2937;">{title.split('. ')[1]}</strong>
                    <div style="font-size: 0.88rem; color: #4b5563; margin-top: 2px;">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_main2:
    st.markdown("### 🎯 UN SDG 2: Zero Hunger")
    st.markdown(
        """
        <div style="background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
            <h4 style="margin: 0 0 8px 0; color: #1b5e20;">Target 2.4 - Sustainable Food Production</h4>
            <p style="font-size: 0.88rem; color: #2e7d32; line-height: 1.5; margin: 0;">
                By 2030, ensure sustainable food production systems and implement resilient agricultural practices 
                that increase productivity and production.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        - **Preventing Crop Losses:** Pests and plant diseases cause an estimated 20–40% loss of global agricultural yield annually.
        - **Reducing Chemical Misuse:** Accurate AI diagnosis helps farmers avoid excessive or incorrect pesticide applications, protecting groundwater and soil health.
        - **Empowering Smallholders:** Brings scientific plant pathology knowledge directly to remote, underserved farming communities at zero marginal cost.
        """
    )

st.markdown("---")

# Supported Pathologies
st.markdown("### 🍅 Supported Tomato Leaf Categories")
st.write("KrishiDrishti is trained to recognize all 10 standard tomato categories from the PlantVillage dataset:")

cat_cols = st.columns(2)
categories = [
    ("Tomato Bacterial Spot", "Bacterial infection causing necrotic spots on foliage and fruit."),
    ("Tomato Early Blight", "Fungal pathogen producing characteristic bullseye concentric rings."),
    ("Tomato Late Blight", "Devastating water mold spreading rapidly in humid, cool conditions."),
    ("Tomato Leaf Mold", "Fungal issue producing pale yellow spots with olive-green underside mold."),
    ("Tomato Septoria Leaf Spot", "Severe fungal defoliation with tiny dark fruiting bodies."),
    ("Tomato Spider Mites", "Pest damage causing yellow stippling and fine foliage webbing."),
    ("Tomato Target Spot", "Fungal circular lesions with prominent concentric zoning."),
    ("Tomato Yellow Leaf Curl Virus", "Whitefly-transmitted virus causing severe stunting and leaf cupping."),
    ("Tomato Mosaic Virus", "Highly contagious mechanically transmitted viral mosaic pattern."),
    ("Tomato Healthy Leaf", "Vibrant, unblemished foliage with optimal physiological health."),
]

for idx, (name, brief) in enumerate(categories):
    col = cat_cols[idx % 2]
    with col:
        badge = "🌿 Healthy" if "Healthy" in name else "⚠️ Diseased"
        badge_color = "#2e7d32" if "Healthy" in name else "#c62828"
        st.markdown(
            f"""
            <div style="padding: 10px 14px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 8px;">
                <span style="float: right; font-size: 0.75rem; font-weight: 700; color: {badge_color};">{badge}</span>
                <strong style="color: #1f2937;">{name}</strong>
                <div style="font-size: 0.83rem; color: #6b7280; margin-top: 2px;">{brief}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Disclaimer Footer
st.markdown(
    f"""
    <div class="kd-disclaimer">
        <strong>Academic & Screening Disclaimer:</strong> {AGRICULTURAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True,
)
