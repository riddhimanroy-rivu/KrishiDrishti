"""
KrishiDrishti - About Project Page
Academic documentation, SDG 2 justification, architecture breakdown, and student metadata.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROJECT_NAME,
    TAGLINE,
    ACADEMIC_SUBJECT,
    UN_SDG_TARGET,
)

st.set_page_config(
    page_title=f"About | {PROJECT_NAME}",
    page_icon="ℹ️",
    layout="wide",
)

# Load CSS
css_file = PROJECT_ROOT / "assets" / "css" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("ℹ️ About KrishiDrishti")
st.caption(f"An Academic Deep Learning Project for {ACADEMIC_SUBJECT} | Third-Year CSE / CSE (AI)")

st.markdown("---")

col_a, col_b = st.columns([3, 2], gap="large")

with col_a:
    st.markdown("### 📋 Executive Summary")
    st.write(
        """
        **KrishiDrishti** (Sanskrit: *Vision for Agriculture*) is an AI-powered agricultural screening system 
        engineered to detect and diagnose plant leaf diseases in tomato crops using Convolutional Neural Networks (CNNs). 
        By employing **MobileNetV2 Transfer Learning**, the application delivers high-accuracy disease diagnosis 
        at minimal computational cost, making it suitable for deployment on edge devices and standard laptop hardware 
        without demanding expensive dedicated GPUs.
        """
    )

    st.markdown("### ⚠️ Problem Statement")
    st.write(
        """
        Crop pathologies and pest infestations account for **20% to 40% of global agricultural yield losses** annually. 
        In developing regions, smallholder farmers often lack prompt access to certified agronomists or plant pathologists. 
        Consequently, diseases are either diagnosed too late—leading to extensive harvest failure—or misidentified, 
        prompting indiscriminate broad-spectrum chemical sprays that degrade soil fertility, pollute groundwater, 
        and foster antimicrobial resistance.
        """
    )

    st.markdown("### 🎯 Academic Objectives")
    st.write(
        """
        1. **Automated Visual Screening:** Implement an end-to-end computer vision pipeline capable of classifying foliar anomalies into 10 categories.
        2. **Transfer Learning Efficiency:** Fine-tune ImageNet-pretrained MobileNetV2 feature extractors to prevent overfitting on specialized botanical imagery.
        3. **Confidence-Calibrated Diagnosis:** Produce calibrated class probabilities to warn users when confidence is low or ambiguous.
        4. **Agronomic IPM Guidance:** Map predicted pathologies to organic, preventive Integrated Pest Management (IPM) practices to curtail chemical misuse.
        5. **Rigorous Academic Validation:** Evaluate using stratified hold-out validation with genuine confusion matrices, precision, recall, and F1-scores.
        """
    )

with col_b:
    st.markdown(
        f"""
        <div style="background: #e8f5e9; border: 2px solid #81c784; border-radius: 14px; padding: 22px; margin-bottom: 20px;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🌍</div>
            <h4 style="margin: 0 0 8px 0; color: #1b5e20;">UN SDG 2: Zero Hunger</h4>
            <p style="font-size: 0.9rem; color: #2e7d32; line-height: 1.6; margin: 0 0 10px 0;">
                <strong>Target 2.4:</strong> By 2030, ensure sustainable food production systems and implement resilient agricultural practices 
                that increase productivity and production, help maintain ecosystems, and strengthen capacity for adaptation to climate change.
            </p>
            <p style="font-size: 0.85rem; color: #388e3c; margin: 0;">
                By mitigating early-stage fungal and bacterial leaf blights before whole-field infestation occurs, KrishiDrishti directly safeguards rural farm income, 
                stabilizes regional food supply chains, and curtails toxic chemical runoff into local watersheds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 💻 Technology Stack")
    st.markdown(
        """
        - **Deep Learning Framework:** TensorFlow 2.x & Keras 3.x
        - **Backbone Architecture:** MobileNetV2 (ImageNet Pretrained)
        - **Web Frontend:** Streamlit with Custom Responsive CSS
        - **Image Processing:** Pillow (PIL) & NumPy
        - **Model Evaluation:** Scikit-Learn, Matplotlib, Seaborn
        - **Quality Assurance:** Pytest Test Suite
        """
    )

st.markdown("---")

# Model Architecture Breakdown
st.markdown("### 🧠 Convolutional Neural Network Architecture")

st.markdown(
    """
    KrishiDrishti adopts **MobileNetV2** as its foundational backbone, selected specifically for its balance of high feature extraction fidelity 
    and lightweight computational footprint (inverted residual blocks with linear bottlenecks).
    """
)

arch_cols = st.columns(5)
arch_steps = [
    ("1. Input Layer", "(224, 224, 3) RGB tensor scaled to [-1, 1] range"),
    ("2. MobileNetV2 Backbone", "Pretrained weights frozen for feature extraction"),
    ("3. Global Avg Pooling", "Reduces spatial 7x7 feature maps to 1280-dim vector"),
    ("4. Dense Head + Dropout", "256 ReLU units with L2 regularization and 30% Dropout"),
    ("5. Softmax Output", "10-class probability distribution matching tomato categories"),
]

for idx, (title, detail) in enumerate(arch_steps):
    with arch_cols[idx]:
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; min-height: 130px;">
                <div style="font-weight: 700; color: #1b5e20; font-size: 0.95rem; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.82rem; color: #6b7280;">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# Dataset Attribution & Limitations
c_data, c_limits = st.columns(2)

with c_data:
    st.markdown("### 📚 Dataset Attribution")
    st.write(
        """
        The model is trained on the publicly accessible **PlantVillage** benchmark agricultural dataset.
        - **Original Creators:** Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.*
        - **Kaggle Distribution:** Emma Rex (`emmarex/plantdisease`)
        - **Classes Utilized:** 10 Tomato classes (9 pathologies + 1 healthy control).
        - **Split Ratio:** 70% Training (with rotation, zoom, flip augmentations), 15% Validation, 15% Hold-out Testing.
        """
    )

with c_limits:
    st.markdown("### 🚧 System Limitations")
    st.write(
        """
        - **Crop Scope:** Currently focused on tomato leaves (*Solanum lycopersicum*). Non-tomato leaves will yield out-of-distribution classifications.
        - **Visual Occlusion:** Extreme camera glare, severe motion blur, or overlapping multi-leaf clutter can reduce prediction confidence.
        - **Screening Nature:** AI predictions serve as early-warning screening, not a definitive chemical prescription. Laboratory confirmation is recommended for high-stakes quarantine pests.
        """
    )

st.markdown("---")

# Student / Team Details Placeholder
st.markdown("### 👥 Academic Project Team")
st.caption("College Practical Submission: PCCCS582 - Neural Network & Deep Learning")

team_c1, team_c2, team_c3 = st.columns(3)

with team_c1:
    st.markdown(
        """
        <div class="kd-card">
            <h4 style="margin: 0 0 4px 0; color: #1b5e20;">Student Investigator 1</h4>
            <div style="font-size: 0.85rem; color: #6b7280;">Role: Deep Learning & Data Pipeline</div>
            <hr style="margin: 10px 0;">
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Name:</strong> [Insert Student Name]</p>
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Roll No:</strong> [Insert University Roll No]</p>
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Department:</strong> CSE (AI & ML) - 3rd Year</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with team_c2:
    st.markdown(
        """
        <div class="kd-card">
            <h4 style="margin: 0 0 4px 0; color: #1b5e20;">Student Investigator 2</h4>
            <div style="font-size: 0.85rem; color: #6b7280;">Role: UI/UX & Application Integration</div>
            <hr style="margin: 10px 0;">
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Name:</strong> [Insert Student Name]</p>
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Roll No:</strong> [Insert University Roll No]</p>
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Department:</strong> CSE (AI & ML) - 3rd Year</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with team_c3:
    st.markdown(
        """
        <div class="kd-card">
            <h4 style="margin: 0 0 4px 0; color: #1b5e20;">Faculty Supervisor</h4>
            <div style="font-size: 0.85rem; color: #6b7280;">Role: Academic Guidance & Review</div>
            <hr style="margin: 10px 0;">
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Name:</strong> [Insert Professor Name]</p>
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Designation:</strong> Assistant Professor</p>
            <p style="font-size: 0.85rem; margin: 4px 0;"><strong>Subject:</strong> Neural Network & DL (PCCCS582)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
