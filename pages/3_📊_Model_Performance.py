"""
KrishiDrishti - Model Performance & Evaluation Dashboard
Displays authentic evaluation results, confusion matrices, and training curves.
Never presents fabricated metrics.
"""

import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROJECT_NAME,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    PLOTS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
)
from src.evaluate import load_cached_evaluation_metrics, evaluate_model

st.set_page_config(
    page_title=f"Model Performance | {PROJECT_NAME}",
    page_icon="📊",
    layout="wide",
)

# Load CSS
css_file = PROJECT_ROOT / "assets" / "css" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Deep Learning Model Performance & Evaluation")
st.caption("Quantitative validation metrics, confusion matrix, and loss/accuracy dynamics evaluated on the hold-out test split.")

st.markdown("---")

# Check if model exists
model_exists = BEST_MODEL_PATH.exists()
metrics_file = METRICS_DIR / "evaluation_metrics.json"
metrics_exist = metrics_file.exists()

if not model_exists:
    st.warning(
        """
        ### ⚠️ Model Not Yet Trained
        
        No trained model weights were found at `models/best_model.keras`.
        To generate genuine performance metrics, please train the model first by executing:
        
        ```bash
        python -m src.train
        ```
        """,
        icon="ℹ️",
    )
    st.stop()

# If model exists but evaluation hasn't been executed yet, provide an on-demand button
if not metrics_exist:
    st.info(
        """
        The model weights are trained, but evaluation on the test split has not been recorded yet.
        Click the button below to compute test accuracy, precision, recall, and generate the confusion matrix.
        """
    )
    if st.button("🚀 Run Test Set Evaluation Now", type="primary"):
        with st.spinner("Evaluating model across hold-out test set..."):
            try:
                metrics_summary = evaluate_model()
                st.success("Test evaluation completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Evaluation error: {str(e)}")
                st.stop()
    else:
        st.stop()

# Load genuine cached evaluation metrics
metrics_data = load_cached_evaluation_metrics()
if not metrics_data:
    st.error("Could not load evaluation metrics.")
    st.stop()

# Metric Summary Cards
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    acc = metrics_data.get("test_accuracy", 0.0)
    st.markdown(
        f"""
        <div class="kd-metric-box">
            <div class="kd-metric-value">{acc * 100:.2f}%</div>
            <div class="kd-metric-label">Test Set Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    prec = metrics_data.get("weighted_precision", 0.0)
    st.markdown(
        f"""
        <div class="kd-metric-box">
            <div class="kd-metric-value">{prec * 100:.2f}%</div>
            <div class="kd-metric-label">Weighted Precision</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    rec = metrics_data.get("weighted_recall", 0.0)
    st.markdown(
        f"""
        <div class="kd-metric-box">
            <div class="kd-metric-value">{rec * 100:.2f}%</div>
            <div class="kd-metric-label">Weighted Recall</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m4:
    f1 = metrics_data.get("weighted_f1", 0.0)
    st.markdown(
        f"""
        <div class="kd-metric-box">
            <div class="kd-metric-value">{f1 * 100:.2f}%</div>
            <div class="kd-metric-label">Weighted F1-Score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for detailed reports
tab_cm, tab_curves, tab_report = st.tabs([
    "🔲 Confusion Matrix Heatmap",
    "📈 Training & Validation Curves",
    "📋 Detailed Classification Report",
])

with tab_cm:
    st.markdown("### 🔲 Multi-Class Confusion Matrix")
    st.write(
        "Displays the correspondence between ground truth pathology labels and model predictions. "
        "High diagonal values indicate robust discriminative capacity across classes."
    )
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Seaborn Confusion Matrix on Test Split", use_container_width=True)
    else:
        st.info("Confusion matrix plot file not found. Re-run evaluation to generate.")

with tab_curves:
    st.markdown("### 📈 Learning Dynamics")
    st.write("Tracking loss minimization and generalization accuracy across epochs.")

    c_curv1, c_curv2 = st.columns(2)
    acc_plot = PLOTS_DIR / "accuracy_history.png"
    loss_plot = PLOTS_DIR / "loss_history.png"

    with c_curv1:
        if acc_plot.exists():
            st.image(str(acc_plot), caption="Training vs Validation Accuracy", use_container_width=True)
        else:
            st.info("Accuracy plot pending model training.")

    with c_curv2:
        if loss_plot.exists():
            st.image(str(loss_plot), caption="Training vs Validation Categorical Crossentropy Loss", use_container_width=True)
        else:
            st.info("Loss plot pending model training.")

with tab_report:
    st.markdown("### 📋 Per-Class Classification Report")
    raw_report = metrics_data.get("classification_report", {})
    if raw_report:
        report_rows = []
        for class_k, metrics_v in raw_report.items():
            if isinstance(metrics_v, dict):
                clean_name = class_k.replace("Tomato___", "").replace("_", " ")
                report_rows.append({
                    "Class / Pathology": clean_name,
                    "Precision": f"{metrics_v.get('precision', 0):.4f}",
                    "Recall": f"{metrics_v.get('recall', 0):.4f}",
                    "F1-Score": f"{metrics_v.get('f1-score', 0):.4f}",
                    "Support Samples": int(metrics_v.get("support", 0)),
                })
        df_report = pd.DataFrame(report_rows)
        st.dataframe(df_report, use_container_width=True, hide_index=True)

        txt_report_path = REPORTS_DIR / "classification_report.txt"
        if txt_report_path.exists():
            with open(txt_report_path, "r") as fp:
                with st.expander("📄 View Raw Text Report"):
                    st.text(fp.read())
    else:
        st.info("Classification report details not available.")
