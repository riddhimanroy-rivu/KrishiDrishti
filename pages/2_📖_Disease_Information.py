"""
KrishiDrishti - Disease Information Knowledge Base Page
Educational reference catalog for all 10 supported tomato pathologies.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import PROJECT_NAME
from src.disease_info import (
    DISEASE_KNOWLEDGE_BASE,
    AGRICULTURAL_DISCLAIMER,
    get_disease_info,
)

st.set_page_config(
    page_title=f"Disease Library | {PROJECT_NAME}",
    page_icon="📖",
    layout="wide",
)

# Load CSS
css_file = PROJECT_ROOT / "assets" / "css" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📖 Crop Pathology & Disease Information Library")
st.write(
    "Explore comprehensive, scientifically verified diagnostic profiles, symptoms, and organic prevention "
    "methods for all tomato crop conditions recognized by KrishiDrishti."
)

st.markdown("---")

# Filter and Search
col_search, col_filter = st.columns([2, 1])
with col_search:
    search_query = st.text_input("🔍 Search disease by name or symptom:", placeholder="e.g. Blight, Mosaic, Mites...")
with col_filter:
    status_filter = st.selectbox("Filter condition status:", ["All Conditions", "Diseased Foliage", "Healthy Leaves"])

# Filter diseases
filtered_items = []
for key, info in DISEASE_KNOWLEDGE_BASE.items():
    # Filter by condition
    if status_filter == "Diseased Foliage" and info["status"] != "Diseased":
        continue
    if status_filter == "Healthy Leaves" and info["status"] != "Healthy":
        continue

    # Filter by search
    if search_query:
        query = search_query.lower()
        match_title = query in info["display_name"].lower()
        match_agent = query in info["causal_agent"].lower()
        match_desc = query in info["description"].lower()
        match_sym = any(query in s.lower() for s in info["symptoms"])
        if not (match_title or match_agent or match_desc or match_sym):
            continue

    filtered_items.append((key, info))

st.write(f"Showing **{len(filtered_items)}** condition profile(s):")

for key, info in filtered_items:
    is_healthy = info["status"] == "Healthy"
    status_color = "#2e7d32" if is_healthy else "#c62828"
    sev = info.get("severity", "N/A")
    sev_badge = (
        "kd-tag-critical" if "High" in sev or "Critical" in sev
        else "kd-tag-moderate" if "Moderate" in sev
        else "kd-tag-low"
    )

    with st.expander(f"{'🌿' if is_healthy else '⚠️'} {info['display_name']} ({info['status']})", expanded=False):
        c_meta1, c_meta2 = st.columns([3, 1])
        with c_meta1:
            st.markdown(f"**Causal Organism:** *{info['causal_agent']}*")
        with c_meta2:
            st.markdown(f"**Severity Level:** <span class='{sev_badge}'>{sev}</span>", unsafe_allow_html=True)

        st.markdown("#### **Overview**")
        st.write(info["description"])

        c_sym, c_prev = st.columns(2)
        with c_sym:
            st.markdown("#### 🔍 **Diagnostic Symptoms**")
            for s in info["symptoms"]:
                st.write(f"• {s}")

        with c_prev:
            st.markdown("#### 🛡️ **Cultural & Preventive Practices**")
            for p in info["preventive_practices"]:
                st.write(f"• {p}")

        st.markdown("#### 💡 **Agronomic & Integrated Pest Management Recommendations**")
        for r in info["recommendations"]:
            st.write(f"• {r}")

st.markdown("---")
st.markdown(
    f"""
    <div class="kd-disclaimer">
        <strong>Agronomic Disclaimer:</strong> {AGRICULTURAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True,
)
