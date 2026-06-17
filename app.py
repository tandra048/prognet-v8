"""
ProGNet v8 — Brain Tumor Growth Predictor
Streamlit Web Application
Author: Tandra Biswas | IIT, Jahangirnagar University
"""
import streamlit as st

st.set_page_config(
    page_title="ProGNet — Brain Tumor Growth Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────
with open("assets/style.css") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-mark">🧠</div>
        <div>
            <div class="sb-name">ProGNet</div>
            <div class="sb-sub">Brain Tumor Growth Predictor</div>
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Home", "📤  Upload MRI", "📊  Results",
         "📈  Growth Graph", "🔍  XAI Explanation", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="sb-info">
        <div class="sbi-l">Model</div>
        <div class="sbi-v">ProGNet v8 (2D CNN+LSTM)</div>
        <div class="sbi-l" style="margin-top:8px">Architecture</div>
        <div class="sbi-v">TimeDistributed(ResNet) → LSTM</div>
        <div class="sbi-l" style="margin-top:8px">XAI</div>
        <div class="sbi-v">Grad-CAM (cnn_b3_conv2)</div>
        <div class="sbi-l" style="margin-top:8px">Datasets</div>
        <div class="sbi-v">BraTS · Yale · Nickparvar</div>
    </div>""", unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────
if   "🏠" in page:  from pages import home;    home.render()
elif "📤" in page:  from pages import upload;  upload.render()
elif "📊" in page:  from pages import results; results.render()
elif "📈" in page:  from pages import growth;  growth.render()
elif "🔍" in page:  from pages import xai;     xai.render()
elif "ℹ️" in page: from pages import about;   about.render()
