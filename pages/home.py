import streamlit as st

def render():
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🔬 Deep Learning · Medical Imaging · Explainable AI</div>
        <h1>AI-Powered <span>Brain Tumor</span><br>Growth Prediction</h1>
        <p>ProGNet v8 combines a <strong>Residual 2D CNN encoder</strong> (TimeDistributed per MRI frame)
        with <strong>LSTM</strong> temporal learning to predict tumor growth rate from sequential MRI scans.
        Predictions are explained by <strong>Grad-CAM</strong> heatmaps.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="feat-g">
        <div class="feat"><div class="fi">📈</div>
            <div class="ft">Growth Prediction</div>
            <div class="fd">Predict tumor growth % from sequential MRI using Residual CNN + LSTM.</div></div>
        <div class="feat"><div class="fi">⏱️</div>
            <div class="ft">Temporal Analysis</div>
            <div class="fd">Physics-based deterministic growth simulation across time points.</div></div>
        <div class="feat"><div class="fi">🔥</div>
            <div class="ft">Grad-CAM XAI</div>
            <div class="fd">Architecture-aware Grad-CAM from cnn_b3_conv2 layer highlights key regions.</div></div>
        <div class="feat"><div class="fi">📊</div>
            <div class="ft">Volume Charts</div>
            <div class="fd">Interactive Plotly charts — actual vs predicted tumor volume over time.</div></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="card-t">Model Architecture — ProGNet v8</div>', unsafe_allow_html=True)
        st.code("""Input  (4, 96×96, 1ch)
       ↓
TimeDistributed(Residual 2D CNN)
  ├─ ResBlock(32)  96→48  [cnn_b1]
  ├─ ResBlock(64)  48→24  [cnn_b2]
  └─ ResBlock(128) 24→12  [cnn_b3] ← Grad-CAM: cnn_b3_conv2
  └─ GlobalAvgPool → Dense(256)
       ↓ (SEQ_LEN=4, 256)
LSTM(128, dropout=0.25)
LSTM(64,  dropout=0.25)
       ↓
Dense(64) → Dense(32) → Dense(1)  ← growth rate""", language="text")

    with col2:
        st.markdown('<div class="card-t">v8 Key Fixes</div>', unsafe_allow_html=True)
        fixes = [
            ("✅ FIX 1", "Physics-based deterministic growth (no random labels)"),
            ("✅ FIX 2", "Residual 2D CNN per slice → LSTM (no 3D CNN)"),
            ("✅ FIX 3", "Z-score per modality — no JPEG/NIfTI domain shift"),
            ("✅ FIX 4", "Normalized regression targets (z-score on y)"),
            ("✅ FIX 5", "Regression-only eval: MAE, R², RMSE, MAPE"),
            ("✅ FIX 6", "Huber loss (δ=0.5) + Adam clipnorm=1.0"),
        ]
        for tag, desc in fixes:
            st.markdown(
                f"<span class='badge bg-green'>{tag}</span> "
                f"<span style='font-size:12px;color:var(--gray500)'>{desc}</span><br>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec-t">Powered by</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="chip-r">
        <span class="chip">TensorFlow-CPU 2.17</span>
        <span class="chip">Residual 2D CNN</span>
        <span class="chip">TimeDistributed</span>
        <span class="chip">LSTM</span>
        <span class="chip">Grad-CAM</span>
        <span class="chip">Huber Loss</span>
        <span class="chip">BraTS 2021</span>
        <span class="chip">Yale Brain Mets</span>
        <span class="chip">Nickparvar MRI</span>
        <span class="chip">Streamlit 1.39</span>
        <span class="chip">Plotly 5.24</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:12px;color:var(--gray400);font-size:12px">
        👩‍💻 <strong style="color:var(--gray700)">Tandra Biswas</strong> (Exam Roll: 240103)
        &nbsp;|&nbsp; Supervisor: <strong style="color:var(--gray700)">Mehrin Anannya</strong>
        &nbsp;|&nbsp; IIT, Jahangirnagar University
    </div>""", unsafe_allow_html=True)
