import streamlit as st

def render():
    st.markdown("""
    <div class="ph">
        <h1>About ProGNet</h1>
        <p>Master's research — IIT, Jahangirnagar University</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#eff6ff,#f8fafc);border:1px solid #bfdbfe;
        border-radius:14px;padding:32px;margin-bottom:24px">
        <h2 style="font-size:20px;font-weight:700;margin:0 0 10px;color:#0f172a">
            ProGNet v8: Prognostic Framework for Brain Tumor Growth
        </h2>
        <p style="font-size:14px;color:#64748b;line-height:1.8;max-width:780px;margin:0">
            ProGNet is a deep learning framework for predicting brain tumor growth from sequential MRI.
            It uses a <strong>Residual 2D CNN encoder</strong> (TimeDistributed per frame) for spatial features
            and an <strong>LSTM network</strong> for temporal evolution learning.
            <strong>Grad-CAM</strong> (from <code>cnn_b3_conv2</code>) provides clinical explainability.
        </p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    cards = [
        ("🤖","#eff6ff","Architecture","Residual 2D CNN<br>TimeDistributed + LSTM"),
        ("🗂️","#ecfdf5","Datasets","BraTS 2021<br>Yale Brain Mets<br>Nickparvar MRI"),
        ("🔥","#fffbeb","XAI Method","Grad-CAM (cnn_b3_conv2)<br>Integrated Gradients<br>Occlusion Sensitivity"),
        ("🌐","#f5f3ff","Web Stack","Streamlit 1.39<br>Plotly 5.24<br>TF-CPU 2.17"),
    ]
    for col,(icon,bg,title,val) in zip([c1,c2,c3,c4],cards):
        col.markdown(f"""
        <div style="background:#fff;border:1px solid var(--gray200);border-radius:12px;
            padding:20px;box-shadow:var(--shadow)">
            <div style="width:40px;height:40px;background:{bg};border-radius:10px;
                display:flex;align-items:center;justify-content:center;font-size:20px;margin-bottom:10px">{icon}</div>
            <div style="font-size:13px;font-weight:700;color:var(--gray900);margin-bottom:6px">{title}</div>
            <div style="font-size:12px;color:var(--gray500);line-height:1.6">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="card-t">Architecture (from Prognet-model.ipynb Cell 9)</div>',
                    unsafe_allow_html=True)
        st.code("""Input: (SEQ_LEN=4, 96, 96, 1)
       ↓
TimeDistributed(Residual 2D CNN)
  ResBlock(32)  96→48  cnn_b1_conv1/conv2
  ResBlock(64)  48→24  cnn_b2_conv1/conv2
  ResBlock(128) 24→12  cnn_b3_conv1/conv2 ← Grad-CAM
  GlobalAvgPool → Dense(256, relu)
       ↓
LSTM(128, dropout=0.25, rec_drop=0.15)
LSTM(64,  dropout=0.25, rec_drop=0.15)
       ↓
Dense(64, relu)
Dense(32, relu)
Dense(1, linear)  ← growth_rate
Loss: Huber(delta=0.5)
Optimizer: Adam(lr=3e-4, clipnorm=1.0)""", language="text")

    with cb:
        st.markdown('<div class="card-t">v8 Fix Summary (Cell 15)</div>',
                    unsafe_allow_html=True)
        fixes = [
            ("✅ FIX 1","Deterministic physics-based growth — no random labels"),
            ("✅ FIX 2","2D CNN per slice + LSTM — no 3D CNN inside TimeDistributed"),
            ("✅ FIX 3","Z-score per modality — no JPEG/NIfTI domain shift"),
            ("✅ FIX 4","Normalized regression targets (z-score on y)"),
            ("✅ FIX 5","Regression-only eval: MAE, R², RMSE, MAPE"),
            ("✅ FIX 6","Huber loss (δ=0.5) + Adam clipnorm=1.0"),
            ("✅ FIX 7","Sample weights — balance glioma dominance"),
            ("✅ FIX 8","Grad-CAM from cnn_b3_conv2 (last residual conv)"),
        ]
        for tag,desc in fixes:
            st.markdown(
                f"<span class='badge bg-green'>{tag}</span> "
                f"<span style='font-size:12px;color:var(--gray500)'>{desc}</span><br>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card-t">Evaluation Metrics</div>', unsafe_allow_html=True)
    e1,e2,e3,e4 = st.columns(4)
    for col,(m,d) in zip([e1,e2,e3,e4],[
        ("MAE","Mean Absolute Error — growth rate accuracy"),
        ("R² Score","Regression fit quality over time"),
        ("RMSE","Root Mean Squared Error"),
        ("MAPE","Mean Absolute Percentage Error"),
    ]):
        col.markdown(f"""
        <div style="background:var(--gray50);border:1px solid var(--gray200);
            border-radius:8px;padding:14px">
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                letter-spacing:.06em;color:var(--gray400);margin-bottom:6px">{m}</div>
            <div style="font-size:12px;color:var(--gray500);line-height:1.5">{d}</div>
        </div>""", unsafe_allow_html=True)

    
