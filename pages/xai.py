import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

def render():
    st.markdown("""
    <div class="ph">
        <h1>AI Explanation (XAI)</h1>
        <p>Grad-CAM from <code>cnn_b3_conv2</code> — which MRI regions influenced the prediction most</p>
    </div>""", unsafe_allow_html=True)

    r = st.session_state.get('result')
    if not r:
        st.markdown("""<div style="text-align:center;padding:60px 0">
            <div style="font-size:48px;margin-bottom:16px">🔍</div>
            <div style="font-size:16px;font-weight:600;color:var(--gray700)">No results yet — upload MRI scans first.</div>
        </div>""", unsafe_allow_html=True)
        return

    heatmaps = r.get('heatmaps', [])
    seq      = r.get('seq')
    g        = r['growth_rate']
    region   = r['xai_region']
    cls      = r['tumor_class']
    n        = len(heatmaps)

    if not heatmaps or seq is None:
        st.warning("No Grad-CAM data available.")
        return

    st.markdown('<div style="font-size:14px;font-weight:600;color:var(--gray900);margin-bottom:10px">Select Time Step</div>',
                unsafe_allow_html=True)
    t = st.slider("", 0, max(n-1, 0), max(n-1, 0),
                  format="t%d", label_visibility="collapsed")

    hm  = heatmaps[t]
    img = seq[0, t, :, :, 0]

    from utils.model import overlay_gradcam
    ov = overlay_gradcam(img, hm, alpha=0.45)

    # ── 3 image columns ──
    st.markdown('<div style="font-size:14px;font-weight:600;color:var(--gray900);margin-bottom:10px">Grad-CAM Visualization</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    img_pil = Image.fromarray((img*255).clip(0,255).astype(np.uint8))
    hm_jet  = cv2.applyColorMap((hm*255).clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_pil  = Image.fromarray(cv2.cvtColor(hm_jet, cv2.COLOR_BGR2RGB))
    ov_pil  = Image.fromarray(ov)

    with c1:
        st.markdown(f'<div style="font-size:13px;font-weight:600;margin-bottom:6px;color:var(--gray700)">🧠 Original MRI (t{t+1})</div>',
                    unsafe_allow_html=True)
        st.image(img_pil.resize((280,280), Image.NEAREST), width="stretch")

    with c2:
        st.markdown(f'<div style="font-size:13px;font-weight:600;margin-bottom:6px;color:var(--gray700)">🔥 Grad-CAM Heatmap (t{t+1})</div>',
                    unsafe_allow_html=True)
        st.image(hm_pil.resize((280,280), Image.NEAREST), width="stretch")

    with c3:
        st.markdown(f'<div style="font-size:13px;font-weight:600;margin-bottom:6px;color:var(--gray700)">🎯 Overlay (t{t+1})</div>',
                    unsafe_allow_html=True)
        st.image(ov_pil.resize((280,280), Image.NEAREST), width="stretch")

    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin:10px 0 20px">
        <span style="font-size:12px;color:var(--gray500)">Low Activation</span>
        <div class="imp-b" style="flex:1"></div>
        <span style="font-size:12px;color:var(--gray500)">High Activation</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Explanation + findings
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="card-t">Explanation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <p style="font-size:13px;color:var(--gray500);line-height:1.75;margin-bottom:14px">
            The Residual CNN focused on the
            <strong style="color:var(--gray900)">{region}</strong>,
            which shows significant intensity changes across time points.
            This region contributed most to the predicted growth rate of
            <strong style="color:var(--blue)">{g:.1f}%</strong>.<br><br>
            Red areas in the Grad-CAM indicate highest activation in the
            last convolutional layer (<code>cnn_b3_conv2</code>) of the
            residual CNN encoder.
        </p>
        <div style="font-size:12px;font-weight:600;margin-bottom:6px">Feature Importance</div>
        <div class="imp-b"></div>
        <div class="imp-s"><span>Low</span><span>High</span></div>""",
        unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="card-t">Key Findings</div>', unsafe_allow_html=True)
        bc = r['badge']
        findings = [
            ("🔴", "High-activation region",   region.capitalize()),
            ("📈", "Predicted growth rate",     f"{g:.1f}% over 3 months"),
            ("🎯", "Model confidence",          f"{r['confidence']*100:.1f}%"),
            ("🧠", "Tumor classification",      cls.capitalize()),
            ("🔬", "Status",                    f'<span class="badge {bc}">{r["status_label"]}</span>'),
            ("⚙️", "Architecture",            "TimeDistributed(ResNet2D) → LSTM"),
            ("📐", "Grad-CAM layer",            "cnn_b3_conv2 (last residual conv)"),
            ("🕐", "Explained time step",       f"t{t+1} of {n}"),
        ]
        for icon, lbl, val in findings:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:10px;padding:9px 0;
                border-bottom:1px solid var(--gray100)">
                <span style="font-size:16px;flex-shrink:0">{icon}</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:var(--gray900)">{lbl}</div>
                    <div style="font-size:12px;color:var(--gray500);margin-top:1px">{val}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Activation over time
    st.markdown('<div style="font-size:14px;font-weight:600;color:var(--gray900);margin-bottom:10px">CNN Attention Intensity Over Time</div>',
                unsafe_allow_html=True)
    avg_int = [float(hm.mean()) for hm in heatmaps]
    lbs     = [f"t{i+1}" for i in range(len(avg_int))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lbs, y=avg_int, mode='lines+markers', name='Mean CAM Intensity',
        line=dict(color='#dc2626', width=2.5),
        marker=dict(size=8, color='#fff', line=dict(color='#dc2626', width=2)),
        fill='tozeroy', fillcolor='rgba(220,38,38,0.07)',
        hovertemplate='<b>%{x}</b><br>Intensity: %{y:.4f}<extra></extra>'))
    fig.update_layout(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', height=220,
        margin=dict(l=10,r=10,t=20,b=10),
        xaxis=dict(title='Time Step', gridcolor='#f1f5f9', linecolor='#e2e8f0'),
        yaxis=dict(title='Avg Grad-CAM Activation', gridcolor='#f1f5f9', linecolor='#e2e8f0'),
        font=dict(family='Plus Jakarta Sans, sans-serif', size=12),
        showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("""
    <div style="font-size:12px;color:var(--gray400);text-align:center;margin-top:4px">
        Higher activation = model relies more on this frame for growth prediction
    </div>""", unsafe_allow_html=True)
