import streamlit as st
import pandas as pd
from datetime import datetime

def render():
    st.markdown("""
    <div class="ph">
        <h1>Prediction Results</h1>
        <p>ProGNet v8 — Residual 2D CNN + LSTM analysis output</p>
    </div>""", unsafe_allow_html=True)

    r = st.session_state.get('result')
    if not r:
        st.markdown("""
        <div style="text-align:center;padding:60px 0">
            <div style="font-size:48px;margin-bottom:16px">📤</div>
            <div style="font-size:18px;font-weight:600;color:var(--gray700);margin-bottom:8px">No results yet</div>
            <div style="font-size:14px;color:var(--gray400)">Go to <strong>Upload MRI</strong> first.</div>
        </div>""", unsafe_allow_html=True)
        return

    g    = r['growth_rate']
    conf = r['confidence'] * 100
    lbl  = r['status_label']
    bc   = r['badge']
    cls  = r['tumor_class']

    kpi_color = {'Rapid Growth':'red','Moderate Growth':'orange',
                 'Slow Growth':'green','Stable / No Tumor':''}.get(lbl,'')
    icon = {'Rapid Growth':'🔴','Moderate Growth':'🟡',
            'Slow Growth':'🟢','Stable / No Tumor':'⚪'}.get(lbl,'🔵')

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"""<div class="kpi"><div class="kpi-l">Predicted Growth</div>
        <div class="kpi-v {kpi_color}">{g:.1f}%</div>
        <div class="kpi-s">Next 3 months</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi"><div class="kpi-l">Confidence</div>
        <div class="kpi-v blue">{conf:.1f}%</div>
        <div class="kpi-s">Model certainty</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi"><div class="kpi-l">Status</div>
        <div style="margin-top:8px"><span class="badge {bc}">{icon} {lbl}</span></div>
        <div class="kpi-s" style="margin-top:8px">Type: {cls}</div></div>""",
        unsafe_allow_html=True)
    c4.markdown(f"""<div class="kpi"><div class="kpi-l">Scans</div>
        <div class="kpi-v">{r['num_scans']}</div>
        <div class="kpi-s">Sequential MRI frames</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_m, col_s = st.columns([1.2, 1])

    with col_m:
        st.markdown('<div class="card-t">Growth Rate Details</div>', unsafe_allow_html=True)
        val_color = {'red':'var(--red)','orange':'var(--orange)',
                     'green':'var(--green)','':'var(--gray900)'}.get(kpi_color,'var(--gray900)')
        st.markdown(f"""
        <div style="font-size:11px;color:var(--gray400);font-weight:600;margin-bottom:4px">PREDICTED GROWTH RATE</div>
        <div style="font-size:58px;font-weight:800;line-height:1;color:{val_color};font-family:var(--mono)">{g:.1f}%</div>
        <div style="font-size:12px;color:var(--gray400);margin-top:6px">(Next 3 months)</div>
        <div style="margin-top:10px"><span class="badge {bc}">{icon} {lbl}</span></div>
        <div style="margin-top:16px">
            <div style="font-size:12px;color:var(--gray500);margin-bottom:6px">
                Confidence: <strong>{conf:.1f}%</strong></div>
            <div class="cb-bg"><div class="cb-fill" style="width:{conf:.0f}%"></div></div>
        </div>
        <div class="info-s" style="margin-top:16px">
            ℹ️ ProGNet v8 predicts the tumor will grow by <strong>{g:.1f}%</strong>
            in the next 3 months based on {r['num_scans']} sequential MRI scans
            (TimeDistributed Residual CNN → LSTM).
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            if st.button("📈 Growth Graph", use_container_width=True):
                st.info("Navigate to **Growth Graph** in the sidebar.")
        with cb:
            if st.button("🔍 XAI Explanation", use_container_width=True):
                st.info("Navigate to **XAI Explanation** in the sidebar.")

    with col_s:
        st.markdown('<div class="card-t">Summary</div>', unsafe_allow_html=True)
        today = datetime.now().strftime("%d %b %Y")
        rows  = [
            ("Total Scans",        str(r['num_scans'])),
            ("Scan Interval",      "30 days (est.)"),
            ("Prediction Horizon", "3 months"),
            ("Analysis Date",      today),
            ("Tumor Class",        cls.capitalize()),
            ("XAI Region",         r['xai_region'].title()),
            ("Model",              "ProGNet v8"),
            ("Architecture",       "ResNet2D + LSTM"),
            ("Grad-CAM Layer",     "cnn_b3_conv2"),
        ]
        html = "<table class='s-tbl'>"
        for k, v in rows:
            html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # Per-frame table
    st.markdown('<div class="card-t">Per-Frame Volume Analysis</div>', unsafe_allow_html=True)
    vols = r.get('volumes', [])
    pvols = r.get('pred_vols', [])
    if vols:
        base = vols[0]
        df = pd.DataFrame([{
            "Frame":          f"t{i+1}",
            "Actual (cc)":    v,
            "Predicted (cc)": round(pvols[i], 1),
            "Δ vs t0 (%)":    f"{'+' if (v-base)/max(base,1e-6)*100 >= 0 else ''}{(v-base)/max(base,1e-6)*100:.2f}%",
            "Status":         "↗ Growing" if (v-base)/max(base,1e-6)*100 > 2 else "→ Stable",
        } for i, v in enumerate(vols)])
        st.dataframe(df, hide_index=True, use_container_width=True)
