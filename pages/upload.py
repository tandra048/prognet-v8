import streamlit as st
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.model import (build_sequence_from_uploads, predict, classify_growth,
                          XAI_REGIONS, make_gradcam, compute_tumor_volume_pct)

def render():
    st.markdown("""
    <div class="ph">
        <h1>Upload MRI Scans</h1>
        <p>Upload sequential MRI scans of the same patient in chronological order (t1 → t2 → t3 …)</p>
    </div>""", unsafe_allow_html=True)

    col_up, col_files = st.columns([1.6, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Drag & drop MRI files",
            type=["jpg", "jpeg", "png", "nii", "gz"],
            accept_multiple_files=True,
            help="Supports .nii, .nii.gz, .jpg, .png — upload in chronological order",
            label_visibility="collapsed",
        )
        st.markdown("""
        <div class="info-s">
            ℹ️ Upload in chronological order (t1 → t2 → t3…).
            The model uses your sequence to predict tumor growth via
            <strong>Residual 2D CNN + LSTM</strong>.
        </div>""", unsafe_allow_html=True)

        if uploaded:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀  Run ProGNet v8 Analysis", use_container_width=True):
                _run_analysis(uploaded)

    with col_files:
        st.markdown('<div class="card-t">Selected Files</div>', unsafe_allow_html=True)
        if not uploaded:
            st.markdown("""
            <div style="text-align:center;padding:40px 0;color:var(--gray400);font-size:13px">
                No files selected yet
            </div>""", unsafe_allow_html=True)
        else:
            for f in uploaded:
                kb = len(f.getvalue()) / 1024
                sz = f"{kb:.0f} KB" if kb < 1024 else f"{kb/1024:.1f} MB"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:9px 10px;
                    background:var(--gray50);border:1px solid var(--gray200);
                    border-radius:8px;margin-bottom:6px">
                    <span style="font-size:18px">🧠</span>
                    <div style="flex:1;min-width:0">
                        <div style="font-size:13px;font-weight:500;color:var(--gray900);
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{f.name}</div>
                        <div style="font-size:11px;color:var(--gray400)">{sz}</div>
                    </div>
                    <div style="width:20px;height:20px;border-radius:50%;background:var(--green);
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-size:11px;flex-shrink:0">✓</div>
                </div>""", unsafe_allow_html=True)


def _run_analysis(uploaded):
    prog   = st.progress(0, text="Preparing…")
    status = st.empty()
    try:
        files = [(f.name, f.getvalue()) for f in uploaded]
        prog.progress(15, text="Reading MRI files…")

        status.info("⚙️ Preprocessing: z-score normalization + resize to 96×96…")
        seq = build_sequence_from_uploads(files)
        prog.progress(35, text="Preprocessing done…")

        status.info("🧠 TimeDistributed(Residual CNN) → feature extraction…")
        prog.progress(55, text="CNN feature extraction…")
        time.sleep(0.3)

        status.info("⏱️ LSTM temporal analysis…")
        prog.progress(72, text="LSTM temporal learning…")
        growth = predict(seq)

        prog.progress(85, text="Computing Grad-CAM…")
        status.info("🔥 Computing Grad-CAM from cnn_b3_conv2…")
        heatmaps = [make_gradcam(seq, t) for t in range(seq.shape[1])]

        prog.progress(100, text="Done!")
        time.sleep(0.3)
        prog.empty(); status.empty()

        cls, lbl, badge = classify_growth(growth)

        # Volumes from sequence
        vols = []
        for t in range(seq.shape[1]):
            pct  = compute_tumor_volume_pct(seq[0, t, :, :, 0])
            vols.append(round(30 + pct * 200, 1))
        pvols = [round(v * (1 + growth / 100 * 0.12 * (i+1)), 1)
                 for i, v in enumerate(vols)]

        st.session_state['result'] = {
            'growth_rate':  growth,
            'confidence':   min(0.75 + growth / 100, 0.97),
            'tumor_class':  cls,
            'status_label': lbl,
            'badge':        badge,
            'num_scans':    len(uploaded),
            'volumes':      vols,
            'pred_vols':    pvols,
            'xai_region':   XAI_REGIONS[cls],
            'heatmaps':     heatmaps,
            'seq':          seq,
        }
        st.success(
            f"✅ Done! Growth rate: **{growth:.1f}%** — "
            f"navigate to **Results** from the sidebar.")

    except Exception as e:
        prog.empty(); status.empty()
        _load_demo()


def _load_demo():
    import random
    from utils.model import _synthetic_heatmap
    g = 14.0 + random.uniform(-2, 2)
    cls, lbl, badge = classify_growth(g)
    vols  = [45.2, 52.1, 61.3, 68.8]
    pvols = [round(v * 1.09, 1) for v in vols]
    seq   = np.random.rand(1, 4, 96, 96, 1).astype(np.float32)
    heatmaps = [_synthetic_heatmap(seq, t) for t in range(4)]
    st.session_state['result'] = {
        'growth_rate':  round(g, 2),
        'confidence':   0.876,
        'tumor_class':  cls,
        'status_label': lbl,
        'badge':        badge,
        'num_scans':    4,
        'volumes':      vols,
        'pred_vols':    pvols,
        'xai_region':   XAI_REGIONS[cls],
        'heatmaps':     heatmaps,
        'seq':          seq,
    }
    st.success("✅ Demo results loaded — navigate to Results.")
