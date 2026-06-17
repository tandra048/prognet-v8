import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render():
    st.markdown("""
    <div class="ph">
        <h1>Tumor Growth Over Time</h1>
        <p>Actual vs Predicted tumor volume across sequential scan time points</p>
    </div>""", unsafe_allow_html=True)

    r = st.session_state.get('result')
    if not r:
        st.markdown("""<div style="text-align:center;padding:60px 0">
            <div style="font-size:48px;margin-bottom:16px">📈</div>
            <div style="font-size:16px;font-weight:600;color:var(--gray700)">No data yet — upload MRI scans first.</div>
        </div>""", unsafe_allow_html=True)
        return

    vols  = r.get('volumes', [45.2, 52.1, 61.3, 68.8])
    pvols = r.get('pred_vols', [v*1.09 for v in vols])
    g     = r['growth_rate']
    lbs   = [f"t{i+1}" for i in range(len(vols))]

    # ── Main chart ──
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lbs, y=vols, fill='tozeroy',
        fillcolor='rgba(37,99,235,0.07)',
        line=dict(color='rgba(37,99,235,0)', width=0),
        showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=lbs, y=vols, mode='lines+markers', name='Actual Volume',
        line=dict(color='#2563eb', width=2.5),
        marker=dict(size=8, color='#fff', line=dict(color='#2563eb', width=2)),
        hovertemplate='<b>%{x}</b><br>Actual: %{y:.1f} cc<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=lbs, y=pvols, mode='lines+markers', name='Predicted Volume',
        line=dict(color='#dc2626', width=2, dash='dash'),
        marker=dict(size=6, color='#dc2626'),
        hovertemplate='<b>%{x}</b><br>Predicted: %{y:.1f} cc<extra></extra>'))
    fig.update_layout(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        font=dict(family='Plus Jakarta Sans, sans-serif', size=12, color='#334155'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(title='Time Point', gridcolor='#f1f5f9', linecolor='#e2e8f0',
                   showline=True, zeroline=False),
        yaxis=dict(title='Tumor Volume (cc)', gridcolor='#f1f5f9', linecolor='#e2e8f0',
                   showline=True, zeroline=False),
        hovermode='x unified', margin=dict(l=10,r=10,t=40,b=10), height=300)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Stats
    c1,c2,c3,c4 = st.columns(4)
    for col, lbl, val, clr in [
        (c1, "Current Volume (t0)",   f"{vols[0]:.1f} cc",   ""),
        (c2, "Predicted Vol (t+90d)", f"{pvols[-1]:.1f} cc", "blue"),
        (c3, "Predicted Growth",      f"+{g:.1f}%",          "green" if g < 10 else "red"),
        (c4, "Trend",                 "↗ Increasing" if g > 1 else "→ Stable",
              "red" if g > 5 else ""),
    ]:
        col.markdown(
            f'<div class="kpi"><div class="kpi-l">{lbl}</div>'
            f'<div class="kpi-v {clr}" style="font-size:22px">{val}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar + Delta charts ──
    cb, cd = st.columns(2)
    with cb:
        st.markdown('<div style="font-size:14px;font-weight:600;color:var(--gray900);margin-bottom:10px">Volume per Frame</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=lbs, y=vols, name='Actual',
            marker_color='#2563eb', opacity=0.85,
            hovertemplate='%{x}: %{y:.1f} cc<extra>Actual</extra>'))
        fig2.add_trace(go.Bar(x=lbs, y=pvols, name='Predicted',
            marker_color='#dc2626', opacity=0.6,
            hovertemplate='%{x}: %{y:.1f} cc<extra>Predicted</extra>'))
        fig2.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', height=240,
            barmode='group', showlegend=True, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation='h', y=1.0, x=0),
            xaxis=dict(gridcolor='#f1f5f9'),
            yaxis=dict(gridcolor='#f1f5f9', title='Volume (cc)'),
            font=dict(family='Plus Jakarta Sans, sans-serif', size=11))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with cd:
        st.markdown('<div style="font-size:14px;font-weight:600;color:var(--gray900);margin-bottom:10px">Growth Rate per Frame (Δ%)</div>',
                    unsafe_allow_html=True)
        base   = vols[0]
        deltas = [(v-base)/max(base,1e-6)*100 for v in vols]
        colors = ['#dc2626' if d>2 else '#059669' if d<-2 else '#d97706' for d in deltas]
        fig3 = go.Figure(go.Bar(x=lbs, y=deltas, marker_color=colors, opacity=0.85,
            hovertemplate='%{x}: %{y:.2f}%<extra></extra>'))
        fig3.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', height=240,
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(gridcolor='#f1f5f9'),
            yaxis=dict(gridcolor='#f1f5f9', title='Δ Growth %',
                       zeroline=True, zerolinecolor='#94a3b8'),
            font=dict(family='Plus Jakarta Sans, sans-serif', size=11))
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    # Table
    st.markdown('<div style="font-size:14px;font-weight:600;color:var(--gray900);margin:8px 0 12px">Per-Frame Analysis</div>',
                unsafe_allow_html=True)
    df = pd.DataFrame([{
        "Frame": f"t{i+1}", "Actual (cc)": v, "Predicted (cc)": round(pvols[i],1),
        "Δ vs t0 (%)": f"{'+' if (v-base)/max(base,1e-6)*100>=0 else ''}{(v-base)/max(base,1e-6)*100:.2f}%",
        "Status": "↗ Growing" if (v-base)/max(base,1e-6)*100>2 else "→ Stable",
    } for i,v in enumerate(vols)])
    st.dataframe(df, hide_index=True, use_container_width=True)
