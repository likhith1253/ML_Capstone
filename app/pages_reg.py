import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from theme import CLAY, INK, PINE, STONE, OCHRE

KNOBS = ["fwd_pkts_tot", "bwd_pkts_tot", "fwd_pkts_payload.tot", "bwd_pkts_payload.tot",
         "fwd_header_size_tot", "bwd_header_size_tot", "fwd_PSH_flag_count"]


def human(sec):
    sec = float(sec)
    if abs(sec) < 1e-3:
        return f"{sec * 1e6:.1f} µs"
    if abs(sec) < 1:
        return f"{sec * 1e3:.1f} ms"
    if abs(sec) < 120:
        return f"{sec:.1f} s"
    if abs(sec) < 7200:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.2f} h"


def flow_lab(bundle):
    t, rng = bundle["template"], bundle["ranges"]
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("##### Describe a network flow")
        st.caption("Features you do not touch stay at the training median. Slider ends are the 1st and 99th percentile of the training rows.")
        c1, c2 = st.columns(2)
        proto = c1.selectbox("Protocol", bundle["cat_values"]["proto"], index=bundle["cat_values"]["proto"].index(t["proto"]))
        service = c2.selectbox("Service", bundle["cat_values"]["service"], index=bundle["cat_values"]["service"].index(t["service"]))
        row = dict(t)
        row.update(proto=proto, service=service)
        for f in KNOBS:
            lo, hi, _ = rng[f]
            row[f] = st.slider(f, float(lo), float(max(hi, lo + 1)), float(t[f]), step=1.0, key=f"reg_{f}")
        row["total_packets"] = row["fwd_pkts_tot"] + row["bwd_pkts_tot"]
        st.caption(f"total_packets = {row['total_packets']:.0f} (forward + backward)")

    x = pd.DataFrame([row])[bundle["models"]["Decision Tree"].feature_names_in_]
    preds = {n: float(m.predict(x)[0]) for n, m in bundle["models"].items()}
    best = bundle["metrics"].sort_values("R2").iloc[-1]["Model"]
    with right:
        c1, c2 = st.columns(2)
        c1.metric(f"{best} prediction", human(preds[best]))
        c2.metric("Test R² of that model", f"{bundle['metrics'].set_index('Model').loc[best, 'R2']:.3f}")
        s = pd.Series(preds).sort_values()
        fig = go.Figure(go.Bar(x=np.maximum(np.abs(s.values), 1e-7), y=s.index, orientation="h",
                               marker_color=[CLAY if n == best else (OCHRE if v < 0 else STONE) for n, v in s.items()],
                               text=[human(v) for v in s.values], textposition="outside"))
        fig.update_layout(title="Predicted flow duration, every model", height=470, xaxis=dict(title="Seconds, plotted by magnitude (log axis)", type="log"), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Ochre bars are negative predictions (the linear models can produce them), drawn by their size. Most models do poorly on this target; see the comparison tab.")


def comparison(bundle, data):
    m = bundle["metrics"]
    metric = st.radio("Metric", ["R2", "RMSE", "MAE"], horizontal=True, key="reg_metric")
    zoom = st.toggle("Zoom to models with R² above -5", value=True, key="reg_zoom") if metric == "R2" else False
    s = m.sort_values(metric, ascending=(metric == "R2"))
    if zoom:
        s = s[s["R2"] > -5]
    better_high = metric == "R2"
    best = s[metric].max() if better_high else s[metric].min()
    left, right = st.columns([6, 5], gap="large")
    with left:
        fig = go.Figure(go.Bar(x=s[metric], y=s["Model"], orientation="h",
                               marker_color=[CLAY if v == best else PINE for v in s[metric]],
                               text=[f"{v:.3f}" for v in s[metric]], textposition="outside"))
        hidden = len(m) - len(s)
        fig.update_layout(title=f"{metric} on the held-out test set" + (f" ({hidden} models off the scale)" if hidden else ""),
                          height=440, yaxis_title=None, xaxis_title=metric)
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(go.Scatter(x=m["Fit time (s)"], y=m["MAE"], mode="markers+text", text=m["Model"], textposition="top center",
                                   textfont=dict(size=10), marker=dict(size=13, color=PINE, line=dict(color=INK, width=1))))
        fig.update_layout(title="MAE against fit time", height=440, xaxis=dict(title="Fit time (s)", type="log"),
                          yaxis=dict(title="MAE (s)", type="log"))
        st.plotly_chart(fig, width="stretch")
    st.dataframe(m.set_index("Model").style.format({"R2": "{:.4f}", "RMSE": "{:.3f}", "MAE": "{:.4f}", "Fit time (s)": "{:.2f}"}),
                 width="stretch")

    st.markdown("##### Predicted against actual")
    a, b = st.columns([1, 3])
    model = a.selectbox("Model", [n for n in m["Model"]], index=list(m["Model"]).index("Decision Tree"), key="reg_pva")
    log = a.toggle("Log scale", value=True)
    n_show = a.slider("Points shown", 500, 6000, 2500, step=500)
    y = data["y_te"].to_numpy()
    p = bundle["preds"][model]
    idx = np.random.RandomState(42).choice(len(y), size=n_show, replace=False)
    xa, ya = y[idx], p[idx]
    if log:
        xa, ya = np.log1p(np.clip(xa, 0, None)), np.log1p(np.clip(ya, 0, None))
    lim = [0, max(xa.max(), ya.max())]
    fig = go.Figure([go.Scattergl(x=xa, y=ya, mode="markers", marker=dict(size=5, color=PINE, opacity=0.35), name="Test flows"),
                     go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=CLAY, dash="dash"), name="Perfect prediction")])
    fig.update_layout(title=f"{model}: predicted vs actual", height=480,
                      xaxis_title="log1p(actual seconds)" if log else "Actual seconds", yaxis_title="log1p(predicted seconds)" if log else "Predicted seconds")
    b.plotly_chart(fig, width="stretch")


def diagnostics(bundle, data):
    a, b = st.columns(2, gap="large")
    with a:
        model = st.selectbox("Model", list(bundle["models"]), index=5, key="reg_res_model")
        res = data["y_te"].to_numpy() - bundle["preds"][model]
        clip = st.slider("Residual range shown (seconds)", 10, 2000, 300, step=10)
        fig = go.Figure(go.Histogram(x=np.clip(res, -clip, clip), nbinsx=80, marker_color=PINE))
        fig.update_layout(title=f"Residuals, {model} (clipped to ±{clip}s)", height=420, yaxis=dict(type="log", title="Count (log)"),
                          xaxis_title="Actual − predicted (s)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Mean {res.mean():.3f} s, std {res.std():.2f} s, largest miss {np.abs(res).max():.0f} s over {len(res):,} test flows.")
    with b:
        name = st.selectbox("Model with importances", list(bundle["importances"]), index=1, key="reg_imp_model")
        top = st.slider("Features shown", 5, 25, 15)
        imp = bundle["importances"][name].sort_values().tail(top)
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=CLAY))
        fig.update_layout(title=f"{name}: feature importance", height=420 + (top - 15) * 8, xaxis_title="Importance", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Importance is statistical association, not cause.")
