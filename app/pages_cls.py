import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import auc as metrics_auc, confusion_matrix, roc_curve

import core
import theme
from theme import CATS, CARD, CLAY, INK, PINE, RULE, STONE

SIZE = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea", "EquivDiameter"]
SHAPE = ["AspectRation", "Eccentricity", "Extent", "Solidity", "roundness", "Compactness"]
FACTORS = ["ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]
RAW = SIZE + SHAPE + FACTORS


def _set_defaults(medians):
    for f in RAW:
        st.session_state[f"bean_{f}"] = float(medians[f])


def _row(values):
    row = dict(values)
    row["Area_Perimeter_Ratio"] = row["Area"] / row["Perimeter"]
    return row


def bean_lab(bundle, d):
    df, classes = d["df"], d["classes"]
    lo, hi = df[RAW].quantile(0.01), df[RAW].quantile(0.99)
    med = {c: df[df["Class"] == c][RAW].median() for c in classes}
    if "bean_Area" not in st.session_state:
        _set_defaults(med["SIRA"])

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("##### Bean measurements")
        st.caption("Slide the measurements, or start from the typical bean of a variety. Range: 1st to 99th percentile of the data.")
        st.selectbox("Start from the typical bean of", [c.title() for c in classes], index=classes.index("SIRA"), key="bean_start",
                     on_change=lambda: _set_defaults(med[st.session_state["bean_start"].upper()]))
        vals = {}
        for title, group in (("Size (pixels)", SIZE), ("Shape", SHAPE), ("Shape factors", FACTORS)):
            with st.expander(title, expanded=(title == "Size (pixels)")):
                for f in group:
                    step = float((hi[f] - lo[f]) / 200)
                    vals[f] = st.slider(f, float(lo[f]), float(hi[f]), key=f"bean_{f}", step=step, format="%.4g")
    row = _row(vals)
    x = pd.DataFrame([row])[d["X"].columns]
    xs = d["scaler"].transform(x)
    probs = {n: bundle["models"][n].predict_proba(xs if core.CLASSIFIERS[n][1] else x)[0] for n in bundle["models"]}

    with right:
        model = st.selectbox("Model", list(probs), index=list(probs).index("SVM (RBF)"))
        p = probs[model]
        top = int(np.argmax(p))
        c1, c2 = st.columns(2)
        c1.metric("Predicted variety", classes[top].title())
        c2.metric("Confidence", f"{p[top] * 100:.1f}%")

        order = np.argsort(p)
        steps = 14
        fig = go.Figure(
            data=[go.Bar(x=p[order], y=[classes[i].title() for i in order], orientation="h",
                         marker_color=[CLAY if i == top else STONE for i in order])],
            frames=[go.Frame(data=[go.Bar(x=p[order] * (s / steps), y=[classes[i].title() for i in order], orientation="h",
                                          marker_color=[CLAY if i == top else STONE for i in order])]) for s in range(steps + 1)])
        fig.update_layout(title=f"Class probabilities, {model}", height=360, margin=dict(b=95), xaxis=dict(range=[0, 1.02], title="Probability"),
                          yaxis_title=None, showlegend=False)
        st.plotly_chart(theme.play_controls(fig, 60, "Replay"), width="stretch")

    a, b = st.columns(2, gap="large")
    with a:
        scale = ((pd.Series(vals) - lo) / (hi - lo)).clip(0, 1)
        ref = ((med[classes[top]] - lo) / (hi - lo)).clip(0, 1)
        fig = go.Figure()
        for name, series, colour, fill in (("Your bean", scale, CLAY, "rgba(181,83,47,0.25)"),
                                           (f"Typical {classes[top].title()}", ref, PINE, "rgba(47,93,98,0.15)")):
            fig.add_trace(go.Scatterpolar(r=list(series) + [series.iloc[0]], theta=RAW + [RAW[0]], name=name,
                                          line=dict(color=colour, width=2), fill="toself", fillcolor=fill))
        fig.update_layout(title="Your bean against the predicted variety", height=470,
                          polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor=RULE),
                                     angularaxis=dict(tickfont=dict(size=10)), bgcolor="rgba(0,0,0,0)"),
                          legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, width="stretch")
    with b:
        mat = pd.DataFrame(probs, index=[c.title() for c in classes]).T
        fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, zmin=0, zmax=1, xgap=2, ygap=2,
                                   colorscale=[[0, CARD], [1, PINE]], text=np.round(mat.values, 2), texttemplate="%{text}",
                                   colorbar=dict(title="P")))
        fig.update_layout(title="Every model on the same bean", height=470, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        votes = pd.Series({n: classes[int(np.argmax(v))] for n, v in probs.items()}).value_counts()
        st.caption("Votes: " + ", ".join(f"{k.title()} {v}/{len(probs)}" for k, v in votes.items()))


def comparison(bundle):
    m = bundle["metrics"]
    metric = st.radio("Metric", ["Accuracy", "F1 (weighted)", "F1 (macro)", "ROC-AUC", "Precision", "Recall"],
                      horizontal=True, key="cls_metric")
    s = m.sort_values(metric)
    lo = max(0.0, float(s[metric].min()) - 0.05)
    left, right = st.columns([6, 5], gap="large")
    with left:
        fig = go.Figure(go.Bar(x=s[metric], y=s["Model"], orientation="h",
                               marker_color=[CLAY if v == s[metric].max() else PINE for v in s[metric]],
                               text=[f"{v:.4f}" for v in s[metric]], textposition="outside"))
        fig.update_layout(title=f"{metric} on the held-out test set", height=420, xaxis=dict(range=[lo, min(1.02, s[metric].max() + 0.04)]),
                          yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(go.Scatter(x=m["Fit time (s)"], y=m["Accuracy"], mode="markers+text", text=m["Model"], textposition="top center",
                                   marker=dict(size=(m["ROC-AUC"] - m["ROC-AUC"].min() + 0.01) * 900 + 10, color=PINE, opacity=0.65,
                                               line=dict(color=INK, width=1)), textfont=dict(size=10)))
        fig.update_layout(title="Accuracy against fit time (bubble = ROC-AUC)", height=420, xaxis=dict(title="Fit time (s)", type="log"),
                          yaxis_title="Accuracy")
        st.plotly_chart(fig, width="stretch")
    st.dataframe(m.set_index("Model").style.format("{:.4f}").highlight_max(axis=0, color="#f0dcc9", subset=m.columns[1:7]),
                 width="stretch")
    chips = "".join(f'<span class="chip">{n}</span>' for n in m["Model"]) + "".join(f'<span class="chip pending">{n} (pending)</span>' for n in core.PENDING_CLASSIFIERS)
    st.markdown(f"<div class='small' style='margin-top:.6rem'>Model slots: {chips}</div>", unsafe_allow_html=True)


def diagnostics(bundle, d):
    classes = [c.title() for c in d["classes"]]
    model = st.selectbox("Model", list(bundle["models"]), index=list(bundle["models"]).index("SVM (RBF)"), key="cls_diag")
    y, pred, prob = d["y_te"], bundle["preds"][model], bundle["probs"][model]
    a, b = st.columns(2, gap="large")
    with a:
        norm = st.toggle("Show row percentages", value=True)
        cm = confusion_matrix(y, pred)
        z = cm / cm.sum(axis=1, keepdims=True) if norm else cm
        fig = go.Figure(go.Heatmap(z=z, x=classes, y=classes, colorscale=[[0, CARD], [1, CLAY]], xgap=2, ygap=2,
                                   text=np.round(z, 2) if norm else z, texttemplate="%{text}", showscale=False))
        fig.update_layout(title=f"Confusion matrix, {model}", height=460, xaxis_title="Predicted", yaxis=dict(title="True", autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = go.Figure()
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y == i, prob[:, i])
            auc = metrics_auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{c} ({auc:.3f})", line=dict(color=CATS[i], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=STONE, dash="dash"), name="Chance", hoverinfo="skip"))
        fig.update_layout(title="One-vs-rest ROC curves", height=460, xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2, gap="large")
    with a:
        pca = PCA(2, random_state=core.SEED).fit(d["scaler"].transform(d["X_te"]))
        xy = pca.transform(d["scaler"].transform(d["X_te"]))
        wrong = pred != y
        fig = go.Figure()
        for i, c in enumerate(classes):
            k = (y == i) & ~wrong
            fig.add_trace(go.Scattergl(x=xy[k, 0], y=xy[k, 1], mode="markers", name=c, marker=dict(color=CATS[i], size=5, opacity=0.6)))
        fig.add_trace(go.Scattergl(x=xy[wrong, 0], y=xy[wrong, 1], mode="markers", name="Misclassified",
                                   marker=dict(color=INK, size=8, symbol="x"), text=[f"True {classes[t]}, predicted {classes[p]}" for t, p in zip(y[wrong], pred[wrong])]))
        v = pca.explained_variance_ratio_ * 100
        fig.update_layout(title=f"Test set in PCA space, {int(wrong.sum())} errors", height=460,
                          xaxis_title=f"PC1 ({v[0]:.1f}%)", yaxis_title=f"PC2 ({v[1]:.1f}%)")
        st.plotly_chart(fig, width="stretch")
    with b:
        m = bundle["models"][model]
        if hasattr(m, "feature_importances_"):
            imp = pd.Series(m.feature_importances_, index=d["X"].columns)
        elif hasattr(m, "coef_"):
            imp = pd.Series(np.abs(m.coef_).mean(axis=0), index=d["X"].columns)
        else:
            imp = None
        if imp is None:
            st.info(f"{model} has no built-in importance. Pick a tree model or Logistic Regression.")
        else:
            imp = imp.sort_values().tail(12)
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=PINE))
            fig.update_layout(title="Feature importance (top 12)", height=460, xaxis_title="Importance", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
