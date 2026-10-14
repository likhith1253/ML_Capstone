import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.decomposition import PCA

import core
import theme
from theme import CATS, CARD, CLAY, INK, PINE, RULE, STONE

CL10 = CATS + ["#c98f6b", "#5b8a72", "#a39a4a"]
PROFILE = ["lead_time", "adr", "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest",
           "previous_cancellations", "booking_changes", "days_in_waiting_list", "total_of_special_requests"]


def _discrete(n=10):
    out = []
    for i in range(n):
        out += [[i / n, CL10[i]], [(i + 1) / n, CL10[i]]]
    return out


def _scatter(xy, labels, size=4, name=None):
    return go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="markers", name=name,
                        marker=dict(size=size, opacity=0.6, color=labels, colorscale=_discrete(), cmin=0, cmax=10, showscale=False),
                        hoverinfo="skip")


def explore_k(b):
    sw, xy = b["sweep"], b["coords"]
    idx = np.random.RandomState(core.SEED).choice(len(xy), 4000, replace=False)
    ks = list(sw["k"])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], subplot_titles=("Bookings in PCA space", "Elbow curve"))
    fig.add_trace(_scatter(xy[idx], b["km_labels"][4][idx]), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=sw["Inertia"], mode="lines+markers", line=dict(color=PINE, width=2), marker=dict(size=6), name="Inertia"), row=1, col=2)
    cur = sw[sw["k"] == 4].iloc[0]
    fig.add_trace(go.Scatter(x=[4], y=[cur["Inertia"]], mode="markers", marker=dict(size=15, color=CLAY, line=dict(color=INK, width=1.5)),
                             name="Selected k"), row=1, col=2)
    frames = []
    for k in ks:
        r = sw[sw["k"] == k].iloc[0]
        frames.append(go.Frame(name=str(k), traces=[0, 2], data=[_scatter(xy[idx], b["km_labels"][k][idx]),
                                                                go.Scatter(x=[k], y=[r["Inertia"]], mode="markers")]))
    fig.frames = frames
    fig.update_layout(height=520, showlegend=False,
                      sliders=[dict(active=2, x=0.2, len=0.75, y=-0.05, pad=dict(t=10), currentvalue=dict(prefix="k = "),
                                    steps=[dict(label=str(k), method="animate",
                                                args=[[str(k)], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=0))])
                                           for k in ks])])
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=2, dtick=1)
    fig.update_yaxes(title_text="Inertia", row=1, col=2)
    st.plotly_chart(theme.play_controls(fig, 900), width="stretch")
    st.caption("Press Play or drag the slider to watch the clusters re-form as k changes. 4,000 of the 15,000 bookings are drawn; all 15,000 were clustered.")

    a, c = st.columns([1, 2], gap="large")
    k = a.slider("Inspect one k", 2, 10, 4)
    row = sw[sw["k"] == k].iloc[0]
    a.metric("Silhouette (higher is better)", f"{row['Silhouette']:.3f}")
    a.metric("Davies-Bouldin (lower is better)", f"{row['Davies-Bouldin']:.3f}")
    a.metric("Calinski-Harabasz (higher is better)", f"{row['Calinski-Harabasz']:.0f}")
    sizes = pd.Series(b["km_labels"][k]).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=[f"Cluster {i}" for i in sizes.index], y=sizes.values, marker_color=[CL10[i] for i in sizes.index],
                           text=[f"{v:,}" for v in sizes.values], textposition="outside"))
    fig.update_layout(title=f"Cluster sizes at k = {k}", height=360, yaxis=dict(title="Bookings", type="log"), xaxis_title=None)
    c.plotly_chart(fig, width="stretch")
    st.caption("The elbow curve bends gently rather than sharply, so k is chosen with the silhouette and Davies-Bouldin scores as well. k = 4 was used in the notebook.")


def _dendro(b, cut):
    Z = b["linkage"]
    d = dendrogram(Z, color_threshold=cut, above_threshold_color="gray", no_plot=True)
    names = [c for c in dict.fromkeys(d["color_list"]) if c != "gray"]
    fig = go.Figure()
    for ic, dc, col in zip(d["icoord"], d["dcoord"], d["color_list"]):
        colour = STONE if col == "gray" else CATS[names.index(col) % len(CATS)]
        fig.add_trace(go.Scatter(x=ic, y=dc, mode="lines", line=dict(color=colour, width=1.2), hoverinfo="skip", showlegend=False))
    fig.add_hline(y=cut, line=dict(color=CLAY, dash="dash"))
    fig.update_layout(title="Dendrogram (Ward linkage, 1,000 sampled bookings)", height=420, xaxis=dict(showticklabels=False, title="Bookings"),
                      yaxis_title="Ward merge distance")
    return fig, int(fcluster(Z, cut, "distance").max())


def compare(b):
    m = b["metrics"]
    cols = st.columns(3)
    for col, name in zip(cols, ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]):
        km, ag = m["K-Means"][name], m["Agglomerative"][name]
        better_high = name != "Davies-Bouldin"
        winner = "K-Means" if (km > ag) == better_high else "Agglomerative"
        col.markdown(f'<div class="card"><div class="small">{name}</div>'
                     f'<div class="small" style="margin-top:.4rem">K-Means</div><div class="big" style="font-size:1.5rem">{km:,.3f}</div>'
                     f'<div class="small" style="margin-top:.3rem">Agglomerative</div><div class="big" style="font-size:1.5rem">{ag:,.3f}</div>'
                     f'<div class="small" style="margin-top:.4rem;color:{CLAY}">better: {winner}</div></div>', unsafe_allow_html=True)
    st.write("")
    view = st.radio("Projection", ["PCA", "t-SNE (5,000 bookings)"], horizontal=True, key="clu_view")
    if view == "PCA":
        xy, sel = b["coords"], slice(None)
        v = b["var"] * 100
        axes = (f"PC1 ({v[0]:.1f}%)", f"PC2 ({v[1]:.1f}%)")
    else:
        xy, sel = b["tsne"], b["tsne_idx"]
        axes = ("t-SNE 1", "t-SNE 2")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("K-Means (k = 4)", "Agglomerative (Ward, 4 clusters)"), shared_yaxes=True)
    for i, lab in enumerate([b["km4"].labels_, b["agg"]]):
        for c in range(4):
            keep = lab[sel] == c
            fig.add_trace(go.Scattergl(x=xy[keep, 0], y=xy[keep, 1], mode="markers", name=f"Cluster {c}", legendgroup=str(c),
                                       showlegend=(i == 0), marker=dict(size=4, color=CL10[c], opacity=0.6)), row=1, col=i + 1)
    fig.update_xaxes(title_text=axes[0])
    fig.update_yaxes(title_text=axes[1], col=1)
    fig.update_layout(height=470, legend=dict(itemsizing="constant"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Click a legend entry to hide a cluster. Both methods give one very large cluster and a few small ones.")

    Z = b["linkage"]
    default = float((Z[-3, 2] + Z[-4, 2]) / 2)
    cut = st.slider("Dendrogram cut height", float(Z[-12, 2]), float(Z[-1, 2]), default, step=0.1)
    fig, n = _dendro(b, cut)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Cutting at {cut:.1f} leaves {n} clusters in this 1,000-booking subset. The default cut gives the 4 clusters used in the notebook.")


def profiles(b):
    algo = st.radio("Algorithm", ["K-Means", "Agglomerative"], horizontal=True, key="clu_prof")
    labels = b["km4"].labels_ if algo == "K-Means" else b["agg"]
    s = b["sample"].copy()
    s["cluster"] = labels
    means = s.groupby("cluster")[PROFILE].mean()
    z = (means - means.mean()) / means.std(ddof=0)
    a, c = st.columns([3, 1], gap="large")
    with a:
        fig = go.Figure(go.Heatmap(z=z.T.values, x=[f"Cluster {i}" for i in means.index], y=PROFILE, zmin=-2, zmax=2, xgap=2, ygap=2,
                                   colorscale=[[0, PINE], [0.5, CARD], [1, CLAY]], text=np.round(means.T.values, 2), texttemplate="%{text}",
                                   colorbar=dict(title="vs. other<br>clusters")))
        fig.update_layout(title=f"What sets each {algo} cluster apart (cell = mean value)", height=470, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with c:
        counts = s["cluster"].value_counts().sort_index()
        fig = go.Figure(go.Pie(labels=[f"Cluster {i}" for i in counts.index], values=counts.values, hole=0.55, sort=False,
                               marker=dict(colors=CL10[:len(counts)], line=dict(color=CARD, width=2)), textinfo="value"))
        fig.update_layout(title="Cluster sizes", height=470, legend=dict(orientation="h", y=-0.05), margin=dict(t=50, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

    a, c = st.columns(2, gap="large")
    with a:
        feat = st.selectbox("Distribution of", PROFILE, index=1)
        fig = go.Figure()
        for i in sorted(s["cluster"].unique()):
            fig.add_trace(go.Violin(y=s.loc[s["cluster"] == i, feat], name=f"Cluster {i}", line_color=CL10[i], fillcolor=CL10[i], opacity=0.55,
                                    meanline_visible=True, points=False, spanmode="hard"))
        fig.update_layout(title=f"{feat} by cluster", height=400, showlegend=False, yaxis_title=feat)
        st.plotly_chart(fig, width="stretch")
    with c:
        cat = st.selectbox("Category share", ["market_segment", "customer_type", "hotel", "deposit_type"])
        share = pd.crosstab(s["cluster"], s[cat], normalize="index")
        fig = go.Figure([go.Bar(name=str(col), x=[f"Cluster {i}" for i in share.index], y=share[col], marker_color=CATS[j % len(CATS)])
                         for j, col in enumerate(share.columns)])
        fig.update_layout(title=f"{cat} within each cluster", barmode="stack", height=400, yaxis=dict(title="Share", tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")

    if st.toggle("Reveal the booking outcome (never used to form the clusters)", value=False):
        rate = s.groupby("cluster")["is_canceled"].mean()
        fig = go.Figure(go.Bar(x=[f"Cluster {i}" for i in rate.index], y=rate.values, marker_color=STONE,
                               text=[f"{v:.0%}" for v in rate.values], textposition="outside"))
        fig.update_layout(title="Cancellation rate per cluster", height=330, yaxis=dict(range=[0, 0.6], tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")
        st.caption("This only describes the groups. Clustering was unsupervised and does not predict cancellations.")


@st.cache_resource
def _pca(_b):
    return PCA(n_components=2, random_state=core.SEED).fit(_b["Z"])


def try_booking(b):
    s = b["sample"]
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("##### Describe a booking")
        c1, c2 = st.columns(2)
        row = {"hotel": c1.selectbox("Hotel", sorted(s["hotel"].unique())),
               "market_segment": c2.selectbox("Market segment", sorted(s["market_segment"].unique()), index=5 if len(s["market_segment"].unique()) > 5 else 0),
               "customer_type": c1.selectbox("Customer type", sorted(s["customer_type"].unique()), index=2),
               "deposit_type": c2.selectbox("Deposit", sorted(s["deposit_type"].unique())),
               "meal": c1.selectbox("Meal", sorted(s["meal"].unique())),
               "distribution_channel": c2.selectbox("Channel", sorted(s["distribution_channel"].unique())),
               "reserved_room_type": c1.selectbox("Room type", sorted(s["reserved_room_type"].unique()))}
        row["lead_time"] = st.slider("Lead time (days)", 0, 500, 80)
        row["adr"] = st.slider("Average daily rate", 0.0, 400.0, 100.0, step=1.0)
        row["stays_in_week_nights"] = st.slider("Week nights", 0, 15, 2)
        row["stays_in_weekend_nights"] = st.slider("Weekend nights", 0, 8, 1)
        c1, c2, c3 = st.columns(3)
        row["adults"] = c1.slider("Adults", 1, 6, 2)
        row["children"] = c2.slider("Children", 0, 4, 0)
        row["babies"] = c3.slider("Babies", 0, 2, 0)
        c1, c2 = st.columns(2)
        row["is_repeated_guest"] = int(c1.toggle("Repeated guest"))
        row["previous_cancellations"] = c2.slider("Previous cancellations", 0, 5, 0)
        row["previous_bookings_not_canceled"] = c1.slider("Previous completed bookings", 0, 20, 0)
        row["booking_changes"] = c2.slider("Booking changes", 0, 5, 0)
        row["days_in_waiting_list"] = c1.slider("Days on waiting list", 0, 200, 0)
        row["required_car_parking_spaces"] = c2.slider("Parking spaces", 0, 3, 0)
        row["total_of_special_requests"] = st.slider("Special requests", 0, 5, 0)

    frame = pd.DataFrame([row])
    _, z = core._hotel_matrix(frame, fit=b["prep"])
    km = b["km4"]
    dist = np.linalg.norm(km.cluster_centers_ - z, axis=1)
    cluster = int(np.argmin(dist))
    pca = _pca(b)
    pt = pca.transform(z)[0]
    with right:
        st.metric("Nearest K-Means cluster", f"Cluster {cluster}")
        idx = np.random.RandomState(core.SEED).choice(len(b["coords"]), 4000, replace=False)
        fig = go.Figure(_scatter(b["coords"][idx], km.labels_[idx], size=4))
        fig.add_trace(go.Scatter(x=[pt[0]], y=[pt[1]], mode="markers", name="Your booking",
                                 marker=dict(size=18, symbol="star", color=INK, line=dict(color="white", width=1.5))))
        fig.update_layout(title="Where the booking lands", height=400, xaxis_title="PC1", yaxis_title="PC2", showlegend=False)
        st.plotly_chart(fig, width="stretch")
        fig = go.Figure(go.Bar(x=dist, y=[f"Cluster {i}" for i in range(len(dist))], orientation="h",
                               marker_color=[CLAY if i == cluster else STONE for i in range(len(dist))],
                               text=[f"{v:.1f}" for v in dist], textposition="outside"))
        fig.update_layout(title="Distance to each cluster centre", height=250, xaxis_title="Euclidean distance in the 48-feature space",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Assignment is by distance to the k = 4 centroids. Clusters 0, 2 and 3 are small, so most ordinary bookings land in cluster 1.")
