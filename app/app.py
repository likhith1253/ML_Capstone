"""Capstone Lab: interactive front end for the 23CSE301 notebooks.

Run from the repository root:  streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import streamlit.components.v1 as components

import core
import pages_clu
import pages_cls
import pages_reg
import theme

st.set_page_config(page_title="Capstone Lab", layout="wide", initial_sidebar_state="collapsed")
theme.apply()


@st.cache_resource(show_spinner="Training the classifiers (first launch only, about two minutes)...")
def cls_data():
    return core.classification_bundle(), core.load_beans()


@st.cache_resource(show_spinner="Training the regressors (first launch only, about five minutes)...")
def reg_data():
    return core.regression_bundle(), core.regression_data()


@st.cache_resource(show_spinner="Clustering the bookings (first launch only, about three minutes)...")
def clu_data():
    return core.clustering_bundle()


HERO = f"""
<style>
body {{ margin:0; font-family:{theme.FONT}; color:{theme.INK}; background:transparent; }}
.row {{ display:flex; gap:14px; }}
.cell {{ flex:1; border-top:2px solid {theme.INK}; padding-top:10px; opacity:0; transform:translateY(10px); animation:in .7s ease forwards; }}
.cell:nth-child(2) {{ animation-delay:.12s; }} .cell:nth-child(3) {{ animation-delay:.24s; }} .cell:nth-child(4) {{ animation-delay:.36s; }}
.n {{ font-family:{theme.SERIF}; font-size:38px; line-height:1; }}
.l {{ font-size:12px; color:#5d686e; margin-top:6px; letter-spacing:.02em; }}
@keyframes in {{ to {{ opacity:1; transform:none; }} }}
</style>
<div class="row">
  <div class="cell"><div class="n" data-to="119390">0</div><div class="l">hotel bookings in the raw file</div></div>
  <div class="cell"><div class="n" data-to="13611">0</div><div class="l">dry beans, 7 varieties</div></div>
  <div class="cell"><div class="n" data-to="123117">0</div><div class="l">IoT network flows in the raw file</div></div>
  <div class="cell"><div class="n" data-to="{len(core.CLASSIFIERS) + 10 + 2}">0</div><div class="l">models in this dashboard</div></div>
</div>
<script>
document.querySelectorAll('.n').forEach(function(el) {{
  var to = +el.dataset.to, t0 = null;
  function step(t) {{ if (!t0) t0 = t; var p = Math.min((t - t0) / 1400, 1); var e = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(to * e).toLocaleString('en-US'); if (p < 1) requestAnimationFrame(step); }}
  requestAnimationFrame(step);
}});
</script>
"""


def overview():
    theme.kicker("23CSE301 Machine Learning Capstone")
    st.markdown("# Capstone Lab")
    st.markdown('<p class="lede">Three tracks, three datasets. Move the sliders to try your own inputs on the trained models, '
                'or open the comparison tabs to see how the algorithms score and where they fail.</p>', unsafe_allow_html=True)
    components.html(HERO, height=110)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        theme.card("Classification", "<div class='small'>Dry Bean dataset, 7 varieties, 16 shape measurements.</div>"
                   f"<div style='margin-top:.7rem'>{''.join(f'<span class=chip>{n}</span>' for n in core.CLASSIFIERS)}"
                   f"{''.join(f'<span class=\"chip pending\">{n}</span>' for n in core.PENDING_CLASSIFIERS)}</div>"
                   f"<div class='small'>{len(core.CLASSIFIERS)} of 10 required models loaded. Dashed slots are still to be added.</div>")
    with c2:
        theme.card("Regression", "<div class='small'>RT-IoT2022 network flows. Predict how long a flow lasts from its packet counts.</div>"
                   "<div style='margin-top:.7rem'>" + "".join(f"<span class=chip>{n}</span>" for n in
                                                              ["Linear", "Ridge", "Lasso", "ElasticNet", "Polynomial", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR", "KNN"])
                   + "</div><div class='small'>10 of 10 required models loaded.</div>")
    with c3:
        theme.card("Clustering", "<div class='small'>Hotel Booking Demand. Group bookings without using any labels.</div>"
                   "<div style='margin-top:.7rem'><span class=chip>K-Means</span><span class=chip>Agglomerative (Ward)</span></div>"
                   "<div class='small'>2 of 2 required algorithms loaded.</div>")
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("##### How this dashboard stays in step with the notebooks")
    st.markdown("Each page rebuilds its data with the same cleaning, split and random seed as the notebook, then trains the models "
                "with the tuned settings from the notebook. The first launch trains and caches them; later launches open instantly. "
                "A new model is one line in `CLASSIFIERS` or `REGRESSORS` inside `app/core.py`.")


def classification():
    theme.kicker("Track 1")
    st.markdown("## Dry Bean classification")
    bundle, d = cls_data()
    page = theme.nav(["Bean lab", "Model comparison", "Diagnostics"], "nav_cls")
    {"Bean lab": lambda: pages_cls.bean_lab(bundle, d), "Model comparison": lambda: pages_cls.comparison(bundle),
     "Diagnostics": lambda: pages_cls.diagnostics(bundle, d)}[page]()


def regression():
    theme.kicker("Track 2")
    st.markdown("## Network flow duration")
    bundle, data = reg_data()
    page = theme.nav(["Flow lab", "Model comparison", "Diagnostics"], "nav_reg")
    {"Flow lab": lambda: pages_reg.flow_lab(bundle), "Model comparison": lambda: pages_reg.comparison(bundle, data),
     "Diagnostics": lambda: pages_reg.diagnostics(bundle, data)}[page]()


def clustering():
    theme.kicker("Track 3")
    st.markdown("## Hotel booking clusters")
    b = clu_data()
    page = theme.nav(["Explore k", "Compare methods", "Cluster profiles", "Try a booking"], "nav_clu")
    {"Explore k": lambda: pages_clu.explore_k(b), "Compare methods": lambda: pages_clu.compare(b),
     "Cluster profiles": lambda: pages_clu.profiles(b), "Try a booking": lambda: pages_clu.try_booking(b)}[page]()


PAGES = {"Overview": overview, "Classification": classification, "Regression": regression, "Clustering": clustering}
choice = st.radio("page", list(PAGES), horizontal=True, key="page", label_visibility="collapsed")
st.markdown('<hr class="rule" style="margin-top:.6rem">', unsafe_allow_html=True)
PAGES[choice]()
