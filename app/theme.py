"""Colours, plotly template and page CSS. One warm, low-contrast palette; no gradients."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK, PAPER, CARD, RULE = "#1f2a30", "#f5f1e8", "#fffdf8", "#e3dccb"
CLAY, PINE, OCHRE, MOSS, PLUM, STEEL, STONE = "#b5532f", "#2f5d62", "#d9a441", "#6b7f4f", "#7a4e63", "#3f6c9a", "#8a8578"
CATS = [CLAY, PINE, OCHRE, MOSS, PLUM, STEEL, STONE]
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"
SERIF = "Georgia, Cambria, Times New Roman, serif"

pio.templates["capstone"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=CATS, margin=dict(l=50, r=20, t=50, b=45),
    xaxis=dict(gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE, ticks="outside", tickcolor=RULE),
    yaxis=dict(gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE, ticks="outside", tickcolor=RULE),
    title=dict(font=dict(family=SERIF, size=17), x=0.01),
    legend=dict(bgcolor="rgba(255,253,248,0.8)", bordercolor=RULE, borderwidth=1),
    hoverlabel=dict(bgcolor=CARD, bordercolor=RULE, font=dict(family=FONT, color=INK)),
))
pio.templates.default = "capstone"

CSS = f"""
<style>
.stApp {{ background: {PAPER}; color: {INK}; font-family: {FONT}; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 4rem; max-width: 1280px; }}
h1, h2, h3 {{ font-family: {SERIF}; font-weight: 600; letter-spacing: -0.01em; color: {INK}; }}
h1 {{ font-size: 2.1rem; }} h2 {{ font-size: 1.45rem; }} h3 {{ font-size: 1.15rem; }}
p, li, label, .stMarkdown {{ font-size: 0.95rem; }}
.kicker {{ text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.72rem; color: {CLAY}; font-weight: 600; margin-bottom: 0.2rem; }}
.lede {{ color: #4b565c; max-width: 62ch; line-height: 1.55; }}
.card {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 10px; padding: 1.1rem 1.25rem; animation: rise .5s ease both; }}
.card h4 {{ font-family: {SERIF}; margin: 0 0 .3rem 0; font-size: 1.1rem; }}
.big {{ font-family: {SERIF}; font-size: 2.2rem; line-height: 1.1; color: {INK}; }}
.small {{ color: #5d686e; font-size: 0.82rem; }}
.chip {{ display: inline-block; padding: 2px 10px; margin: 0 6px 6px 0; border-radius: 999px; font-size: 0.75rem;
        border: 1px solid {PINE}; color: {PINE}; background: transparent; }}
.chip.pending {{ border: 1px dashed {STONE}; color: {STONE}; }}
.rule {{ border: 0; border-top: 1px solid {RULE}; margin: 1.2rem 0; }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
div[role="radiogroup"] {{ gap: 0.4rem; }}
div[role="radiogroup"] label {{ background: transparent; border: 1px solid {RULE}; border-radius: 6px; padding: 4px 14px; }}
div[role="radiogroup"] label:has(input:checked) {{ background: {INK}; border-color: {INK}; }}
div[role="radiogroup"] label:has(input:checked) p {{ color: {PAPER}; }}
div[role="radiogroup"] label > div:first-of-type {{ display: none; }}
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 10px; padding: .7rem .9rem; }}
[data-testid="stMetricValue"] {{ font-family: {SERIF}; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {CLAY}; box-shadow: none; }}
button[kind="secondary"] {{ border-radius: 6px; border: 1px solid {RULE}; background: {CARD}; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def kicker(text):
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def nav(options, key):
    return st.radio("section", options, horizontal=True, key=key, label_visibility="collapsed")


def play_controls(fig, frame_ms=450, label="Play"):
    """Play / pause buttons for a figure that has frames."""
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", x=0.0, y=-0.3, xanchor="left", yanchor="top", pad=dict(t=0, r=8),
        bgcolor=CARD, bordercolor=RULE, font=dict(color=INK),
        buttons=[
            dict(label=label, method="animate", args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                                                  transition=dict(duration=frame_ms // 2), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
        ])])
    return fig
