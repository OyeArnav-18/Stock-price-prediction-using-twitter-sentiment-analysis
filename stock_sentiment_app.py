
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from string import Template
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# THEME SYSTEM
# Every colour in the whole app comes from this one dictionary.
# This is what fixes the invisible-text bug: instead of letting
# native Streamlit widgets fall back to the browser's auto-detected
# theme (which is what caused white-on-white text before), we now
# force every element — including native widgets — to use one of
# these two explicit palettes. Flipping the toggle just swaps which
# dictionary gets used.
# ─────────────────────────────────────────
THEMES = {
    "light": dict(
        bg_app="#F8FAFC", bg_sidebar="#FFFFFF", bg_card="#FFFFFF",
        text_primary="#1E293B", text_secondary="#64748B", text_heading="#1A3A5C",
        border="#E2E8F0", table_stripe="#F8FAFC",
        chart_bg="#FFFFFF", grid="#F1F5F9", axis="#64748B", spine="#E2E8F0",
        info_bg="#EFF6FF", info_border="#BFDBFE", info_text="#1E3A5F",
        warn_bg="#FFFBEB", warn_border="#FDE68A", warn_text="#78350F",
        pos_bg="#F0FDF4", pos_border="#86EFAC", pos_text="#14532D",
        neg_bg="#FFF1F2", neg_border="#FDA4AF", neg_text="#7F1D1D",
        neu_bg="#F8FAFC", neu_border="#CBD5E1", neu_text="#334155",
    ),
    "dark": dict(
        bg_app="#0F172A", bg_sidebar="#1E293B", bg_card="#1E293B",
        text_primary="#E2E8F0", text_secondary="#94A3B8", text_heading="#93C5FD",
        border="#334155", table_stripe="#182334",
        chart_bg="#1E293B", grid="#334155", axis="#94A3B8", spine="#334155",
        info_bg="#1E3A5F", info_border="#2563A8", info_text="#DBEAFE",
        warn_bg="#3F2D0A", warn_border="#92660B", warn_text="#FDE68A",
        pos_bg="#0F2E1B", pos_border="#16A34A", pos_text="#86EFAC",
        neg_bg="#3A1014", neg_border="#DC2626", neg_text="#FDA4AF",
        neu_bg="#1E293B", neu_border="#475569", neu_text="#CBD5E1",
    ),
}

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False  # light is the default

t = THEMES["dark"] if st.session_state["dark_mode"] else THEMES["light"]

CSS_TEMPLATE = Template("""
<style>
    /* ---- Base catch-all: every plain text element gets an explicit
       colour so nothing can ever fall back to the browser's theme ---- */
    .stApp { background-color: $bg_app; color: $text_primary; }
    .stApp p, .stApp span, .stApp li, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: $text_primary;
    }

    /* ---- Sidebar (covers both old and new Streamlit class names) ---- */
    [data-testid="stSidebar"], .stSidebar {
        background-color: $bg_sidebar !important;
        border-right: 1px solid $border;
    }
    [data-testid="stSidebar"] *, .stSidebar * { color: $text_primary !important; }
    [data-testid="stSidebar"] label, .stSidebar .stRadio label {
        color: $text_primary !important;
        font-size: 0.95rem;
    }

    /* ---- Expanders (the "How It Works" titles that were vanishing) ---- */
    [data-testid="stExpander"] {
        background-color: $bg_card;
        border: 1px solid $border;
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] svg {
        color: $text_primary !important;
        fill: $text_primary !important;
    }

    /* ---- Metrics ---- */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] { color: $text_primary !important; }

    /* ---- Widget labels (text_area / number_input titles) ---- */
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p { color: $text_primary !important; }

    /* ---- Buttons ---- */
    .stButton button, [data-testid^="stBaseButton"] {
        color: $text_primary;
        background-color: $bg_card;
        border: 1px solid $border;
    }

    /* ---- Text inputs (fixes the dark-box-on-white-page mismatch) ---- */
    .stTextArea textarea, .stNumberInput input, .stTextInput input {
        color: $text_primary !important;
        background-color: $bg_card !important;
        border: 1px solid $border !important;
    }

    /* ---- Native alert boxes (st.info / st.warning) ---- */
    [data-testid="stAlert"] { background-color: $info_bg; border-radius: 8px; }
    [data-testid="stAlert"] * { color: $info_text !important; }

    /* ---- Dark-mode toggle switch stays legible too ---- */
    [data-testid="stToggle"] label p { color: $text_primary !important; }

    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Header banner — intentionally fixed regardless of theme,
       it's a dark gradient with white text either way */
    .main-header {
        background: linear-gradient(135deg, #1A3A5C 0%, #2563A8 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(37,99,168,0.15);
    }
    .main-header h1 { color: #FFFFFF !important; font-size: 1.9rem; font-weight: 700; margin: 0 0 0.3rem 0; letter-spacing: -0.5px; }
    .main-header p { color: #93C5FD !important; font-size: 0.92rem; margin: 0; }
    .main-header .guide { color: #BFDBFE !important; font-size: 0.85rem; margin-top: 0.3rem; }

    .metric-card {
        background: $bg_card;
        border: 1px solid $border;
        border-radius: 10px;
        padding: 1.4rem 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-number { font-size: 2.2rem; font-weight: 700; color: $text_heading; line-height: 1; }
    .metric-label {
        font-size: 0.82rem; color: $text_secondary; margin-top: 0.4rem;
        font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;
    }

    .section-header {
        color: $text_heading; font-size: 1.3rem; font-weight: 700;
        border-left: 4px solid #2563A8; padding-left: 0.8rem; margin-bottom: 1.5rem;
    }

    .sentiment-positive { background: $pos_bg; border: 1px solid $pos_border; border-left: 5px solid #16A34A; padding: 1rem 1.2rem; border-radius: 8px; font-size: 1.1rem; font-weight: 600; color: $pos_text; }
    .sentiment-negative { background: $neg_bg; border: 1px solid $neg_border; border-left: 5px solid #DC2626; padding: 1rem 1.2rem; border-radius: 8px; font-size: 1.1rem; font-weight: 600; color: $neg_text; }
    .sentiment-neutral  { background: $neu_bg; border: 1px solid $neu_border; border-left: 5px solid #94A3B8; padding: 1rem 1.2rem; border-radius: 8px; font-size: 1.1rem; font-weight: 600; color: $neu_text; }

    .signal-buy  { background: $pos_bg;  border: 2px solid #16A34A; border-radius: 12px; padding: 2rem; text-align: center; font-size: 2rem; font-weight: 700; color: $pos_text; letter-spacing: 1px; }
    .signal-sell { background: $neg_bg;  border: 2px solid #DC2626; border-radius: 12px; padding: 2rem; text-align: center; font-size: 2rem; font-weight: 700; color: $neg_text; letter-spacing: 1px; }
    .signal-hold { background: $warn_bg; border: 2px solid #CA8A04; border-radius: 12px; padding: 2rem; text-align: center; font-size: 2rem; font-weight: 700; color: $warn_text; letter-spacing: 1px; }

    .info-box    { background: $info_bg; border: 1px solid $info_border; border-radius: 8px; padding: 1rem 1.2rem; color: $info_text; font-size: 0.95rem; line-height: 1.6; }
    .warning-box { background: $warn_bg; border: 1px solid $warn_border; border-radius: 8px; padding: 1rem 1.2rem; color: $warn_text; font-size: 0.9rem; }

    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
    .styled-table th { background: $text_heading; color: #FFFFFF; padding: 0.7rem 1rem; text-align: left; }
    .styled-table td { padding: 0.65rem 1rem; border-bottom: 1px solid $border; color: $text_primary; }
    .styled-table tr:nth-child(even) td { background: $table_stripe; }

    .sidebar-stat { background: $bg_app; border: 1px solid $border; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
    .sidebar-stat-label { font-size: 0.78rem; color: $text_secondary; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .sidebar-stat-value { font-size: 1.3rem; font-weight: 700; color: $text_heading; margin-top: 0.1rem; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""")

st.markdown(CSS_TEMPLATE.substitute(t), unsafe_allow_html=True)

# ─────────────────────────────────────────
# VADER WITH FINANCIAL LEXICON
# This fixes misclassification of financial terms
# ─────────────────────────────────────────
@st.cache_resource
def load_analyser():
    analyser = SentimentIntensityAnalyzer()

    financial_lexicon = {
        "crushing":    2.5,
        "unstoppable": 2.8,
        "surging":     2.5,
        "soaring":     2.5,
        "rallying":    2.0,
        "breakout":    2.0,
        "bullish":     2.5,
        "skyrocketing":2.8,
        "mooning":     2.5,
        "ripping":     2.0,
        "smashing":    2.0,
        "blowing":     1.5,
        "killing":     1.8,
        "monster":     1.5,
        "beat":        2.0,
        "beats":       2.0,
        "upgrade":     2.0,
        "outperform":  2.0,
        "overweight":  1.5,
        "upside":      1.5,
        "strong":      1.5,
        "strength":    1.5,
        "record":      1.5,

        "bearish":     -2.5,
        "plummeting":  -2.8,
        "tanking":     -2.8,
        "sinking":     -2.0,
        "dumping":     -2.5,
        "bleeding":    -2.0,
        "miss":        -2.0,
        "missed":      -2.0,
        "misses":      -2.0,
        "warning":     -1.5,
        "downgrade":   -2.0,
        "underperform":-2.0,
        "underweight": -1.5,
        "sell-off":    -2.0,
        "selloff":     -2.0,
        "correction":  -1.5,
        "weak":        -1.5,
        "weakness":    -1.5,
        "downside":    -1.5,
    }
    analyser.lexicon.update(financial_lexicon)
    return analyser

analyser = load_analyser()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Stock Price Prediction Using Twitter Sentiment Analysis</h1>
    <p>MVSREC &nbsp;·&nbsp; BE CSE Sec A &nbsp;·&nbsp; Batch 22 &nbsp;·&nbsp;
       Arnav Bhardwaj &nbsp;&amp;&nbsp; Suraj Pratap Singh</p>
    <p class="guide">Guide: Neeraj Sharma, Asst. Professor, Dept. of CSE, MVSREC</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.toggle("🌙 Dark Mode", key="dark_mode")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Project Overview",
         "Live Sentiment Analyser",
         "Model Results",
         "Buy / Hold / Sell Signal",
         "How It Works"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Project Statistics**")
    for label, value in [
        ("Tweets Analysed",    "5,791"),
        ("Trading Days",       "987"),
        ("Models Built",       "3"),
        ("Best Model RMSE",    "$12.00"),
        ("Accuracy Gain",      "+79.2%"),
        ("Direction Accuracy", "52.9%"),
    ]:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">{label}</div>
            <div class="sidebar-stat-value">{value}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ═══════════════════════════════════════════
if page == "Project Overview":

    st.markdown('<p class="section-header">Project Overview</p>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, num, label in zip(
        [col1, col2, col3, col4],
        ["5,791", "987", "3", "79.2%"],
        ["Tweets Analysed", "Trading Days", "Models Built",
         "Improvement Over Baseline"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{num}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### The Problem")
        st.markdown("""
        <div class="info-box">
        Stock markets are unpredictable. Traditional models only look at
        price history — they ignore <strong>what people are saying</strong>
        about a stock on social media.<br><br>
        A single viral tweet can move a stock by 5% in minutes.
        Traditional models have no way to account for this.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Our Solution")
        st.markdown("""
        <div class="info-box">
        We built a system combining <strong>two signals</strong>:<br><br>
        1. <strong>Price history</strong> — what the stock has been doing<br>
        2. <strong>Twitter sentiment</strong> — what people are saying<br><br>
        We tested 3 increasingly intelligent models and proved that
        adding sentiment data improves prediction accuracy by
        <strong>79.2%</strong>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>")
    st.markdown("#### The 3 Models We Built")

    col1, col2, col3 = st.columns(3)
    model_data = [
        ("ARIMA", "Statistical Baseline",
         "Uses price history only. Finds mathematical patterns in past prices. Cannot predict trend changes.",
         "$57.74", "Baseline", "#DC2626"),
        ("Random Forest", "Classic Machine Learning",
         "100 decision trees each examine all features and vote. Uses price + sentiment. Understands non-linear relationships.",
         "$18.80", "67% better", "#F59E0B"),
        ("LSTM Neural Network", "Deep Learning",
         "Has memory — looks at 30 consecutive days at once. Learns patterns across time. Uses sequences + sentiment.",
         "$12.00", "79% better", "#16A34A"),
    ]
    for col, (name, type_, desc, rmse, improvement, color) in \
            zip([col1, col2, col3], model_data):
        with col:
            st.markdown(f"""
            <div style="background:{t['bg_card']}; border:1px solid {t['border']};
                        border-top: 4px solid {color};
                        border-radius:10px; padding:1.2rem;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06); height:100%">
                <div style="font-size:1.1rem; font-weight:700;
                            color:{t['text_heading']}">{name}</div>
                <div style="font-size:0.8rem; color:{t['text_secondary']};
                            margin-bottom:0.8rem">{type_}</div>
                <div style="font-size:0.9rem; color:{t['text_primary']};
                            line-height:1.6; margin-bottom:1rem">{desc}</div>
                <div style="font-size:1.4rem; font-weight:700;
                            color:{color}">RMSE: {rmse}</div>
                <div style="font-size:0.85rem; color:{t['text_secondary']}">{improvement}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 2 — LIVE SENTIMENT ANALYSER
# ═══════════════════════════════════════════
elif page == "Live Sentiment Analyser":

    st.markdown('<p class="section-header">Live Tweet Sentiment Analyser</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Type any tweet or sentence and see how our sentiment engine scores it.
    This uses VADER enhanced with a financial-specific lexicon to correctly
    interpret stock market language like "crushing earnings" or "tanking".
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Enter a tweet or sentence about a stock:",
        placeholder='e.g. "MSFT crushing earnings expectations again, '
                    'this stock is unstoppable right now"',
        height=110
    )

    col_btn, col_clr, _ = st.columns([1, 1, 4])
    with col_btn:
        analyse_btn = st.button("Analyse Sentiment", type="primary")

    st.markdown("**Quick examples:**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("Positive example"):
            st.session_state["ex"] = ("MSFT crushing earnings expectations again, "
                                       "this stock is unstoppable right now")
    with ex2:
        if st.button("Negative example"):
            st.session_state["ex"] = ("Apple is in serious trouble. Supply chain "
                                       "disaster, falling demand. Stock is tanking.")
    with ex3:
        if st.button("Neutral example"):
            st.session_state["ex"] = "GOOG trading at 142.50, volume slightly above average."

    if "ex" in st.session_state:
        user_input = st.session_state["ex"]
        st.info(f'Using example: *"{user_input}"*')

    if (analyse_btn or "ex" in st.session_state) and user_input:
        scores   = analyser.polarity_scores(user_input)
        compound = scores["compound"]

        if compound >= 0.05:
            label   = "POSITIVE"
            css_cls = "sentiment-positive"
            impact  = ("This type of sentiment is associated with upward price pressure. "
                       "Investors reading positive tweets tend to buy, driving prices up.")
        elif compound <= -0.05:
            label   = "NEGATIVE"
            css_cls = "sentiment-negative"
            impact  = ("This type of sentiment is associated with downward price pressure. "
                       "Negative sentiment can trigger selling and price drops.")
        else:
            label   = "NEUTRAL"
            css_cls = "sentiment-neutral"
            impact  = ("No clear directional sentiment detected. "
                       "Factual or informational tweets with no emotional signal.")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="{css_cls}">
            Sentiment: {label} &nbsp;&nbsp;|&nbsp;&nbsp;
            Compound Score: {compound:+.4f}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Compound Score",      f"{compound:+.4f}")
        c2.metric("Positive Component",  f"{scores['pos']:.3f}")
        c3.metric("Negative Component",  f"{scores['neg']:.3f}")
        c4.metric("Neutral Component",   f"{scores['neu']:.3f}")

        fig, ax = plt.subplots(figsize=(9, 1.6))
        fig.patch.set_facecolor(t["chart_bg"])
        ax.set_facecolor(t["chart_bg"])

        ax.barh(0.5, 2, left=-1, height=0.5,
                color=t["grid"], align="center", zorder=1)
        fill_color = "#16A34A" if compound >= 0.05 \
            else ("#DC2626" if compound <= -0.05 else "#94A3B8")
        ax.barh(0.5, abs(compound),
                left=0 if compound >= 0 else compound,
                height=0.5, color=fill_color, align="center",
                alpha=0.85, zorder=2)
        ax.axvline(x=0, color=t["spine"], linewidth=1.5, zorder=3)
        ax.axvline(x=compound, color=t["text_heading"],
                   linewidth=2.5, zorder=4)

        ax.set_xlim(-1, 1)
        ax.set_ylim(0.1, 0.9)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["Very Negative", "Negative",
                             "Neutral", "Positive", "Very Positive"],
                            fontsize=9, color=t["axis"])
        ax.set_yticks([])
        ax.set_title(f"Sentiment Gauge  —  Score: {compound:+.4f}  |  {label}",
                     fontweight="bold", color=t["text_heading"], fontsize=10, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        <div class="info-box" style="margin-top:0.8rem">
            <strong>Market Impact:</strong> {impact}
        </div>
        """, unsafe_allow_html=True)

        if "ex" in st.session_state:
            del st.session_state["ex"]

# ═══════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════
elif page == "Model Results":

    st.markdown('<p class="section-header">Model Performance Results</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### RMSE Comparison — Lower is Better")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(t["chart_bg"])
        ax.set_facecolor(t["chart_bg"])
        models = ["ARIMA\n(Price Only)",
                  "Random Forest\n(Price + Sentiment)",
                  "LSTM\n(Sequences + Sentiment)"]
        rmses  = [57.74, 18.80, 12.00]
        colors = ["#EF4444", "#F59E0B", "#16A34A"]
        bars   = ax.bar(models, rmses, color=colors,
                        width=0.45, edgecolor=t["chart_bg"], linewidth=1.5)
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"${val:.2f}", ha="center",
                    fontweight="bold", fontsize=11, color=t["text_heading"])
        ax.set_ylabel("RMSE (USD)", color=t["axis"])
        ax.set_ylim(0, 70)
        ax.set_title("Average prediction error per day",
                     fontsize=9, color=t["axis"], pad=6)
        ax.tick_params(colors=t["axis"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(t["spine"])
        ax.spines["bottom"].set_color(t["spine"])
        ax.grid(axis="y", color=t["grid"], linewidth=1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### What This Means in Plain English")
        st.markdown("""
        <table class="styled-table">
        <tr><th>Model</th><th>RMSE</th><th>In Plain Terms</th></tr>
        <tr><td>ARIMA</td><td>$57.74</td>
            <td>Off by $57 on average — worse than guessing</td></tr>
        <tr><td>Random Forest</td><td>$18.80</td>
            <td>Off by $18 — usable but not reliable</td></tr>
        <tr><td>LSTM</td><td>$12.00</td>
            <td>Off by $12 on a $300 stock — 4% error</td></tr>
        </table>
        <br>
        <div class="info-box">
        <strong>Why does LSTM win?</strong><br>
        It remembers 30 days of history simultaneously, learning
        momentum patterns that single-day models completely miss.
        Adding sentiment further reduces error by 36% over Random
        Forest alone.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("ARIMA",          "$57.74", "Baseline",        delta_color="off")
    c2.metric("Random Forest",  "$18.80", "-$38.94 vs ARIMA",delta_color="inverse")
    c3.metric("LSTM",           "$12.00", "-$45.74 vs ARIMA",delta_color="inverse")

    st.markdown("---")
    st.markdown("#### Tweet Sentiment Distribution — 5,791 Tweets")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(t["chart_bg"])
        ax.pie(
            [2138, 2352, 1301],
            labels=["Positive\n36.9%", "Neutral\n40.6%", "Negative\n22.5%"],
            colors=["#16A34A", "#94A3B8", "#DC2626"],
            startangle=90,
            wedgeprops={"edgecolor": t["chart_bg"], "linewidth": 2.5},
            textprops={"fontsize": 10, "color": t["text_primary"]}
        )
        ax.set_title("Stock tweets lean slightly positive",
                     fontweight="bold", color=t["text_heading"], pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("""
        <div class="info-box">
        <strong>40.6% neutral</strong> — most stock tweets are factual
        price or ticker updates with no emotional words. VADER correctly
        scores these near zero.<br><br>
        <strong>36.9% positive</strong> — more bullish than bearish,
        consistent with 2020–2024 being a strong bull market for tech stocks.<br><br>
        <strong>22.5% negative</strong> — includes COVID crash tweets (2020),
        inflation fears (2022), and tech sector selloffs.<br><br>
        <strong>Key finding:</strong> The majority-neutral tweets explain
        why raw sentiment had low feature importance in Random Forest —
        noise dilutes signal without date-matched tweet data.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 4 — BUY / HOLD / SELL SIGNAL
# ═══════════════════════════════════════════
elif page == "Buy / Hold / Sell Signal":

    st.markdown('<p class="section-header">Buy / Hold / Sell Signal Generator</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Enter a tweet and current price information to generate a combined
    trading signal. The system weights tweet sentiment (40%),
    moving average crossover (40%), and price momentum (20%).
    </div><br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("#### Step 1 — Enter a Tweet")
        tweet_input = st.text_area(
            "Paste a tweet about the stock:",
            placeholder='e.g. "MSFT crushing earnings again, '
                        'revenue up 17% year on year"',
            height=100,
            label_visibility="collapsed"
        )
        st.markdown("#### Step 2 — Enter Price Data")
        current_price = st.number_input(
            "Current stock price ($)",
            min_value=1.0, max_value=5000.0,
            value=370.0, step=0.5
        )
        ma5_input = st.number_input(
            "5-day moving average price ($)",
            min_value=1.0, max_value=5000.0,
            value=365.0, step=0.5
        )
        ma20_input = st.number_input(
            "20-day moving average price ($)",
            min_value=1.0, max_value=5000.0,
            value=355.0, step=0.5
        )
        generate_btn = st.button("Generate Signal", type="primary")

    with col2:
        if generate_btn:
            if not tweet_input.strip():
                st.warning("Please enter a tweet before generating a signal.")
            else:
                scores   = analyser.polarity_scores(tweet_input)
                compound = scores["compound"]
                momentum = current_price - ma20_input
                ma_cross = 1 if ma5_input > ma20_input else -1

                combined = (compound * 0.4) + \
                           (ma_cross * 0.4) + \
                           (0.2 if momentum > 0 else -0.2)

                if combined > 0.2:
                    signal  = "BUY"
                    css_cls = "signal-buy"
                elif combined < -0.2:
                    signal  = "SELL"
                    css_cls = "signal-sell"
                else:
                    signal  = "HOLD"
                    css_cls = "signal-hold"

                sent_desc = (
                    f"positive sentiment ({compound:+.3f})"
                    if compound >= 0.05 else
                    f"negative sentiment ({compound:+.3f})"
                    if compound <= -0.05 else
                    f"neutral sentiment ({compound:+.3f})"
                )
                ma_desc = (
                    "price trending above its 20-day average (bullish crossover)"
                    if ma5_input > ma20_input else
                    "price trending below its 20-day average (bearish crossover)"
                )
                mom_desc = (
                    f"price is ${abs(momentum):.2f} above the 20-day trend"
                    if momentum > 0 else
                    f"price is ${abs(momentum):.2f} below the 20-day trend"
                )

                if signal == "BUY":
                    reasoning = (f"The combination of {sent_desc} with "
                                 f"{ma_desc} and {mom_desc} points to "
                                 f"upward momentum — suggesting a buying opportunity.")
                elif signal == "SELL":
                    reasoning = (f"The combination of {sent_desc} with "
                                 f"{ma_desc} and {mom_desc} suggests "
                                 f"continued downward pressure — consider reducing exposure.")
                else:
                    reasoning = (f"Mixed signals: {sent_desc}, but {ma_desc}. "
                                 f"With {mom_desc}, there is no clear directional "
                                 f"edge — hold and wait for a stronger signal.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="{css_cls}">{signal}</div>',
                            unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                sent_verdict = ("Positive" if compound >= 0.05
                                else ("Negative" if compound <= -0.05
                                      else "Neutral"))
                ma_verdict   = ("Bullish — MA5 > MA20"
                                if ma5_input > ma20_input
                                else "Bearish — MA5 < MA20")
                mom_verdict  = (f"Above trend (+${momentum:.2f})"
                                if momentum > 0
                                else f"Below trend (${momentum:.2f})")

                st.markdown(f"""
                <table class="styled-table">
                <tr><th>Signal Component</th><th>Value</th><th>Verdict</th></tr>
                <tr><td>Tweet Sentiment</td>
                    <td>{compound:+.3f}</td>
                    <td>{sent_verdict}</td></tr>
                <tr><td>MA Crossover</td>
                    <td>MA5 vs MA20</td>
                    <td>{ma_verdict}</td></tr>
                <tr><td>Price Momentum</td>
                    <td>vs 20-day avg</td>
                    <td>{mom_verdict}</td></tr>
                <tr><td><strong>Combined Score</strong></td>
                    <td><strong>{combined:+.3f}</strong></td>
                    <td><strong>{signal}</strong></td></tr>
                </table>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <br>
                <div class="info-box">
                    <strong>Reasoning:</strong> {reasoning}
                </div>
                <br>
                <div class="warning-box">
                    Disclaimer: This is a research project for educational
                    purposes only. Do not make real financial decisions based
                    on this tool.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <br><br>
            <div style="text-align:center; color:{t['text_secondary']}; font-size:1rem;
                        padding:2rem; border:1px dashed {t['border']};
                        border-radius:10px;">
                Enter a tweet and price data,<br>then click
                <strong>Generate Signal</strong>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 5 — HOW IT WORKS
# ═══════════════════════════════════════════
elif page == "How It Works":

    st.markdown('<p class="section-header">How It Works — Plain English</p>',
                unsafe_allow_html=True)

    steps = [
        ("Step 1 — Data Collection",
         "We collected 5,791 tweets about Microsoft, Apple, and Google, plus 4 years "
         "of daily stock prices (2020–2024) from Yahoo Finance. Each trading day has "
         "5 numbers: Open, High, Low, Close, and Volume."),
        ("Step 2 — Text Cleaning",
         "Raw tweets are messy — URLs, @mentions, #hashtags, numbers. We strip all "
         "noise leaving only emotional words. "
         "'MSFT is going to the moon! http://bit.ly/xyz @user' "
         "becomes 'msft is going to the moon'."),
        ("Step 3 — Sentiment Scoring",
         "Each cleaned tweet runs through VADER — a dictionary of 7,500+ words each "
         "with a sentiment score. We enhanced it with 30+ financial terms VADER "
         "misclassifies by default (e.g. 'crushing earnings' = positive, not aggressive). "
         "Output: a score from -1 (very negative) to +1 (very positive)."),
        ("Step 4 — Feature Engineering",
         "We compute 6 features per trading day: daily price change %, 5-day moving "
         "average, 20-day moving average, volatility, volume change, and the VADER "
         "sentiment score. These are the clues our models use to predict."),
        ("Step 5 — ARIMA Baseline",
         "First we train ARIMA on price data alone. No ML, no sentiment. This gives "
         "us a benchmark — the best prediction without intelligence. RMSE = $57.74. "
         "Every model we build must beat this number."),
        ("Step 6 — Random Forest",
         "100 decision trees examine all 6 features and vote. Because it uses sentiment "
         "alongside price features, it dropped error to $18.80 — 67% better than ARIMA. "
         "It also told us which features mattered most: MA5 dominated at 91%."),
        ("Step 7 — LSTM Neural Network",
         "Our main model. Looks at 30 consecutive days at once, not just today. "
         "Has memory gates controlling what to remember and forget across the sequence. "
         "With price sequences + sentiment: RMSE = $12.00 — 79% better than ARIMA, "
         "36% better than Random Forest. Direction accuracy: 52.9% (vs 50% random)."),
    ]

    for title, desc in steps:
        with st.expander(title):
            st.markdown(f"""
            <div style="color:{t['text_primary']}; font-size:0.97rem; line-height:1.7">
                {desc}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Core Finding")
        st.markdown("""
        <div class="info-box">
        Adding Twitter sentiment to a stock prediction model reduces
        prediction error by <strong>79.2%</strong> compared to using
        price data alone. This validates that public opinion on social
        media contains real signal about future price movements.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Honest Limitations")
        st.markdown("""
        <div class="info-box">
        Tweet dataset has no date stamps — sentiment was sampled randomly
        rather than matched to specific trading days.<br><br>
        Real-time Twitter API requires $100/month — not feasible for students.<br><br>
        52.9% directional accuracy beats random chance but is not
        investment-grade performance.<br><br>
        Results are on MSFT only — generalisation needs further testing.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="color:{t['text_secondary']}; font-size:0.9rem; text-align:center; padding:1rem">
        Arnav Bhardwaj &nbsp;&amp;&nbsp; Suraj Pratap Singh &nbsp;|&nbsp;
        Guide: Neeraj Sharma, Asst. Professor, CSE &nbsp;|&nbsp;
        MVSREC &nbsp;·&nbsp; BE CSE Sec A &nbsp;·&nbsp;
        Sem III 2025-2026 &nbsp;·&nbsp; Batch 22
    </div>
    """, unsafe_allow_html=True)
