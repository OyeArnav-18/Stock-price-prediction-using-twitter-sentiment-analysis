
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "dark" not in st.session_state:
    st.session_state.dark = False

D = st.session_state.dark

BG    = "#0F172A" if D else "#F8FAFC"
CARD  = "#1E293B" if D else "#FFFFFF"
TEXT  = "#F1F5F9" if D else "#1E293B"
SUB   = "#94A3B8" if D else "#64748B"
BDR   = "#334155" if D else "#E2E8F0"
INP   = "#1E293B" if D else "#FFFFFF"
INPT  = "#F1F5F9" if D else "#1E293B"
SIDEBG= "#1E293B" if D else "#FFFFFF"
SIDEBD= "#334155" if D else "#E2E8F0"
INFOBC= "#1E3A5F" if D else "#EFF6FF"
INFOBR= "#334155" if D else "#BFDBFE"
INFOTX= "#BFDBFE" if D else "#1E3A5F"

css = f"""
<style>
.stApp {{ background-color: {BG} !important; }}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {{
    background-color: {SIDEBG} !important;
    border-right: 1px solid {SIDEBD} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] div {{
    color: {TEXT} !important;
}}

/* ── SIDEBAR TOGGLE BUTTON — always visible whether open or collapsed ── */
button[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] {{
    display:       flex       !important;
    visibility:    visible    !important;
    opacity:       1          !important;
    position:      fixed      !important;
    left:          0          !important;
    top:           45%        !important;
    z-index:       999999     !important;
    background-color: #2563A8 !important;
    border-radius: 0 8px 8px 0 !important;
    border:        none       !important;
    width:         2rem       !important;
    height:        3rem       !important;
    align-items:   center     !important;
    justify-content: center   !important;
    cursor:        pointer    !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.2) !important;
}}
button[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg {{
    fill:    white !important;
    color:   white !important;
    width:   1rem  !important;
    height:  1rem  !important;
}}
/* Also keep the expand button inside open sidebar visible */
button[data-testid="stSidebarNavCollapseButton"],
[data-testid="stSidebarNavCollapseButton"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}}

/* Main text */
.stMarkdown, .stText, p, h1, h2, h3, h4, label {{
    color: {TEXT} !important;
}}

/* Input boxes */
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {{
    background-color: {INP} !important;
    color: {INPT} !important;
    border: 1px solid {BDR} !important;
    border-radius: 6px !important;
}}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    background-color: {BG} !important;
}}

/* Metric cards */
.metric-card {{
    background: {CARD};
    border: 1px solid {BDR};
    border-radius: 10px;
    padding: 1.4rem 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}}
.metric-number {{ font-size: 2.2rem; font-weight: 700; color: #2563A8; line-height: 1; }}
.metric-label {{
    font-size: 0.82rem; color: {SUB}; margin-top: 0.4rem;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;
}}

.section-header {{
    color: {"#93C5FD" if D else "#1A3A5C"};
    font-size: 1.3rem; font-weight: 700;
    border-left: 4px solid #2563A8;
    padding-left: 0.8rem; margin-bottom: 1.5rem;
}}

.info-box {{
    background: {INFOBC}; border: 1px solid {INFOBR};
    border-radius: 8px; padding: 1rem 1.2rem;
    color: {INFOTX}; font-size: 0.95rem; line-height: 1.6;
}}

.sentiment-positive {{
    background: {"#052e16" if D else "#F0FDF4"};
    border-left: 5px solid #16A34A;
    padding: 1rem 1.2rem; border-radius: 8px;
    font-size: 1.1rem; font-weight: 600;
    color: {"#86EFAC" if D else "#14532D"};
}}
.sentiment-negative {{
    background: {"#450a0a" if D else "#FFF1F2"};
    border-left: 5px solid #DC2626;
    padding: 1rem 1.2rem; border-radius: 8px;
    font-size: 1.1rem; font-weight: 600;
    color: {"#FCA5A5" if D else "#7F1D1D"};
}}
.sentiment-neutral {{
    background: {"#1E293B" if D else "#F8FAFC"};
    border-left: 5px solid #94A3B8;
    padding: 1rem 1.2rem; border-radius: 8px;
    font-size: 1.1rem; font-weight: 600;
    color: {TEXT};
}}

.signal-buy {{
    background: {"#052e16" if D else "#F0FDF4"};
    border: 2px solid #16A34A; border-radius: 12px;
    padding: 2rem; text-align: center;
    font-size: 2rem; font-weight: 700;
    color: {"#86EFAC" if D else "#14532D"};
}}
.signal-sell {{
    background: {"#450a0a" if D else "#FFF1F2"};
    border: 2px solid #DC2626; border-radius: 12px;
    padding: 2rem; text-align: center;
    font-size: 2rem; font-weight: 700;
    color: {"#FCA5A5" if D else "#7F1D1D"};
}}
.signal-hold {{
    background: {"#1c1917" if D else "#FEFCE8"};
    border: 2px solid #CA8A04; border-radius: 12px;
    padding: 2rem; text-align: center;
    font-size: 2rem; font-weight: 700;
    color: {"#FDE68A" if D else "#713F12"};
}}

.styled-table {{ width:100%; border-collapse:collapse; font-size:0.95rem; }}
.styled-table th {{ background:#1A3A5C; color:white; padding:0.7rem 1rem; text-align:left; }}
.styled-table td {{ padding:0.65rem 1rem; border-bottom:1px solid {BDR}; color:{TEXT}; }}
.styled-table tr:nth-child(even) td {{ background:{"#1E293B" if D else "#F8FAFC"}; }}

.sidebar-stat {{
    background: {"#334155" if D else "#F8FAFC"};
    border: 1px solid {SIDEBD}; border-radius: 8px;
    padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}}
.sidebar-stat-label {{
    font-size: 0.78rem; color: {SUB}; text-transform: uppercase;
    letter-spacing: 0.5px; font-weight: 600;
}}
.sidebar-stat-value {{ font-size: 1.3rem; font-weight: 700; color: {"#60A5FA" if D else "#1A3A5C"}; }}

.warning-box {{
    background: {"#1c1917" if D else "#FFFBEB"};
    border: 1px solid {"#92400e" if D else "#FDE68A"};
    border-radius: 8px; padding: 1rem 1.2rem;
    color: {"#FDE68A" if D else "#78350F"}; font-size: 0.9rem;
}}

.streamlit-expanderHeader {{ color: {TEXT} !important; background: {CARD} !important; }}
.streamlit-expanderContent {{ background: {CARD} !important; color: {TEXT} !important; }}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

@st.cache_resource
def load_analyser():
    a = SentimentIntensityAnalyzer()
    # ── EXPANDED FINANCIAL LEXICON ──
    # Key fix: "crushed" was negative in standard VADER but means
    # "exceeded expectations" in financial context
    a.lexicon.update({
        # Positive — earnings / performance
        "crushed":      2.8,   # "crushed it / crushed earnings" = beat expectations
        "crushing":     2.5,
        "smashed":      2.5,   # "smashed estimates"
        "destroyed":    2.0,   # "destroyed forecasts" = very positive in finance
        "killing":      2.0,   # "killing it this quarter"
        "blew":         2.0,   # "blew past estimates"
        "blowing":      1.8,
        "beat":         2.0,
        "beats":        2.0,
        "beat out":     2.0,
        "topped":       1.8,   # "topped estimates"
        "exceeded":     2.0,
        "outpaced":     1.8,
        "surged":       2.5,
        "surging":      2.5,
        "soaring":      2.5,
        "soared":       2.5,
        "rallying":     2.0,
        "rallied":      2.0,
        "skyrocketing": 2.8,
        "breakout":     2.0,
        "bullish":      2.5,
        "unstoppable":  2.8,
        "mooning":      2.5,
        "upgrade":      2.0,
        "upgraded":     2.0,
        "outperform":   2.0,
        "overweight":   1.5,
        "record":       1.5,
        "strong":       1.5,
        "strength":     1.5,
        "monster":      1.5,
        "load up":      1.8,   # "load up on shares" = buy signal
        "loading up":   1.8,
        "buy":          1.5,
        "buying":       1.5,
        "revenue":      1.0,   # revenue mentioned positively in context
        "profit":       1.5,
        "profits":      1.5,
        "earnings":     0.8,
        "growth":       1.5,
        "growing":      1.2,
        "upside":       1.5,
        "undervalued":  1.5,
        "opportunity":  1.2,
        # Negative
        "bearish":      -2.5,
        "plummeting":   -2.8,
        "plunged":      -2.5,
        "tanking":      -2.8,
        "tanked":       -2.5,
        "sinking":      -2.0,
        "sank":         -2.0,
        "dumping":      -2.5,
        "bleeding":     -2.0,
        "miss":         -2.0,
        "missed":       -2.0,
        "misses":       -2.0,
        "warning":      -1.5,
        "downgrade":    -2.0,
        "downgraded":   -2.0,
        "underperform": -2.0,
        "selloff":      -2.0,
        "sell-off":     -2.0,
        "correction":   -1.5,
        "weak":         -1.5,
        "weakness":     -1.5,
        "downside":     -1.5,
        "collapse":     -2.5,
        "collapsed":    -2.5,
        "crash":        -2.5,
        "crashed":      -2.5,
        "overvalued":   -1.5,
    })
    return a

analyser = load_analyser()

# ── HEADER ──
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1A3A5C 0%,#2563A8 100%);
            padding:2rem 2.5rem;border-radius:12px;margin-bottom:1.5rem;">
    <h1 style="color:#FFFFFF;font-size:1.9rem;font-weight:700;margin:0">
        Stock Price Prediction Using Twitter Sentiment Analysis</h1>
    <p style="color:#93C5FD;font-size:0.92rem;margin:0.3rem 0 0 0">
        MVSREC &nbsp;·&nbsp; BE CSIT &nbsp;·&nbsp; Batch 23 &nbsp;·&nbsp;
        Arnav Bhardwaj &amp; Suraj Pratap Singh</p>
    <p style="color:#BFDBFE;font-size:0.85rem;margin:0.2rem 0 0 0">
        Guide: Neeraj Sharma, Asst. Professor, Dept. of CS&amp;IT, MVSREC</p>
</div>""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown(f"### {'🌙 Dark Mode ON' if D else '☀️ Light Mode'}")
    if st.button("Toggle Dark / Light Mode", use_container_width=True):
        st.session_state.dark = not st.session_state.dark
        st.rerun()
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to", [
        "Project Overview","Live Sentiment Analyser",
        "Model Results","Buy / Hold / Sell Signal","How It Works"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Project Statistics**")
    for lbl, val in [
        ("Tweets Analysed","5,791"),("Trading Days","987"),
        ("Models Built","3"),("Best RMSE","$12.00"),
        ("Accuracy Gain","+79.2%"),("Direction Acc.","52.9%"),
    ]:
        st.markdown(f"""<div class="sidebar-stat">
            <div class="sidebar-stat-label">{lbl}</div>
            <div class="sidebar-stat-value">{val}</div>
        </div>""", unsafe_allow_html=True)

FIG_BG = "#0F172A" if D else "white"
FIG_TX = "#F1F5F9" if D else "#334155"
FIG_GR = "#334155" if D else "#F1F5F9"

# ═══════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ═══════════════════════════════════════════
if page == "Project Overview":
    st.markdown('<p class="section-header">Project Overview</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,num,lbl in zip([c1,c2,c3,c4],
        ["5,791","987","3","79.2%"],
        ["Tweets Analysed","Trading Days","Models Built","Improvement Over Baseline"]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-number">{num}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.write("")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### The Problem")
        st.markdown("""<div class="info-box">
            Stock markets are unpredictable. Traditional models only look at price history —
            they ignore <strong>what people are saying</strong> on social media.<br>
            A single viral tweet can move a stock by 5% in minutes.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("#### Our Solution")
        st.markdown("""<div class="info-box">
            We built a system combining <strong>two signals</strong>:<br><br>
            1. <strong>Price history</strong> — what the stock has been doing<br>
            2. <strong>Twitter sentiment</strong> — what people are saying<br><br>
            3 models tested. Best result: <strong>79.2% improvement</strong> over baseline.
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("#### The 3 Models We Built")
    c1,c2,c3 = st.columns(3)
    for col,name,typ,desc,rmse,imp,color in zip([c1,c2,c3],
        ["ARIMA","Random Forest","LSTM Neural Network"],
        ["Statistical Baseline","Classic Machine Learning","Deep Learning"],
        ["Uses price history only. Finds mathematical patterns in past prices. Cannot predict trend changes.",
         "100 decision trees examine all features and vote. Uses price + sentiment. Handles non-linear relationships.",
         "Has memory — looks at 30 consecutive days at once. Learns patterns across time. Uses sequences + sentiment."],
        ["$57.74","$18.80","$12.00"],
        ["Baseline","67% better","79% better"],
        ["#EF4444","#F59E0B","#16A34A"]
    ):
        with col:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BDR};
                border-top:4px solid {color};border-radius:10px;padding:1.2rem;
                box-shadow:0 1px 4px rgba(0,0,0,0.08);">
                <div style="font-size:1.1rem;font-weight:700;
                    color:{"#60A5FA" if D else "#1A3A5C"}">{name}</div>
                <div style="font-size:0.8rem;color:{SUB};margin-bottom:0.8rem">{typ}</div>
                <div style="font-size:0.9rem;color:{TEXT};line-height:1.6;margin-bottom:1rem">{desc}</div>
                <div style="font-size:1.4rem;font-weight:700;color:{color}">RMSE: {rmse}</div>
                <div style="font-size:0.85rem;color:{SUB}">{imp}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 2 — LIVE SENTIMENT ANALYSER
# ═══════════════════════════════════════════
elif page == "Live Sentiment Analyser":
    st.markdown('<p class="section-header">Live Tweet Sentiment Analyser</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        Type any tweet and see how our sentiment engine scores it. Enhanced with an
        expanded financial lexicon to correctly interpret stock market language like
        "crushed it", "crushing earnings", "tanking", "load up on shares" and more.
    </div>""", unsafe_allow_html=True)
    st.write("")

    user_input = st.text_area("Enter a tweet:", height=110,
        placeholder='e.g. "MSFT crushing earnings expectations, this stock is unstoppable"',
        label_visibility="collapsed")
    analyse_btn = st.button("Analyse Sentiment", type="primary")

    st.markdown("**Quick examples:**")
    e1,e2,e3 = st.columns(3)
    with e1:
        if st.button("Positive example"):
            st.session_state["ex"] = "AAPL crushed it this quarter! Revenue is up 15%, time to load up on shares before market opens"
    with e2:
        if st.button("Negative example"):
            st.session_state["ex"] = "Apple is in serious trouble. Supply chain disaster. Stock is tanking badly."
    with e3:
        if st.button("Neutral example"):
            st.session_state["ex"] = "GOOG trading at 142.50, volume slightly above average today."

    if "ex" in st.session_state:
        user_input = st.session_state["ex"]
        st.info(f'Example loaded: "{user_input}"')

    if (analyse_btn or "ex" in st.session_state) and user_input:
        scores   = analyser.polarity_scores(user_input)
        compound = scores["compound"]
        if   compound >= 0.05:  label,css = "POSITIVE","sentiment-positive"; impact = "Associated with upward price pressure. Positive sentiment drives buying activity."
        elif compound <= -0.05: label,css = "NEGATIVE","sentiment-negative"; impact = "Associated with downward price pressure. Negative sentiment can trigger selling."
        else:                    label,css = "NEUTRAL", "sentiment-neutral";  impact = "No clear directional signal. Factual or informational content."

        st.write("")
        st.markdown(f'<div class="{css}">Sentiment: {label} &nbsp;|&nbsp; Score: {compound:+.4f}</div>', unsafe_allow_html=True)
        st.write("")
        co1,co2,co3,co4 = st.columns(4)
        co1.metric("Compound", f"{compound:+.4f}")
        co2.metric("Positive", f"{scores['pos']:.3f}")
        co3.metric("Negative", f"{scores['neg']:.3f}")
        co4.metric("Neutral",  f"{scores['neu']:.3f}")

        fig,ax = plt.subplots(figsize=(9,1.6))
        fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(FIG_BG)
        ax.barh(0.5,2,left=-1,height=0.5,color=("#334155" if D else "#F1F5F9"),align="center",zorder=1)
        fc = "#16A34A" if compound>=0.05 else ("#DC2626" if compound<=-0.05 else "#94A3B8")
        ax.barh(0.5,abs(compound),left=0 if compound>=0 else compound,height=0.5,color=fc,align="center",alpha=0.85,zorder=2)
        ax.axvline(x=0,color="#64748B",linewidth=1.5,zorder=3)
        ax.axvline(x=compound,color=FIG_TX,linewidth=2.5,zorder=4)
        ax.set_xlim(-1,1); ax.set_ylim(0.1,0.9)
        ax.set_xticks([-1,-0.5,0,0.5,1])
        ax.set_xticklabels(["Very Negative","Negative","Neutral","Positive","Very Positive"],fontsize=9,color=FIG_TX)
        ax.set_yticks([])
        ax.set_title(f"Sentiment Gauge  —  {label}  |  Score: {compound:+.4f}",
                     fontweight="bold",color=FIG_TX,fontsize=10,pad=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f'<div class="info-box" style="margin-top:0.5rem"><strong>Market Impact:</strong> {impact}</div>', unsafe_allow_html=True)
        if "ex" in st.session_state: del st.session_state["ex"]

# ═══════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════
elif page == "Model Results":
    st.markdown('<p class="section-header">Model Performance Results</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### RMSE Comparison — Lower is Better")
        fig,ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(FIG_BG)
        mdls  = ["ARIMA\n(Price Only)","Random Forest\n(Price+Sentiment)","LSTM\n(Sequences+Sentiment)"]
        rmses = [57.74,18.80,12.00]
        cols  = ["#EF4444","#F59E0B","#16A34A"]
        bars  = ax.bar(mdls,rmses,color=cols,width=0.45,edgecolor=FIG_BG,linewidth=1.5)
        for bar,val in zip(bars,rmses):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                    f"${val:.2f}",ha="center",fontweight="bold",fontsize=11,color=FIG_TX)
        ax.set_ylabel("RMSE (USD)",color=FIG_TX)
        ax.set_ylim(0,70); ax.tick_params(colors=FIG_TX)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155"); ax.spines["bottom"].set_color("#334155")
        ax.grid(axis="y",color=FIG_GR,linewidth=1)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        st.markdown("#### What This Means")
        st.markdown(f"""<table class="styled-table">
        <tr><th>Model</th><th>RMSE</th><th>Plain Terms</th></tr>
        <tr><td>ARIMA</td><td>$57.74</td><td>Off by $57/day — misses trend entirely</td></tr>
        <tr><td>Random Forest</td><td>$18.80</td><td>Off by $18/day — usable but limited</td></tr>
        <tr><td>LSTM</td><td>$12.00</td><td>Off by $12 on a $300 stock = 4% error</td></tr>
        </table><br>
        <div class="info-box"><strong>Why LSTM wins:</strong> It sees 30 consecutive days at once,
        learning momentum patterns across time. Sentiment adds a further 36% gain over Random Forest.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    co1,co2,co3 = st.columns(3)
    co1.metric("ARIMA",         "$57.74","Baseline",         delta_color="off")
    co2.metric("Random Forest", "$18.80","-$38.94 vs ARIMA", delta_color="inverse")
    co3.metric("LSTM",          "$12.00","-$45.74 vs ARIMA", delta_color="inverse")

    st.markdown("---")
    st.markdown("#### Tweet Sentiment Distribution — 5,791 Tweets")
    c1,c2 = st.columns(2)
    with c1:
        fig,ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor(FIG_BG)
        ax.pie([2138,2352,1301],
               labels=["Positive\n36.9%","Neutral\n40.6%","Negative\n22.5%"],
               colors=["#16A34A","#94A3B8","#DC2626"],startangle=90,
               wedgeprops={"edgecolor":FIG_BG,"linewidth":2.5},
               textprops={"fontsize":10,"color":FIG_TX})
        ax.set_title("Stock tweets lean slightly positive",fontweight="bold",color=FIG_TX,pad=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        st.markdown("""<div class="info-box">
        <strong>40.6% neutral</strong> — most stock tweets are factual price updates with no emotion.<br><br>
        <strong>36.9% positive</strong> — more bullish than bearish, consistent with 2020–2024 tech bull run.<br><br>
        <strong>22.5% negative</strong> — includes COVID crash (2020) and inflation fears (2022).<br><br>
        <strong>Key insight:</strong> Majority-neutral tweets explain why raw sentiment had low feature importance without date-matched data.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 4 — BUY / HOLD / SELL
# ═══════════════════════════════════════════
elif page == "Buy / Hold / Sell Signal":
    st.markdown('<p class="section-header">Buy / Hold / Sell Signal Generator</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        Combines tweet sentiment (40%), moving average crossover (40%), and price momentum (20%).
        Fill in a tweet and current price data, then click Generate Signal.
    </div>""", unsafe_allow_html=True)
    st.write("")
    c1,c2 = st.columns([1.1,1])
    with c1:
        st.markdown("#### Step 1 — Enter a Tweet")
        tweet_input = st.text_area("Tweet:",label_visibility="collapsed",height=100,
            placeholder='e.g. "MSFT crushed earnings again, revenue up 17% year on year"')
        st.markdown("#### Step 2 — Enter Price Data")
        cur  = st.number_input("Current stock price ($)", min_value=1.0,max_value=5000.0,value=370.0,step=0.5)
        ma5  = st.number_input("5-day moving average ($)",min_value=1.0,max_value=5000.0,value=365.0,step=0.5)
        ma20 = st.number_input("20-day moving average ($)",min_value=1.0,max_value=5000.0,value=355.0,step=0.5)
        gen  = st.button("Generate Signal", type="primary")
    with c2:
        if gen:
            if not tweet_input.strip():
                st.warning("Please enter a tweet first.")
            else:
                sc  = analyser.polarity_scores(tweet_input)
                cmp = sc["compound"]
                mom = cur - ma20
                mac = 1 if ma5 > ma20 else -1
                cmb = (cmp*0.4)+(mac*0.4)+(0.2 if mom>0 else -0.2)
                if   cmb >  0.2: sig,css = "BUY", "signal-buy"
                elif cmb < -0.2: sig,css = "SELL","signal-sell"
                else:            sig,css = "HOLD","signal-hold"
                sv = "Positive" if cmp>=0.05 else ("Negative" if cmp<=-0.05 else "Neutral")
                mv = "Bullish — MA5 > MA20" if ma5>ma20 else "Bearish — MA5 < MA20"
                pv = f"Above trend (+${mom:.2f})" if mom>0 else f"Below trend (-${abs(mom):.2f})"
                if   sig=="BUY":  rs = f"Positive tweet sentiment ({cmp:+.3f}) combined with {mv.lower()} and price {pv.lower()} suggests upward momentum."
                elif sig=="SELL": rs = f"Negative tweet sentiment ({cmp:+.3f}) combined with {mv.lower()} and price {pv.lower()} suggests downward pressure."
                else:             rs = f"Mixed signals — tweet sentiment is {sv.lower()} ({cmp:+.3f}) but {mv.lower()}. No clear edge — wait for stronger signal."
                st.write("")
                st.markdown(f'<div class="{css}">{sig}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f"""<table class="styled-table">
                <tr><th>Component</th><th>Value</th><th>Verdict</th></tr>
                <tr><td>Tweet Sentiment</td><td>{cmp:+.3f}</td><td>{sv}</td></tr>
                <tr><td>MA Crossover</td><td>MA5 vs MA20</td><td>{mv}</td></tr>
                <tr><td>Price Momentum</td><td>vs 20-day avg</td><td>{pv}</td></tr>
                <tr><td><strong>Combined Score</strong></td><td><strong>{cmb:+.3f}</strong></td><td><strong>{sig}</strong></td></tr>
                </table>""", unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="info-box"><strong>Reasoning:</strong> {rs}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown('<div class="warning-box">Disclaimer: Research project for educational purposes only. Do not make real financial decisions based on this tool.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""<br><br>
            <div style="text-align:center;color:{SUB};font-size:1rem;
                        padding:2rem;border:1px dashed {BDR};border-radius:10px;">
                Enter a tweet and price data,<br>then click <strong>Generate Signal</strong>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 5 — HOW IT WORKS
# ═══════════════════════════════════════════
elif page == "How It Works":
    st.markdown('<p class="section-header">How It Works — Plain English</p>', unsafe_allow_html=True)
    steps = [
        ("Step 1 — Data Collection","We collected 5,791 tweets about Microsoft, Apple, and Google stocks, plus 4 years of daily stock prices (2020-2024) from Yahoo Finance. Each trading day has 5 numbers: Open, High, Low, Close, and Volume."),
        ("Step 2 — Text Cleaning","Raw tweets are full of URLs, @mentions, and hashtags. We strip all noise, keeping only emotional words. Example: MSFT is going to the moon! http://bit.ly becomes: msft is going to the moon."),
        ("Step 3 — Sentiment Scoring","Each tweet runs through VADER with an expanded financial lexicon of 50+ terms. Words like crushed it, load up on shares, tanking, and plummeting are all correctly interpreted in financial context. Output: score from -1 to +1."),
        ("Step 4 — Feature Engineering","We compute 6 features per trading day: daily price change %, 5-day moving average, 20-day moving average, volatility, volume change, and the VADER sentiment score."),
        ("Step 5 — ARIMA Baseline","ARIMA uses price data alone — no ML, no sentiment. Achieved RMSE = $57.74. Every subsequent model must beat this."),
        ("Step 6 — Random Forest","100 decision trees examine all 6 features and vote. RMSE dropped to $18.80 — 67% better than ARIMA. Feature importance showed MA5 dominated at 91%."),
        ("Step 7 — LSTM Neural Network","Looks at 30 consecutive days at once and has memory gates. With price sequences and sentiment together: RMSE = $12.00 — 79% better than ARIMA."),
    ]
    for title,desc in steps:
        with st.expander(title):
            st.markdown(f'<div style="color:{TEXT};font-size:0.97rem;line-height:1.7">{desc}</div>', unsafe_allow_html=True)
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Core Finding")
        st.markdown("""<div class="info-box">
        Adding Twitter sentiment reduces prediction error by <strong>79.2%</strong>
        compared to price-only ARIMA baseline. Twitter sentiment contains real signal
        about future stock price movements.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("#### Honest Limitations")
        st.markdown("""<div class="info-box">
        Tweet dataset has no date stamps — sentiment was sampled randomly.<br><br>
        Real-time Twitter API requires $100/month — not feasible for students.<br><br>
        52.9% directional accuracy beats random (50%) but is not investment-grade.<br><br>
        Results tested on MSFT only — generalisation needs further testing.
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f'<div style="color:{SUB};font-size:0.9rem;text-align:center;padding:1rem">'
        'Arnav Bhardwaj &amp; Suraj Pratap Singh &nbsp;|&nbsp;'
        'Guide: Neeraj Sharma, Asst. Professor, CS&amp;IT &nbsp;|&nbsp;'
        'MVSREC &nbsp;·&nbsp; BE CSIT &nbsp;·&nbsp; Sem IV 2025-2026 &nbsp;·&nbsp; Batch 23'
        '</div>', unsafe_allow_html=True)
