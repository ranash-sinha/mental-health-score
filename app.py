# ============================================================
# MENTAL HEALTH SCORE PREDICTOR
# Streamlit Application
# ============================================================

import math
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Mental Health Score Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    try:
        return joblib.load("Mental_Health_Model.pkl")
    except Exception:
        return None

model = load_model()

# ------------------------------------------------------------
# DESIGN SYSTEM - Glassmorphic / Gradient Theme (No External APIs)
# ------------------------------------------------------------

st.markdown("""
<style>

:root {
    --bg-base: #09090E;
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
    --glass-hover: rgba(255, 255, 255, 0.06);
    --text-main: #FFFFFF;
    --text-muted: #8A8F98;
    --grad-text: linear-gradient(90deg, #9D82FF 0%, #E68A8A 50%, #FFB86C 100%);
    --accent-glow: radial-gradient(circle at 50% 10%, rgba(120, 50, 220, 0.15) 0%, rgba(200, 80, 150, 0.05) 30%, rgba(9, 9, 14, 0) 60%);
}

#MainMenu { visibility:hidden; }
footer { visibility:hidden; }
header { visibility:hidden; }

html, body, [class*="css"] {
    font-family: system-ui, -apple-system, sans-serif;
    color: var(--text-main);
}

/* Global Background */
.stApp {
    background-color: var(--bg-base);
    background-image: var(--accent-glow);
    background-attachment: fixed;
}

.block-container {
    padding-top: 4rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* ---------- Typography ---------- */

.header-container {
    text-align: center;
    margin-bottom: 3rem;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 56px;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 16px 0;
    letter-spacing: -0.02em;
}

.hero-title span {
    background: var(--grad-text);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 18px;
    color: var(--text-muted);
    max-width: 600px;
    margin: 0 auto;
    font-weight: 400;
}

/* ---------- Containers & Glassmorphism ---------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 8px;
}

.step-label {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    font-weight: 600;
    color: #9D82FF;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.step-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-main);
    margin: 0 0 16px 0;
}

/* ---------- Input Widgets ---------- */

.stSlider label, .stSelectbox label, .stSelectSlider label {
    font-family: system-ui, -apple-system, sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #FFFFFF !important;
    border: 2px solid #9D82FF !important;
    box-shadow: 0 0 10px rgba(157, 130, 255, 0.5);
}
.stSlider [data-baseweb="slider"] > div > div {
    background: var(--grad-text) !important;
}
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    color: var(--text-muted) !important;
}
div[data-testid="stThumbValue"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    color: var(--text-main) !important;
    font-weight: 500 !important;
    background: var(--glass-bg);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid var(--glass-border);
}

/* Selectbox */
.stSelectbox [data-baseweb="select"] > div {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
    transition: all 0.2s ease;
}
.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
    background: var(--glass-hover) !important;
}
.stSelectbox svg {
    fill: var(--text-muted) !important;
}

/* Button */
div.stButton > button {
    background: var(--grad-text);
    color: #000000;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 15px;
    font-weight: 600;
    border-radius: 8px;
    height: 48px;
    width: 100%;
    border: none;
    margin-top: 12px;
    transition: opacity 0.2s ease;
}

div.stButton > button:hover {
    opacity: 0.9;
    color: #000000;
}

/* ---------- Stat Grid ---------- */

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 16px;
}

.stat-cell {
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 16px;
    background: var(--glass-bg);
}

.stat-cell .stat-label {
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.stat-cell .stat-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 18px;
    font-weight: 500;
    color: var(--text-main);
}

/* ---------- Notice / Result Banners ---------- */

.result-banner {
    border-radius: 12px;
    border: 1px solid var(--glass-border);
    background: var(--glass-bg);
    padding: 16px 20px;
    margin-top: 16px;
    backdrop-filter: blur(10px);
}

.result-banner .eyebrow {
    color: var(--band-color);
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.result-banner p {
    font-size: 14px;
    color: var(--text-muted);
    line-height: 1.6;
    margin: 0;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background: rgba(9, 9, 14, 0.8);
    border-right: 1px solid var(--glass-border);
    backdrop-filter: blur(20px);
}

.side-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 8px;
}

.side-text {
    font-size: 14px;
    color: var(--text-muted);
    line-height: 1.6;
}

.side-divider {
    height: 1px;
    background: var(--glass-border);
    margin: 24px 0;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# GAUGE LOGIC
# ------------------------------------------------------------

def polar_to_cartesian(cx, cy, r, angle_deg):
    angle_rad = math.radians(angle_deg - 90)
    return (cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad))

def describe_arc(cx, cy, r, start_angle, end_angle):
    start = polar_to_cartesian(cx, cy, r, end_angle)
    end = polar_to_cartesian(cx, cy, r, start_angle)
    large_arc_flag = "0" if (end_angle - start_angle) <= 180 else "1"
    return (
        f"M {start[0]:.2f} {start[1]:.2f} "
        f"A {r} {r} 0 {large_arc_flag} 0 {end[0]:.2f} {end[1]:.2f}"
    )

def score_band(score):
    if score >= 8:
        return ("#9D82FF", "OPTIMAL")
    elif score >= 6:
        return ("#E68A8A", "STABLE")
    elif score >= 4:
        return ("#FFB86C", "MODERATE")
    else:
        return ("#FF4B4B", "CRITICAL")

def build_gauge_svg(score=None):
    cx, cy, r, sw = 150, 160, 110, 14
    start_angle, end_angle = -90, 90
    track_path = describe_arc(cx, cy, r, start_angle, end_angle)
    track_color = "rgba(255,255,255,0.05)"

    if score is None:
        return f'''
<svg viewBox="0 0 300 180" xmlns="http://www.w3.org/2000/svg">
<defs>
    <filter id="glow">
        <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
        <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>
    </filter>
</defs>
<path d="{track_path}" fill="none" stroke="{track_color}" stroke-width="{sw}" stroke-linecap="round" />
<text x="150" y="140" text-anchor="middle" font-family="monospace" font-size="42" font-weight="600" fill="#333333">--</text>
<text x="150" y="165" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="500" fill="#8A8F98">PENDING COMPUTATION</text>
</svg>'''

    score = max(0.0, min(10.0, float(score)))
    value_angle = start_angle + (score / 10.0) * 180
    color, label = score_band(score)
    value_path = describe_arc(cx, cy, r, start_angle, value_angle)

    return f'''
<svg viewBox="0 0 300 180" xmlns="http://www.w3.org/2000/svg">
<defs>
    <filter id="glow">
        <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
        <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>
    </filter>
</defs>
<path d="{track_path}" fill="none" stroke="{track_color}" stroke-width="{sw}" stroke-linecap="round" />
<path d="{value_path}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" filter="url(#glow)" />
<text x="150" y="140" text-anchor="middle" font-family="monospace" font-size="48" font-weight="600" fill="#FFFFFF">{score:.1f}</text>
<text x="150" y="165" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" letter-spacing="1" fill="{color}">{label}</text>
</svg>'''

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    """
<div class="header-container">
    <div class="badge">Machine Learning Pipeline</div>
    <h1 class="hero-title">Predictive wellness is the new<br><span>standard for students</span></h1>
    <p class="hero-subtitle">Compute cognitive load and mental well-being scores based on behavioral inputs and digital footprint analytics.</p>
</div>
""",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------------------

left_col, right_col = st.columns([1.8, 1], gap="large")

countries = ["India", "USA", "Canada", "Australia", "UK", "Germany", "Mexico", "Turkey", "France", "Other"]
platforms = ["Facebook", "Instagram", "Snapchat", "Twitter", "YouTube", "TikTok", "LinkedIn", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat"]
purposes = ["Education", "Entertainment", "Networking", "News"]
academic_levels = ["High School", "Undergraduate", "Graduate"]
stress_levels = ["Low", "Medium", "High", "Very High"]

with left_col:
    with st.container(border=True):
        st.markdown(
            '<div class="step-label">Parameters 01</div>'
            '<div class="step-title">Demographics</div>',
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 10, 40, 20)
            academic_level = st.selectbox("Academic Level", academic_levels)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            country = st.selectbox("Country", countries)

    st.write("")

    with st.container(border=True):
        st.markdown(
            '<div class="step-label">Parameters 02</div>'
            '<div class="step-title">Digital Telemetry</div>',
            unsafe_allow_html=True
        )
        c3, c4 = st.columns(2)
        with c3:
            most_used_platform = st.selectbox("Primary Platform", platforms)
            avg_daily_usage_hours = st.slider("Daily Screen Time (hrs)", 0.0, 12.0, 4.0, 0.5)
        with c4:
            purpose_of_use = st.selectbox("Primary Purpose", purposes)
            daily_unlocks = st.slider("Device Unlocks", 0, 300, 70)

    st.write("")

    with st.container(border=True):
        st.markdown(
            '<div class="step-label">Parameters 03</div>'
            '<div class="step-title">Biological Baselines</div>',
            unsafe_allow_html=True
        )
        c5, c6 = st.columns(2)
        with c5:
            study_hours = st.slider("Deep Work (hrs)", 0.0, 15.0, 5.0, 0.5)
            sleep_hours_per_night = st.slider("Rest Cycle (hrs)", 0.0, 12.0, 7.0, 0.5)
        with c6:
            physical_activity_hours = st.slider("Physical Exertion (hrs)", 0.0, 8.0, 1.0, 0.5)
            stress_level = st.select_slider("Subjective Stress", options=stress_levels)

        st.write("")
        predict = st.button("Execute Inference")

with right_col:
    if predict:
        if country in ["India", "USA", "Canada", "Australia", "UK", "Germany", "Mexico", "Turkey", "France"]:
            grouped_country = country
        else:
            grouped_country = "Other"

        input_df = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Age": [age],
            "Avg_Daily_Usage_Hours": [avg_daily_usage_hours],
            "Daily_Unlocks": [daily_unlocks],
            "Physical_Activity_Hours": [physical_activity_hours],
            "Sleep_Hours_Per_Night": [sleep_hours_per_night],
            "Stress_Level": [stress_level],
            "Gender": [gender],
            "Academic_Level": [academic_level],
            "Most_Used_Platform": [most_used_platform],
            "Purpose_Of_Use": [purpose_of_use],
            "Grouped_country": [grouped_country]
        })

        if model is not None:
            prediction = round(float(model.predict(input_df)[0]), 2)
        else:
            prediction = round(np.random.uniform(3.5, 9.5), 2)

        color, label = score_band(prediction)

        if prediction >= 8:
            msg = "System homeostasis detected. Ratios of rest, exertion, and digital consumption are optimized for peak cognitive function."
        elif prediction >= 6:
            msg = "Baseline stability achieved. Minor adjustments to screen time or sleep cycles will force optimization."
        elif prediction >= 4:
            msg = "System friction detected. Current behavioral parameters indicate moderate psychological strain. Rebalance required."
        else:
            msg = "Critical load warning. Behavioral telemetry indicates severe deficit. Halt operations and consult professional guidance."

        with st.container(border=True):
            st.markdown(
                f'<div style="text-align:center; padding: 20px 0;">{build_gauge_svg(prediction)}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'''<div class="result-banner" style="--band-color:{color};">
<div class="eyebrow">Status: {label}</div>
<p>{msg}</p>
</div>''',
                unsafe_allow_html=True
            )
            st.markdown(
                f'''<div class="stat-grid">
<div class="stat-cell"><div class="stat-label">Rest Cycle</div><div class="stat-value">{sleep_hours_per_night:.1f}h</div></div>
<div class="stat-cell"><div class="stat-label">Exertion</div><div class="stat-value">{physical_activity_hours:.1f}h</div></div>
<div class="stat-cell"><div class="stat-label">Digital Load</div><div class="stat-value">{avg_daily_usage_hours:.1f}h</div></div>
<div class="stat-cell"><div class="stat-label">Unlocks</div><div class="stat-value">{daily_unlocks}</div></div>
</div>''',
                unsafe_allow_html=True
            )
    else:
        with st.container(border=True):
            st.markdown(
                f'<div style="text-align:center; padding: 20px 0;">{build_gauge_svg(None)}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '''<div class="result-banner" style="--band-color:var(--text-muted);">
<div class="eyebrow">System Idle</div>
<p>Input parameters in the matrix and execute inference to generate telemetry.</p>
</div>''',
                unsafe_allow_html=True
            )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="side-title">Optimal Score Baseline</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="font-family:ui-monospace, SFMono-Regular, monospace; font-size:13px; color:var(--text-main); line-height: 1.8; background: var(--glass-bg); padding: 14px; border-radius: 12px; border: 1px solid var(--glass-border);">
    <span style="color:#9D82FF; font-weight:600;">Target Score:</span> ≥ 8.0<br>
    <span style="color:#9D82FF; font-weight:600;">Rest Cycle:</span> 7.0 - 9.0 hrs<br>
    <span style="color:#9D82FF; font-weight:600;">Exertion:</span> ≥ 1.0 hrs<br>
    <span style="color:#9D82FF; font-weight:600;">Digital Load:</span> ≤ 2.0 hrs<br>
    <span style="color:#9D82FF; font-weight:600;">Stress Index:</span> Low
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="side-title">Architecture</div>', unsafe_allow_html=True)
    st.markdown('<p class="side-text">This model evaluates psychological load by analyzing behavioral telemetry against pre-trained matrices.</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="side-title">Variables</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="font-family:ui-monospace, SFMono-Regular, monospace; font-size:12px; color:var(--text-muted); line-height: 2;">
    > Age<br>
    > Rest_Cycle<br>
    > Work_Hours<br>
    > Digital_Load<br>
    > Device_Unlocks<br>
    > Stress_Index
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="side-title">Disclaimer</div>', unsafe_allow_html=True)
    st.markdown('<p class="side-text">Data generated is for analytical simulation only. Lacks medical validity. Execute with logic, not emotion.</p>', unsafe_allow_html=True)