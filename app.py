"""
AUDIOGUARD - AI Audio Forensic Platform
Commercial-grade AI Audio Deepfake Detection Application
Run with: streamlit run app.py
"""

import os
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config
from predict import predict_audio_file
from audio_utils import load_and_preprocess_audio, compute_mel_spectrogram, compute_mfcc_visualization

# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION & COMMERCIAL SAAS THEME
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AudioGuard | AI Audio Forensic Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Commercial-Grade AI Audio Forensic Platform (Inspired by AUTHENTICITY.AI)
st.markdown("""
<style>
    /* Base Palette & Global Styling */
    .stApp {
        background-color: #070B14;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit Default UI Elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1160px;
    }

    /* Fixed Left Sidebar (270px) */
    section[data-testid="stSidebar"] {
        background-color: #0B1220 !important;
        border-right: 1px solid #1D2940 !important;
        padding-top: 1.5rem;
        width: 270px !important;
    }
    .sidebar-brand-wrapper {
        padding: 0 10px 18px 10px;
        border-bottom: 1px solid #1D2940;
        margin-bottom: 24px;
    }
    .sidebar-logo-text {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sidebar-logo-mark {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #6366F1 0%, #38BDF8 100%);
        border-radius: 6px;
        display: inline-block;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
    }
    .sidebar-subtitle-text {
        font-size: 0.74rem;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .sidebar-nav-header {
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 12px 10px 12px;
    }

    /* Custom Radio Navigation Pill Items */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 10px 14px;
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 0.92rem;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: #101827;
        color: #F8FAFC !important;
        border-color: #1D2940;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label[aria-checked="true"] {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.2) 0%, rgba(56, 189, 248, 0.05) 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        font-weight: 700;
    }

    /* Sidebar Bottom Status Card */
    .sidebar-status-box {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 50px;
    }
    .status-dot-green {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        margin-right: 6px;
    }

    /* Top Content Header Bar */
    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 24px;
        background: #0B1220;
        border: 1px solid #1D2940;
        border-radius: 14px;
        margin-bottom: 28px;
    }
    .top-bar-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .top-bar-sub {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-left: 10px;
        font-weight: 400;
    }
    .top-bar-badge-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .top-bar-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        color: #34D399;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
    }
    .top-bar-badge {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 600;
    }

    /* Cards & Containers */
    .info-card {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .info-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .info-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .info-badge {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: #1E293B;
        color: #6366F1;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #334155;
    }
    .info-desc {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.55;
    }

    /* Large Hero Section Card */
    .hero-card {
        background: linear-gradient(135deg, #101827 0%, #0B1220 100%);
        border: 1px solid #1D2940;
        border-radius: 20px;
        padding: 38px 34px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .hero-pill-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 16px;
    }
    .hero-h1-text {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
        color: #FFFFFF;
        margin-bottom: 14px;
    }
    .hero-p-text {
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.6;
        margin-bottom: 26px;
        max-width: 820px;
    }

    /* Compact Metrics & Information Badges */
    .compact-metric-card {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        height: 100%;
    }
    .compact-metric-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .compact-metric-lbl {
        font-size: 0.76rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }

    /* Process Flow Container */
    .flow-wrapper {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 18px;
        padding: 26px;
        margin: 24px 0;
    }
    .flow-grid {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 18px;
    }
    .flow-step {
        background: #0B1220;
        border: 1px solid #1D2940;
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        flex: 1;
    }
    .flow-num {
        font-size: 0.75rem;
        font-weight: 800;
        color: #6366F1;
        margin-bottom: 4px;
    }
    .flow-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .flow-arrow {
        color: #475569;
        font-weight: 800;
        font-size: 1.1rem;
    }

    /* Result Panels */
    .result-panel-real {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(16, 24, 39, 0.95) 100%);
        border: 1px solid #10B981;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
    }
    .result-panel-fake {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(16, 24, 39, 0.95) 100%);
        border: 1px solid #EF4444;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
    }
    .result-panel-uncertain {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.45) 0%, rgba(16, 24, 39, 0.95) 100%);
        border: 1px solid #F59E0B;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.15);
    }
    .result-status-indicator {
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-header-main {
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        margin-bottom: 8px;
    }
    .result-desc-text {
        font-size: 1.02rem;
        color: #E2E8F0;
        max-width: 580px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Custom Streamlit Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 12px 24px;
        font-size: 0.95rem;
        transition: all 0.2s ease-in-out;
        width: 100%;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45);
    }

    /* Download Report Button */
    div.stDownloadButton > button {
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 12px;
        font-weight: 700;
        padding: 12px 24px;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    div.stDownloadButton > button:hover {
        background: #334155;
        color: #FFFFFF;
        border-color: #475569;
    }

    /* Code Container Box */
    .code-container-box {
        background: #0B1220;
        border: 1px solid #1D2940;
        border-radius: 14px;
        padding: 20px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.86rem;
        color: #38BDF8;
        line-height: 1.6;
        margin: 16px 0;
    }

    /* Style Tabs */
    button[data-baseweb="tab"] {
        background: transparent;
        color: #94A3B8 !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 16px;
    }
    button[aria-selected="true"] {
        color: #38BDF8 !important;
        background: #101827 !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Navigation State
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Dashboard"
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0


# -------------------------------------------------------------------
# SIDEBAR NAVIGATION (STRICTLY 4 ITEMS)
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-wrapper">
        <div class="sidebar-logo-text">
            <span class="sidebar-logo-mark"></span> AudioGuard
        </div>
        <div class="sidebar-subtitle-text">AI Audio Forensic Platform</div>
    </div>
    <div class="sidebar-nav-header">NAVIGATION</div>
    """, unsafe_allow_html=True)

    nav_options = ["Dashboard", "Audio Analysis", "System Architecture", "About"]
    current_idx = nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0

    nav_selection = st.radio(
        "Navigation",
        nav_options,
        index=current_idx,
        label_visibility="collapsed"
    )
    st.session_state.active_nav = nav_selection

    st.markdown("""
    <div class="sidebar-status-box">
        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">Detection Engine</div>
        <div style="font-size: 0.88rem; font-weight: 700; color: #10B981; margin-top: 4px;">
            <span class="status-dot-green"></span> Operational
        </div>
        <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 4px; font-weight: 500;">MFCC + Random Forest</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# COMPACT TOP HEADER BAR
# -------------------------------------------------------------------
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-title">
        {st.session_state.active_nav} <span class="top-bar-sub">AI Audio Deepfake Detection</span>
    </div>
    <div class="top-bar-badge-box">
        <div class="top-bar-status">
            <span class="status-dot-green"></span> Engine Operational
        </div>
        <div class="top-bar-badge">
            MFCC + Random Forest
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 1: DASHBOARD
# -------------------------------------------------------------------
if st.session_state.active_nav == "Dashboard":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-pill-badge">AI AUDIO FORENSICS</div>
        <div class="hero-h1-text">Detect AI-Generated Voices</div>
        <div class="hero-p-text">Analyze speech recordings using acoustic feature analysis and a trained machine-learning model to identify characteristics associated with synthetic or AI-generated speech.</div>
    </div>
    """, unsafe_allow_html=True)

    h_btn1, h_btn2 = st.columns(2)
    with h_btn1:
        if st.button("Analyze Audio →", key="btn_hero_an"):
            st.session_state.active_nav = "Audio Analysis"
            st.rerun()
    with h_btn2:
        if st.button("Record Audio", key="btn_hero_rec"):
            st.session_state.active_nav = "Audio Analysis"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### System Overview")

    # 4 System Overview Grid Cards
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown("""
        <div class="compact-metric-card">
            <div class="compact-metric-val" style="color: #818CF8;">MFCC + Random Forest</div>
            <div class="compact-metric-lbl">Detection Method</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="compact-metric-card">
            <div class="compact-metric-val">Speech Audio</div>
            <div class="compact-metric-lbl">Input</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="compact-metric-card">
            <div class="compact-metric-val">WAV • MP3 • FLAC • OGG • M4A</div>
            <div class="compact-metric-lbl">Supported Formats</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        st.markdown("""
        <div class="compact-metric-card">
            <div class="compact-metric-val" style="color: #38BDF8;">60 seconds</div>
            <div class="compact-metric-lbl">Maximum Duration</div>
        </div>
        """, unsafe_allow_html=True)

    # How AudioGuard Works Connected Flow
    st.markdown("""
    <div class="flow-wrapper">
        <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC;">How AudioGuard Works</div>
        <div class="flow-grid">
            <div class="flow-step">
                <div class="flow-num">01</div>
                <div class="flow-title">Audio Input</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">02</div>
                <div class="flow-title">Preprocessing</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">03</div>
                <div class="flow-title">MFCC Extraction</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">04</div>
                <div class="flow-title">Random Forest</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">05</div>
                <div class="flow-title">Detection Result</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # System Navigation Cards (Matching Authenticity.AI 3 Grid Cards)
    st.markdown("### System Navigation")
    n1, n2, n3 = st.columns(3)

    with n1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-header">
                <div class="info-title">Audio Analysis</div>
                <div class="info-badge">PRIMARY WORKSPACE</div>
            </div>
            <div class="info-desc">Inspect speech recordings for acoustic characteristics, MFCC feature anomalies, and synthetic speech probabilities.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Section →", key="btn_nav_sec1"):
            st.session_state.active_nav = "Audio Analysis"
            st.rerun()

    with n2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-header">
                <div class="info-title">System Architecture</div>
                <div class="info-badge">DOCUMENTATION</div>
            </div>
            <div class="info-desc">Explore technical specifications of feature extraction, 16 kHz standardization, and Random Forest classifier logic.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Section →", key="btn_nav_sec2"):
            st.session_state.active_nav = "System Architecture"
            st.rerun()

    with n3:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-header">
                <div class="info-title">About</div>
                <div class="info-badge">OVERVIEW</div>
            </div>
            <div class="info-desc">Learn about the acoustic feature approach, local inference design, scope of classification, and platform details.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Section →", key="btn_nav_sec3"):
            st.session_state.active_nav = "About"
            st.rerun()


# -------------------------------------------------------------------
# PAGE 2: AUDIO ANALYSIS (MAIN FORENSIC WORKSPACE)
# -------------------------------------------------------------------
elif st.session_state.active_nav == "Audio Analysis":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">Audio Analysis</h2>
        <p style="color: #94A3B8; font-size: 1.02rem;">Upload or record speech audio for forensic analysis</p>
    </div>
    """, unsafe_allow_html=True)

    input_tab = st.radio(
        "Input Source:",
        ["Upload Audio File", "Record Live Voice"],
        horizontal=True
    )

    audio_file_buffer = None
    audio_filename = "sample.wav"

    if input_tab == "Upload Audio File":
        st.markdown("""
        <div class="info-card">
            <div class="info-title" style="margin-bottom: 4px;">Single Audio Forensics Workspace</div>
            <div class="info-desc" style="margin-bottom: 20px;">Upload an audio file (WAV, MP3, FLAC, OGG, M4A) to execute acoustic inspection</div>
        </div>
        """, unsafe_allow_html=True)

        audio_file_buffer = st.file_uploader(
            "Drop audio file here or Browse Local Files",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Supports WAV, MP3, FLAC, OGG and M4A • Maximum duration: 60 seconds",
            key=f"uploader_{st.session_state.reset_count}"
        )
        if audio_file_buffer:
            audio_filename = audio_file_buffer.name

    else:
        st.markdown("### Record Your Voice")
        if hasattr(st, "audio_input"):
            audio_file_buffer = st.audio_input("Ready to Record:")
            if audio_file_buffer:
                audio_filename = "recorded_voice.wav"
        else:
            st.warning("Live audio recording is not supported in this environment. Please use Upload Audio File.")

    # Awaiting Audio File State vs Selected Audio Preview State
    if audio_file_buffer is None:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 48px 20px; margin-top: 20px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">Ready for Analysis</div>
            <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 8px;">Upload or record an audio file to begin forensic inspection.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        audio_bytes_data = audio_file_buffer.getvalue() if hasattr(audio_file_buffer, "getvalue") else audio_file_buffer.read()
        file_size_kb = len(audio_bytes_data) / 1024
        file_fmt = Path(audio_filename).suffix.upper().replace(".", "") or "WAV"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title" style="margin-bottom: 12px;">Selected Audio</div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px;">
                <div style="font-size: 0.92rem; color: #CBD5E1;"><strong>Filename:</strong> <code>{audio_filename}</code></div>
                <div style="font-size: 0.92rem; color: #CBD5E1;"><strong>Format:</strong> <code>{file_fmt}</code></div>
                <div style="font-size: 0.92rem; color: #CBD5E1;"><strong>Size:</strong> <code>{file_size_kb:.1f} KB</code></div>
                <div style="font-size: 0.92rem; color: #CBD5E1;"><strong>Sample Rate:</strong> <code>16 kHz Mono</code></div>
            </div>
        """, unsafe_allow_html=True)

        st.audio(audio_bytes_data, format="audio/wav")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Run Audio Forensic Analysis →", type="primary", key="btn_run_forensic"):
            status_box = st.empty()
            status_box.markdown("""
            <div class="info-card">
                <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">Analyzing Audio</div>
                <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 12px;">Running acoustic forensic analysis...</div>
                <div style="font-size: 0.85rem; color: #34D399;">✓ Audio preprocessing</div>
                <div style="font-size: 0.85rem; color: #34D399;">✓ MFCC feature extraction</div>
                <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700;">● Random Forest classification</div>
                <div style="font-size: 0.85rem; color: #64748B;">○ Generating forensic report</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.4)

            try:
                suffix = Path(audio_filename).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes_data)
                    temp_path = tmp.name

                result = predict_audio_file(temp_path, backend="mfcc")

                try:
                    os.remove(temp_path)
                except Exception:
                    pass

                status_box.empty()

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("# Analysis Complete")

                pred = result["prediction"]
                fake_p = result["fake_probability"]
                real_p = result["real_probability"]
                conf = result["confidence"]
                is_uncertain = result["is_uncertain"]
                proc_time = result["processing_time_sec"]

                if is_uncertain:
                    st.markdown("""
                    <div class="result-panel-uncertain">
                        <div class="result-status-indicator" style="color: #F59E0B;">● UNCERTAIN</div>
                        <div class="result-header-main" style="color: #F59E0B;">RESULT INCONCLUSIVE</div>
                        <div class="result-desc-text">The acoustic characteristics are close to the decision threshold.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred == "REAL":
                    st.markdown("""
                    <div class="result-panel-real">
                        <div class="result-status-indicator" style="color: #10B981;">● AUTHENTIC SPEECH</div>
                        <div class="result-header-main" style="color: #10B981;">AUTHENTIC / REAL AUDIO</div>
                        <div class="result-desc-text">The audio is classified as likely genuine human speech.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-panel-fake">
                        <div class="result-status-indicator" style="color: #EF4444;">● SYNTHETIC SPEECH DETECTED</div>
                        <div class="result-header-main" style="color: #EF4444;">AI-GENERATED AUDIO</div>
                        <div class="result-desc-text">The audio is classified as likely synthetic or AI-generated speech.</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Detection Confidence Visualization
                st.markdown("### Detection Confidence")
                st.markdown(f"<h1 style='text-align: center; color: #38BDF8; font-size: 3.6rem; font-weight: 900; margin-bottom: 8px;'>{conf:.1f}%</h1>", unsafe_allow_html=True)
                st.progress(conf / 100.0)

                b1, b2, b3 = st.columns(3)
                with b1:
                    st.markdown(f"**REAL:** `{real_p:.1f}%`")
                with b2:
                    st.markdown(f"<div style='text-align: center;'><strong>Analysis Time:</strong> <code>{proc_time:.2f} s</code></div>", unsafe_allow_html=True)
                with b3:
                    st.markdown(f"<div style='text-align: right;'><strong>AI-GENERATED:</strong> <code>{fake_p:.1f}%</code></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Forensic Analysis Summary Grid (6 Grid Cards)
                st.markdown("### Forensic Analysis Summary")
                r1, r2, r3, r4, r5, r6 = st.columns(6)
                with r1:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: {'#10B981' if pred=='REAL' else '#EF4444'};">{'REAL' if pred=='REAL' else 'AI-GENERATED'}</div>
                        <div class="compact-metric-lbl">Classification</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val">{conf:.1f}%</div>
                        <div class="compact-metric-lbl">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r3:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #10B981;">{real_p:.1f}%</div>
                        <div class="compact-metric-lbl">Real Prob</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r4:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #EF4444;">{fake_p:.1f}%</div>
                        <div class="compact-metric-lbl">AI Prob</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r5:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #A855F7;">{result['audio_duration_sec']:.2f} s</div>
                        <div class="compact-metric-lbl">Duration</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r6:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #38BDF8;">MFCC + RF</div>
                        <div class="compact-metric-lbl">Method</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Acoustic Diagnostics Section
                st.markdown("### Acoustic Diagnostics")
                waveform, sr, _ = load_and_preprocess_audio(audio_bytes_data)

                plt.style.use('dark_background')
                v1, v2, v3 = st.tabs(["Waveform", "MFCC Heatmap", "Mel-Spectrogram"])

                with v1:
                    st.caption("Time-domain representation of the analyzed speech signal.")
                    fig, ax = plt.subplots(figsize=(9, 2.6))
                    fig.patch.set_facecolor('#101827')
                    ax.set_facecolor('#0B1220')
                    time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
                    ax.plot(time_axis, waveform, color="#38BDF8", alpha=0.85, linewidth=0.8)
                    ax.set_xlabel("Time (seconds)", color="#94A3B8")
                    ax.set_ylabel("Amplitude", color="#94A3B8")
                    ax.set_title("Audio Amplitude Waveform", color="#F8FAFC", fontsize=10, fontweight="bold")
                    ax.grid(True, color="#1D2940", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)

                with v2:
                    st.caption("Mel-frequency cepstral representation used by the detection pipeline.")
                    mfccs = compute_mfcc_visualization(waveform, sr=sr)
                    fig, ax = plt.subplots(figsize=(9, 2.8))
                    fig.patch.set_facecolor('#101827')
                    ax.set_facecolor('#0B1220')
                    img = ax.imshow(mfccs, aspect="auto", origin="lower", cmap="viridis")
                    fig.colorbar(img, ax=ax)
                    ax.set_title("MFCC Feature Heatmap", color="#F8FAFC", fontsize=10, fontweight="bold")
                    ax.set_xlabel("Time Frames", color="#94A3B8")
                    ax.set_ylabel("MFCC Coefficients", color="#94A3B8")
                    plt.tight_layout()
                    st.pyplot(fig)

                with v3:
                    st.caption("Time-frequency representation of the speech signal.")
                    mel_db = compute_mel_spectrogram(waveform, sr=sr)
                    fig, ax = plt.subplots(figsize=(9, 2.8))
                    fig.patch.set_facecolor('#101827')
                    ax.set_facecolor('#0B1220')
                    img = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
                    fig.colorbar(img, ax=ax, format="%+2.0f dB")
                    ax.set_title("Log Mel-Spectrogram Frequency Spectrum", color="#F8FAFC", fontsize=10, fontweight="bold")
                    ax.set_xlabel("Time Frames", color="#94A3B8")
                    ax.set_ylabel("Mel Frequency Bands", color="#94A3B8")
                    plt.tight_layout()
                    st.pyplot(fig)

                # Segment-Level Analysis (If audio duration > 4 seconds)
                if len(result["windows"]) > 1:
                    st.markdown("### Segment-Level Analysis")
                    win_df = pd.DataFrame(result["windows"])
                    win_df = win_df.rename(columns={
                        "start_sec": "Start (s)",
                        "end_sec": "End (s)",
                        "label": "Classification",
                        "fake_prob": "AI Probability (%)",
                        "real_prob": "Real Prob (%)"
                    })
                    st.dataframe(win_df[["Start (s)", "End (s)", "Classification", "AI Probability (%)"]], use_container_width=True)

                    st.markdown("#### AI Probability Over Time")
                    fig, ax = plt.subplots(figsize=(8, 2.6))
                    fig.patch.set_facecolor('#101827')
                    ax.set_facecolor('#0B1220')

                    times = [(w["start_sec"] + w["end_sec"]) / 2 for w in result["windows"]]
                    probs = [w["fake_prob"] for w in result["windows"]]

                    ax.plot(times, probs, marker="o", color="#EF4444" if pred == "FAKE" else "#10B981", linewidth=2.5)
                    ax.axhline(result['decision_threshold'] * 100, color="#64748B", linestyle="--", label="Threshold")
                    ax.set_ylim([0, 100])
                    ax.set_xlabel("Time (seconds)", color="#94A3B8")
                    ax.set_ylabel("AI Probability (%)", color="#94A3B8")
                    ax.tick_params(colors="#94A3B8")
                    ax.set_title("AI Probability Over Time", color="#F8FAFC", fontsize=10, fontweight="bold")
                    ax.grid(True, color="#1D2940", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)

                # Result Actions
                st.markdown("<br>", unsafe_allow_html=True)
                act1, act2 = st.columns(2)
                with act1:
                    if st.button("Analyze Another Audio", key="btn_reset_forensic"):
                        st.session_state.reset_count += 1
                        st.rerun()

                with act2:
                    report_text = f"""==================================================
AUDIOGUARD FORENSIC ANALYSIS REPORT
==================================================
Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Audio Filename:     {audio_filename}
Audio Duration:     {result['audio_duration_sec']:.2f} seconds

CLASSIFICATION:     {result['prediction']}
Confidence Score:   {result['confidence']:.2f}%
REAL Probability:   {result['real_probability']:.2f}%
AI Probability:     {result['fake_probability']:.2f}%

Detection Method:   MFCC + Random Forest Classifier
Processing Time:    {result['processing_time_sec']:.2f} seconds
==================================================
"""
                    st.download_button(
                        label="Download Analysis Report",
                        data=report_text,
                        file_name=f"audioguard_report_{Path(audio_filename).stem}.txt",
                        mime="text/plain"
                    )

                # Disclaimer
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("Disclaimer: AudioGuard provides probabilistic machine-learning predictions. Results should be treated as automated indicators and not as absolute proof of authenticity.")

            except Exception as e:
                st.error("Unable to Analyze Audio. Please upload a valid supported audio file and try again.")

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 3: SYSTEM ARCHITECTURE (INSPIRED BY AUTHENTICITY.AI ARCHITECTURE)
# -------------------------------------------------------------------
elif st.session_state.active_nav == "System Architecture":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">System Architecture</h2>
        <p style="color: #94A3B8; font-size: 1.05rem;">Audio Forensic Detection Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture Flowchart Sitemap
    st.markdown("""
    <div class="flow-wrapper">
        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">Pipeline Architecture Sitemap</div>
        <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 4px;">Sequential acoustic inspection & classification workflow</div>
        <div class="flow-grid" style="flex-wrap: wrap; margin-top: 20px;">
            <div class="flow-step">
                <div class="flow-num">STEP 01</div>
                <div class="flow-title">Audio Input</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 02</div>
                <div class="flow-title">Preprocessing</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 03</div>
                <div class="flow-title">16 kHz Mono Standard</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 04</div>
                <div class="flow-title">MFCC Extraction</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 05</div>
                <div class="flow-title">Random Forest</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 06</div>
                <div class="flow-title">REAL / FAKE Verdict</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Technical Specs Code Block
    st.markdown("""
    <div class="info-card">
        <div class="info-title">RANDOM FOREST CLASSIFICATION PIPELINE MATH & SPECS</div>
        <div class="code-container-box">
// Acoustic Feature Vector Construction (172-Dimensional):<br>
Features = [ MFCC(1..13), Delta_MFCC(1..13), Delta2_MFCC(1..13), SpectralCentroid, SpectralRolloff, ZCR, RMS, Flatness ]<br><br>
// Decision Logic Threshold:<br>
if (Overall_Fake_Probability >= 0.50) {<br>
&nbsp;&nbsp;&nbsp;&nbsp;Verdict = "AI-GENERATED AUDIO";<br>
&nbsp;&nbsp;&nbsp;&nbsp;Confidence = Overall_Fake_Probability * 100;<br>
} else {<br>
&nbsp;&nbsp;&nbsp;&nbsp;Verdict = "AUTHENTIC / REAL AUDIO";<br>
&nbsp;&nbsp;&nbsp;&nbsp;Confidence = Overall_Real_Probability * 100;<br>
}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Technical Pipeline Cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <div class="info-card-header">
                <div class="info-title">Audio Preprocessing</div>
                <div class="info-badge">STAGE 1</div>
            </div>
            <div class="info-desc">
                Standardizes raw input audio into uniform acoustic signals:
                <ul style="margin-top: 10px; color: #CBD5E1; padding-left: 18px;">
                    <li>Single-channel Mono conversion</li>
                    <li>16,000 Hz resampling target</li>
                    <li>Peak amplitude normalization</li>
                    <li>Sliding windowing (4.0s window, 2.0s hop)</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <div class="info-card-header">
                <div class="info-title">Feature Extraction</div>
                <div class="info-badge">STAGE 2</div>
            </div>
            <div class="info-desc">
                Computes 172-dimensional acoustic feature representations:
                <ul style="margin-top: 10px; color: #CBD5E1; padding-left: 18px;">
                    <li>13 Mel-Frequency Cepstral Coefficients</li>
                    <li>Delta & Delta-Delta MFCC derivatives</li>
                    <li>Spectral Centroid & Spectral Rolloff</li>
                    <li>Zero-Crossing Rate (ZCR) & RMS Energy</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <div class="info-card-header">
                <div class="info-title">Random Forest Classifier</div>
                <div class="info-badge">STAGE 3</div>
            </div>
            <div class="info-desc">
                Ensemble decision tree model trained for binary speech classification:
                <ul style="margin-top: 10px; color: #CBD5E1; padding-left: 18px;">
                    <li>CPU-based optimized inference</li>
                    <li>Class probability estimation</li>
                    <li>Aggregated sliding window voting</li>
                    <li>Local private model execution</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 4: ABOUT (INSPIRED BY AUTHENTICITY.AI ABOUT PAGE)
# -------------------------------------------------------------------
elif st.session_state.active_nav == "About":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">About AudioGuard</h2>
        <p style="color: #94A3B8; font-size: 1.05rem;">Platform Overview & Technical Implementation Specifications</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-title" style="font-size: 1.2rem; margin-bottom: 8px;">About AudioGuard</div>
        <div class="info-desc" style="font-size: 1.02rem;">
            AudioGuard is an AI-assisted audio forensic application designed to analyze speech recordings and identify acoustic characteristics associated with AI-generated or synthetic speech.
        </div>
    </div>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Detection Approach</div>
            <div class="info-desc" style="font-size: 1rem; color: #38BDF8; font-weight: 700; margin-top: 6px;">MFCC Feature Analysis</div>
            <div class="info-desc" style="margin-top: 6px;">Extracts spectral envelop and cepstral properties from audio signals to capture subtle synthetic artifacts.</div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Classification Model</div>
            <div class="info-desc" style="font-size: 1rem; color: #818CF8; font-weight: 700; margin-top: 6px;">Random Forest</div>
            <div class="info-desc" style="margin-top: 6px;">Leverages an ensemble decision tree pipeline trained on acoustic vectors to calculate probabilities.</div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Processing</div>
            <div class="info-desc" style="font-size: 1rem; color: #10B981; font-weight: 700; margin-top: 6px;">Local Audio Analysis</div>
            <div class="info-desc" style="margin-top: 6px;">Executes local CPU-based inference with privacy-focused audio processing.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-title" style="margin-bottom: 12px;">System Specifications</div>
        <table style="width: 100%; border-collapse: collapse; color: #CBD5E1; font-size: 0.95rem;">
            <tr style="border-bottom: 1px solid #1D2940;">
                <td style="padding: 10px 0; font-weight: 700; color: #94A3B8; width: 35%;">Detection Method</td>
                <td style="padding: 10px 0; font-weight: 600; color: #F8FAFC;">MFCC + Random Forest Classifier</td>
            </tr>
            <tr style="border-bottom: 1px solid #1D2940;">
                <td style="padding: 10px 0; font-weight: 700; color: #94A3B8;">Input Types</td>
                <td style="padding: 10px 0; font-weight: 600; color: #F8FAFC;">Speech Audio (WAV, MP3, FLAC, OGG, M4A)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1D2940;">
                <td style="padding: 10px 0; font-weight: 700; color: #94A3B8;">Output Labels</td>
                <td style="padding: 10px 0; font-weight: 600; color: #F8FAFC;">AUTHENTIC / REAL AUDIO | AI-GENERATED AUDIO</td>
            </tr>
            <tr>
                <td style="padding: 10px 0; font-weight: 700; color: #94A3B8;">Model Storage</td>
                <td style="padding: 10px 0; font-weight: 600; color: #F8FAFC;">models/mfcc_model.joblib & models/config.json</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 24px;">
        <p style="color: #64748B; font-size: 0.85rem;"><strong>Disclaimer:</strong> AudioGuard provides probabilistic machine-learning predictions. Results should be treated as automated indicators and not as absolute proof of authenticity.</p>
    </div>
    """, unsafe_allow_html=True)
