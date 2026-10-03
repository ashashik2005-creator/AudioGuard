"""
AUDIOGUARD - AI Audio Forensic Platform
Run with: streamlit run app.py
"""

import os
import time
import tempfile
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config
from predict import predict_audio_file
from audio_utils import load_and_preprocess_audio, compute_mel_spectrogram, compute_mfcc_visualization

# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION & ENTERPRISE DARK SAAS THEME
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AudioGuard | AI Audio Forensic Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Enterprise AI Forensic SaaS Platform (No Emojis, Sleek SVG/Text Layout)
st.markdown("""
<style>
    /* Dark Theme Base Palette */
    .stApp {
        background-color: #070B14;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Default Streamlit Headers & Footers */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1120px;
    }

    /* Narrow Fixed Left Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0B1220 !important;
        border-right: 1px solid #1D2940 !important;
        padding-top: 1.2rem;
        width: 260px !important;
    }
    .sidebar-brand-wrapper {
        padding: 0 8px 16px 8px;
        border-bottom: 1px solid #1D2940;
        margin-bottom: 20px;
    }
    .sidebar-logo-text {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-logo-mark {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #6366F1 0%, #38BDF8 100%);
        border-radius: 6px;
        display: inline-block;
    }
    .sidebar-subtitle-text {
        font-size: 0.76rem;
        color: #94A3B8;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-top: 3px;
    }

    /* Compact Sidebar Bottom Status */
    .sidebar-status-box {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 12px;
        padding: 14px;
        margin-top: 36px;
    }
    .status-dot-green {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 6px;
    }
    .version-tag {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 16px;
        text-align: center;
    }

    /* Top Content Header Bar */
    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: #0B1220;
        border: 1px solid #1D2940;
        border-radius: 14px;
        margin-bottom: 24px;
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
    .top-bar-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        color: #34D399;
        font-weight: 600;
    }

    /* Hero Section (65% / 35% Split) */
    .hero-card {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 20px;
        padding: 36px 30px;
        height: 100%;
    }
    .hero-h1-text {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.25;
        color: #FFFFFF;
        margin-bottom: 14px;
    }
    .hero-p-text {
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.55;
        margin-bottom: 28px;
    }

    /* Audio Intelligence Graphic Card (Right 35%) */
    .intel-card {
        background: linear-gradient(135deg, #0B1220 0%, #101827 100%);
        border: 1px solid #1D2940;
        border-radius: 20px;
        padding: 30px 24px;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .intel-wave-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        height: 40px;
        margin: 16px 0;
    }
    .intel-wave-bar {
        width: 4px;
        background: linear-gradient(180deg, #6366F1 0%, #38BDF8 100%);
        border-radius: 2px;
        animation: pulse 1.4s ease-in-out infinite alternate;
    }
    .intel-wave-bar:nth-child(2) { height: 28px; animation-delay: 0.2s; }
    .intel-wave-bar:nth-child(3) { height: 40px; animation-delay: 0.4s; }
    .intel-wave-bar:nth-child(4) { height: 20px; animation-delay: 0.1s; }
    .intel-wave-bar:nth-child(5) { height: 32px; animation-delay: 0.3s; }

    /* Compact Metrics Row */
    .compact-metric-card {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
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
        margin-top: 4px;
    }

    /* How AudioGuard Works Connected Flow */
    .flow-wrapper {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 18px;
        padding: 24px;
        margin-top: 28px;
    }
    .flow-grid {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 16px;
    }
    .flow-step {
        background: #0B1220;
        border: 1px solid #1D2940;
        border-radius: 12px;
        padding: 14px;
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
    }

    /* Primary Result Cards */
    .result-panel-real {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(16, 24, 39, 0.9) 100%);
        border: 1px solid #10B981;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
    }
    .result-panel-fake {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(16, 24, 39, 0.9) 100%);
        border: 1px solid #EF4444;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
    }
    .result-panel-uncertain {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.45) 0%, rgba(16, 24, 39, 0.9) 100%);
        border: 1px solid #F59E0B;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
    }
    .result-status-indicator {
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-header-main {
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        margin-bottom: 8px;
    }
    .result-desc-text {
        font-size: 1.05rem;
        color: #E2E8F0;
        max-width: 580px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 12px 24px;
        font-size: 0.98rem;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# Initialize Navigation State
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Dashboard"
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0


# -------------------------------------------------------------------
# NARROW FIXED SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-wrapper">
        <div class="sidebar-logo-text">
            <span class="sidebar-logo-mark"></span> AudioGuard
        </div>
        <div class="sidebar-subtitle-text">AI Audio Forensic Platform</div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        ["Dashboard", "Analyze Audio", "About"],
        index=["Dashboard", "Analyze Audio", "About"].index(st.session_state.active_nav) if st.session_state.active_nav in ["Dashboard", "Analyze Audio", "About"] else 0,
        label_visibility="collapsed"
    )
    st.session_state.active_nav = nav_selection

    st.markdown("""
    <div class="sidebar-status-box">
        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Detection Engine</div>
        <div style="font-size: 0.88rem; font-weight: 700; color: #10B981; margin-top: 4px;">
            <span class="status-dot-green"></span> Operational
        </div>
        <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 4px; font-weight: 500;">MFCC + Random Forest</div>
    </div>
    <div class="version-tag">AudioGuard Platform v1.0</div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# COMPACT TOP HEADER BAR
# -------------------------------------------------------------------
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-title">
        {st.session_state.active_nav} <span class="top-bar-sub">AI Audio Deepfake Detection</span>
    </div>
    <div class="top-bar-status">
        <span class="status-dot-green"></span> Engine Operational
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 1: DASHBOARD
# -------------------------------------------------------------------
if st.session_state.active_nav == "Dashboard":
    col_main_hero, col_intel_hero = st.columns([65, 35])

    with col_main_hero:
        st.markdown("""
        <div class="hero-card">
            <div class="hero-h1-text">Detect AI-Generated Voices</div>
            <div class="hero-p-text">Analyze speech recordings for acoustic characteristics associated with synthetic or AI-generated speech.</div>
        </div>
        """, unsafe_allow_html=True)

        h_btn1, h_btn2 = st.columns(2)
        with h_btn1:
            if st.button("Analyze Audio →", key="btn_hero_an"):
                st.session_state.active_nav = "Analyze Audio"
                st.rerun()
        with h_btn2:
            if st.button("Record Audio", key="btn_hero_rec"):
                st.session_state.active_nav = "Analyze Audio"
                st.rerun()

    with col_intel_hero:
        st.markdown("""
        <div class="intel-card">
            <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC;">AudioGuard</div>
            <div style="font-size: 0.82rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">Acoustic Intelligence</div>
            <div class="intel-wave-container">
                <div class="intel-wave-bar"></div>
                <div class="intel-wave-bar"></div>
                <div class="intel-wave-bar"></div>
                <div class="intel-wave-bar"></div>
                <div class="intel-wave-bar"></div>
            </div>
            <div style="font-size: 0.82rem; color: #38BDF8; font-weight: 700;">MFCC Analysis Engine</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Compact Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="compact-metric-card">
            <div class="compact-metric-val" style="color: #818CF8;">MFCC + Random Forest</div>
            <div class="compact-metric-lbl">Detection Method</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="compact-metric-card">
            <div class="compact-metric-val">Speech Audio</div>
            <div class="compact-metric-lbl">Input Type</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="compact-metric-card">
            <div class="compact-metric-val">WAV / MP3 / FLAC / OGG / M4A</div>
            <div class="compact-metric-lbl">Supported Formats</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="compact-metric-card">
            <div class="compact-metric-val" style="color: #38BDF8;">60 sec</div>
            <div class="compact-metric-lbl">Max Duration</div>
        </div>
        """, unsafe_allow_html=True)

    # How AudioGuard Works Section
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
                <div class="flow-title">REAL / FAKE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dashboard Feature Section (3 Cards)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Acoustic Analysis</div>
            <div class="info-desc">Analyze speech characteristics using MFCC-based acoustic features.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Machine Learning Detection</div>
            <div class="info-desc">Classification is performed using the existing trained Random Forest model.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Forensic Visualization</div>
            <div class="info-desc">Inspect waveform, MFCC and available acoustic visualizations.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Recent Analysis Status Card
    st.markdown("""
    <div class="info-card" style="text-align: center; padding: 28px;">
        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-bottom: 6px;">Ready for Analysis</div>
        <div style="font-size: 0.92rem; color: #94A3B8; margin-bottom: 16px;">Upload a speech recording to begin your first forensic analysis.</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 2: ANALYZE AUDIO
# -------------------------------------------------------------------
elif st.session_state.active_nav == "Analyze Audio":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">Audio Analysis</h2>
        <p style="color: #94A3B8; font-size: 1.02rem;">Upload or record speech audio for forensic analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # 2-Column Workspace Layout (Left: Input | Right: Status & Action)
    col_input_left, col_status_right = st.columns([50, 50])

    audio_file_buffer = None
    audio_filename = "sample.wav"

    with col_input_left:
        st.markdown("### Audio Input")
        input_tab = st.radio("Input Source:", ["Upload", "Record"], horizontal=True, label_visibility="collapsed")

        if input_tab == "Upload":
            audio_file_buffer = st.file_uploader(
                "Drop your audio file here or browse from your computer",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                help="WAV • MP3 • FLAC • OGG • M4A | Maximum: 60 seconds",
                key=f"uploader_{st.session_state.reset_count}"
            )
            if audio_file_buffer:
                audio_filename = audio_file_buffer.name
        else:
            st.markdown("#### Record Your Voice")
            if hasattr(st, "audio_input"):
                audio_file_buffer = st.audio_input("Ready to Record:")
                if audio_file_buffer:
                    audio_filename = "recorded_voice.wav"
            else:
                st.warning("Live audio recording is not supported in this environment. Please use Upload.")

    with col_status_right:
        st.markdown("### Analysis Status")
        if audio_file_buffer is None:
            st.markdown("""
            <div class="info-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC;">Ready for Analysis</div>
                <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 6px;">Upload or record audio to begin.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            audio_bytes_data = audio_file_buffer.getvalue() if hasattr(audio_file_buffer, "getvalue") else audio_file_buffer.read()
            file_size_kb = len(audio_bytes_data) / 1024
            file_fmt = Path(audio_filename).suffix.upper().replace(".", "") or "WAV"

            st.markdown(f"""
            <div class="info-card">
                <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">Audio Ready</div>
                <div style="font-size: 0.9rem; color: #CBD5E1;"><strong>Filename:</strong> <code>{audio_filename}</code></div>
                <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 4px;"><strong>Format:</strong> <code>{file_fmt}</code></div>
                <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 4px;"><strong>Size:</strong> <code>{file_size_kb:.1f} KB</code></div>
            </div>
            """, unsafe_allow_html=True)

            st.audio(audio_bytes_data, format="audio/wav")

            if st.button("Run Forensic Analysis →", type="primary", key="btn_run_forensic"):
                # Processing State Animation
                status_box = st.empty()
                status_box.markdown("""
                <div class="info-card">
                    <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">Analyzing Audio</div>
                    <div style="font-size: 0.88rem; color: #94A3B8;">Running acoustic forensic analysis...</div>
                    <div style="font-size: 0.85rem; color: #34D399; margin-top: 10px;">✓ Audio loaded</div>
                    <div style="font-size: 0.85rem; color: #34D399;">✓ Audio preprocessing</div>
                    <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700;">● MFCC feature extraction</div>
                    <div style="font-size: 0.85rem; color: #64748B;">○ Random Forest classification</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.4)

                try:
                    # Save temp file for prediction
                    suffix = Path(audio_filename).suffix or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(audio_bytes_data)
                        temp_path = tmp.name

                    # Run existing MFCC Random Forest prediction pipeline
                    result = predict_audio_file(temp_path, backend="mfcc")

                    # Remove temp file
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

                    status_box.empty()

                    # RESULT SECTION
                    st.markdown("---")
                    st.markdown("### Analysis Complete")

                    pred = result["prediction"]
                    fake_p = result["fake_probability"]
                    real_p = result["real_probability"]
                    conf = result["confidence"]
                    is_uncertain = result["is_uncertain"]

                    if is_uncertain:
                        st.markdown("""
                        <div class="result-panel-uncertain">
                            <div class="result-status-indicator" style="color: #F59E0B;">● INCONCLUSIVE</div>
                            <div class="result-header-main" style="color: #F59E0B;">RESULT INCONCLUSIVE</div>
                            <div class="result-desc-text">The acoustic characteristics are close to the decision threshold. This audio should be treated as uncertain.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif pred == "REAL":
                        st.markdown("""
                        <div class="result-panel-real">
                            <div class="result-status-indicator" style="color: #10B981;">● GENUINE HUMAN SPEECH</div>
                            <div class="result-header-main" style="color: #10B981;">REAL AUDIO</div>
                            <div class="result-desc-text">The model classified this recording as likely genuine human speech.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-panel-fake">
                            <div class="result-status-indicator" style="color: #EF4444;">● SYNTHETIC SPEECH DETECTED</div>
                            <div class="result-header-main" style="color: #EF4444;">AI-GENERATED AUDIO</div>
                            <div class="result-desc-text">The model classified this recording as likely synthetic or AI-generated speech.</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # CONFIDENCE VISUALIZATION
                    st.markdown("### Detection Confidence")
                    st.markdown(f"<h1 style='text-align: center; color: #38BDF8; font-size: 3.8rem; font-weight: 900; margin-bottom: 8px;'>{conf:.1f}%</h1>", unsafe_allow_html=True)
                    st.progress(conf / 100.0)

                    b1, b2 = st.columns(2)
                    with b1:
                        st.markdown(f"**REAL:** `{real_p:.1f}%`")
                    with b2:
                        st.markdown(f"<div style='text-align: right;'><strong>AI-GENERATED:</strong> <code>{fake_p:.1f}%</code></div>", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # RESULT INFORMATION CARDS (4 CARDS)
                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.markdown(f"""
                        <div class="compact-metric-card">
                            <div class="compact-metric-val" style="color: {'#10B981' if pred=='REAL' else '#EF4444'};">{'REAL' if pred=='REAL' else 'AI-GENERATED'}</div>
                            <div class="compact-metric-lbl">Prediction</div>
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
                            <div class="compact-metric-val" style="color: #A855F7;">{result['audio_duration_sec']:.2f} sec</div>
                            <div class="compact-metric-lbl">Audio Duration</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with r4:
                        st.markdown(f"""
                        <div class="compact-metric-card">
                            <div class="compact-metric-val" style="color: #F59E0B;">{result['processing_time_sec']:.2f} sec</div>
                            <div class="compact-metric-lbl">Processing Time</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ACOUSTIC FORENSICS SECTION
                    st.markdown("# Acoustic Forensics")
                    waveform, sr, _ = load_and_preprocess_audio(audio_bytes_data)

                    plt.style.use('dark_background')
                    v1, v2, v3 = st.tabs(["Waveform", "MFCC", "Mel-Spectrogram"])

                    with v1:
                        fig, ax = plt.subplots(figsize=(9, 2.8))
                        fig.patch.set_facecolor('#101827')
                        ax.set_facecolor('#0B1220')
                        time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
                        ax.plot(time_axis, waveform, color="#38BDF8", alpha=0.85, linewidth=0.8)
                        ax.set_xlabel("Time (seconds)", color="#94A3B8")
                        ax.set_ylabel("Amplitude", color="#94A3B8")
                        ax.set_title("Audio Amplitude Waveform", color="#F8FAFC")
                        ax.grid(True, color="#1D2940", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)

                    with v2:
                        mfccs = compute_mfcc_visualization(waveform, sr=sr)
                        fig, ax = plt.subplots(figsize=(9, 3))
                        fig.patch.set_facecolor('#101827')
                        ax.set_facecolor('#0B1220')
                        img = ax.imshow(mfccs, aspect="auto", origin="lower", cmap="viridis")
                        fig.colorbar(img, ax=ax)
                        ax.set_title("MFCC Feature Heatmap", color="#F8FAFC")
                        ax.set_xlabel("Time Frames", color="#94A3B8")
                        ax.set_ylabel("MFCC Coefficients", color="#94A3B8")
                        plt.tight_layout()
                        st.pyplot(fig)

                    with v3:
                        mel_db = compute_mel_spectrogram(waveform, sr=sr)
                        fig, ax = plt.subplots(figsize=(9, 3))
                        fig.patch.set_facecolor('#101827')
                        ax.set_facecolor('#0B1220')
                        img = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
                        fig.colorbar(img, ax=ax, format="%+2.0f dB")
                        ax.set_title("Log Mel-Spectrogram Frequency Spectrum", color="#F8FAFC")
                        ax.set_xlabel("Time Frames", color="#94A3B8")
                        ax.set_ylabel("Mel Frequency Bands", color="#94A3B8")
                        plt.tight_layout()
                        st.pyplot(fig)

                    # OPTIONAL SEGMENT ANALYSIS (If audio > 4 seconds)
                    if len(result["windows"]) > 1:
                        st.markdown("### Segment Analysis")
                        win_df = pd.DataFrame(result["windows"])
                        win_df = win_df.rename(columns={
                            "start_sec": "Start (s)",
                            "end_sec": "End (s)",
                            "label": "Classification",
                            "fake_prob": "AI Probability (%)",
                            "real_prob": "Real Prob (%)"
                        })
                        st.dataframe(win_df[["Start (s)", "End (s)", "Classification", "AI Probability (%)"]], use_container_width=True)

                        fig, ax = plt.subplots(figsize=(8, 2.8))
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
                        ax.set_title("AI Probability Over Time", color="#F8FAFC", fontsize=11, fontweight="bold")
                        ax.grid(True, color="#1D2940", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)

                    # RESULT ACTIONS
                    st.markdown("---")
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
                            label="Download Report",
                            data=report_text,
                            file_name=f"audioguard_report_{Path(audio_filename).stem}.txt",
                            mime="text/plain"
                        )

                    # DISCLAIMER
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption("Important: AudioGuard provides an automated machine-learning prediction based on acoustic characteristics. The result should be treated as an analytical indicator and not as absolute proof of authenticity.")

                except Exception as e:
                    st.error("Unable to Analyze Audio. Please upload a valid supported audio file and try again.")


# -------------------------------------------------------------------
# PAGE 3: ABOUT
# -------------------------------------------------------------------
elif st.session_state.active_nav == "About":
    st.markdown("""
    <div style="margin-bottom: 28px;">
        <h2 style="font-size: 2.3rem; font-weight: 800; margin-bottom: 8px;">About AudioGuard</h2>
        <p style="color: #94A3B8; font-size: 1.05rem;">AudioGuard is an AI audio forensic application designed to analyze speech recordings and identify acoustic characteristics associated with synthetic or AI-generated speech.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Pipeline Overview

    ```text
    Audio → Preprocessing → MFCC Features → Random Forest → Prediction
    ```

    ---

    ### Detection Technology

    - **Detection Method:** MFCC + Random Forest
    - **Input:** Speech Audio
    - **Output:** REAL / AI-GENERATED
    - **Processing:** Local CPU-based inference
    - **Supported Formats:** WAV • MP3 • FLAC • OGG • M4A
    """)
