"""
AUDIOGUARD - AI Audio Forensic Platform
Top Navigation Bar Architecture (No Left Sidebar)
Run with: streamlit run app.py
"""

import os
import json
import time
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config
from predict import predict_audio_file
from audio_utils import load_and_preprocess_audio, compute_mel_spectrogram, compute_mfcc_visualization

# -------------------------------------------------------------------
# DYNAMIC BACKEND INSPECTION
# -------------------------------------------------------------------
def get_backend_info() -> Tuple[str, int, str]:
    """
    Reads the real classifier type, feature vector dimension, and backend from models/config.json.
    Prevents hardcoding values while remaining faithful to the trained model.
    """
    config_file = config.MODELS_DIR / "config.json"
    classifier_name = "Random Forest"
    feature_dim = 172
    backend_name = "MFCC"

    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                cdata = json.load(f)
                b = cdata.get("backend", "mfcc").upper()
                c_type = cdata.get("classifier_type", "rf").lower()
                feature_dim = cdata.get("feature_dim", 172)
                backend_name = b

                if c_type in ["rf", "random_forest", "randomforest"]:
                    classifier_name = "Random Forest"
                else:
                    classifier_name = c_type.upper()
        except Exception:
            pass

    return classifier_name, feature_dim, backend_name


CLASSIFIER_NAME, FEATURE_DIM, BACKEND_NAME = get_backend_info()
DETECTION_METHOD_STR = f"{BACKEND_NAME} + {CLASSIFIER_NAME}"


# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIG & TOP NAVIGATION BAR THEME (NO SIDEBAR)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AudioGuard | AI Audio Forensic Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Full-Width Sticky Top Navigation Bar & Dark SaaS Palette
st.markdown(f"""
<style>
    /* Dark Theme Base (#080B14) */
    .stApp {{
        background-color: #080B14;
        background-image: radial-gradient(circle at 50% 0%, rgba(109, 94, 248, 0.07) 0%, transparent 75%);
        color: #F8FAFC;
        font-family: 'Manrope', 'Inter', 'Plus Jakarta Sans', -apple-system, sans-serif;
    }}

    /* Hide Streamlit Default Headers, Footers & Left Sidebar */
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    section[data-testid="stSidebar"] {{
        display: none !important;
    }}
    
    .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }}

    /* Full-Width Sticky Top Navigation Bar Container */
    .top-navbar-wrapper {{
        background: rgba(11, 18, 32, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid #1D2940;
        border-radius: 16px;
        padding: 12px 24px;
        margin-bottom: 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }}
    .top-nav-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .top-nav-logo-mark {{
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #6D5EF8 0%, #38BDF8 100%);
        border-radius: 8px;
        display: inline-block;
        box-shadow: 0 0 14px rgba(109, 94, 248, 0.5);
    }}
    .top-nav-brand-title {{
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        line-height: 1.1;
    }}
    .top-nav-brand-tagline {{
        font-size: 0.68rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    /* Top Navigation Radio Pill Buttons Styling */
    div[data-testid="stRadio"] > div {{
        display: flex;
        flex-direction: row;
        gap: 10px;
        justify-content: center;
    }}
    div[data-testid="stRadio"] label {{
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 10px;
        padding: 8px 18px;
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }}
    div[data-testid="stRadio"] label:hover {{
        background: #1E293B;
        color: #F8FAFC !important;
        border-color: #334155;
    }}
    div[data-testid="stRadio"] label[aria-checked="true"] {{
        background: linear-gradient(90deg, rgba(109, 94, 248, 0.25) 0%, rgba(56, 189, 248, 0.08) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(109, 94, 248, 0.5) !important;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(109, 94, 248, 0.3);
    }}

    /* Status Badges */
    .top-status-badge {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        color: #34D399;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
    }}
    .top-method-badge {{
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 600;
    }}
    .status-dot-green {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        margin-right: 4px;
    }}

    /* Cards & Panels */
    .info-card {{
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }}
    .info-card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }}
    .info-title {{
        font-size: 1.05rem;
        font-weight: 800;
        color: #F8FAFC;
    }}
    .info-badge {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: #1E293B;
        color: #6D5EF8;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #334155;
    }}
    .info-desc {{
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.55;
    }}

    /* Hero Card */
    .hero-card {{
        background: linear-gradient(135deg, #101827 0%, #0B1220 100%);
        border: 1px solid #1D2940;
        border-radius: 20px;
        padding: 40px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }}
    .hero-eyebrow {{
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
    }}
    .hero-h1-text {{
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.25;
        color: #FFFFFF;
        margin-bottom: 14px;
    }}
    .hero-p-text {{
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.6;
        margin-bottom: 26px;
    }}

    /* Waveform Graphic */
    .hero-wave-graphic {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        height: 60px;
        opacity: 0.85;
    }}
    .hero-wave-bar {{
        width: 4px;
        background: linear-gradient(180deg, #6D5EF8 0%, #38BDF8 100%);
        border-radius: 2px;
        animation: wavePulse 1.4s ease-in-out infinite alternate;
    }}
    .hero-wave-bar:nth-child(1) {{ height: 20px; animation-delay: 0.1s; }}
    .hero-wave-bar:nth-child(2) {{ height: 45px; animation-delay: 0.3s; }}
    .hero-wave-bar:nth-child(3) {{ height: 30px; animation-delay: 0.2s; }}
    .hero-wave-bar:nth-child(4) {{ height: 55px; animation-delay: 0.4s; }}
    .hero-wave-bar:nth-child(5) {{ height: 25px; animation-delay: 0.15s; }}
    .hero-wave-bar:nth-child(6) {{ height: 40px; animation-delay: 0.35s; }}

    @keyframes wavePulse {{
        0% {{ transform: scaleY(0.8); opacity: 0.7; }}
        100% {{ transform: scaleY(1.2); opacity: 1; }}
    }}

    /* Capability Cards */
    .capability-card {{
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 16px;
        padding: 22px;
        height: 100%;
        transition: transform 0.2s, border-color 0.2s;
    }}
    .capability-card:hover {{
        transform: translateY(-2px);
        border-color: #334155;
    }}
    .capability-num {{
        font-size: 0.78rem;
        font-weight: 800;
        color: #6D5EF8;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }}
    .capability-title {{
        font-size: 0.98rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 6px;
    }}

    /* Pipeline Flow */
    .flow-wrapper {{
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 18px;
        padding: 26px;
        margin: 24px 0;
    }}
    .flow-grid {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 18px;
    }}
    .flow-step {{
        background: #0B1220;
        border: 1px solid #1D2940;
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        flex: 1;
    }}
    .flow-num {{
        font-size: 0.72rem;
        font-weight: 800;
        color: #6D5EF8;
        margin-bottom: 4px;
    }}
    .flow-title {{
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
    }}
    .flow-arrow {{
        color: #475569;
        font-weight: 800;
        font-size: 1.1rem;
    }}

    /* Result Panels */
    .result-panel-real {{
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(16, 24, 39, 0.95) 100%);
        border: 1px solid #10B981;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
    }}
    .result-panel-fake {{
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(16, 24, 39, 0.95) 100%);
        border: 1px solid #EF4444;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
    }}
    .result-panel-uncertain {{
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.45) 0%, rgba(16, 24, 39, 0.95) 100%);
        border: 1px solid #F59E0B;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.15);
    }}
    .result-status-indicator {{
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .result-header-main {{
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        margin-bottom: 8px;
    }}
    .result-desc-text {{
        font-size: 1.02rem;
        color: #E2E8F0;
        max-width: 580px;
        margin: 0 auto;
        line-height: 1.5;
    }}

    /* Buttons */
    div.stButton > button {{
        background: linear-gradient(135deg, #6D5EF8 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 12px 24px;
        font-size: 0.95rem;
        transition: all 0.2s ease-in-out;
        width: 100%;
        box-shadow: 0 4px 14px rgba(109, 94, 248, 0.3);
    }}
    div.stButton > button:hover {{
        background: linear-gradient(135deg, #5B4CE0 0%, #3730A3 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(109, 94, 248, 0.45);
    }}

    /* Download Report Button */
    div.stDownloadButton > button {{
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 12px;
        font-weight: 700;
        padding: 12px 24px;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }}
    div.stDownloadButton > button:hover {{
        background: #334155;
        color: #FFFFFF;
        border-color: #475569;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        background: transparent;
        color: #94A3B8 !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 16px;
    }}
    button[aria-selected="true"] {{
        color: #38BDF8 !important;
        background: #101827 !important;
        font-weight: 700;
    }}
</style>
""", unsafe_allow_html=True)


# Initialize Navigation State
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Dashboard"
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0


# -------------------------------------------------------------------
# STICKY TOP NAVIGATION BAR (NO LEFT SIDEBAR)
# -------------------------------------------------------------------
nav_col1, nav_col2, nav_col3 = st.columns([30, 45, 25])

with nav_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0;">
        <span class="top-nav-logo-mark"></span>
        <div>
            <div class="top-nav-brand-title">AUDIOGUARD</div>
            <div class="top-nav-brand-tagline">AI AUDIO FORENSIC PLATFORM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    nav_options = ["Dashboard", "Audio Analysis", "System Architecture", "About"]
    current_idx = nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0

    nav_selection = st.radio(
        "Navigation",
        nav_options,
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state.active_nav = nav_selection

with nav_col3:
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: flex-end; gap: 10px; padding-top: 4px;">
        <div class="top-status-badge">
            <span class="status-dot-green"></span> Engine Operational
        </div>
        <div class="top-method-badge">
            {DETECTION_METHOD_STR}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: none; border-bottom: 1px solid #1D2940; margin-top: 8px; margin-bottom: 28px;'>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 1: DASHBOARD
# -------------------------------------------------------------------
if st.session_state.active_nav == "Dashboard":
    col_hero_left, col_hero_right = st.columns([72, 28])

    with col_hero_left:
        st.markdown("""
        <div class="hero-card">
            <div class="hero-eyebrow">AI AUDIO FORENSICS</div>
            <div class="hero-h1-text">Detect AI-Generated Voices<br><span style="color: #38BDF8;">with confidence.</span></div>
            <div class="hero-p-text">Analyze speech recordings using acoustic feature analysis and a trained machine-learning model to identify characteristics associated with synthetic or AI-generated speech.</div>
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

        st.markdown("</div>", unsafe_allow_html=True)

    with col_hero_right:
        st.markdown("""
        <div class="info-card" style="height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 32px 20px;">
            <div style="font-size: 0.76rem; font-weight: 800; color: #38BDF8; letter-spacing: 0.08em; text-transform: uppercase;">ACOUSTIC INTELLIGENCE</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin-top: 6px;">AudioGuard</div>
            <div class="hero-wave-graphic" style="margin: 20px 0;">
                <div class="hero-wave-bar"></div>
                <div class="hero-wave-bar"></div>
                <div class="hero-wave-bar"></div>
                <div class="hero-wave-bar"></div>
                <div class="hero-wave-bar"></div>
                <div class="hero-wave-bar"></div>
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600;">MFCC Feature Engine</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### CAPABILITIES")

    # 4 Capability Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="capability-card">
            <div class="capability-num">01</div>
            <div class="capability-title">MFCC FEATURE ANALYSIS</div>
            <div class="info-desc">Acoustic feature extraction for speech analysis.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="capability-card">
            <div class="capability-num">02</div>
            <div class="capability-title">{CLASSIFIER_NAME.upper()} CLASSIFICATION</div>
            <div class="info-desc">Fast CPU-based inference using the trained classifier.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="capability-card">
            <div class="capability-num">03</div>
            <div class="capability-title">ACOUSTIC VISUALIZATION</div>
            <div class="info-desc">Waveform, MFCC and Mel-Spectrogram inspection.</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="capability-card">
            <div class="capability-num">04</div>
            <div class="capability-title">LOCAL PROCESSING</div>
            <div class="info-desc">Audio analysis performed locally without external AI APIs.</div>
        </div>
        """, unsafe_allow_html=True)

    # How AudioGuard Analyzes Audio Flow Pipeline
    st.markdown(f"""
    <div class="flow-wrapper">
        <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC;">How AudioGuard Analyzes Audio</div>
        <div class="flow-grid">
            <div class="flow-step">
                <div class="flow-num">STAGE 01</div>
                <div class="flow-title">AUDIO INPUT</div>
                <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 4px;">Speech File Load</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STAGE 02</div>
                <div class="flow-title">PREPROCESSING</div>
                <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 4px;">16 kHz Standard</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STAGE 03</div>
                <div class="flow-title">MFCC FEATURES</div>
                <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 4px;">{FEATURE_DIM}-Dim Extraction</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STAGE 04</div>
                <div class="flow-title">{CLASSIFIER_NAME.upper()}</div>
                <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 4px;">Model Inference</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STAGE 05</div>
                <div class="flow-title">DETECTION RESULT</div>
                <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 4px;">REAL / FAKE Verdict</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 2: AUDIO ANALYSIS (MAIN FORENSIC WORKSPACE)
# -------------------------------------------------------------------
elif st.session_state.active_nav == "Audio Analysis":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">Audio Analysis</h2>
        <p style="color: #94A3B8; font-size: 1.02rem;">Upload or record speech audio for forensic inspection.</p>
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
        <div class="info-card" style="text-align: center; padding: 36px 20px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-bottom: 6px;">AUDIO ANALYSIS</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-bottom: 4px;">Drop your audio here</div>
            <div style="font-size: 0.9rem; color: #94A3B8;">Drag & drop your file or Browse Files</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 12px;">WAV, MP3, FLAC, OGG, M4A - maximum 60 seconds</div>
        </div>
        """, unsafe_allow_html=True)

        audio_file_buffer = st.file_uploader(
            "Browse Files",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="WAV, MP3, FLAC, OGG, M4A - maximum 60 seconds",
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

    # Empty State vs Selected Audio Workspace
    if audio_file_buffer is None:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 48px 20px; margin-top: 20px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">READY FOR FORENSIC ANALYSIS</div>
            <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 8px;">Upload a speech recording to begin acoustic inspection.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        audio_bytes_data = audio_file_buffer.getvalue() if hasattr(audio_file_buffer, "getvalue") else audio_file_buffer.read()
        file_size_mb = len(audio_bytes_data) / (1024 * 1024)
        file_fmt = Path(audio_filename).suffix.upper().replace(".", "") or "WAV"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title" style="margin-bottom: 12px;">Selected Audio</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #38BDF8; margin-bottom: 14px;"><code>{audio_filename}</code></div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 18px;">
                <div style="font-size: 0.88rem; color: #CBD5E1;"><strong>FORMAT:</strong> <code>{file_fmt}</code></div>
                <div style="font-size: 0.88rem; color: #CBD5E1;"><strong>SAMPLE RATE:</strong> <code>16 kHz</code></div>
                <div style="font-size: 0.88rem; color: #CBD5E1;"><strong>CHANNELS:</strong> <code>MONO</code></div>
                <div style="font-size: 0.88rem; color: #CBD5E1;"><strong>FILE SIZE:</strong> <code>{file_size_mb:.2f} MB</code></div>
            </div>
        """, unsafe_allow_html=True)

        st.audio(audio_bytes_data, format="audio/wav")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("RUN FORENSIC ANALYSIS →", type="primary", key="btn_run_forensic"):
            status_box = st.empty()
            status_box.markdown(f"""
            <div class="info-card">
                <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">Analyzing Audio</div>
                <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 12px;">Running acoustic forensic analysis...</div>
                <div style="font-size: 0.85rem; color: #34D399;">✓ Audio preprocessing</div>
                <div style="font-size: 0.85rem; color: #34D399;">✓ Feature extraction</div>
                <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700;">● {CLASSIFIER_NAME} classification</div>
                <div style="font-size: 0.85rem; color: #64748B;">○ Result generation</div>
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
                        <div class="result-desc-text">Borderline result: The acoustic confidence is close to the decision threshold. Verification recommended.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred == "REAL":
                    st.markdown("""
                    <div class="result-panel-real">
                        <div class="result-status-indicator" style="color: #10B981;">● AUTHENTIC AUDIO</div>
                        <div class="result-header-main" style="color: #10B981;">AUTHENTIC AUDIO</div>
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

                # Large Circular / Numerical Confidence Gauge
                st.markdown(f"<h1 style='text-align: center; color: #38BDF8; font-size: 3.8rem; font-weight: 900; margin-bottom: 2px;'>{conf:.1f}%</h1>", unsafe_allow_html=True)
                st.markdown("<div style='text-align: center; font-size: 0.8rem; font-weight: 800; color: #94A3B8; letter-spacing: 0.1em; margin-bottom: 16px;'>CONFIDENCE</div>", unsafe_allow_html=True)
                st.progress(conf / 100.0)

                b1, b2 = st.columns(2)
                with b1:
                    st.markdown(f"**REAL:** `{real_p:.1f}%`")
                with b2:
                    st.markdown(f"<div style='text-align: right;'><strong>AI-GENERATED:</strong> <code>{fake_p:.1f}%</code></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Result Summary Grid
                st.markdown("### RESULT SUMMARY")
                r1, r2, r3, r4, r5, r6, r7 = st.columns(7)
                with r1:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: {'#10B981' if pred=='REAL' else '#EF4444'};">{'REAL' if pred=='REAL' else 'AI-GENERATED'}</div>
                        <div class="compact-metric-lbl">CLASSIFICATION</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val">{conf:.1f}%</div>
                        <div class="compact-metric-lbl">CONFIDENCE</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r3:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #10B981;">{real_p:.1f}%</div>
                        <div class="compact-metric-lbl">REAL PROB</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r4:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #EF4444;">{fake_p:.1f}%</div>
                        <div class="compact-metric-lbl">AI PROB</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r5:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #F59E0B;">{proc_time:.2f} s</div>
                        <div class="compact-metric-lbl">PROCESSING TIME</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r6:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #A855F7;">{result['audio_duration_sec']:.2f} s</div>
                        <div class="compact-metric-lbl">AUDIO DURATION</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r7:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #38BDF8;">{DETECTION_METHOD_STR}</div>
                        <div class="compact-metric-lbl">METHOD</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Acoustic Forensics Section
                st.markdown("### ACOUSTIC FORENSICS")
                waveform, sr, _ = load_and_preprocess_audio(audio_bytes_data)

                plt.style.use('dark_background')
                v1, v2, v3 = st.tabs(["WAVEFORM", "MFCC", "MEL-SPECTROGRAM"])

                with v1:
                    st.caption("Time-domain representation of the analyzed audio.")
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
                    st.caption("Mel-frequency cepstral representation used for feature extraction.")
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

                # Segment Analysis (If audio duration > 4 seconds)
                if len(result["windows"]) > 1:
                    st.markdown("### SEGMENT ANALYSIS")
                    st.caption("AI PROBABILITY OVER TIME")
                    
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

                    # Compact Segment Cards
                    seg_cols = st.columns(min(len(result["windows"]), 4))
                    for idx, w in enumerate(result["windows"][:4]):
                        with seg_cols[idx % 4]:
                            st.markdown(f"""
                            <div class="compact-metric-card">
                                <div style="font-size: 0.78rem; font-weight: 800; color: #6D5EF8;">{w['start_sec']:.1f}s–{w['end_sec']:.1f}s</div>
                                <div style="font-size: 0.95rem; font-weight: 800; color: {'#EF4444' if w['fake_prob'] >= 50 else '#10B981'}; margin-top: 4px;">{w['fake_prob']:.1f}%</div>
                                <div class="compact-metric-lbl">AI Probability</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Forensic Report Actions
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### ANALYSIS REPORT")
                
                act_col1, act_col2 = st.columns(2)
                with act_col1:
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

Detection Method:   {DETECTION_METHOD_STR}
Processing Time:    {result['processing_time_sec']:.2f} seconds
==================================================
"""
                    st.download_button(
                        label="Download Analysis Report",
                        data=report_text,
                        file_name=f"audioguard_report_{Path(audio_filename).stem}.txt",
                        mime="text/plain"
                    )

                with act_col2:
                    if st.button("Analyze Another Audio", key="btn_reset_forensic"):
                        st.session_state.reset_count += 1
                        st.rerun()

                # Disclaimer
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("AudioGuard provides probabilistic machine-learning predictions. Results should be treated as automated indicators rather than absolute proof of authenticity.")

            except Exception:
                st.error("ANALYSIS FAILED - Unable to process this audio file. Check that the format is supported and try again.")

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 3: SYSTEM ARCHITECTURE
# -------------------------------------------------------------------
elif st.session_state.active_nav == "System Architecture":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">System Architecture</h2>
        <p style="color: #94A3B8; font-size: 1.05rem;">AudioGuard Detection Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture Flowchart Diagram
    st.markdown(f"""
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
                <div class="flow-title">Audio Preprocessing</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 03</div>
                <div class="flow-title">16 kHz Mono Standardization</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 04</div>
                <div class="flow-title">MFCC Feature Extraction</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 05</div>
                <div class="flow-title">{FEATURE_DIM}-Dimensional Feature Vector</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 06</div>
                <div class="flow-title">{CLASSIFIER_NAME} Classifier</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 07</div>
                <div class="flow-title">Probability Estimation</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-num">STEP 08</div>
                <div class="flow-title">REAL / AI-GENERATED</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Technical Sections
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <div class="info-card-header">
                <div class="info-title">AUDIO PREPROCESSING</div>
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
        st.markdown(f"""
        <div class="info-card" style="height: 100%;">
            <div class="info-card-header">
                <div class="info-title">FEATURE EXTRACTION</div>
                <div class="info-badge">STAGE 2</div>
            </div>
            <div class="info-desc">
                Computes {FEATURE_DIM}-dimensional acoustic feature representations:
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
        st.markdown(f"""
        <div class="info-card" style="height: 100%;">
            <div class="info-card-header">
                <div class="info-title">CLASSIFICATION</div>
                <div class="info-badge">STAGE 3</div>
            </div>
            <div class="info-desc">
                {CLASSIFIER_NAME} classifier trained for binary speech classification:
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
# PAGE 4: ABOUT
# -------------------------------------------------------------------
elif st.session_state.active_nav == "About":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">About AudioGuard</h2>
        <p style="color: #94A3B8; font-size: 1.05rem;">AudioGuard AI Audio Forensic Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
        <div class="info-title" style="font-size: 1.2rem; margin-bottom: 8px;">About AudioGuard</div>
        <div class="info-desc" style="font-size: 1.02rem;">
            AudioGuard is an AI-assisted audio forensic application that analyzes speech recordings using acoustic feature extraction and a trained {CLASSIFIER_NAME} classifier to identify characteristics associated with AI-generated speech.
        </div>
    </div>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Detection Method</div>
            <div class="info-desc" style="font-size: 1rem; color: #38BDF8; font-weight: 700; margin-top: 6px;">{DETECTION_METHOD_STR}</div>
            <div class="info-desc" style="margin-top: 6px;">Extracts spectral envelop and cepstral properties from audio signals to capture subtle synthetic artifacts.</div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Input</div>
            <div class="info-desc" style="font-size: 1rem; color: #818CF8; font-weight: 700; margin-top: 6px;">Speech Audio</div>
            <div class="info-desc" style="margin-top: 6px;">Supports WAV, MP3, FLAC, OGG, and M4A audio files up to 60 seconds duration.</div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Processing</div>
            <div class="info-desc" style="font-size: 1rem; color: #10B981; font-weight: 700; margin-top: 6px;">Local Analysis</div>
            <div class="info-desc" style="margin-top: 6px;">Executes local CPU-based inference with privacy-focused audio processing.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 24px;">
        <p style="color: #64748B; font-size: 0.85rem;">AudioGuard provides probabilistic machine-learning predictions. Results should be treated as automated indicators rather than absolute proof of authenticity.</p>
    </div>
    """, unsafe_allow_html=True)
