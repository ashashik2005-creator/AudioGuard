"""
AUDIOGUARD - AI Audio Forensic Platform
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
# STREAMLIT PAGE CONFIGURATION & ENTERPRISE DARK SAAS THEME
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AudioGuard | AI Audio Forensic Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Commercial AI Forensic SaaS Product
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
    #MainMenu {visibility: hidden;}
    
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1140px;
    }

    /* Narrow Fixed Left Sidebar */
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
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-subtitle-text {
        font-size: 0.74rem;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Modern Styled Sidebar Radio Navigation */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 10px 14px;
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 0.95rem;
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
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.18) 0%, rgba(56, 189, 248, 0.06) 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        font-weight: 700;
    }

    /* Compact Sidebar Bottom Status Box */
    .sidebar-status-box {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 48px;
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
    }
    .top-bar-badge {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.76rem;
        color: #94A3B8;
        font-weight: 600;
    }

    /* General Card Containers */
    .info-card {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .info-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    .info-desc {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.55;
    }

    /* Hero Section Card */
    .hero-card {
        background: linear-gradient(135deg, #101827 0%, #0B1220 100%);
        border: 1px solid #1D2940;
        border-radius: 20px;
        padding: 38px 34px;
        margin-bottom: 24px;
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
        margin-bottom: 24px;
    }

    /* Compact Metrics Row */
    .compact-metric-card {
        background: #101827;
        border: 1px solid #1D2940;
        border-radius: 14px;
        padding: 18px;
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
        margin-top: 6px;
    }

    /* Primary Result Panels */
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

    /* Style Download Button */
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

    /* Style Streamlit File Uploader Box */
    section[data-testid="stFileUploader"] {
        background: #0B1220;
        border: 1px dashed #334155;
        border-radius: 16px;
        padding: 20px;
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
    st.session_state.active_nav = "🏠 Dashboard"
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0


# -------------------------------------------------------------------
# SIDEBAR NAVIGATION (STRICTLY 3 ITEMS)
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-wrapper">
        <div class="sidebar-logo-text">
            🎙️ AUDIOGUARD
        </div>
        <div class="sidebar-subtitle-text">AI Audio Forensic Platform</div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = ["🏠 Dashboard", "🎙️ Analyze Audio", "ℹ️ About"]
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
        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Detection Engine</div>
        <div style="font-size: 0.88rem; font-weight: 700; color: #10B981; margin-top: 4px;">
            <span class="status-dot-green"></span> 🟢 Operational
        </div>
        <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 4px; font-weight: 500;">MFCC + Random Forest</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# COMPACT TOP HEADER BAR
# -------------------------------------------------------------------
st.markdown("""
<div class="top-bar">
    <div class="top-bar-title">
        AudioGuard <span class="top-bar-sub">AI Audio Deepfake Detection</span>
    </div>
    <div class="top-bar-badge-box">
        <div class="top-bar-status">
            <span class="status-dot-green"></span> 🟢 Detection Engine Ready
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
if st.session_state.active_nav == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-h1-text">Detect AI-Generated Voices</div>
        <div class="hero-p-text">Analyze speech recordings using acoustic feature analysis and a trained machine-learning model to identify characteristics associated with synthetic or AI-generated speech.</div>
    </div>
    """, unsafe_allow_html=True)

    h_btn1, h_btn2 = st.columns(2)
    with h_btn1:
        if st.button("🎙️ Analyze Audio", key="btn_hero_an"):
            st.session_state.active_nav = "🎙️ Analyze Audio"
            st.rerun()
    with h_btn2:
        if st.button("🎤 Record Audio", key="btn_hero_rec"):
            st.session_state.active_nav = "🎙️ Analyze Audio"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Dashboard Information Cards
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">🎙️ Audio Analysis</div>
            <div class="info-desc">Upload or record speech audio for automated analysis.</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">🧠 Acoustic Features</div>
            <div class="info-desc">The system analyzes MFCC-based acoustic characteristics.</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">⚡ Fast Detection</div>
            <div class="info-desc">The trained Random Forest model provides CPU-based inference.</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">🔒 Privacy Focused</div>
            <div class="info-desc">Audio is processed through the local application.</div>
        </div>
        """, unsafe_allow_html=True)

    # Supported Audio Information Section
    st.markdown("""
    <div class="info-card" style="margin-top: 10px;">
        <div class="info-title">Supported Audio</div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-top: 6px;">WAV • MP3 • FLAC • OGG • M4A</div>
        <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 8px;"><strong>Maximum Duration:</strong> 60 seconds</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 2: ANALYZE AUDIO
# -------------------------------------------------------------------
elif st.session_state.active_nav == "🎙️ Analyze Audio":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">Audio Analysis</h2>
        <p style="color: #94A3B8; font-size: 1.02rem;">Upload or record speech audio for forensic analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    input_tab = st.radio(
        "Input Method:",
        ["Upload Audio", "Record Audio"],
        horizontal=True
    )

    audio_file_buffer = None
    audio_filename = "sample.wav"

    if input_tab == "Upload Audio":
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 28px;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🎙️</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC;">Drop your audio file here</div>
            <div style="font-size: 0.9rem; color: #94A3B8;">or browse files from your computer</div>
            <div style="font-size: 0.82rem; color: #64748B; margin-top: 10px;">WAV • MP3 • FLAC • OGG • M4A | Maximum duration: 60 seconds</div>
        </div>
        """, unsafe_allow_html=True)

        audio_file_buffer = st.file_uploader(
            "Select Audio File",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility="collapsed",
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
            st.warning("Live audio recording is not supported in this environment. Please use Upload Audio.")

    # Audio Preview & Analysis Trigger Section
    if audio_file_buffer is not None:
        audio_bytes_data = audio_file_buffer.getvalue() if hasattr(audio_file_buffer, "getvalue") else audio_file_buffer.read()
        file_size_kb = len(audio_bytes_data) / 1024
        file_fmt = Path(audio_filename).suffix.upper().replace(".", "") or "WAV"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Selected Audio</div>
            <div style="font-size: 0.92rem; color: #CBD5E1;"><strong>Filename:</strong> <code>{audio_filename}</code></div>
            <div style="font-size: 0.92rem; color: #CBD5E1; margin-top: 4px;"><strong>Format:</strong> <code>{file_fmt}</code></div>
            <div style="font-size: 0.92rem; color: #CBD5E1; margin-top: 4px; margin-bottom: 16px;"><strong>File Size:</strong> <code>{file_size_kb:.1f} KB</code></div>
        """, unsafe_allow_html=True)

        st.audio(audio_bytes_data, format="audio/wav")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍 Analyze Audio", type="primary", key="btn_run_forensic"):
            status_box = st.empty()
            status_box.markdown("""
            <div class="info-card">
                <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">Analyzing Audio</div>
                <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 12px;">Processing your audio through the detection engine...</div>
                <div style="font-size: 0.85rem; color: #34D399;">✓ Audio loaded</div>
                <div style="font-size: 0.85rem; color: #34D399;">✓ Audio preprocessing</div>
                <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700;">● MFCC feature extraction</div>
                <div style="font-size: 0.85rem; color: #64748B;">○ Random Forest classification</div>
                <div style="font-size: 0.85rem; color: #64748B;">○ Probability calculation</div>
                <div style="font-size: 0.85rem; color: #64748B;">○ Generating analysis result</div>
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

                if is_uncertain:
                    st.markdown("""
                    <div class="result-panel-uncertain">
                        <div class="result-status-indicator" style="color: #F59E0B;">🟡 INCONCLUSIVE</div>
                        <div class="result-header-main" style="color: #F59E0B;">RESULT INCONCLUSIVE</div>
                        <div class="result-desc-text">The acoustic characteristics are close to the decision threshold.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred == "REAL":
                    st.markdown("""
                    <div class="result-panel-real">
                        <div class="result-status-indicator" style="color: #10B981;">🟢 REAL AUDIO</div>
                        <div class="result-header-main" style="color: #10B981;">REAL AUDIO</div>
                        <div class="result-desc-text">The audio is classified as likely genuine human speech.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-panel-fake">
                        <div class="result-status-indicator" style="color: #EF4444;">🔴 AI-GENERATED AUDIO</div>
                        <div class="result-header-main" style="color: #EF4444;">AI-GENERATED AUDIO</div>
                        <div class="result-desc-text">The audio is classified as likely synthetic or AI-generated speech.</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence Display
                st.markdown("### Detection Confidence")
                st.markdown(f"<h1 style='text-align: center; color: #38BDF8; font-size: 3.6rem; font-weight: 900; margin-bottom: 8px;'>{conf:.1f}%</h1>", unsafe_allow_html=True)
                st.progress(conf / 100.0)

                b1, b2 = st.columns(2)
                with b1:
                    st.markdown(f"**REAL:** `{real_p:.1f}%`")
                with b2:
                    st.markdown(f"<div style='text-align: right;'><strong>AI-GENERATED:</strong> <code>{fake_p:.1f}%</code></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Analysis Details Grid
                r1, r2, r3, r4, r5 = st.columns(5)
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
                with r5:
                    st.markdown(f"""
                    <div class="compact-metric-card">
                        <div class="compact-metric-val" style="color: #38BDF8;">MFCC + Random Forest</div>
                        <div class="compact-metric-lbl">Detection Method</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Acoustic Forensic Visualizations
                st.markdown("### Acoustic Analysis")
                waveform, sr, _ = load_and_preprocess_audio(audio_bytes_data)

                plt.style.use('dark_background')
                v1, v2, v3 = st.tabs(["Waveform", "MFCC", "Mel-Spectrogram"])

                with v1:
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
                    if st.button("🔄 Analyze Another Audio", key="btn_reset_forensic"):
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
                        label="📄 Download Report",
                        data=report_text,
                        file_name=f"audioguard_report_{Path(audio_filename).stem}.txt",
                        mime="text/plain"
                    )

                # Disclaimer
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("Important: AudioGuard provides an automated machine-learning prediction based on acoustic characteristics. The result should be treated as an analytical indicator and not as absolute proof of authenticity.")

            except Exception as e:
                st.error("Unable to Analyze Audio. Please upload a valid supported audio file and try again.")

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 3: ABOUT
# -------------------------------------------------------------------
elif st.session_state.active_nav == "ℹ️ About":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;">About AudioGuard</h2>
        <p style="color: #94A3B8; font-size: 1.05rem;">AudioGuard is an AI-powered audio forensic application designed to analyze speech recordings and identify characteristics associated with synthetic or AI-generated speech.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-title">Detection Pipeline Flow</div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; font-family: monospace;">Audio → Preprocessing → MFCC Features → Random Forest → Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-title">Detection Technology</div>
        <table style="width: 100%; border-collapse: collapse; color: #CBD5E1; font-size: 0.95rem;">
            <tr style="border-bottom: 1px solid #1D2940;">
                <td style="padding: 10px 0; font-weight: 700; color: #94A3B8; width: 35%;">Detection Method</td>
                <td style="padding: 10px 0; font-weight: 600; color: #F8FAFC;">MFCC + Random Forest</td>
            </tr>
            <tr style="border-bottom: 1px solid #1D2940;">
                <td style="padding: 10px 0; font-weight: 700; color: #94A3B8;">Input</td>
                <td style="padding: 10px 0; font-weight: 600; color: #F8FAFC;">Speech Audio</td>
            </tr>
            <tr style="border-bottom: 1px solid #1D2940;">
                <td style="padding: 10px 0; font-weight: 700; color: #94A3B8;">Output</td>
                <td style="padding: 10px 0; font-weight: 600; color: #F8FAFC;">REAL / AI-GENERATED</td>
            </tr>
            <tr>
                <td style="padding: 10px 0; font-weight: 700; color: #94A3B8;">Processing</td>
                <td style="padding: 10px 0; font-weight: 600; color: #F8FAFC;">Local CPU-based inference</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
