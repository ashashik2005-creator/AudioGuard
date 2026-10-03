"""
AudioGuard - Premium AI Audio Deepfake Detection Web Application
Run with: streamlit run app.py
"""

import os
import io
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
# STREAMLIT CONFIGURATION & HYPER-MODERN DARK SAAS STYLING
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AudioGuard | AI Audio Deepfake Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Stunning Dark Navy / Purple Gradient SaaS Interface
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit padding & header elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }

    /* Glassmorphic Brand Header */
    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 28px;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.6);
    }
    .brand-name {
        font-size: 1.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-sub {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Hero Section */
    .hero-wrapper {
        text-align: center;
        padding: 52px 24px 36px 24px;
        background: radial-gradient(circle at 50% 30%, rgba(99, 102, 241, 0.22) 0%, rgba(168, 85, 247, 0.08) 45%, rgba(11, 15, 25, 0) 70%);
        border-radius: 24px;
        margin-bottom: 28px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hero-badge {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 30px;
        color: #A5B4FC;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 18px;
    }
    .hero-h1 {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.025em;
        line-height: 1.2;
        color: #FFFFFF;
        margin-bottom: 16px;
    }
    .hero-p {
        font-size: 1.22rem;
        color: #94A3B8;
        max-width: 680px;
        margin: 0 auto 28px auto;
        line-height: 1.55;
    }

    /* Equalizer Waveform Graphic */
    .eq-visualizer {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        height: 46px;
        margin: 20px 0;
    }
    .eq-bar {
        width: 5px;
        background: linear-gradient(180deg, #6366F1 0%, #A855F7 100%);
        border-radius: 3px;
        animation: pulse 1.4s ease-in-out infinite alternate;
    }
    .eq-bar:nth-child(2) { height: 34px; animation-delay: 0.2s; }
    .eq-bar:nth-child(3) { height: 46px; animation-delay: 0.4s; }
    .eq-bar:nth-child(4) { height: 22px; animation-delay: 0.1s; }
    .eq-bar:nth-child(5) { height: 40px; animation-delay: 0.5s; }
    .eq-bar:nth-child(6) { height: 28px; animation-delay: 0.3s; }

    /* Feature Cards */
    .feature-item {
        background: #151D2A;
        border: 1px solid #1E293B;
        border-radius: 18px;
        padding: 24px;
        height: 100%;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .feature-item:hover {
        transform: translateY(-4px);
        border-color: #6366F1;
        box-shadow: 0 12px 24px -10px rgba(99, 102, 241, 0.3);
    }
    .feature-icon-box {
        font-size: 2.2rem;
        margin-bottom: 12px;
    }
    .feature-title-box {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    .feature-desc-box {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* Metric Highlights Strip */
    .stats-strip {
        display: flex;
        align-items: center;
        justify-content: space-around;
        background: #151D2A;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 20px;
        margin-top: 32px;
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .stat-lbl {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Result Cards */
    .result-real-card {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #10B981;
        border-radius: 24px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 20px 35px -10px rgba(16, 185, 129, 0.25);
    }
    .result-fake-card {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.5) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #EF4444;
        border-radius: 24px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 20px 35px -10px rgba(239, 68, 68, 0.25);
    }
    .result-uncertain-card {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.5) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #F59E0B;
        border-radius: 24px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 20px 35px -10px rgba(245, 158, 11, 0.25);
    }
    .result-symbol {
        font-size: 3.6rem;
        margin-bottom: 8px;
    }
    .result-main-header {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        margin-bottom: 8px;
    }
    .result-sub-header {
        font-size: 1.15rem;
        color: #E2E8F0;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Summary Metric Box */
    .metric-summary-card {
        background: #151D2A;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .metric-summary-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-summary-lbl {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* Custom Gradient Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 14px 28px;
        font-size: 1.05rem;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px 0 rgba(99, 102, 241, 0.55);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        background-color: #151D2A;
        border-radius: 12px;
        color: #94A3B8;
        font-weight: 600;
        border: 1px solid #1E293B;
        padding: 0 24px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: white !important;
        border: none !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        padding: 36px 0 16px 0;
        border-top: 1px solid #1E293B;
        margin-top: 56px;
        color: #64748B;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State Variables
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "reset_id" not in st.session_state:
    st.session_state.reset_id = 0
if "preset_sample" not in st.session_state:
    st.session_state.preset_sample = None


# -------------------------------------------------------------------
# BRAND HEADER & TOP NAVIGATION
# -------------------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div>
        <div class="brand-name">🎙️ AudioGuard</div>
        <div class="brand-sub">AI-Powered Audio Deepfake Detection</div>
    </div>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns([1, 1, 1])
with nav1:
    if st.button("🏠 Home", key="n_home"):
        st.session_state.page = "Home"
        st.session_state.preset_sample = None
        st.rerun()
with nav2:
    if st.button("🎙️ Analyze", key="n_analyze"):
        st.session_state.page = "Analyze"
        st.rerun()
with nav3:
    if st.button("ℹ️ About", key="n_about"):
        st.session_state.page = "About"
        st.session_state.preset_sample = None
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# 1. HOME PAGE
# -------------------------------------------------------------------
if st.session_state.page == "Home":
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-badge">⚡ REAL-TIME ACOUSTIC VOICE VERIFICATION</div>
        <div class="hero-h1">Detect AI-Generated Voices</div>
        <div class="hero-p">Analyze speech recordings with AI and discover whether the voice is genuine human speech or synthetic AI deepfake.</div>
        <div class="eq-visualizer">
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎙️ Analyze Audio File", key="cta_an"):
            st.session_state.page = "Analyze"
            st.session_state.preset_sample = None
            st.rerun()
    with c2:
        if st.button("🎤 Record Voice Audio", key="cta_rec"):
            st.session_state.page = "Analyze"
            st.session_state.preset_sample = None
            st.rerun()

    # 1-Click Sample Pre-load Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.95rem; margin-bottom: 12px;'><strong>Or test instantly with 1-click sample audio clips:</strong></div>", unsafe_allow_html=True)

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        if st.button("🧪 Try Sample Real Human Voice", key="sample_real"):
            st.session_state.page = "Analyze"
            real_files = list((config.TEST_DIR / "real").glob("*.*"))
            if real_files:
                st.session_state.preset_sample = str(real_files[0])
            st.rerun()
    with s_col2:
        if st.button("🧪 Try Sample AI Synthetic Voice", key="sample_fake"):
            st.session_state.page = "Analyze"
            fake_files = list((config.TEST_DIR / "fake").glob("*.*"))
            if fake_files:
                st.session_state.preset_sample = str(fake_files[0])
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Feature Cards
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon-box">🎧</div>
            <div class="feature-title-box">Audio Analysis</div>
            <div class="feature-desc-box">Analyze speech recordings using acoustic characteristics extracted from the audio.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon-box">🧠</div>
            <div class="feature-title-box">AI Detection</div>
            <div class="feature-desc-box">The existing MFCC + Random Forest model analyzes the uploaded audio.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon-box">⚡</div>
            <div class="feature-title-box">Fast Results</div>
            <div class="feature-desc-box">Get an accurate detection result within seconds on CPU.</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon-box">🔒</div>
            <div class="feature-title-box">Local Processing</div>
            <div class="feature-desc-box">Audio is processed locally by the application and is not permanently stored.</div>
        </div>
        """, unsafe_allow_html=True)

    # Stats Strip
    st.markdown("""
    <div class="stats-strip">
        <div style="text-align: center;">
            <div class="stat-val">&lt; 1.5s</div>
            <div class="stat-lbl">Inference Speed</div>
        </div>
        <div style="text-align: center;">
            <div class="stat-val">172-dim</div>
            <div class="stat-lbl">MFCC Features</div>
        </div>
        <div style="text-align: center;">
            <div class="stat-val">100%</div>
            <div class="stat-lbl">Local Privacy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# 2. ANALYZE PAGE
# -------------------------------------------------------------------
elif st.session_state.page == "Analyze":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 28px;">
        <h2 style="font-size: 2.4rem; font-weight: 800; margin-bottom: 8px;">Analyze Your Audio</h2>
        <p style="color: #94A3B8; font-size: 1.1rem;">Upload a recording or record your voice to check its authenticity.</p>
    </div>
    """, unsafe_allow_html=True)

    audio_file_buffer = None
    audio_filename = "sample.wav"

    # Check if pre-loaded sample was triggered
    preset_path = st.session_state.preset_sample

    if preset_path and os.path.exists(preset_path):
        st.info(f"📁 Pre-loaded sample selected: `{Path(preset_path).name}`")
        with open(preset_path, "rb") as f:
            audio_bytes_data = f.read()
            audio_file_buffer = io.BytesIO(audio_bytes_data)
            audio_filename = Path(preset_path).name
    else:
        input_mode = st.radio(
            "Choose Input Method:",
            ["Upload Audio", "Record Audio"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if input_mode == "Upload Audio":
            audio_file_buffer = st.file_uploader(
                "🎵 Drop your audio here or Browse Files",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                help="Supported formats: WAV • MP3 • FLAC • OGG • M4A | Maximum length: 60 seconds",
                key=f"uploader_{st.session_state.reset_id}"
            )
            if audio_file_buffer:
                audio_filename = audio_file_buffer.name
        else:
            if hasattr(st, "audio_input"):
                audio_file_buffer = st.audio_input("🎤 Record Audio:")
                if audio_file_buffer:
                    audio_filename = "recorded_voice.wav"
            else:
                st.warning("Live audio recording is not supported in this environment. Please use File Upload.")

    if audio_file_buffer is not None:
        st.markdown("---")
        st.markdown("### Audio Preview")

        col_player, col_meta = st.columns([2, 1])
        audio_bytes_content = audio_file_buffer.getvalue() if hasattr(audio_file_buffer, "getvalue") else audio_file_buffer.read()

        with col_player:
            st.audio(audio_bytes_content, format="audio/wav")
        with col_meta:
            file_size_kb = len(audio_bytes_content) / 1024
            file_fmt = Path(audio_filename).suffix.upper().replace(".", "") or "WAV"
            st.markdown(f"**File Name:** `{audio_filename}`")
            st.markdown(f"**Format:** `{file_fmt}`")
            st.markdown(f"**File Size:** `{file_size_kb:.1f} KB`")

        if st.button("🔍 Analyze Audio", type="primary", key="btn_run"):
            # Polished Animated Progress Loader
            status_box = st.empty()

            status_box.markdown("◉ **Analyzing Audio...**")
            time.sleep(0.3)
            status_box.markdown("✓ **Processing 16 kHz Mono Waveform**")
            time.sleep(0.3)
            status_box.markdown("✓ **Extracting Mel-Frequency Cepstral Coefficients (MFCC)**")
            time.sleep(0.3)
            status_box.markdown("✓ **Running Random Forest Deepfake Classifier**")
            time.sleep(0.2)
            status_box.markdown("✓ **Preparing Verdict & Confidence Metrics**")
            time.sleep(0.2)

            try:
                # Save temporary file for model prediction
                suffix = Path(audio_filename).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes_content)
                    temp_path = tmp.name

                # Run existing trained MFCC model inference
                result = predict_audio_file(temp_path, backend="mfcc")

                # Remove temporary file
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

                status_box.empty()

                # MAIN RESULT DISPLAY CARD
                pred = result["prediction"]
                fake_p = result["fake_probability"]
                real_p = result["real_probability"]
                conf = result["confidence"]
                is_uncertain = result["is_uncertain"]

                st.markdown("<br>", unsafe_allow_html=True)

                if is_uncertain:
                    st.markdown("""
                    <div class="result-uncertain-card">
                        <div class="result-symbol">⚠️</div>
                        <div class="result-main-header" style="color: #F59E0B;">RESULT INCONCLUSIVE</div>
                        <div class="result-sub-header">The acoustic characteristics are close to the decision threshold. This audio should be treated as uncertain.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred == "REAL":
                    st.markdown("""
                    <div class="result-real-card">
                        <div class="result-symbol">✓</div>
                        <div class="result-main-header" style="color: #10B981;">REAL AUDIO</div>
                        <div class="result-sub-header">This audio appears to be genuine human speech.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-fake-card">
                        <div class="result-symbol">⚠</div>
                        <div class="result-main-header" style="color: #EF4444;">AI-GENERATED AUDIO</div>
                        <div class="result-sub-header">This audio shows characteristics associated with synthetic speech.</div>
                    </div>
                    """, unsafe_allow_html=True)

                # DETECTION CONFIDENCE DISPLAY
                st.markdown("### Detection Confidence")
                st.markdown(f"<h1 style='text-align: center; color: #38BDF8; font-size: 3.8rem; font-weight: 900; margin-bottom: 8px;'>{conf:.1f}%</h1>", unsafe_allow_html=True)
                st.progress(conf / 100.0)

                b1, b2 = st.columns(2)
                with b1:
                    st.markdown(f"**REAL — {real_p:.1f}%**")
                with b2:
                    st.markdown(f"<div style='text-align: right;'><strong>FAKE — {fake_p:.1f}%</strong></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # SUMMARY METRIC CARDS (3 CARDS)
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(f"""
                    <div class="metric-summary-card">
                        <div class="metric-summary-val" style="color: {'#10B981' if pred=='REAL' else '#EF4444'};">{'REAL' if pred=='REAL' else 'AI-GENERATED'}</div>
                        <div class="metric-summary-lbl">Classification</div>
                    </div>
                    """, unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""
                    <div class="metric-summary-card">
                        <div class="metric-summary-val">{conf:.1f}%</div>
                        <div class="metric-summary-lbl">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""
                    <div class="metric-summary-card">
                        <div class="metric-summary-val" style="color: #A855F7;">{result['audio_duration_sec']:.2f}s</div>
                        <div class="metric-summary-lbl">Audio Duration</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # AUDIO INSIGHTS (Waveform, MFCC, Mel-Spectrogram)
                with st.expander("🔊 Audio Insights (Waveform, MFCC, Mel-Spectrogram)"):
                    waveform, sr, _ = load_and_preprocess_audio(audio_bytes_content)

                    plt.style.use('dark_background')
                    v1, v2, v3 = st.tabs(["Waveform", "MFCC", "Mel-Spectrogram"])

                    with v1:
                        fig, ax = plt.subplots(figsize=(9, 2.8))
                        fig.patch.set_facecolor('#151D2A')
                        ax.set_facecolor('#0B0F19')
                        time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
                        ax.plot(time_axis, waveform, color="#38BDF8", alpha=0.85, linewidth=0.8)
                        ax.set_xlabel("Time (seconds)", color="#94A3B8")
                        ax.set_ylabel("Amplitude", color="#94A3B8")
                        ax.set_title("Audio Amplitude Waveform", color="#F8FAFC")
                        ax.grid(True, color="#1E293B", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)

                    with v2:
                        mfccs = compute_mfcc_visualization(waveform, sr=sr)
                        fig, ax = plt.subplots(figsize=(9, 3))
                        fig.patch.set_facecolor('#151D2A')
                        ax.set_facecolor('#0B0F19')
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
                        fig.patch.set_facecolor('#151D2A')
                        ax.set_facecolor('#0B0F19')
                        img = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
                        fig.colorbar(img, ax=ax, format="%+2.0f dB")
                        ax.set_title("Log Mel-Spectrogram Frequency Spectrum", color="#F8FAFC")
                        ax.set_xlabel("Time Frames", color="#94A3B8")
                        ax.set_ylabel("Mel Frequency Bands", color="#94A3B8")
                        plt.tight_layout()
                        st.pyplot(fig)

                # SEGMENT ANALYSIS FOR LONG AUDIO
                if len(result["windows"]) > 1:
                    with st.expander("⏱️ Segment Analysis"):
                        win_df = pd.DataFrame(result["windows"])
                        win_df = win_df.rename(columns={
                            "start_sec": "Start (s)",
                            "end_sec": "End (s)",
                            "label": "Prediction",
                            "fake_prob": "Fake Prob (%)",
                            "real_prob": "Real Prob (%)"
                        })
                        st.dataframe(win_df[["Start (s)", "End (s)", "Prediction", "Fake Prob (%)"]], use_container_width=True)

                        fig, ax = plt.subplots(figsize=(8, 3))
                        fig.patch.set_facecolor('#151D2A')
                        ax.set_facecolor('#0B0F19')

                        times = [(w["start_sec"] + w["end_sec"]) / 2 for w in result["windows"]]
                        probs = [w["fake_prob"] for w in result["windows"]]

                        ax.plot(times, probs, marker="o", color="#EF4444" if pred == "FAKE" else "#10B981", linewidth=2.5)
                        ax.axhline(result['decision_threshold'] * 100, color="#64748B", linestyle="--", label="Threshold")
                        ax.set_ylim([0, 100])
                        ax.set_xlabel("Time (seconds)", color="#94A3B8")
                        ax.set_ylabel("Fake Probability (%)", color="#94A3B8")
                        ax.tick_params(colors="#94A3B8")
                        ax.set_title("AI Probability Over Time", color="#F8FAFC", fontsize=11, fontweight="bold")
                        ax.grid(True, color="#1E293B", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)

                # ANALYZE ANOTHER AUDIO BUTTON
                st.markdown("---")
                if st.button("🔄 Analyze Another Audio", key="btn_reset_all"):
                    st.session_state.reset_id += 1
                    st.session_state.preset_sample = None
                    st.rerun()

            except Exception as e:
                st.error("Unable to Analyze Audio. Please upload a valid supported audio file and try again.")


# -------------------------------------------------------------------
# 3. ABOUT PAGE
# -------------------------------------------------------------------
elif st.session_state.page == "About":
    st.markdown("""
    <div style="margin-bottom: 28px;">
        <h2 style="font-size: 2.3rem; font-weight: 800; margin-bottom: 8px;">About AudioGuard</h2>
        <p style="color: #94A3B8; font-size: 1.1rem;">AudioGuard analyzes speech recordings using acoustic characteristics extracted from audio and an existing trained machine-learning model to identify whether the audio appears to be genuine or AI-generated.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Processing Pipeline

    ```text
    Audio Input
        ↓
    Audio Preprocessing (16 kHz Mono)
        ↓
    MFCC Feature Extraction (172-dim Stats)
        ↓
    Random Forest Classifier
        ↓
    REAL / AI-GENERATED PREDICTION
    ```

    ---

    ### Technology Card

    - **Detection Method:** MFCC + Random Forest
    - **Input:** Speech Audio
    - **Supported Formats:** WAV • MP3 • FLAC • OGG • M4A
    - **Model Status:** ✓ Trained Model Active
    """)


# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("""
<div class="footer-text">
    <strong>AudioGuard</strong> — AI-Powered Audio Deepfake Detection<br>
    Supported formats: WAV • MP3 • FLAC • OGG • M4A
</div>
""", unsafe_allow_html=True)
