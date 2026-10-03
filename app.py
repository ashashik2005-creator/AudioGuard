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
# STREAMLIT PAGE CONFIGURATION & FORENSIC DARK THEME STYLING
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AudioGuard | AI Audio Forensic Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark SaaS Forensic Platform Styling
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit Default Elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1080px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
        padding-top: 1.5rem;
    }
    .sidebar-brand {
        padding: 0 12px 20px 12px;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 24px;
    }
    .sidebar-title {
        font-size: 1.6rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .sidebar-status-card {
        background: #151D2A;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 16px;
        margin-top: 40px;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        font-weight: 700;
        color: #10B981;
    }

    /* Top Header Bar */
    .top-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 24px;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid #1E293B;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .top-header-left {
        font-size: 1.2rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .top-header-left span {
        font-size: 0.88rem;
        color: #94A3B8;
        font-weight: 400;
        margin-left: 10px;
    }
    .top-header-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .tech-badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #A5B4FC;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 52px 24px 36px 24px;
        background: radial-gradient(circle at 50% 30%, rgba(99, 102, 241, 0.2) 0%, rgba(11, 15, 25, 0) 70%);
        border-radius: 24px;
        margin-bottom: 32px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hero-h1 {
        font-size: 3.0rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        margin-bottom: 16px;
    }
    .hero-p {
        font-size: 1.2rem;
        color: #94A3B8;
        max-width: 680px;
        margin: 0 auto 32px auto;
        line-height: 1.55;
    }

    /* Feature Info Cards */
    .info-card {
        background: #151D2A;
        border: 1px solid #1E293B;
        border-radius: 18px;
        padding: 24px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        border-color: #6366F1;
    }
    .info-icon {
        font-size: 2.2rem;
        margin-bottom: 12px;
    }
    .info-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    .info-desc {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* Supported Strip */
    .supported-strip {
        background: #151D2A;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 20px 28px;
        margin-top: 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Result Cards */
    .result-card-real {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #10B981;
        border-radius: 24px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 20px 30px -10px rgba(16, 185, 129, 0.25);
    }
    .result-card-fake {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.5) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #EF4444;
        border-radius: 24px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 20px 30px -10px rgba(239, 68, 68, 0.25);
    }
    .result-card-uncertain {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.5) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #F59E0B;
        border-radius: 24px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 20px 30px -10px rgba(245, 158, 11, 0.25);
    }
    .result-symbol {
        font-size: 3.8rem;
        margin-bottom: 8px;
    }
    .result-h1 {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        margin-bottom: 8px;
    }
    .result-p {
        font-size: 1.15rem;
        color: #E2E8F0;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Grid Metric Card */
    .grid-metric-card {
        background: #151D2A;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .grid-metric-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .grid-metric-lbl {
        font-size: 0.78rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 4px;
    }

    /* Custom Primary Button */
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
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# SIDEBAR NAVIGATION & ENGINE STATUS
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-title">🎙️ AUDIOGUARD</div>
        <div class="sidebar-sub">AI Audio Forensic Platform</div>
    </div>
    """, unsafe_allow_html=True)

    nav_page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎙️ Analyze Audio", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="sidebar-status-card">
        <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Detection Engine</div>
        <div class="status-indicator" style="margin-top: 6px;">
            <span>🟢</span> Operational
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px; font-weight: 600;">MFCC + Random Forest</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# TOP HEADER BAR
# -------------------------------------------------------------------
st.markdown("""
<div class="top-header-bar">
    <div class="top-header-left">
        AudioGuard <span>AI Audio Deepfake Detection</span>
    </div>
    <div class="top-header-right">
        <div class="status-badge">🟢 Detection Engine Ready</div>
        <div class="tech-badge">MFCC + Random Forest</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 1: DASHBOARD
# -------------------------------------------------------------------
if nav_page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-h1">Detect AI-Generated Voices</div>
        <div class="hero-p">Analyze speech recordings using acoustic feature analysis and a trained machine-learning model to identify characteristics associated with synthetic or AI-generated speech.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎙️ Analyze Audio", key="dash_an"):
            st.session_state["nav_override"] = "🎙️ Analyze Audio"
            st.rerun()
    with c2:
        if st.button("🎤 Record Audio", key="dash_rec"):
            st.session_state["nav_override"] = "🎙️ Analyze Audio"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 4 Information Cards
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">🎙️</div>
            <div class="info-title">Audio Analysis</div>
            <div class="info-desc">Upload or record speech audio for automated analysis.</div>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">🧠</div>
            <div class="info-title">Acoustic Features</div>
            <div class="info-desc">The system analyzes MFCC-based acoustic characteristics.</div>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">⚡</div>
            <div class="info-title">Fast Detection</div>
            <div class="info-desc">The trained Random Forest model provides CPU-based inference.</div>
        </div>
        """, unsafe_allow_html=True)
    with i4:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">🔒</div>
            <div class="info-title">Privacy Focused</div>
            <div class="info-desc">Audio is processed through the local application.</div>
        </div>
        """, unsafe_allow_html=True)

    # Supported Audio Section
    st.markdown("""
    <div class="supported-strip">
        <div>
            <span style="color: #94A3B8; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; display: block; margin-bottom: 4px;">Supported Audio</span>
            <span style="font-weight: 800; font-size: 1.1rem; color: #F8FAFC;">WAV &nbsp;•&nbsp; MP3 &nbsp;•&nbsp; FLAC &nbsp;•&nbsp; OGG &nbsp;•&nbsp; M4A</span>
        </div>
        <div>
            <span style="color: #94A3B8; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; display: block; margin-bottom: 4px;">Maximum Duration</span>
            <span style="font-weight: 800; font-size: 1.1rem; color: #38BDF8;">60 Seconds</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 2: ANALYZE AUDIO
# -------------------------------------------------------------------
elif nav_page == "🎙️ Analyze Audio":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 28px;">
        <h2 style="font-size: 2.3rem; font-weight: 800; margin-bottom: 8px;">Audio Analysis</h2>
        <p style="color: #94A3B8; font-size: 1.1rem;">Upload or record speech audio for forensic analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    input_segmented = st.radio(
        "Input Method:",
        ["Upload Audio", "Record Audio"],
        horizontal=True,
        label_visibility="collapsed"
    )

    audio_file_buffer = None
    audio_filename = "sample.wav"

    if input_segmented == "Upload Audio":
        audio_file_buffer = st.file_uploader(
            "🎙️ Drop your audio file here or browse files from your computer",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="WAV • MP3 • FLAC • OGG • M4A | Maximum duration: 60 seconds"
        )
        if audio_file_buffer:
            audio_filename = audio_file_buffer.name
    else:
        st.markdown("### Record Your Voice")
        if hasattr(st, "audio_input"):
            audio_file_buffer = st.audio_input("🎙️ Ready to Record:")
            if audio_file_buffer:
                audio_filename = "recorded_voice.wav"
        else:
            st.warning("Live audio recording is not supported in this environment. Please use Upload Audio.")

    if audio_file_buffer is not None:
        st.markdown("---")
        st.markdown("### Selected Audio")

        audio_bytes_data = audio_file_buffer.getvalue() if hasattr(audio_file_buffer, "getvalue") else audio_file_buffer.read()
        file_size_kb = len(audio_bytes_data) / 1024
        file_fmt = Path(audio_filename).suffix.upper().replace(".", "") or "WAV"

        col_p, col_m = st.columns([2, 1])
        with col_p:
            st.audio(audio_bytes_data, format="audio/wav")
        with col_m:
            st.markdown(f"**Filename:** `{audio_filename}`")
            st.markdown(f"**Format:** `{file_fmt}`")
            st.markdown(f"**File Size:** `{file_size_kb:.1f} KB`")

        if st.button("🔍 Analyze Audio", type="primary", key="btn_run_forensic"):
            # Polished Processing Stages Animation
            status_box = st.empty()

            status_box.markdown("⏳ **Analyzing Audio...**\n*Processing your audio through the detection engine...*")
            time.sleep(0.3)
            status_box.markdown("✓ **Audio loaded**")
            time.sleep(0.2)
            status_box.markdown("✓ **Audio preprocessing**")
            time.sleep(0.2)
            status_box.markdown("✓ **MFCC feature extraction**")
            time.sleep(0.3)
            status_box.markdown("✓ **Random Forest classification**")
            time.sleep(0.2)
            status_box.markdown("✓ **Probability calculation**")
            time.sleep(0.2)
            status_box.markdown("✓ **Generating analysis result**")
            time.sleep(0.2)

            try:
                # Save temp file for prediction
                suffix = Path(audio_filename).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes_data)
                    temp_path = tmp.name

                # Execute existing MFCC Random Forest prediction pipeline
                result = predict_audio_file(temp_path, backend="mfcc")

                # Remove temp file
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

                status_box.empty()

                # RESULT SECTION
                st.markdown("### Analysis Complete")

                pred = result["prediction"]
                fake_p = result["fake_probability"]
                real_p = result["real_probability"]
                conf = result["confidence"]
                is_uncertain = result["is_uncertain"]

                if is_uncertain:
                    st.markdown("""
                    <div class="result-card-uncertain">
                        <div class="result-symbol">⚠️</div>
                        <div class="result-h1" style="color: #F59E0B;">RESULT INCONCLUSIVE</div>
                        <div class="result-p">The acoustic characteristics are close to the decision threshold. This audio should be treated as uncertain.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred == "REAL":
                    st.markdown("""
                    <div class="result-card-real">
                        <div class="result-symbol">🟢</div>
                        <div class="result-h1" style="color: #10B981;">REAL AUDIO</div>
                        <div class="result-p">The audio is classified as likely genuine human speech.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card-fake">
                        <div class="result-symbol">🔴</div>
                        <div class="result-h1" style="color: #EF4444;">AI-GENERATED AUDIO</div>
                        <div class="result-p">The audio is classified as likely synthetic or AI-generated speech.</div>
                    </div>
                    """, unsafe_allow_html=True)

                # CONFIDENCE DISPLAY
                st.markdown("### Detection Confidence")
                st.markdown(f"<h1 style='text-align: center; color: #38BDF8; font-size: 3.8rem; font-weight: 900; margin-bottom: 8px;'>{conf:.1f}%</h1>", unsafe_allow_html=True)
                st.progress(conf / 100.0)

                p1, p2 = st.columns(2)
                with p1:
                    st.markdown(f"**REAL:** `{real_p:.1f}%`")
                with p2:
                    st.markdown(f"<div style='text-align: right;'><strong>AI-GENERATED:</strong> <code>{fake_p:.1f}%</code></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ANALYSIS DETAILS GRID (5 CARDS)
                g1, g2, g3, g4, g5 = st.columns(5)
                with g1:
                    st.markdown(f"""
                    <div class="grid-metric-card">
                        <div class="grid-metric-val" style="color: {'#10B981' if pred=='REAL' else '#EF4444'};">{'REAL' if pred=='REAL' else 'AI-GENERATED'}</div>
                        <div class="grid-metric-lbl">Prediction</div>
                    </div>
                    """, unsafe_allow_html=True)
                with g2:
                    st.markdown(f"""
                    <div class="grid-metric-card">
                        <div class="grid-metric-val">{conf:.1f}%</div>
                        <div class="grid-metric-lbl">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with g3:
                    st.markdown(f"""
                    <div class="grid-metric-card">
                        <div class="grid-metric-val" style="color: #A855F7;">{result['audio_duration_sec']:.2f} sec</div>
                        <div class="grid-metric-lbl">Audio Duration</div>
                    </div>
                    """, unsafe_allow_html=True)
                with g4:
                    st.markdown(f"""
                    <div class="grid-metric-card">
                        <div class="grid-metric-val" style="color: #F59E0B;">{result['processing_time_sec']:.2f} sec</div>
                        <div class="grid-metric-lbl">Processing Time</div>
                    </div>
                    """, unsafe_allow_html=True)
                with g5:
                    st.markdown(f"""
                    <div class="grid-metric-card">
                        <div class="grid-metric-val" style="font-size: 0.95rem; line-height: 1.8;">MFCC + Random Forest</div>
                        <div class="grid-metric-lbl">Detection Method</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ACOUSTIC ANALYSIS VISUALIZATIONS
                st.markdown("# Acoustic Analysis")
                waveform, sr, _ = load_and_preprocess_audio(audio_bytes_data)

                plt.style.use('dark_background')
                t1, t2, t3 = st.tabs(["Waveform", "MFCC", "Mel-Spectrogram"])

                with t1:
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

                with t2:
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

                with t3:
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
                    fig.patch.set_facecolor('#151D2A')
                    ax.set_facecolor('#0B0F19')

                    times = [(w["start_sec"] + w["end_sec"]) / 2 for w in result["windows"]]
                    probs = [w["fake_prob"] for w in result["windows"]]

                    ax.plot(times, probs, marker="o", color="#EF4444" if pred == "FAKE" else "#10B981", linewidth=2.5)
                    ax.axhline(result['decision_threshold'] * 100, color="#64748B", linestyle="--", label="Threshold")
                    ax.set_ylim([0, 100])
                    ax.set_xlabel("Time (seconds)", color="#94A3B8")
                    ax.set_ylabel("AI Probability (%)", color="#94A3B8")
                    ax.tick_params(colors="#94A3B8")
                    ax.set_title("AI Probability Over Time", color="#F8FAFC", fontsize=11, fontweight="bold")
                    ax.grid(True, color="#1E293B", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)

                # RESULT ACTIONS
                st.markdown("---")
                act1, act2 = st.columns(2)
                with act1:
                    if st.button("🔄 Analyze Another Audio", key="btn_reset_forensic"):
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

                # DISCLAIMER
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("🔒 **Important:** AudioGuard provides an automated machine-learning prediction based on acoustic characteristics. The result should be treated as an analytical indicator and not as absolute proof of authenticity.")

            except Exception as e:
                st.error("Unable to Analyze Audio. Please upload a valid supported audio file and try again.")


# -------------------------------------------------------------------
# PAGE 3: ABOUT
# -------------------------------------------------------------------
elif nav_page == "ℹ️ About":
    st.markdown("""
    <div style="margin-bottom: 28px;">
        <h2 style="font-size: 2.3rem; font-weight: 800; margin-bottom: 8px;">About AudioGuard</h2>
        <p style="color: #94A3B8; font-size: 1.1rem;">AudioGuard is an AI-powered audio forensic application designed to analyze speech recordings and identify characteristics associated with synthetic or AI-generated speech.</p>
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
