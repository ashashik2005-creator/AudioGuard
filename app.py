"""
AUDIOGUARD - AI Audio Forensic Platform
Top navigation bar, no left sidebar.   Run with:  streamlit run app.py

Only the front end is different in this version. The backend calls are unchanged:
    predict_audio_file(...), load_and_preprocess_audio(...), compute_mel_spectrogram(...),
    compute_mfcc_visualization(...)
Needs Streamlit >= 1.41  (pip install -U streamlit)
"""

import html
import io
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import config
from predict import predict_audio_file
from audio_utils import load_and_preprocess_audio, compute_mel_spectrogram, compute_mfcc_visualization


# =====================================================================================
# BACKEND INFO (read from models/config.json, never hard-coded)
# =====================================================================================
def get_backend_info() -> Tuple[str, int, str]:
    config_file = config.MODELS_DIR / "config.json"
    classifier_name, feature_dim, backend_name = "Random Forest", 172, "MFCC"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                cdata = json.load(f)
            backend_name = cdata.get("backend", "mfcc").upper()
            c_type = cdata.get("classifier_type", "rf").lower()
            feature_dim = cdata.get("feature_dim", 172)
            if c_type in ["rf", "random_forest", "randomforest"]:
                classifier_name = "Random Forest"
            else:
                classifier_name = c_type.upper()
        except Exception:
            pass
    return classifier_name, feature_dim, backend_name


CLASSIFIER_NAME, FEATURE_DIM, BACKEND_NAME = get_backend_info()
DETECTION_METHOD_STR = f"{BACKEND_NAME} + {CLASSIFIER_NAME}"
E = html.escape

st.set_page_config(
    page_title="AudioGuard | AI Audio Forensic Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================================
# STYLE
# Rules used to keep everything aligned:
#   1. Cards that contain only text are ONE html block laid out with CSS grid (equal heights, equal gaps).
#   2. Cards that contain Streamlit widgets are st.container(key=...) so the widgets really sit inside the card.
#   3. Every card, gap and radius comes from the same few variables below.
# =====================================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#080B14; --surface:#101827; --surface2:#0B1220; --line:#1D2940; --line2:#2A3A58;
  --text:#F8FAFC; --muted:#94A3B8; --dim:#64748B;
  --indigo:#6D5EF8; --blue:#4F46E5; --cyan:#38BDF8;
  --green:#10B981; --red:#EF4444; --amber:#F59E0B;
  --radius:16px; --gap:20px;
}
html, body, .stApp, [data-testid="stAppViewContainer"], button, input, textarea{
  font-family:'Manrope','Inter','Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif !important;
}
.stApp{
  background-color:var(--bg);
  background-image:
    radial-gradient(900px 420px at 50% -120px, rgba(109,94,248,.18), transparent 70%),
    radial-gradient(700px 380px at 100% 8%, rgba(56,189,248,.07), transparent 70%);
  background-attachment:fixed;
  color:var(--text);
}
header, [data-testid="stHeader"], footer, #MainMenu{ display:none !important; }
section[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"]{ display:none !important; }
.block-container{ max-width:1240px; padding:1.1rem 1.5rem 3.5rem !important; }
[data-testid="stVerticalBlock"]{ gap:var(--gap); }

/* ---------- cards that hold Streamlit widgets ---------- */
.st-key-navbar, .st-key-hero, .st-key-workspace, .st-key-selected, .st-key-viz_wave, .st-key-viz_mfcc,
.st-key-viz_mel, .st-key-segment, .st-key-report{
  background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  padding:22px 24px !important; box-shadow:0 10px 30px rgba(0,0,0,.28);
}
.st-key-navbar{
  background:rgba(16,24,39,.86); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
  padding:12px 20px !important; margin-bottom:4px;
}
.st-key-hero{
  background:radial-gradient(520px 260px at 85% 50%, rgba(109,94,248,.16), transparent 70%),
             linear-gradient(135deg,#121C32 0%,#0B1220 100%);
  padding:44px 40px !important; overflow:hidden;
}

/* ---------- top navigation ---------- */
.brand{ display:flex; align-items:center; gap:12px; }
.brand svg{ width:38px; height:38px; flex:none; filter:drop-shadow(0 0 10px rgba(109,94,248,.55)); }
.brand-name{ font-size:1.3rem; font-weight:800; letter-spacing:-.01em; line-height:1.1; color:#fff; }
.brand-tag{ font-size:.62rem; font-weight:700; letter-spacing:.09em; color:var(--muted); margin-top:3px; white-space:nowrap; }
.status{ display:flex; flex-direction:column; align-items:flex-end; gap:3px; }
.status-line{ display:flex; align-items:center; gap:8px; font-size:.82rem; font-weight:700; color:#34D399; white-space:nowrap; }
.status-dot{ width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 9px var(--green); }
.status-method{ font-size:.74rem; font-weight:600; color:var(--muted); white-space:nowrap; }

.st-key-navbar div.stButton > button{
  min-height:42px; padding:0 14px; border-radius:10px; box-shadow:none; white-space:nowrap;
}
.st-key-navbar div.stButton > button[kind="secondary"], .st-key-navbar div.stButton > button[data-testid="stBaseButton-secondary"]{
  background:transparent; border:1px solid transparent; color:var(--muted);
}
.st-key-navbar div.stButton > button[kind="secondary"]:hover, .st-key-navbar div.stButton > button[data-testid="stBaseButton-secondary"]:hover{
  background:rgba(255,255,255,.06); color:#fff; border-color:transparent; transform:none;
}
.st-key-navbar div.stButton > button[kind="primary"], .st-key-navbar div.stButton > button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(90deg, rgba(109,94,248,.40), rgba(56,189,248,.14));
  border:1px solid rgba(109,94,248,.65); color:#fff; box-shadow:0 0 16px rgba(109,94,248,.28);
}
.st-key-navbar div.stButton > button[kind="primary"]:hover, .st-key-navbar div.stButton > button[data-testid="stBaseButton-primary"]:hover{
  transform:none; box-shadow:0 0 18px rgba(109,94,248,.4);
}

/* ---------- headings ---------- */
.pagehead{ margin:10px 0 2px; }
.ph1{ font-size:2.1rem; font-weight:800; letter-spacing:-.02em; line-height:1.15; }
.ph2{ color:var(--muted); font-size:1.02rem; margin-top:6px; }
.sec{ margin:14px 0 -4px; }
.sec-t{ font-size:1.25rem; font-weight:800; letter-spacing:-.01em; }
.cardtitle{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin-bottom:2px; }
.cardtitle b{ font-size:1rem; font-weight:800; }
.cardtitle span{ font-size:.82rem; color:var(--muted); }

/* ---------- hero ---------- */
.eyebrow{
  display:inline-block; font-size:.72rem; font-weight:800; letter-spacing:.09em; color:var(--cyan);
  background:rgba(56,189,248,.10); border:1px solid rgba(56,189,248,.32); padding:5px 13px; border-radius:999px;
}
.hero-h1{ font-size:clamp(2rem,3.6vw,3.1rem); font-weight:800; letter-spacing:-.025em; line-height:1.12; margin:18px 0 14px; color:#fff; }
.hero-h1 .grad{ background:linear-gradient(90deg,#8B7CFF,#38BDF8,#8B7CFF); background-size:200% 100%; animation:shimmer 5s linear infinite; -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-p{ color:var(--muted); font-size:1.04rem; line-height:1.65; max-width:560px; margin-bottom:6px; }
.art{
  height:100%; min-height:250px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:22px;
  background:rgba(8,11,20,.55); border:1px solid var(--line); border-radius:14px; padding:26px 18px;
}
.art-top{ font-size:.7rem; font-weight:800; letter-spacing:.1em; color:var(--cyan); }
.bars{ display:flex; align-items:center; justify-content:center; gap:5px; height:112px; width:100%; }
.bars i{
  display:block; width:5px; height:var(--h); border-radius:3px;
  background:linear-gradient(180deg,#8B7CFF,#38BDF8); opacity:.9;
  animation:eq 1.7s ease-in-out infinite alternate; animation-delay:calc(var(--i) * -.17s);
}
@keyframes eq{ from{ transform:scaleY(.45); } to{ transform:scaleY(1); } }
.art-bot{ font-size:.82rem; font-weight:700; color:var(--muted); text-align:center; }

/* ---------- button rows inside cards ---------- */
div.stButton > button, div.stDownloadButton > button{
  width:100%; min-height:48px; border-radius:12px; font-weight:700; font-size:.95rem;
  transition:transform .15s, box-shadow .2s, background .2s, border-color .2s;
}
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#6D5EF8 0%,#4F46E5 55%,#2F6FE0 100%); color:#fff; border:none;
  box-shadow:0 6px 18px rgba(109,94,248,.35);
}
div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="stBaseButton-primary"]:hover{
  transform:translateY(-1px); box-shadow:0 10px 24px rgba(109,94,248,.5); color:#fff;
}
div.stButton > button[kind="secondary"], div.stButton > button[data-testid="stBaseButton-secondary"],
div.stDownloadButton > button{
  background:rgba(255,255,255,.03); color:var(--text); border:1px solid var(--line2);
}
div.stButton > button[kind="secondary"]:hover, div.stButton > button[data-testid="stBaseButton-secondary"]:hover,
div.stDownloadButton > button:hover{
  background:rgba(255,255,255,.07); border-color:var(--indigo); color:#fff; transform:translateY(-1px);
}
div.stButton > button p, div.stDownloadButton > button p{ font-weight:700; }

/* ---------- capability / pipeline / architecture / about / metrics (pure html grids) ---------- */
.grid{ display:grid; gap:var(--gap); }
.g4{ grid-template-columns:repeat(4,minmax(0,1fr)); }
.g3{ grid-template-columns:repeat(3,minmax(0,1fr)); }
.g5{ grid-template-columns:repeat(5,minmax(0,1fr)); }
.tile{
  background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:22px;
  display:flex; flex-direction:column; gap:10px; transition:transform .2s, border-color .2s;
}
.tile:hover{ transform:translateY(-2px); border-color:var(--line2); }
.tile-top{ display:flex; align-items:center; justify-content:space-between; }
.ico{
  width:38px; height:38px; border-radius:11px; display:grid; place-items:center; color:var(--cyan);
  background:linear-gradient(135deg,rgba(109,94,248,.25),rgba(56,189,248,.10)); border:1px solid rgba(109,94,248,.35);
}
.ico svg{ width:20px; height:20px; }
.num{ font-size:.78rem; font-weight:800; color:var(--indigo); letter-spacing:.06em; }
.tile-t{ font-size:.92rem; font-weight:800; letter-spacing:.02em; text-transform:uppercase; line-height:1.3; }
.g4 .tile-t{ min-height:2.6em; }
.tile-p{ font-size:.9rem; color:var(--muted); line-height:1.55; }
.lst{ margin-top:2px; color:#CBD5E1; font-size:.9rem; line-height:1.75; }
.lst div{ position:relative; padding-left:16px; }
.lst div::before{ content:""; position:absolute; left:0; top:.72em; width:5px; height:5px; border-radius:50%; background:var(--indigo); }

.panel{ position:relative; overflow:hidden; background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:26px; }
.panel-title{ font-size:1.1rem; font-weight:800; }
.panel-sub{ font-size:.88rem; color:var(--muted); margin-top:4px; }
.steps{ display:grid; gap:18px; margin-top:22px; }
.steps.s5{ grid-template-columns:repeat(5,minmax(0,1fr)); }
.steps.s4{ grid-template-columns:repeat(4,minmax(0,1fr)); }
.step{
  position:relative; background:var(--surface2); border:1px solid var(--line); border-radius:14px; padding:18px 16px;
  display:flex; flex-direction:column; gap:6px; text-align:left;
}
.step .n{ font-size:.72rem; font-weight:800; color:var(--indigo); letter-spacing:.07em; }
.step b{ font-size:.92rem; font-weight:800; line-height:1.3; }
.step span{ font-size:.8rem; color:var(--muted); line-height:1.45; }
.steps .step:not(:last-child)::after{                      /* thin connector line in the gap */
  content:""; position:absolute; top:50%; right:-18px; width:18px; height:2px; transform:translateY(-50%);
  background:linear-gradient(90deg,var(--indigo),var(--cyan)); background-size:200% 100%; animation:flow 2.4s linear infinite;
}
.steps.s4 .step:nth-child(4n)::after{ display:none; }
@keyframes flow{ to{ background-position:-200% 0; } }
.fact-label{ font-size:.78rem; font-weight:800; letter-spacing:.07em; color:var(--muted); text-transform:uppercase; }
.fact-val{ font-size:1.05rem; font-weight:800; }
.badge{ font-size:.7rem; font-weight:800; letter-spacing:.07em; color:var(--indigo); background:#1E293B; border:1px solid #334155; padding:3px 9px; border-radius:7px; }
.note{ color:var(--dim); font-size:.85rem; line-height:1.6; }

/* ---------- selected audio ---------- */
.fname{ font-size:1.12rem; font-weight:800; color:var(--cyan); word-break:break-all; margin:6px 0 16px; }
.meta{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-bottom:6px; }
.meta div{ background:var(--surface2); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }
.meta small{ display:block; font-size:.68rem; font-weight:800; letter-spacing:.08em; color:var(--dim); margin-bottom:4px; }
.meta b{ font-size:.98rem; font-weight:800; }
.empty{ text-align:center; padding:26px 10px 10px; }
.empty .wave{ display:flex; justify-content:center; align-items:center; gap:4px; height:34px; margin-bottom:14px; opacity:.8; }
.empty .wave i{ display:block; width:4px; border-radius:2px; background:linear-gradient(180deg,#8B7CFF,#38BDF8); height:var(--h); animation:eq 1.4s ease-in-out infinite alternate; animation-delay:calc(var(--i) * -.13s); }
.empty b{ font-size:.95rem; font-weight:800; letter-spacing:.06em; }
.empty .sub{ font-size:.9rem; color:var(--muted); margin-top:6px; }
.dz-note{ font-size:.82rem; color:var(--dim); margin:2px 0 0; }

/* ---------- uploader ---------- */
[data-testid="stFileUploaderDropzone"]{
  background:rgba(109,94,248,.05); border:1.5px dashed rgba(109,94,248,.5); border-radius:14px; padding:30px 20px;
  transition:border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--cyan); background:rgba(56,189,248,.06); }
[data-testid="stAudioInput"]{
  background:rgba(109,94,248,.06); border:1.5px dashed rgba(109,94,248,.5); border-radius:16px; padding:8px 10px;
  transition:border-color .2s, background .2s, box-shadow .2s;
}
[data-testid="stAudioInput"]:hover{ border-color:var(--cyan); background:rgba(56,189,248,.06); box-shadow:0 0 22px rgba(56,189,248,.12); }
[data-testid="stAudioInputActionButton"]{
  width:46px !important; height:46px !important; min-height:0 !important; border-radius:50% !important;
  background:linear-gradient(135deg,#F43F5E,#EF4444) !important; color:#fff !important; border:none !important;
  animation:recpulse 2s ease-out infinite;
}
@keyframes recpulse{ 0%{ box-shadow:0 0 0 0 rgba(239,68,68,.55); } 70%{ box-shadow:0 0 0 16px rgba(239,68,68,0); } 100%{ box-shadow:0 0 0 0 rgba(239,68,68,0); } }

/* ---------- source switch (Upload / Record) ---------- */
.st-key-src_upload button, .st-key-src_record button{
  min-height:62px !important; border-radius:14px !important; font-size:1rem !important; gap:8px;
}
.st-key-src_upload button p, .st-key-src_record button p{ font-size:1rem; font-weight:800; }
.st-key-src_upload button[kind="secondary"], .st-key-src_record button[kind="secondary"],
.st-key-src_upload button[data-testid="stBaseButton-secondary"], .st-key-src_record button[data-testid="stBaseButton-secondary"]{
  background:rgba(255,255,255,.03); border:1.5px solid var(--line2); color:var(--muted);
}
.st-key-src_upload button[kind="primary"], .st-key-src_record button[kind="primary"],
.st-key-src_upload button[data-testid="stBaseButton-primary"], .st-key-src_record button[data-testid="stBaseButton-primary"]{
  box-shadow:0 0 0 1px rgba(109,94,248,.6), 0 8px 26px rgba(109,94,248,.4);
}
.st-key-src_record button[kind="primary"], .st-key-src_record button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#F43F5E 0%,#EF4444 55%,#DC2626 100%);
  box-shadow:0 0 0 1px rgba(239,68,68,.6), 0 8px 26px rgba(239,68,68,.4);
}

/* ---------- animated orb ---------- */
.orb{ position:relative; width:104px; height:104px; margin:0 auto 18px; display:grid; place-items:center; }
.orb .core{
  position:relative; z-index:2; width:68px; height:68px; border-radius:50%; display:grid; place-items:center; color:#fff;
  background:linear-gradient(135deg,#6D5EF8,#38BDF8); box-shadow:0 0 30px rgba(109,94,248,.6);
}
.orb .core svg{ width:30px; height:30px; }
.orb .rg{ position:absolute; inset:0; border-radius:50%; border:2px solid rgba(109,94,248,.6); animation:ripple 2.7s ease-out infinite; }
.orb .rg:nth-child(2){ animation-delay:.9s; } .orb .rg:nth-child(3){ animation-delay:1.8s; }
@keyframes ripple{ from{ transform:scale(.6); opacity:.9; } to{ transform:scale(1.4); opacity:0; } }
.orb{ cursor:pointer; transition:transform .2s; }
.orb:hover{ transform:scale(1.07); }
.orb:hover .core{ filter:brightness(1.18); }
.orb:active{ transform:scale(.97); }
.orb.up .core{ animation:floaty 3s ease-in-out infinite; }
@keyframes floaty{ 50%{ transform:translateY(-6px); } }
.orb.rec .core{ background:linear-gradient(135deg,#F43F5E,#EF4444); box-shadow:0 0 30px rgba(239,68,68,.6); animation:beat 1.6s ease-in-out infinite; }
.orb.rec .rg{ border-color:rgba(239,68,68,.65); }
@keyframes beat{ 50%{ transform:scale(1.08); } }

/* ---------- processing ---------- */
.proc{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:26px 28px; }
.proc-t{ font-size:1.3rem; font-weight:800; }
.proc-p{ color:var(--muted); font-size:.92rem; margin:4px 0 16px; }
.proc-li{ padding:6px 0; font-size:.92rem; color:#CBD5E1; display:flex; align-items:center; gap:10px; }
.proc-li::before{ content:""; width:9px; height:9px; border-radius:50%; background:var(--indigo); animation:pulse 1.2s ease-in-out infinite; }
.indet{ height:4px; border-radius:4px; background:var(--line); overflow:hidden; margin-top:18px; }
.indet i{ display:block; height:100%; width:35%; border-radius:4px; background:linear-gradient(90deg,var(--indigo),var(--cyan)); animation:slide 1.2s ease-in-out infinite; }
@keyframes slide{ from{ transform:translateX(-100%); } to{ transform:translateX(300%); } }
@keyframes pulse{ 50%{ opacity:.35; } }

/* ---------- result ---------- */
.result{
  --c:var(--green);
  display:grid; grid-template-columns:230px minmax(0,1fr); gap:34px; align-items:center;
  background:linear-gradient(135deg,color-mix(in srgb,var(--c) 14%, #101827) 0%, #0B1220 100%);
  border:1px solid color-mix(in srgb,var(--c) 55%, transparent); border-radius:20px; padding:32px 36px;
  box-shadow:0 0 34px color-mix(in srgb,var(--c) 14%, transparent); animation:rise .5s ease-out;
}
.result.fake{ --c:var(--red); } .result.unsure{ --c:var(--amber); }
@keyframes rise{ from{ opacity:0; transform:translateY(8px); } }
.ring{ position:relative; width:210px; height:210px; justify-self:center; }
.ring svg{ width:100%; height:100%; transform:rotate(-90deg); }
.ring circle{ fill:none; stroke-width:9; }
.ring .trk{ stroke:rgba(255,255,255,.08); }
.ring .val{ stroke:var(--c); stroke-linecap:round; stroke-dasharray:var(--circ); stroke-dashoffset:var(--off); animation:ringin 1.1s ease-out; }
@keyframes ringin{ from{ stroke-dashoffset:var(--circ); } }
.ring-txt{ position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.ring-txt b{ font-size:2.7rem; font-weight:800; letter-spacing:-.02em; line-height:1; }
.ring-txt small{ font-size:.72rem; font-weight:800; letter-spacing:.12em; color:var(--muted); margin-top:8px; }
.kicker{ font-size:.74rem; font-weight:800; letter-spacing:.1em; color:var(--c); }
.verdict{ font-size:clamp(1.8rem,3.2vw,2.5rem); font-weight:800; letter-spacing:-.02em; line-height:1.1; margin:8px 0 10px; color:var(--c); }
.res-desc{ color:#CBD5E1; font-size:1rem; line-height:1.55; max-width:560px; margin-bottom:22px; }
.probs{ display:grid; gap:14px; max-width:560px; }
.ph{ display:flex; justify-content:space-between; font-size:.8rem; font-weight:800; letter-spacing:.07em; margin-bottom:6px; }
.bar{ height:8px; border-radius:6px; background:rgba(255,255,255,.08); overflow:hidden; }
.bar i{ display:block; height:100%; border-radius:6px; animation:grow .9s ease-out; }
@keyframes grow{ from{ width:0 !important; } }
.bar.r i{ background:var(--green); } .bar.f i{ background:var(--red); }

.metrics{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:var(--gap); }
.metric{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:18px 18px 16px; min-width:0; }
.metric small{ display:block; font-size:.68rem; font-weight:800; letter-spacing:.09em; color:var(--dim); margin-bottom:8px; text-transform:uppercase; }
.metric b{ display:block; font-size:1.45rem; font-weight:800; letter-spacing:-.01em; line-height:1.15; overflow-wrap:anywhere; }
.metric b.txt{ font-size:1.05rem; }
.segs{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:14px; margin-top:6px; }
.seg{ background:var(--surface2); border:1px solid var(--line); border-radius:12px; padding:14px; }
.seg small{ display:block; font-size:.74rem; font-weight:800; color:var(--indigo); }
.seg b{ display:block; font-size:1.2rem; font-weight:800; margin:4px 0 2px; }
.seg span{ font-size:.7rem; font-weight:700; letter-spacing:.07em; color:var(--dim); text-transform:uppercase; }
.err{ background:linear-gradient(135deg,rgba(127,29,29,.35),#101827); border:1px solid rgba(239,68,68,.6); border-radius:var(--radius); padding:24px 28px; }
.err b{ display:block; font-size:1.05rem; font-weight:800; letter-spacing:.06em; color:#F87171; margin-bottom:6px; }
.err .m{ color:#E2E8F0; font-size:.95rem; }


/* ---------- motion: home + about ---------- */
@keyframes riseup{ from{ opacity:0; transform:translateY(18px); } to{ opacity:1; transform:none; } }
@keyframes shimmer{ to{ background-position:-200% 0; } }
@keyframes drift{ from{ transform:translate(0,0) scale(1); } to{ transform:translate(50px,36px) scale(1.18); } }
@keyframes scan{ from{ left:6%; } to{ left:94%; } }
@keyframes sweep{ from{ left:-45%; } to{ left:105%; } }
@keyframes stepglow{
  0%,24%,100%{ border-color:var(--line); box-shadow:none; }
  8%,16%{ border-color:rgba(109,94,248,.85); box-shadow:0 0 24px rgba(109,94,248,.32); }
}
@keyframes bob{ 50%{ transform:translateY(-5px); } }

.pagehead{ animation:riseup .6s ease-out backwards; }
.eyebrow{ animation:riseup .7s .05s ease-out backwards; }
.hero-h1{ animation:riseup .7s .15s ease-out backwards; }
.hero-p{ animation:riseup .7s .25s ease-out backwards; }
.st-key-hero div.stButton{ animation:riseup .7s .38s ease-out backwards; }
.st-key-hero{ position:relative; }
.st-key-hero::before, .st-key-hero::after{
  content:""; position:absolute; border-radius:50%; pointer-events:none; z-index:0;
}
.st-key-hero::before{ width:340px; height:340px; left:-90px; top:-130px;
  background:radial-gradient(circle, rgba(109,94,248,.30), transparent 70%); animation:drift 9s ease-in-out infinite alternate; }
.st-key-hero::after{ width:300px; height:300px; right:-60px; bottom:-120px;
  background:radial-gradient(circle, rgba(56,189,248,.22), transparent 70%); animation:drift 11s ease-in-out infinite alternate-reverse; }
.st-key-hero > *{ position:relative; z-index:1; }

.art{ position:relative; overflow:hidden; animation:riseup .8s .2s ease-out backwards; }
.art::after{
  content:""; position:absolute; top:0; bottom:0; left:6%; width:2px; pointer-events:none;
  background:linear-gradient(180deg, transparent, var(--cyan), transparent); box-shadow:0 0 16px var(--cyan);
  animation:scan 3.6s ease-in-out infinite alternate;
}
.live{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:8px; vertical-align:middle;
  background:var(--cyan); box-shadow:0 0 9px var(--cyan); animation:pulse 1.2s ease-in-out infinite; }

.panel::before{
  content:""; position:absolute; top:0; left:-45%; width:45%; height:2px; pointer-events:none;
  background:linear-gradient(90deg, transparent, var(--cyan), var(--indigo), transparent); animation:sweep 4.5s linear infinite;
}
.panel{ animation:riseup .6s .1s ease-out backwards; }
.tile, .metric{ animation:riseup .6s ease-out backwards; }
.grid > .tile:nth-child(1){ animation-delay:.10s; } .grid > .tile:nth-child(2){ animation-delay:.20s; }
.grid > .tile:nth-child(3){ animation-delay:.30s; } .grid > .tile:nth-child(4){ animation-delay:.40s; }
.tile:hover{ box-shadow:0 14px 34px rgba(109,94,248,.20); }
.tile .ico{ animation:bob 3.4s ease-in-out infinite; }
.steps.s5 .step{
  animation:riseup .6s ease-out backwards, stepglow 9s ease-in-out infinite;
  animation-delay:calc(var(--k) * .12s + .2s), calc(var(--k) * 1.5s);
}
.step{ transition:transform .2s; } .step:hover{ transform:translateY(-3px); }

/* ---------- responsive ---------- */
@media (max-width:1100px){
  .g4{ grid-template-columns:repeat(2,minmax(0,1fr)); }
  .metrics{ grid-template-columns:repeat(2,minmax(0,1fr)); }
  .meta{ grid-template-columns:repeat(3,minmax(0,1fr)); }
  .steps.s5{ grid-template-columns:repeat(2,minmax(0,1fr)); }
  .steps.s4{ grid-template-columns:repeat(2,minmax(0,1fr)); }
  .steps .step::after{ display:none !important; }
}
@media (max-width:760px){
  .block-container{ padding:.8rem .9rem 3rem !important; }
  .st-key-hero{ padding:26px 20px !important; }
  .g4,.g3,.metrics,.steps.s5,.steps.s4{ grid-template-columns:1fr; }
  .g4 .tile-t{ min-height:0; }
  .meta{ grid-template-columns:repeat(2,minmax(0,1fr)); }
  .result{ grid-template-columns:1fr; padding:26px 20px; text-align:center; }
  .probs,.res-desc{ margin-left:auto; margin-right:auto; text-align:left; }
  .status{ align-items:flex-start; }
}
@media (prefers-reduced-motion:reduce){ *{ animation:none !important; transition:none !important; } }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# =====================================================================================
# SMALL HELPERS
# =====================================================================================
def compact(s: str) -> str:
    """Remove newlines + indentation so Markdown never turns indented HTML into a code block."""
    return "".join(line.strip() for line in s.splitlines())


def md(s: str):
    st.markdown(compact(s), unsafe_allow_html=True)


def card(key: str):
    """A real container the widgets sit inside (styled through the .st-key-<key> class)."""
    try:
        return st.container(key=key)
    except TypeError:                    # very old Streamlit: still works, just without the card styling
        return st.container()


def cols(spec, **kw):
    try:
        return st.columns(spec, **kw)
    except TypeError:
        kw.pop("vertical_alignment", None)
        return st.columns(spec, **kw)


def mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def inspect_audio(data: bytes, filename: str) -> dict:
    """Real metadata from the uploaded file. Anything that cannot be read is shown as an em dash."""
    info = {"format": (Path(filename).suffix.lstrip(".") or "audio").upper(), "size": len(data),
            "duration": None, "sr": None, "ch": None}
    try:
        import soundfile as sf
        i = sf.info(io.BytesIO(data))
        info.update(duration=float(i.duration), sr=int(i.samplerate), ch=int(i.channels))
        return info
    except Exception:
        pass
    try:
        with wave.open(io.BytesIO(data)) as w:
            info.update(duration=w.getnframes() / w.getframerate(), sr=w.getframerate(), ch=w.getnchannels())
    except Exception:
        pass
    return info


def fmt_size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / (1024 * 1024):.2f} MB"


AUDIO_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}

ICONS = {
    "mfcc": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h2M7 7v10M11 4v16M15 8v8M19 10v4"/></svg>',
    "tree": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="19" r="2"/><path d="M12 7v4M12 11l-6 6M12 11l6 6"/></svg>',
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 15l3-4 3 2 4-5"/></svg>',
    "lock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>',
}
LOGO = ('<svg viewBox="0 0 40 40"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6D5EF8"/>'
        '<stop offset="1" stop-color="#38BDF8"/></linearGradient></defs><rect width="40" height="40" rx="11" fill="url(#lg)"/>'
        '<path d="M10 20h2M15 14v12M20 9v22M25 15v10M30 18v4" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/></svg>')


def set_source(mode: str):
    st.session_state["input_source"] = mode


def goto(page: str, mode: str = None):
    """Button callback: runs before the next rerun, so it may change the nav widget state."""
    st.session_state["nav_radio"] = page
    if mode:
        st.session_state["input_source"] = mode


# =====================================================================================
# SESSION STATE
# =====================================================================================
NAV = ["Home", "Audio Analysis", "About"]
if st.session_state.get("nav_radio") not in NAV:
    st.session_state["nav_radio"] = "Home"
st.session_state.setdefault("reset_count", 0)
st.session_state.setdefault("analysis", None)


# =====================================================================================
# TOP NAVIGATION  (one row: brand | page buttons | engine status)
# =====================================================================================
with card("navbar"):
    c_brand, c_nav, c_status = cols([3, 5, 3], vertical_alignment="center")
    with c_brand:
        md(f'<div class="brand">{LOGO}<div><div class="brand-name">AUDIOGUARD</div>'
           f'<div class="brand-tag">AI AUDIO FORENSIC PLATFORM</div></div></div>')
    with c_nav:
        nav_cols = cols(len(NAV), gap="small")
        for col, name in zip(nav_cols, NAV):
            with col:
                st.button(name, key=f"nav_{name}", use_container_width=True,
                          type="primary" if st.session_state["nav_radio"] == name else "secondary",
                          on_click=goto, args=(name,))
    with c_status:
        md(f'<div class="status"><div class="status-line"><span class="status-dot"></span>Engine Operational</div>'
           f'<div class="status-method">{E(DETECTION_METHOD_STR)}</div></div>')

page = st.session_state["nav_radio"]


# =====================================================================================
# PAGE 1 - DASHBOARD
# =====================================================================================
def page_dashboard():
    with card("hero"):
        left, right = cols([1.75, 1], vertical_alignment="center", gap="large")
        with left:
            md('<span class="eyebrow">AI AUDIO FORENSICS</span>'
               '<div class="hero-h1">Detect AI-Generated Voices <span class="grad">with confidence.</span></div>'
               '<div class="hero-p">Analyze speech recordings using acoustic feature analysis and a trained machine-learning '
               'model to identify characteristics associated with synthetic or AI-generated speech.</div>')
            b1, b2, _ = cols([1.15, 1, 1.4])
            with b1:
                st.button("Analyze Audio →", key="cta_analyze", type="primary", on_click=goto, args=("Audio Analysis",))
            with b2:
                st.button("Record Audio", key="cta_record", type="secondary", on_click=goto,
                          args=("Audio Analysis", "Record Live Voice"))
        with right:
            heights = [34, 58, 44, 82, 62, 100, 74, 96, 56, 88, 66, 104, 70, 90, 50, 78, 60, 40, 68, 46, 30]
            bars = "".join(f'<i style="--h:{h}px;--i:{i}"></i>' for i, h in enumerate(heights))
            md(f'<div class="art"><div class="art-top"><span class="live"></span>ACOUSTIC SIGNATURE</div><div class="bars">{bars}</div>'
               f'<div class="art-bot">{E(BACKEND_NAME)} Feature Engine</div></div>')

    stages = [
        ("Audio Input", "Speech file or microphone recording"),
        ("Preprocessing", "Mono, 16 kHz, normalized"),
        (f"{BACKEND_NAME} Features", f"{FEATURE_DIM}-dimensional feature vector"),
        (CLASSIFIER_NAME, "Trained classifier inference"),
        ("Detection Result", "REAL or AI-GENERATED with probabilities"),
    ]
    md('<div class="panel"><div class="panel-title">How AudioGuard Analyzes Audio</div>'
       '<div class="steps s5">' + "".join(
           f'<div class="step" style="--k:{i}"><span class="n">STAGE {i + 1}</span><b>{E(t)}</b><span>{E(d)}</span></div>'
           for i, (t, d) in enumerate(stages)) + '</div></div>')


# =====================================================================================
# PAGE 2 - AUDIO ANALYSIS
# =====================================================================================
def build_report_text(a: dict) -> str:
    r = a["result"]
    thr = r.get("decision_threshold", 0.50)
    margin = r.get("uncertainty_margin", 0.05)
    fake_p = r["fake_probability"]
    lower_b = (thr - margin) * 100
    upper_b = (thr + margin) * 100
    uncertain = r.get("is_uncertain", False) or (lower_b <= fake_p <= upper_b)

    if uncertain:
        classification_str = "INCONCLUSIVE"
        conf_str = "Low / Borderline"
    elif fake_p < lower_b:
        classification_str = "AUTHENTIC AUDIO"
        conf_str = f"{r['real_probability']:.2f}%"
    else:
        classification_str = "AI-GENERATED AUDIO"
        conf_str = f"{r['fake_probability']:.2f}%"

    return f"""==================================================
AUDIOGUARD FORENSIC ANALYSIS REPORT
==================================================
Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Audio Filename:     {a['filename']}
Audio Duration:     {r['audio_duration_sec']:.2f} seconds

CLASSIFICATION:     {classification_str}
Real Probability:   {r['real_probability']:.2f}%
AI Probability:     {r['fake_probability']:.2f}%
Confidence Level:   {conf_str}

Detection Method:   {DETECTION_METHOD_STR}
Decision Threshold: {thr * 100:.0f}%
Uncertainty Range:  {lower_b:.0f}%–{upper_b:.0f}%
Processing Time:    {r['processing_time_sec']:.2f} seconds
==================================================
AudioGuard provides probabilistic machine-learning predictions.
Results should be treated as automated indicators rather than absolute proof of authenticity.
"""


def style_axes(fig, ax):
    fig.patch.set_facecolor("#101827")
    ax.set_facecolor("#0B1220")
    ax.tick_params(colors="#94A3B8", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#1D2940")


def render_results(a: dict):
    r, vis = a["result"], a["vis"]
    fake_p, real_p = r["fake_probability"], r["real_probability"]
    thr = r.get("decision_threshold", 0.50)
    margin = r.get("uncertainty_margin", 0.05)
    lower_b = (thr - margin) * 100
    upper_b = (thr + margin) * 100

    uncertain = r.get("is_uncertain", False) or (lower_b <= fake_p <= upper_b)

    if uncertain:
        cls = "unsure"
        title = "INCONCLUSIVE"
        subtitle = "The available acoustic evidence is not strong enough to reliably distinguish authentic from AI-generated speech."
        warning_box = ('<div style="background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.4);border-radius:12px;'
                       'padding:14px 18px;margin-top:16px;color:#FBBF24;font-size:0.9rem;line-height:1.5;font-weight:600">'
                       '⚠️ Model probabilities are too close to the decision boundary for a confident classification.'
                       '<br><span style="color:#CBD5E1;font-weight:400;font-size:0.85rem">Borderline result — the model probabilities fall inside the uncertainty range.'
                       '<br>For best results, use a clear speech recording with minimal background noise and sufficient speech duration.</span></div>')
        conf_display = "Low / Borderline"
        verdict_color = "var(--amber)"
    elif fake_p < lower_b:
        cls = "real"
        title = "AUTHENTIC AUDIO"
        subtitle = "The model found stronger evidence consistent with authentic speech."
        warning_box = ""
        conf_display = f"{real_p:.1f}%"
        verdict_color = "var(--green)"
    else:
        cls = "fake"
        title = "AI-GENERATED AUDIO"
        subtitle = "The model found stronger evidence consistent with synthetic speech."
        warning_box = ""
        conf_display = f"{fake_p:.1f}%"
        verdict_color = "var(--red)"

    circ = 2 * np.pi * 52
    off = circ * (1 - min(max(real_p if title == 'AUTHENTIC AUDIO' else (fake_p if title == 'AI-GENERATED AUDIO' else 50), 0), 100) / 100)
    md(f'''<div class="result {cls}">
        <div class="ring" style="--circ:{circ:.1f};--off:{off:.1f}">
          <svg viewBox="0 0 120 120"><circle class="trk" cx="60" cy="60" r="52"/><circle class="val" cx="60" cy="60" r="52"/></svg>
          <div class="ring-txt"><b>{"—" if uncertain else f"{real_p if title == 'AUTHENTIC AUDIO' else fake_p:.0f}%"}</b><small>CONFIDENCE</small></div>
        </div>
        <div>
          <div class="kicker">FORENSIC ASSESSMENT</div>
          <div class="verdict">{title}</div>
          <div class="res-desc">{subtitle}</div>
          <div class="probs">
            <div><div class="ph"><span>REAL PROBABILITY</span><span>{real_p:.1f}%</span></div><div class="bar r"><i style="width:{real_p:.1f}%"></i></div></div>
            <div><div class="ph"><span>AI PROBABILITY</span><span>{fake_p:.1f}%</span></div><div class="bar f"><i style="width:{fake_p:.1f}%"></i></div></div>
          </div>
          {warning_box}
        </div>
      </div>''')

    tiles = [
        ("Model Assessment", f'<b class="txt" style="color:{verdict_color}">{"INCONCLUSIVE" if uncertain else ("AUTHENTIC AUDIO" if title == "AUTHENTIC AUDIO" else "AI-GENERATED AUDIO")}</b>'),
        ("Confidence", f"<b>{conf_display}</b>"),
        ("Real probability", f'<b style="color:var(--green)">{real_p:.1f}%</b>'),
        ("AI probability", f'<b style="color:var(--red)">{fake_p:.1f}%</b>'),
        ("Processing time", f"<b>{r['processing_time_sec']:.2f} s</b>"),
        ("Audio duration", f"<b>{r['audio_duration_sec']:.1f} s</b>"),
        ("Detection method", f'<b class="txt">{E(DETECTION_METHOD_STR)}</b>'),
        ("Decision threshold", f"<b>{thr * 100:.0f}%</b>"),
        ("Uncertainty range", f"<b>{lower_b:.0f}%–{upper_b:.0f}%</b>"),
    ]

    md('<div class="sec"><div class="sec-t">Result summary</div></div>')
    md('<div class="metrics">' + "".join(f'<div class="metric"><small>{k}</small>{v}</div>' for k, v in tiles) + '</div>')

    # ---- acoustic forensics: visualizations -----------------------------------
    md('<div class="sec"><div class="sec-t">Acoustic forensics</div></div>')
    waveform, sr = vis["waveform"], vis["sr"]
    plt.style.use("dark_background")

    with card("viz_wave"):
        md('<div class="cardtitle"><b>Waveform</b><span>Time-domain representation of the analyzed audio.</span></div>')
        fig, ax = plt.subplots(figsize=(10, 2.6))
        style_axes(fig, ax)
        t = np.linspace(0, len(waveform) / sr, num=len(waveform))
        ax.plot(t, waveform, color="#38BDF8", alpha=.9, linewidth=.7)
        ax.set_xlabel("Time (s)", color="#94A3B8", fontsize=9); ax.set_ylabel("Amplitude", color="#94A3B8", fontsize=9)
        ax.grid(True, color="#1D2940", alpha=.6); ax.set_xlim(0, t[-1] if len(t) else 1)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    cm, cs = cols(2, gap="medium")
    with cm:
        with card("viz_mfcc"):
            md('<div class="cardtitle"><b>MFCC</b><span>Cepstral features used for model inference.</span></div>')
            fig, ax = plt.subplots(figsize=(6, 3.2))
            style_axes(fig, ax)
            img = ax.imshow(vis["mfccs"], aspect="auto", origin="lower", cmap="viridis")
            fig.colorbar(img, ax=ax).ax.tick_params(colors="#94A3B8", labelsize=8)
            ax.set_xlabel("Time frames", color="#94A3B8", fontsize=9); ax.set_ylabel("MFCC coefficient", color="#94A3B8", fontsize=9)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with cs:
        with card("viz_mel"):
            md('<div class="cardtitle"><b>Mel-spectrogram</b><span>Time-frequency view of the speech signal.</span></div>')
            fig, ax = plt.subplots(figsize=(6, 3.2))
            style_axes(fig, ax)
            img = ax.imshow(vis["mel_db"], aspect="auto", origin="lower", cmap="magma")
            fig.colorbar(img, ax=ax, format="%+2.0f dB").ax.tick_params(colors="#94A3B8", labelsize=8)
            ax.set_xlabel("Time frames", color="#94A3B8", fontsize=9); ax.set_ylabel("Mel band", color="#94A3B8", fontsize=9)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ---- segment analysis (only if backend provided multiple windows) -----------
    wins = r.get("windows") or []
    if len(wins) > 1:
        md('<div class="sec"><div class="sec-t">Segment analysis</div></div>')
        with card("segment"):
            md('<div class="cardtitle"><b>AI probability over time</b><span>Window-level forensic assessment.</span></div>')
            
            # Check for mixed segment evidence
            has_fake_win = any(w["fake_prob"] >= thr * 100 for w in wins)
            has_real_win = any(w["fake_prob"] < thr * 100 for w in wins)
            if has_fake_win and has_real_win:
                md('<div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:10px;'
                   'padding:10px 14px;margin-bottom:14px;color:#FBBF24;font-size:0.86rem;font-weight:700">'
                   '⚠️ Mixed evidence across audio segments — speech characteristics vary across different time windows.</div>')

            fig, ax = plt.subplots(figsize=(10, 2.8))
            style_axes(fig, ax)
            xs = [(w["start_sec"] + w["end_sec"]) / 2 for w in wins]
            ys = [w["fake_prob"] for w in wins]
            line_c = "#F59E0B" if uncertain else ("#EF4444" if pred == "FAKE" else "#10B981")
            ax.plot(xs, ys, marker="o", color=line_c, linewidth=2.2)
            ax.fill_between(xs, ys, color=line_c, alpha=.10)
            if thr is not None:
                ax.axhline(thr * 100, color="#64748B", linestyle="--", linewidth=1, label="Threshold")
                ax.legend(loc="upper right", fontsize=8, facecolor="#101827", edgecolor="#1D2940")
            ax.set_ylim(0, 100)
            ax.set_xlabel("Time (s)", color="#94A3B8", fontsize=9); ax.set_ylabel("AI probability (%)", color="#94A3B8", fontsize=9)
            ax.grid(True, color="#1D2940", alpha=.6)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            seg_items = []
            for w in wins:
                fp = w["fake_prob"]
                if abs(fp - thr * 100) <= 5:
                    seg_label = "BORDERLINE"
                    seg_color = "var(--amber)"
                elif fp >= thr * 100:
                    seg_label = "AI-GENERATED"
                    seg_color = "var(--red)"
                else:
                    seg_label = "AUTHENTIC"
                    seg_color = "var(--green)"
                seg_items.append(
                    f'<div class="seg"><small>{mmss(w["start_sec"])}–{mmss(w["end_sec"])}</small>'
                    f'<b style="color:{seg_color}">{fp:.1f}%</b>'
                    f'<span>{seg_label}</span></div>'
                )
            md('<div class="segs">' + "".join(seg_items) + '</div>')

    # ---- report -----------------------------------------------------------------
    md('<div class="sec"><div class="sec-t">Analysis report</div></div>')
    with card("report"):
        md(f'<div class="cardtitle"><b>{E(a["filename"])}</b><span>{E(DETECTION_METHOD_STR)}</span></div>'
           f'<div class="note">The report lists the model assessment, probabilities, duration, and processing time for this recording.</div>')
        d1, d2 = cols(2, gap="medium")
        with d1:
            st.download_button("Download Analysis Report", data=build_report_text(a),
                               file_name=f"audioguard_report_{Path(a['filename']).stem}.txt", mime="text/plain",
                               key="dl_report")
        with d2:
            if st.button("Analyze Another Audio", key="btn_reset", type="secondary"):
                st.session_state["reset_count"] += 1
                st.session_state["analysis"] = None
                st.rerun()

    md('<div class="note" style="margin-top:16px">AudioGuard provides probabilistic machine-learning predictions. Results should be treated as '
       'automated indicators rather than absolute proof of authenticity.</div>')


def page_analysis():
    md('<div class="pagehead"><div class="ph1">Audio Analysis</div><div class="ph2">Upload or record speech audio for forensic inspection.</div></div>')
    st.session_state.setdefault("input_source", "Upload Audio File")

    buf, filename = None, "recording.wav"
    with card("workspace"):
        sel_col, note_col = cols([1.4, 1], vertical_alignment="center")
        with sel_col:
            md('<div class="cardtitle"><b>Drop your audio here</b></div>')
        with note_col:
            md('<div class="dz-note" style="text-align:right">WAV · MP3 · FLAC · OGG · M4A — maximum 60 seconds</div>')
        mode = st.session_state["input_source"]
        s1, s2, _ = cols([1, 1, 1.3])
        with s1:
            st.button(":material/upload_file: Upload Audio File", key="src_upload", use_container_width=True,
                      type="primary" if mode == "Upload Audio File" else "secondary",
                      on_click=set_source, args=("Upload Audio File",))
        with s2:
            st.button(":material/mic: Record Live Voice", key="src_record", use_container_width=True,
                      type="primary" if mode == "Record Live Voice" else "secondary",
                      on_click=set_source, args=("Record Live Voice",))
        if mode == "Upload Audio File":
            buf = st.file_uploader("Browse files", type=["wav", "mp3", "flac", "ogg", "m4a"],
                                   label_visibility="collapsed", key=f"uploader_{st.session_state['reset_count']}")
            if buf:
                filename = buf.name
                suffix = Path(filename).suffix.lower()
                if suffix not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                    md('<div class="err"><b>ANALYSIS FAILED</b><div class="m">Unsupported audio format. Please upload WAV, MP3, FLAC, OGG, or M4A.</div></div>')
                    return
        elif hasattr(st, "audio_input"):
            buf = st.audio_input("Record", label_visibility="collapsed", key=f"recorder_{st.session_state['reset_count']}")
            if buf:
                filename = "recorded_voice.wav"
        else:
            md('<div class="err"><b>RECORDING UNAVAILABLE</b><div class="m">This Streamlit version cannot record audio. Update Streamlit or use Upload Audio File.</div></div>')

        if buf is None:
            bars = "".join(f'<i style="--h:{h}px;--i:{i}"></i>' for i, h in
                           enumerate([10, 18, 28, 14, 32, 22, 12, 26, 16, 30, 20, 10, 24, 14]))
            if mode == "Record Live Voice":
                orb_cls, title = "rec", "READY TO RECORD"
                sub = "Click the red microphone to start recording, speak clearly for a few seconds, then press stop."
                icon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>'
            else:
                orb_cls, title = "up", "READY FOR FORENSIC ANALYSIS"
                sub = "Click the icon to choose a speech recording, or drag a file into the box above."
                icon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg>'
            md(f'<div class="empty"><div class="orb {orb_cls}"><span class="rg"></span><span class="rg"></span><span class="rg"></span>'
               f'<div class="core">{icon}</div></div><div class="wave">{bars}</div><b>{title}</b><div class="sub">{sub}</div></div>')

    if buf is None:
        st.session_state["analysis"] = None
        return

    data = buf.getvalue()
    fkey = f"{filename}:{len(data)}"
    meta = inspect_audio(data, filename)
    dash = "—"
    items = [
        ("Format", meta["format"]),
        ("Duration", f"{meta['duration']:.1f} s" if meta["duration"] is not None else dash),
        ("Sample rate", f"{meta['sr'] / 1000:g} kHz" if meta["sr"] else dash),
        ("Channels", {1: "MONO", 2: "STEREO"}.get(meta["ch"], str(meta["ch"]) if meta["ch"] else dash)),
        ("File size", fmt_size(meta["size"])),
    ]
    with card("selected"):
        md(f'<div class="cardtitle"><b>Selected audio</b></div><div class="fname">{E(filename)}</div>'
           '<div class="meta">' + "".join(f"<div><small>{k.upper()}</small><b>{E(str(v))}</b></div>" for k, v in items) + "</div>")
        st.audio(data, format=AUDIO_MIME.get(Path(filename).suffix.lower(), "audio/wav"))
        run = st.button("RUN FORENSIC ANALYSIS →", type="primary", key="btn_run")

    if run:
        status = st.empty()
        with status.container():
            md('<div class="proc"><div class="proc-t">Analyzing Audio</div><div class="proc-p">Running acoustic forensic analysis…</div>'
               '<div class="proc-li">Audio preprocessing</div><div class="proc-li">Feature extraction</div>'
               f'<div class="proc-li">{E(CLASSIFIER_NAME)} classification</div><div class="proc-li">Result generation</div>'
               '<div class="indet"><i></i></div></div>')
        try:
            suffix = Path(filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                temp_path = tmp.name
            try:
                result = predict_audio_file(temp_path, backend="mfcc")      # unchanged backend call
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            waveform, sr, _ = load_and_preprocess_audio(data)
            vis = {"waveform": waveform, "sr": sr,
                   "mfccs": compute_mfcc_visualization(waveform, sr=sr),
                   "mel_db": compute_mel_spectrogram(waveform, sr=sr)}
            st.session_state["analysis"] = {"key": fkey, "filename": filename, "result": result, "vis": vis}
            status.empty()
        except Exception:
            st.session_state["analysis"] = None
            status.empty()
            md('<div class="err"><b>ANALYSIS FAILED</b><div class="m">Unable to complete the forensic analysis. Please try another recording.</div></div>')
            return

    a = st.session_state.get("analysis")
    if a and a["key"] == fkey:
        render_results(a)



# =====================================================================================
# PAGE 3 - SYSTEM ARCHITECTURE
# =====================================================================================
def page_architecture():
    md('<div class="pagehead"><div class="ph1">System Architecture</div><div class="ph2">AudioGuard Detection Pipeline</div></div>')
    steps = [
        ("Audio Input", "WAV, MP3, FLAC, OGG or M4A"),
        ("Audio Preprocessing", "Decode, mono, normalize"),
        ("16 kHz Mono Standardization", "One fixed format for every file"),
        (f"{BACKEND_NAME} Feature Extraction", "Cepstral and spectral features"),
        (f"{FEATURE_DIM}-Dimensional Feature Vector", "One vector per analysis window"),
        (f"{CLASSIFIER_NAME} Classifier", "Trained binary classifier"),
        ("Probability Estimation", "Real and AI-generated probabilities"),
        ("REAL / AI-GENERATED", "Final verdict"),
    ]
    md('<div class="panel"><div class="panel-title">Pipeline architecture</div>'
       '<div class="panel-sub">Sequential acoustic inspection and classification workflow</div>'
       '<div class="steps s4">' + "".join(
           f'<div class="step"><span class="n">STEP {i + 1}</span><b>{E(t)}</b><span>{E(d)}</span></div>'
           for i, (t, d) in enumerate(steps)) + '</div></div>')

    def lst(items):
        return '<div class="lst">' + "".join(f"<div>{i}</div>" for i in items) + "</div>"

    blocks = [
        ("Audio preprocessing", "STAGE 1", "Standardizes raw input audio into a uniform signal.",
         ["Single-channel mono conversion", "16,000 Hz resampling target", "Peak amplitude normalization",
          "Sliding windows (4.0 s window, 2.0 s hop)"]),
        ("Feature extraction", "STAGE 2", f"Computes a {FEATURE_DIM}-dimensional acoustic representation.",
         ["13 Mel-frequency cepstral coefficients", "Delta and delta-delta derivatives",
          "Spectral centroid and roll-off", "Zero-crossing rate and RMS energy"]),
        ("Classification", "STAGE 3", f"{E(CLASSIFIER_NAME)} classifier trained for binary speech classification.",
         ["CPU-based inference", "Class probability estimation", "Aggregated sliding-window voting",
          "Local, private model execution"]),
    ]
    md('<div class="grid g3">' + "".join(
        f'<div class="tile"><div class="tile-top"><div class="tile-t">{t}</div><span class="badge">{b}</span></div>'
        f'<div class="tile-p">{d}</div>{lst(items)}</div>' for t, b, d, items in blocks) + '</div>')


# =====================================================================================
# PAGE 4 - ABOUT
# =====================================================================================
def page_about():
    md('<div class="pagehead"><div class="ph1">About AudioGuard</div><div class="ph2">AI-assisted audio forensics</div></div>')
    md(f'<div class="panel"><div class="panel-title">What AudioGuard does</div>'
       f'<div class="hero-p" style="margin-top:10px;max-width:820px">AudioGuard is an AI-assisted audio forensic application that analyzes '
       f'speech recordings using acoustic feature extraction and a trained {E(CLASSIFIER_NAME)} classifier to identify '
       f'characteristics associated with AI-generated speech.</div></div>')
    facts = [
        ("mfcc", "Detection method", DETECTION_METHOD_STR, "var(--cyan)",
         "Captures spectral envelope and cepstral properties of the signal to find subtle synthetic artifacts."),
        ("chart", "Input", "Speech Audio", "#A5B4FC", "WAV, MP3, FLAC, OGG and M4A files up to 60 seconds, or a live microphone recording."),
        ("lock", "Processing", "Local Analysis", "#34D399", "CPU-based inference on this computer. No audio is sent to an external AI service."),
    ]
    md('<div class="grid g3">' + "".join(
        f'<div class="tile"><div class="tile-top"><div class="ico">{ICONS[ic]}</div></div><span class="fact-label">{l}</span>'
        f'<span class="fact-val" style="color:{c}">{E(v)}</span><div class="tile-p">{d}</div></div>'
        for ic, l, v, c, d in facts) + '</div>')
    md('<div class="note">AudioGuard provides probabilistic machine-learning predictions. Results should be treated as '
       'automated indicators rather than absolute proof of authenticity.</div>')


{"Home": page_dashboard, "Audio Analysis": page_analysis, "About": page_about}[page]()


# Every run: (1) scroll to the top when the page changed, (2) make the big upload / microphone icons clickable
# by forwarding their click to the real Streamlit file-uploader / recorder button.
_changed = st.session_state.get("_shown_page") != page
st.session_state["_shown_page"] = page
_js = (
    "<script>try{const p=window.parent,d=p.document;"
    + ("const m=d.querySelector('section.stMain')||d.querySelector('[data-testid=\"stMain\"]')||d.querySelector('section.main');"
       "if(m){m.scrollTo(0,0);}p.scrollTo(0,0);" if _changed else "")
    + "if(p.__agOrbH){d.removeEventListener('click',p.__agOrbH);}"
    "const h=function(e){const o=e.target&&e.target.closest?e.target.closest('.orb'):null;if(!o){return;}"
    "let t=null;"
    "if(o.classList.contains('rec')){t=d.querySelector('[data-testid=\"stAudioInputActionButton\"]');}"
    "else{t=d.querySelector('[data-testid=\"stFileUploaderDropzone\"] button')||d.querySelector('[data-testid=\"stFileUploaderDropzone\"] input[type=file]');}"
    "if(t){t.click();}};"
    "p.__agOrbH=h;d.addEventListener('click',h);}catch(e){}</script>"
)
try:
    if hasattr(st, "iframe"):
        st.iframe(_js, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(_js, height=0)
except Exception:
    pass
