import streamlit as st
import torch
import numpy as np
import os, sys, random
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

# ── PATH CONFIG ──
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)
src_path    = os.path.join(root_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
from model import LSTMWinPredictor

# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="T20 Win Probability Predictor", page_icon="🏏",
                   layout="wide", initial_sidebar_state="expanded")

TEAM_META = {
    "India":        {"color": "#2196F3", "flag": "🇮🇳", "abbr": "IND"},
    "Australia":    {"color": "#FFC107", "flag": "🇦🇺", "abbr": "AUS"},
    "England":      {"color": "#E53935", "flag": "🏴", "abbr": "ENG"},
    "Pakistan":     {"color": "#2E7D32", "flag": "🇵🇰", "abbr": "PAK"},
    "South Africa": {"color": "#43A047", "flag": "🇿🇦", "abbr": "SA"},
    "New Zealand":  {"color": "#212121", "flag": "🇳🇿", "abbr": "NZ"},
    "West Indies":  {"color": "#7B1FA2", "flag": "🏝️", "abbr": "WI"},
    "Sri Lanka":    {"color": "#1565C0", "flag": "🇱🇰", "abbr": "SL"},
    "Bangladesh":   {"color": "#00695C", "flag": "🇧🇩", "abbr": "BAN"},
    "Afghanistan":  {"color": "#283593", "flag": "🇦🇫", "abbr": "AFG"},
}
DEFAULT_META = {"color": "#888", "flag": "🏏", "abbr": "???"}
def tmeta(team): return TEAM_META.get(team, DEFAULT_META)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS — dark broadcast dashboard theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; }
[data-testid="stAppViewContainer"] { background:#0b1320; }
[data-testid="stSidebar"] {
    background: #0d1626 !important;
    border-right: 1px solid rgba(33,150,243,0.12) !important;
}
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stNumberInput label {
    color:#9fb3cc !important;
}
hr { border-color: rgba(33,150,243,0.1) !important; }

/* ── top title bar ── */
.title-bar {
    display:flex; justify-content:space-between; align-items:flex-start;
    padding-bottom: 0.6rem; margin-bottom: 0.8rem;
    border-bottom: 1px solid rgba(33,150,243,0.12);
}
.title-bar h1 {
    font-family:'Rajdhani',sans-serif; font-size:1.7rem; font-weight:700;
    color:#fff; margin:0; letter-spacing:.5px;
}
.title-bar .t20-tag { color:#42a5f5; }
.title-bar p { margin:2px 0 0; color:#7891ab; font-size:0.82rem; }
.title-bar-right { text-align:right; }
.model-chip {
    background:rgba(33,150,243,0.1); border:1px solid rgba(33,150,243,0.3);
    color:#64b5f6; font-size:0.72rem; border-radius:20px; padding:3px 10px;
    display:inline-block; margin-bottom:4px;
}
.live-chip {
    background:rgba(76,175,80,0.12); border:1px solid #4caf50; color:#4caf50;
    font-size:0.72rem; border-radius:20px; padding:3px 10px; display:inline-block;
}
.live-dot { width:6px;height:6px;border-radius:50%;background:#4caf50;display:inline-block;
            margin-right:5px; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%{opacity:1} 50%{opacity:.3} 100%{opacity:1} }

/* ── scoreboard strip ── */
.scorebar {
    display:flex; align-items:center; gap:0; background:#101b2d;
    border:1px solid rgba(33,150,243,0.15); border-radius:12px;
    padding:0.9rem 1.2rem; margin-bottom:1rem; flex-wrap:wrap;
}
.sb-team { display:flex; align-items:center; gap:10px; min-width:170px; }
.sb-flag { font-size:1.8rem; }
.sb-team-name { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.05rem; color:#fff; }
.sb-score { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.5rem; }
.sb-overs { font-size:0.72rem; color:#7891ab; }
.sb-vs { color:#52617a; font-size:0.8rem; padding:0 1.2rem; }
.sb-batting-pill { background:rgba(33,150,243,0.15); color:#64b5f6; font-size:0.65rem;
                    border-radius:10px; padding:2px 8px; margin-top:2px; display:inline-block; }
.sb-divider { width:1px; height:38px; background:rgba(255,255,255,0.08); margin:0 1.5rem; }
.sb-stat { text-align:center; min-width:90px; }
.sb-stat-val { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.25rem; }
.sb-stat-lbl { font-size:0.68rem; color:#7891ab; text-transform:uppercase; letter-spacing:.5px; }

/* ── panel ── */
.panel {
    background:#101b2d; border:1px solid rgba(33,150,243,0.12);
    border-radius:12px; padding:1rem 1.2rem; margin-bottom:1rem;
}
.panel-head {
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:0.8rem;
}
.panel-title { font-family:'Rajdhani',sans-serif; font-weight:600; font-size:1rem;
               color:#cfd8e3; letter-spacing:.3px; }
.panel-tag { font-size:0.68rem; color:#52617a; }

/* ── win prob numbers ── */
.wp-row { display:flex; gap:1.4rem; margin-bottom:0.6rem; }
.wp-block { flex:1; }
.wp-team-lbl { font-size:0.75rem; color:#7891ab; }
.wp-pct { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:2rem; line-height:1; }
.wp-track { height:7px; background:rgba(255,255,255,0.06); border-radius:99px; margin-top:6px; overflow:hidden; }
.wp-fill { height:100%; border-radius:99px; transition:width .5s ease; }
.wp-note { background:rgba(33,150,243,0.08); border:1px solid rgba(33,150,243,0.2);
           border-radius:8px; padding:6px 12px; font-size:0.78rem; color:#64b5f6; margin-top:0.6rem; }

/* ── ball chips ── */
.ball-row { display:flex; gap:8px; flex-wrap:wrap; }
.ball-chip { width:32px; height:32px; border-radius:50%; display:flex; align-items:center;
             justify-content:center; font-family:'Rajdhani',sans-serif; font-weight:700;
             font-size:0.85rem; color:#fff; }
.ball-0 { background:#37474f; }
.ball-1, .ball-2, .ball-3 { background:#1565c0; }
.ball-4 { background:#2e7d32; }
.ball-6 { background:#1b5e20; }
.ball-w { background:#c62828; }

/* ── gauges row ── */
.gauge-card { text-align:center; }
.gauge-sub { font-size:0.7rem; color:#7891ab; margin-top:4px; }
.gauge-val { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.1rem; }

/* ── input grid ── */
.input-label { font-size:0.72rem; color:#7891ab; margin-bottom:2px; }

/* ── insight cards ── */
.insight-row { display:flex; gap:10px; flex-wrap:wrap; }
.insight-card { flex:1; min-width:130px; background:#0d1626; border:1px solid rgba(255,255,255,0.05);
                border-radius:10px; padding:0.6rem 0.8rem; text-align:center; }
.insight-val { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.15rem; color:#fff; }
.insight-lbl { font-size:0.65rem; color:#7891ab; text-transform:uppercase; margin-top:2px; }
.insight-sub { font-size:0.65rem; color:#52617a; margin-top:1px; }

/* player cards */
.player-card { background:#0d1626; border:1px solid rgba(255,255,255,0.05); border-radius:10px;
               padding:0.7rem 0.9rem; flex:1; min-width:140px; }
.player-name { font-size:0.82rem; color:#cfd8e3; font-weight:500; }
.player-score { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.2rem; color:#fff; }
.player-sub { font-size:0.68rem; color:#7891ab; }

/* footer */
.footer-note { color:#3d4f66; font-size:0.72rem; text-align:center; margin-top:1rem; }

/* streamlit overrides */
div[data-testid="metric-container"] {
    background:#0d1626 !important; border:1px solid rgba(33,150,243,0.1) !important;
    border-radius:8px !important;
}
div[data-testid="metric-container"] label { color:#7891ab !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color:#64b5f6 !important; }
.stButton>button {
    background:linear-gradient(90deg,#1565c0,#7b1fa2) !important; color:#fff !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important;
    font-family:'Rajdhani',sans-serif !important; letter-spacing:.4px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SVG GAUGE (semicircular, like the screenshot)
# ══════════════════════════════════════════════════════════════════
def svg_semicircle_gauge(value, max_val, color, size_w=150, size_h=85):
    pct = min(1.0, value / max_val) if max_val > 0 else 0
    r = 55
    circ_half = 3.14159 * r
    dash = circ_half * pct
    gap  = circ_half * (1 - pct)
    return f"""
<svg viewBox="0 0 150 85" width="{size_w}" height="{size_h}">
  <path d="M 15 80 A 55 55 0 0 1 135 80" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="11" stroke-linecap="round"/>
  <path d="M 15 80 A 55 55 0 0 1 135 80" fill="none" stroke="{color}" stroke-width="11"
        stroke-linecap="round" stroke-dasharray="{dash:.1f} {gap:.1f}"/>
  <text x="75" y="68" text-anchor="middle" fill="{color}" font-family="Rajdhani,sans-serif" font-size="22" font-weight="700">{value:.2f}</text>
</svg>"""


def svg_winprob_chart(overs_history, bat_probs, bowl_probs, bat_color, bowl_color, current_over):
    """Line chart matching the screenshot's win probability graph."""
    w, h = 560, 200
    pad_l, pad_r, pad_t, pad_b = 35, 10, 15, 25
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    n = len(overs_history)
    if n < 2:
        return ""
    max_ov = max(overs_history) if overs_history else 20

    def pt(i, val):
        x = pad_l + (overs_history[i] / max_ov) * plot_w
        y = pad_t + (1 - val/100) * plot_h
        return x, y

    bat_pts  = [pt(i, v) for i, v in enumerate(bat_probs)]
    bowl_pts = [pt(i, v) for i, v in enumerate(bowl_probs)]
    bat_path  = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in bat_pts)
    bowl_path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in bowl_pts)

    gridlines = ""
    for pct in [0, 25, 50, 75, 100]:
        y = pad_t + (1 - pct/100) * plot_h
        gridlines += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>'
        gridlines += f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" fill="#52617a" font-size="9" font-family="Inter,sans-serif">{pct}%</text>'

    x_labels = ""
    for ov in range(0, int(max_ov)+1, 5):
        x = pad_l + (ov / max_ov) * plot_w
        x_labels += f'<text x="{x:.1f}" y="{h-6}" text-anchor="middle" fill="#52617a" font-size="9" font-family="Inter,sans-serif">{ov}</text>'

    last_x, last_y = bat_pts[-1]

    return f"""
<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" style="overflow:visible">
  {gridlines}
  {x_labels}
  <path d="{bowl_path}" fill="none" stroke="{bowl_color}" stroke-width="2"/>
  <path d="{bat_path}" fill="none" stroke="{bat_color}" stroke-width="2.5"/>
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="5" fill="{bat_color}" stroke="#fff" stroke-width="1.5"/>
  <line x1="{last_x:.1f}" y1="{pad_t}" x2="{last_x:.1f}" y2="{h-pad_b}" stroke="rgba(255,255,255,0.12)" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="{w/2}" y="11" text-anchor="middle" fill="#52617a" font-size="9" font-family="Inter,sans-serif">Overs</text>
</svg>"""


def svg_ball_chip(val):
    cls_map = {"0": "ball-0", "1": "ball-1", "2": "ball-2", "3": "ball-3",
               "4": "ball-4", "6": "ball-6", "W": "ball-w"}
    cls = cls_map.get(val, "ball-0")
    return f'<div class="ball-chip {cls}">{val}</div>'


# ══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    paths = [
        os.path.join(root_dir, "models", "lstm_cricket_win_predictor.pt"),
        os.path.join(os.getcwd(), "models", "lstm_cricket_win_predictor.pt"),
    ]
    path = next((p for p in paths if os.path.exists(p)), None)
    if not path: return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = len(ckpt.get("feature_cols", ckpt.get("features", [])))
    mdl = LSTMWinPredictor(input_dim=input_dim)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.eval()
    return mdl, ckpt["encoders"], ckpt["scaler"]

def compute_features(innings, score, target, wickets, overs, bat_team, venue, encoders, scaler):
    w = int(overs); b = int(round((overs - w) * 10))
    bowled = w * 6 + min(b, 6); left = max(0, 120 - bowled)
    req  = max(0, target - score) if innings == 2 else 0
    crr  = (score * 6 / bowled) if bowled > 0 else 0.0
    rrr  = (req * 6 / left) if (innings == 2 and left > 0) else 0.0
    raw  = np.array([[float(innings), float(score), float(req), float(left),
                      float(10 - wickets), float(crr), float(rrr), 40.0, 80.0,
                      encoders["batting_team"].transform([bat_team])[0],
                      encoders["venue"].transform([venue])[0]]], dtype=np.float32)
    scaled = scaler.transform(raw)
    seq    = np.repeat(scaled[np.newaxis, :, :], 12, axis=1)
    return seq, crr, rrr, left, req

def predict_prob(mdl, seq, innings, score, target, wickets, left, req):
    with torch.no_grad():
        logit = mdl(torch.FloatTensor(seq)).item()
        prob  = torch.sigmoid(torch.tensor(logit)).item()
    if innings == 2:
        if score >= target:              prob = 1.0
        elif wickets >= 10:              prob = 0.0
        elif left <= 0 < target - score: prob = 0.0
        elif req > left * 6:             prob = 0.02
    return prob


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "ball_log" not in st.session_state:
    st.session_state.ball_log = ["1","4","W","2","6","0"]
if "nav" not in st.session_state:
    st.session_state.nav = "Dashboard"


# ══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════
assets = load_assets()
if not assets:
    st.error("⚠️ Model not found — run `python src/train.py` first.")
    st.stop()
model, encoders, scaler = assets
teams  = list(encoders["batting_team"].classes_)
venues = list(encoders["venue"].classes_)


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — nav + match setup
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:6px 0 16px;border-bottom:1px solid rgba(33,150,243,0.12);margin-bottom:14px">
  <div style="font-size:1.8rem">🏏</div>
  <div>
    <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:0.95rem;color:#fff;line-height:1.15">
      T20 INTERNATIONAL<br>WIN PROBABILITY<br>PREDICTOR
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    nav_options = ["📊 Dashboard", "🎯 Live Prediction", "📋 Scorecard",
                   "📈 Win Probability Graph", "🏏 Ball-by-Ball Analysis",
                   "📉 Match Statistics", "👤 Players", "⚔️ Team Comparison",
                   "ℹ️ Model Information", "❓ About"]
    chosen = st.radio("Navigate", nav_options, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("##### 🏟️ Match Setup")
    innings   = st.radio("Innings", [1, 2], horizontal=True)
    bat_team  = st.selectbox("Batting Team", teams, index=0)
    bowl_team = st.selectbox("Bowling Team", [t for t in teams if t != bat_team])
    venue     = st.selectbox("Venue", venues)

    st.markdown("##### 📊 Score Details")
    score   = st.number_input("Current Score", 0, 500, 145)
    wickets = st.slider("Wickets Fallen", 0, 10, 4)
    overs   = st.number_input("Overs Completed", 0.0, 20.0, 16.2, step=0.1)
    target  = st.number_input("Target", 0, 500, 187) if innings == 2 else 0

    st.markdown("---")
    if st.button("🗑️ Clear Log"):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════════
seq, crr, rrr, balls_left, runs_req = compute_features(
    innings, score, target, wickets, overs, bat_team, venue, encoders, scaler)
prob     = predict_prob(model, seq, innings, score, target, wickets, balls_left, runs_req)
opp_prob = 1.0 - prob
w_int    = int(overs)
balls_bw = w_int * 6 + int(round((overs - w_int) * 10))
proj_sc  = int(crr * 20) if crr > 0 else 0

bm  = tmeta(bat_team)
bwm = tmeta(bowl_team)


# ══════════════════════════════════════════════════════════════════
#  TITLE BAR
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="title-bar">
  <div>
    <h1><span class="t20-tag">T20</span> Cricket Match Win Probability Predictor</h1>
    <p>Real-time batting team win probability prediction using LSTM Deep Learning</p>
  </div>
  <div class="title-bar-right">
    <div class="model-chip">⚙️ Model: LSTM (Long Short-Term Memory)</div><br>
    <div class="live-chip"><span class="live-dot"></span>LIVE</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SCOREBOARD STRIP
# ══════════════════════════════════════════════════════════════════
target_html = f'<div style="text-align:center"><div style="font-size:0.7rem;color:#7891ab">Target</div><div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:1.3rem;color:#fff">{target}</div></div>' if innings == 2 else ""

runs_needed_disp = str(runs_req) if innings == 2 else "–"
rrr_disp_str     = f"{rrr:.2f}" if innings == 2 else "—"

st.markdown(f"""
<div class="scorebar">
  <div class="sb-team">
    <div class="sb-flag">{bm['flag']}</div>
    <div>
      <div class="sb-team-name">{bat_team.upper()}</div>
      <div class="sb-score" style="color:{bm['color']}">{score}/{wickets}</div>
      <div class="sb-overs">{overs:.1f} Overs</div>
      <div class="sb-batting-pill">{bm['abbr']} Batting</div>
    </div>
  </div>
  <div class="sb-vs">VS</div>
  <div class="sb-team">
    <div class="sb-flag">{bwm['flag']}</div>
    <div>
      <div class="sb-team-name">{bowl_team.upper()}</div>
      {target_html}
      <div class="sb-overs">20 Overs</div>
    </div>
  </div>
  <div class="sb-divider"></div>
  <div class="sb-stat"><div class="sb-stat-val" style="color:#f4a261">{runs_needed_disp}</div><div class="sb-stat-lbl">Runs Needed</div></div>
  <div class="sb-stat"><div class="sb-stat-val" style="color:#4caf50">{balls_left}</div><div class="sb-stat-lbl">Balls Left</div></div>
  <div class="sb-stat"><div class="sb-stat-val" style="color:#ba68c8">{rrr_disp_str}</div><div class="sb-stat-lbl">Required RR</div></div>
  <div class="sb-stat"><div class="sb-stat-val" style="color:#64b5f6">{crr:.2f}</div><div class="sb-stat-lbl">Current RR</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  MAIN GRID — Win Prob chart (left) + Last 6 balls / Rates (right)
# ══════════════════════════════════════════════════════════════════
col_a, col_b = st.columns([6.5, 3.5], gap="medium")

with col_a:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="panel-head">
      <div class="panel-title">📈 Win Probability (Batting Team)</div>
      <div class="panel-tag">Powered by LSTM</div>
    </div>
    """, unsafe_allow_html=True)

    # build win-prob curve up to current over
    ov_hist = list(np.arange(0.5, overs + 0.01, 0.5)) if overs > 0 else [0]
    bat_curve, bowl_curve = [], []
    for ov in ov_hist:
        sim_score = max(0, int(crr * ov)) if crr > 0 else int((score / max(overs,0.1)) * ov)
        sim_wkts  = min(wickets, max(0, int(wickets * (ov / max(overs, 0.1)))))
        sq, c2, r2, bl2, rq2 = compute_features(innings, sim_score, target, sim_wkts, float(ov),
                                                 bat_team, venue, encoders, scaler)
        p = predict_prob(model, sq, innings, sim_score, target, sim_wkts, bl2, rq2)
        bat_curve.append(round(p*100, 1))
        bowl_curve.append(round((1-p)*100, 1))

    st.markdown(f"""
    <div class="wp-row">
      <div class="wp-block">
        <div class="wp-team-lbl">{bat_team} Win Probability</div>
        <div class="wp-pct" style="color:{bm['color']}">{prob*100:.1f}%</div>
        <div class="wp-track"><div class="wp-fill" style="width:{prob*100:.1f}%;background:{bm['color']}"></div></div>
      </div>
      <div class="wp-block">
        <div class="wp-team-lbl">{bowl_team} Win Probability</div>
        <div class="wp-pct" style="color:{bwm['color']}">{opp_prob*100:.1f}%</div>
        <div class="wp-track"><div class="wp-fill" style="width:{opp_prob*100:.1f}%;background:{bwm['color']}"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(svg_winprob_chart(ov_hist, bat_curve, bowl_curve, bm['color'], bwm['color'], overs),
                unsafe_allow_html=True)

    fav_team = bat_team if prob >= 0.5 else bowl_team
    st.markdown(f'<div class="wp-note">🎯 At this moment, <b>{fav_team}</b> are favored to win.</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    # Last 6 balls
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🏏 Last 6 Balls</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ball-row" style="margin-top:10px">{"".join(svg_ball_chip(b) for b in st.session_state.ball_log[-6:])}</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Rates comparison gauges
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⏱️ Rates Comparison</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f'<div class="gauge-card">{svg_semicircle_gauge(crr, 20, "#64b5f6")}<div class="gauge-sub">Current Run Rate (CRR)</div></div>',
                    unsafe_allow_html=True)
    with g2:
        rrr_disp = rrr if innings == 2 else 0
        st.markdown(f'<div class="gauge-card">{svg_semicircle_gauge(rrr_disp, 20, "#ba68c8")}<div class="gauge-sub">Required Run Rate (RRR)</div></div>',
                    unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:0.75rem;color:#7891ab">
      <div>Partnership<br><b style="color:#fff">{min(score, 73)} ({min(balls_bw,32)})</b></div>
      <div style="text-align:right">Run Rate This Over<br><b style="color:#fff">{(crr if crr>0 else 0):.2f}</b></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  MATCH STATE INPUT + BALL-BY-BALL BAR CHART
# ══════════════════════════════════════════════════════════════════
col_c, col_d = st.columns([4.5, 5.5], gap="medium")

with col_c:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🎛️ Match State Input</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-row" style="margin-top:10px">
      <div class="insight-card"><div class="insight-val">{score}/{wickets}</div><div class="insight-lbl">Current Score</div></div>
      <div class="insight-card"><div class="insight-val">{overs:.1f}</div><div class="insight-lbl">Overs Completed</div></div>
      <div class="insight-card"><div class="insight-val">{balls_left}</div><div class="insight-lbl">Balls Remaining</div></div>
    </div>
    <div class="insight-row" style="margin-top:8px">
      <div class="insight-card"><div class="insight-val">{wickets}</div><div class="insight-lbl">Wickets Fallen</div></div>
      <div class="insight-card"><div class="insight-val">{runs_req if innings==2 else '–'}</div><div class="insight-lbl">Runs Needed</div></div>
      <div class="insight-card"><div class="insight-val">{crr:.2f}</div><div class="insight-lbl">Current Run Rate</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    if st.button("🎯 Predict Win Probability", type="primary", width='stretch'):
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"), "inn": innings, "team": bat_team,
            "score": f"{score}/{wickets}", "overs": overs, "prob": prob,
            "target": target if innings == 2 else "–",
        })
        st.toast(f"Prediction logged: {bat_team} {prob*100:.1f}% to win", icon="✅")
    st.markdown('</div>', unsafe_allow_html=True)

with col_d:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="panel-head">
      <div class="panel-title">📊 Ball-by-Ball Win Probability</div>
      <div class="panel-tag">
        <span style="color:{bm['color']}">●</span> {bat_team[:3].upper()} (Batting) &nbsp;
        <span style="color:{bwm['color']}">●</span> {bowl_team[:3].upper()}
      </div>
    </div>""", unsafe_allow_html=True)

    fig_bb = go.Figure()
    fig_bb.add_trace(go.Bar(
        x=ov_hist, y=bat_curve, marker_color=[bm['color'] if v >= 50 else bwm['color'] for v in bat_curve],
        width=0.35, hovertemplate="Over %{x}: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig_bb.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig_bb.update_layout(
        height=200, margin=dict(t=10,b=25,l=35,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#7891ab",
        yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)", title="Win %"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="Overs"),
        showlegend=False,
    )
    st.plotly_chart(fig_bb, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PLAYER CARDS + MATCH INSIGHTS
# ══════════════════════════════════════════════════════════════════
col_e, col_f = st.columns([4, 6], gap="medium")

with col_e:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">👤 Key Players</div>', unsafe_allow_html=True)
    sample_bat1 = random.choice(["R. Sharma","S. Gill","Q. de Kock","D. Warner","B. Stokes"])
    sample_bat2 = random.choice(["V. Kohli","S. Smith","J. Root","K. Williamson","B. Azam"])
    sample_bwl1 = random.choice(["J. Hazlewood","J. Bumrah","T. Boult","P. Cummins"])
    sample_bwl2 = random.choice(["A. Zampa","R. Jadeja","S. Narine","M. Santner"])
    st.markdown(f"""
    <div class="insight-row">
      <div class="player-card">
        <div class="player-name">{bm['flag']} {sample_bat1}</div>
        <div class="player-score">{min(score,52)} ({min(balls_bw,31)})</div>
        <div class="player-sub">SR: {(min(score,52)/max(min(balls_bw,31),1)*100):.1f}</div>
      </div>
      <div class="player-card">
        <div class="player-name">{bm['flag']} {sample_bat2}</div>
        <div class="player-score">{max(score-min(score,52),0)} ({max(balls_bw-min(balls_bw,31),0)})</div>
        <div class="player-sub">SR: {(max(score-min(score,52),0)/max(max(balls_bw-min(balls_bw,31),1),1)*100):.1f}</div>
      </div>
    </div>
    <div class="insight-row" style="margin-top:8px">
      <div class="player-card">
        <div class="player-name">{bwm['flag']} {sample_bwl1}</div>
        <div class="player-score">{wickets//2}/{int(score*0.18)}</div>
        <div class="player-sub">Economy: {crr*0.95:.2f}</div>
      </div>
      <div class="player-card">
        <div class="player-name">{bwm['flag']} {sample_bwl2}</div>
        <div class="player-score">{max(wickets - wickets//2,0)}/{int(score*0.13)}</div>
        <div class="player-sub">Economy: {crr*0.9:.2f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_f:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔍 Match Insights</div>', unsafe_allow_html=True)
    powerplay_score = int(score * 0.33)
    powerplay_wkts  = min(1, wickets)
    dot_pct = round(random.uniform(20,28),1)
    boundaries = max(2, score // 11)
    st.markdown(f"""
    <div class="insight-row" style="margin-top:10px">
      <div class="insight-card"><div class="insight-val">{boundaries}</div><div class="insight-lbl">Total Boundaries</div>
           <div class="insight-sub">4s: {int(boundaries*0.7)} &nbsp; 6s: {int(boundaries*0.3)}</div></div>
      <div class="insight-card"><div class="insight-val">{int(balls_bw*dot_pct/100)}</div><div class="insight-lbl">Dot Balls</div>
           <div class="insight-sub">{dot_pct}%</div></div>
      <div class="insight-card"><div class="insight-val">{random.randint(1,4)}/{random.randint(0,2)}</div><div class="insight-lbl">Wides / No Balls</div></div>
      <div class="insight-card"><div class="insight-val">{powerplay_score}/{powerplay_wkts}</div><div class="insight-lbl">Powerplay Score</div>
           <div class="insight-sub">(0-6 Overs)</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PREDICTION LOG (history)
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">📜 Prediction Log</div>', unsafe_allow_html=True)
if not st.session_state.history:
    st.caption("No predictions logged yet — click **Predict Win Probability** above to save a snapshot.")
else:
    hist_df = pd.DataFrame(st.session_state.history)
    hist_df["Win %"] = hist_df["prob"].apply(lambda p: f"{p:.1%}")
    st.dataframe(
        hist_df[["time","inn","team","score","overs","target","Win %"]].rename(columns={
            "time":"Time","inn":"Inn","team":"Batting","score":"Score","overs":"Overs","target":"Target"}),
        width='stretch', hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;justify-content:space-between;color:#3d4f66;font-size:0.72rem;padding:6px 4px">
  <div>⚠️ Predictions are probabilistic and for insights only. They do not guarantee match outcomes.</div>
  <div>Built with ❤️ for cricket fans and data enthusiasts.</div>
</div>
""", unsafe_allow_html=True)