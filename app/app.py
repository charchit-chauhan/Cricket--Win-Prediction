import streamlit as st
import torch
import numpy as np
import os
import sys
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_path = os.path.join(root_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from model import LSTMWinPredictor

# ─────────────────────────── THEME & PAGE ───────────────────────────
st.set_page_config(
    page_title="Cricket Win Predictor Pro",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

CRICKET_CSS = """
<style>
/* Google font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark green cricket theme */
:root {
    --green-dark:  #1a472a;
    --green-mid:   #2d6a4f;
    --green-light: #40916c;
    --gold:        #f4a261;
    --red:         #e63946;
    --card-bg:     rgba(255,255,255,0.03);
}

/* Top header bar */
.header-banner {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 60%, #40916c 100%);
    padding: 1.5rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
.header-banner h1 { margin: 0; color: white; font-size: 2rem; font-weight: 700; }
.header-banner p  { margin: 0; color: #a8d8b9; font-size: 0.95rem; }
.header-emoji { font-size: 3rem; }

/* Stat cards */
.stat-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.stat-card {
    background: var(--card-bg);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    flex: 1;
    min-width: 130px;
    text-align: center;
}
.stat-card .val { font-size: 1.8rem; font-weight: 700; color: #74c69d; }
.stat-card .lbl { font-size: 0.8rem; color: #aaa; margin-top: 4px; }

/* Probability gauge colours */
.prob-high   { color: #52b788; }
.prob-medium { color: #f4a261; }
.prob-low    { color: #e63946; }

/* Section card */
.section-card {
    background: var(--card-bg);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #74c69d;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0.5rem;
}

/* Verdict box */
.verdict-box {
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-weight: 600;
    font-size: 1.05rem;
    text-align: center;
    margin-top: 0.5rem;
}
.verdict-win  { background: rgba(82,183,136,0.15); border: 1.5px solid #52b788; color: #52b788; }
.verdict-lose { background: rgba(230,57,70,0.15);  border: 1.5px solid #e63946; color: #e63946; }
.verdict-even { background: rgba(244,162,97,0.15); border: 1.5px solid #f4a261; color: #f4a261; }

/* History table */
.history-row { display: flex; gap: 0.5rem; align-items: center; padding: 0.45rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem; }
</style>
"""
st.markdown(CRICKET_CSS, unsafe_allow_html=True)

# ─────────────────────────── LOAD MODEL ───────────────────────────
@st.cache_resource
def load_assets():
    possible_paths = [
        os.path.join(root_dir, "models", "lstm_cricket_win_predictor.pt"),
        os.path.join(os.getcwd(), "models", "lstm_cricket_win_predictor.pt"),
    ]
    path = next((p for p in possible_paths if os.path.exists(p)), None)
    if not path:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = len(ckpt.get("feature_cols", ckpt.get("features", [])))
    model = LSTMWinPredictor(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["encoders"], ckpt["scaler"]

# ─────────────────────────── HELPERS ───────────────────────────
def compute_features(innings, score, target, wickets, overs, bat_team, venue, encoders, scaler):
    whole_overs      = int(overs)
    balls_in_over    = int(round((overs - whole_overs) * 10))
    balls_bowled     = whole_overs * 6 + min(balls_in_over, 6)
    balls_left       = max(0, 120 - balls_bowled)
    runs_req         = max(0, target - score) if innings == 2 else 0
    crr              = (score * 6 / balls_bowled) if balls_bowled > 0 else 0.0
    rrr              = (runs_req * 6 / balls_left) if (innings == 2 and balls_left > 0) else 0.0
    raw = np.array([[
        float(innings), float(score), float(runs_req), float(balls_left),
        float(10 - wickets), float(crr), float(rrr), 40.0, 80.0,
        encoders["batting_team"].transform([bat_team])[0],
        encoders["venue"].transform([venue])[0],
    ]], dtype=np.float32)
    scaled = scaler.transform(raw)
    seq    = np.repeat(scaled[np.newaxis, :, :], 12, axis=1)
    return seq, crr, rrr, balls_left, runs_req

def predict_prob(model, seq, innings, score, target, wickets, balls_left, runs_req):
    with torch.no_grad():
        logit = model(torch.FloatTensor(seq)).item()
        prob  = torch.sigmoid(torch.tensor(logit)).item()
    if innings == 2:
        if score >= target:                          prob = 1.0
        elif wickets >= 10:                          prob = 0.0
        elif balls_left <= 0 and score < target:     prob = 0.0
        elif runs_req > balls_left * 6:              prob = 0.0
    return prob

def win_prob_color(p):
    if p >= 0.65:  return "prob-high"
    if p >= 0.35:  return "prob-medium"
    return "prob-low"

def verdict_class(p):
    if p >= 0.65:  return "verdict-win"
    if p <= 0.35:  return "verdict-lose"
    return "verdict-even"

def verdict_text(p, team):
    if p >= 0.65:  return f"✅ {team} are in a strong position!"
    if p <= 0.35:  return f"⚠️ {team} are under pressure!"
    return "⚖️ It's anybody's game!"

# ─────────────────────────── SESSION STATE ───────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []          # list of dicts

# ─────────────────────────── HEADER ───────────────────────────
st.markdown("""
<div class="header-banner">
  <div class="header-emoji">🏏</div>
  <div>
    <h1>Cricket Win Predictor Pro</h1>
    <p>AI-powered T20 win probability • LSTM neural network • Live match analytics</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── LOAD ───────────────────────────
assets = load_assets()
if not assets:
    st.error("⚠️ Model file not found. Run `python src/train.py` first.")
    st.stop()

model, encoders, scaler = assets
teams  = list(encoders["batting_team"].classes_)
venues = list(encoders["venue"].classes_)

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown("## 🏟️ Match Setup")
    innings   = st.radio("Innings", [1, 2], horizontal=True)
    bat_team  = st.selectbox("🏏 Batting Team", teams)
    bowl_team = st.selectbox("⚡ Bowling Team", [t for t in teams if t != bat_team])
    venue     = st.selectbox("📍 Venue", venues)

    st.markdown("---")
    st.markdown("## 📊 Score Details")
    score   = st.number_input("Current Score", 0, 500, 100)
    wickets = st.slider("Wickets Lost", 0, 10, 3)
    overs   = st.number_input("Overs Completed (e.g. 15.2)", 0.0, 20.0, 15.0, step=0.1)
    target  = st.number_input("Target (Inn 2)", 0, 500, 180) if innings == 2 else 0

    st.markdown("---")
    simulate_btn = st.checkbox("🔁 Show Over-by-Over Simulation")
    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────── COMPUTE ───────────────────────────
seq, crr, rrr, balls_left, runs_req = compute_features(
    innings, score, target, wickets, overs, bat_team, venue, encoders, scaler
)
prob = predict_prob(model, seq, innings, score, target, wickets, balls_left, runs_req)
opp_prob = 1.0 - prob

# Quick derived stats
whole_overs   = int(overs)
balls_bowled  = whole_overs * 6 + int(round((overs - whole_overs) * 10))
overs_left    = (120 - balls_bowled) / 6

# ─────────────────────────── MAIN LAYOUT ───────────────────────────
col_left, col_right = st.columns([6, 4], gap="large")

with col_left:
    # ── STAT CARDS ──
    st.markdown("### 📋 Current Match State")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score",   f"{score}/{wickets}")
    c2.metric("Overs",   f"{overs}")
    c3.metric("CRR",     f"{crr:.2f}")
    if innings == 2:
        c4.metric("RRR",   f"{rrr:.2f}")
        c5.metric("Need",  f"{runs_req} off {balls_left}b")
    else:
        c4.metric("Balls Left", balls_left)
        c5.metric("Proj. Score", f"{int(crr * 20)}")

    st.markdown("")

    # ── WIN PROBABILITY GAUGE ──
    st.markdown("### 🎯 Win Probability")
    fig_gauge = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = round(prob * 100, 1),
        domain = {"x": [0, 1], "y": [0, 1]},
        title  = {"text": f"<b>{bat_team}</b> Win %", "font": {"size": 16, "color": "#ccc"}},
        number = {"suffix": "%", "font": {"size": 34, "color": "#74c69d"}},
        delta  = {"reference": 50, "increasing": {"color": "#52b788"}, "decreasing": {"color": "#e63946"}},
        gauge  = {
            "axis":      {"range": [0, 100], "tickcolor": "#555"},
            "bar":       {"color": "#2d6a4f"},
            "bgcolor":   "rgba(0,0,0,0)",
            "bordercolor": "#555",
            "steps": [
                {"range": [0,  35], "color": "rgba(230,57,70,0.25)"},
                {"range": [35, 65], "color": "rgba(244,162,97,0.20)"},
                {"range": [65,100], "color": "rgba(82,183,136,0.25)"},
            ],
            "threshold": {"line": {"color": "#f4a261", "width": 3}, "value": 50},
        }
    ))
    fig_gauge.update_layout(
        height=280, margin=dict(t=50, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Verdict box
    vc = verdict_class(prob)
    vt = verdict_text(prob, bat_team)
    st.markdown(f'<div class="verdict-box {vc}">{vt}</div>', unsafe_allow_html=True)

    # ── HEAD-TO-HEAD BAR ──
    st.markdown("### ⚔️ Win Probability Comparison")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[bat_team], y=[round(prob * 100, 1)],
        marker_color="#52b788", name=bat_team, text=[f"{prob:.1%}"], textposition="auto",
    ))
    fig_bar.add_trace(go.Bar(
        x=[bowl_team], y=[round(opp_prob * 100, 1)],
        marker_color="#e63946", name=bowl_team, text=[f"{opp_prob:.1%}"], textposition="auto",
    ))
    fig_bar.update_layout(
        yaxis=dict(range=[0, 110], title="Win Probability (%)"),
        xaxis_title="Team", barmode="group",
        height=260, margin=dict(t=20, b=20, l=30, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", legend_font_color="#ccc",
        yaxis_gridcolor="rgba(255,255,255,0.07)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    # ── LOG PREDICTION ──
    if st.button("📌 Log This Prediction", type="primary", use_container_width=True):
        st.session_state.history.append({
            "time":    datetime.now().strftime("%H:%M:%S"),
            "inn":     innings,
            "team":    bat_team,
            "score":   f"{score}/{wickets}",
            "overs":   overs,
            "prob":    prob,
            "target":  target if innings == 2 else "–",
        })
        st.success("Logged ✓")

    # ── QUICK ANALYTICS ──
    st.markdown("### 📈 Match Analytics")
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚡ Momentum Indicators</div>', unsafe_allow_html=True)
        
        # RRR vs CRR
        if innings == 2 and rrr > 0:
            pressure = rrr / crr if crr > 0 else 10
            pressure_pct = min(100, int(pressure * 50))
            diff = rrr - crr
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("CRR", f"{crr:.2f}", delta=None)
            col_m2.metric("RRR", f"{rrr:.2f}", delta=f"{diff:+.2f}", delta_color="inverse")
            st.caption(f"Pressure Index: {pressure:.2f}x  {'🔴 High' if pressure > 1.3 else '🟡 Moderate' if pressure > 1.0 else '🟢 Low'}")
        else:
            proj = int(crr * 20) if crr > 0 else 0
            st.metric("Projected Score", proj)
            st.caption(f"CRR: {crr:.2f} | Balls Remaining: {balls_left}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Wicket Risk
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Resource Usage</div>', unsafe_allow_html=True)
        wkts_used_pct = wickets / 10
        overs_used_pct = balls_bowled / 120
        col_r1, col_r2 = st.columns(2)
        col_r1.metric("Wickets Used", f"{wickets}/10", f"{wkts_used_pct:.0%}")
        col_r2.metric("Overs Used", f"{overs}/20", f"{overs_used_pct:.0%}")
        # Visual resource bar
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            x=["Wickets", "Overs"],
            y=[wkts_used_pct * 100, overs_used_pct * 100],
            marker_color=["#e63946" if wkts_used_pct > 0.7 else "#f4a261" if wkts_used_pct > 0.4 else "#52b788",
                          "#e63946" if overs_used_pct > 0.8 else "#f4a261" if overs_used_pct > 0.5 else "#52b788"],
            text=[f"{wkts_used_pct:.0%}", f"{overs_used_pct:.0%}"],
            textposition="auto",
        ))
        fig_res.update_layout(
            height=160, margin=dict(t=5, b=5, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", showlegend=False,
            yaxis=dict(range=[0, 110], showgrid=False),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SCORE REQUIRED TIMELINE (Inn 2) ──
    if innings == 2 and target > 0:
        st.markdown("### 📉 Score Required Over Time")
        overs_range  = np.arange(int(overs) + 1, 21)
        scores_ahead = [int(target - (crr * ov)) for ov in overs_range]
        scores_req   = [max(0, target - score - int(crr * (ov - overs))) for ov in overs_range]

        fig_req = go.Figure()
        fig_req.add_trace(go.Scatter(
            x=overs_range, y=scores_req, mode="lines+markers",
            line=dict(color="#e63946", width=2),
            name="Runs Still Needed",
            fill="tozeroy", fillcolor="rgba(230,57,70,0.1)",
        ))
        fig_req.add_hline(y=0, line_dash="dash", line_color="#52b788", annotation_text="Target Reached")
        fig_req.update_layout(
            height=200, margin=dict(t=20, b=20, l=30, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", xaxis_title="Over", yaxis_title="Runs Required",
            yaxis_gridcolor="rgba(255,255,255,0.07)",
        )
        st.plotly_chart(fig_req, use_container_width=True)

# ─────────────────────────── SIMULATION (collapsible) ───────────────────────────
if simulate_btn:
    st.markdown("---")
    st.markdown("## 🔁 Over-by-Over Win Probability Simulation")
    st.info("Simulates win probability at each over mark from the current position (keeping wickets and run rate constant).")

    over_marks = list(range(int(overs) + 1, 21))
    probs_sim  = []
    for ov in over_marks:
        seq_s, crr_s, rrr_s, bl_s, rr_s = compute_features(
            innings, int(crr * ov), target, wickets, float(ov), bat_team, venue, encoders, scaler
        )
        p = predict_prob(model, seq_s, innings, int(crr * ov), target, wickets, bl_s, rr_s)
        probs_sim.append(round(p * 100, 1))

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=over_marks, y=probs_sim, mode="lines+markers",
        line=dict(color="#74c69d", width=2.5),
        marker=dict(size=7, color=probs_sim,
                    colorscale=[[0,"#e63946"],[0.5,"#f4a261"],[1,"#52b788"]],
                    showscale=True, colorbar=dict(title="Win %", thickness=12)),
        name=f"{bat_team} Win %",
        hovertemplate="Over %{x}: <b>%{y}%</b><extra></extra>",
    ))
    fig_sim.add_hline(y=50, line_dash="dot", line_color="#f4a261", annotation_text="50% (Even)")
    fig_sim.update_layout(
        height=320, margin=dict(t=20, b=30, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", xaxis_title="Over", yaxis_title="Win Probability (%)",
        yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    # Summary table
    sim_df = pd.DataFrame({"Over": over_marks, f"{bat_team} Win %": probs_sim})
    sim_df[f"{bowl_team} Win %"] = (100 - sim_df[f"{bat_team} Win %"]).round(1)
    st.dataframe(sim_df.set_index("Over"), use_container_width=True)

# ─────────────────────────── HISTORY ───────────────────────────
st.markdown("---")
st.markdown("## 📜 Prediction Log")
if not st.session_state.history:
    st.caption("No predictions logged yet. Use **Log This Prediction** to record snapshots.")
else:
    hist_df = pd.DataFrame(st.session_state.history)
    hist_df["prob_pct"] = hist_df["prob"].apply(lambda p: f"{p:.1%}")
    hist_df["verdict"] = hist_df["prob"].apply(
        lambda p: "✅ Favourite" if p >= 0.65 else ("⚠️ Underdog" if p <= 0.35 else "⚖️ Even")
    )
    display_cols = ["time", "inn", "team", "score", "overs", "target", "prob_pct", "verdict"]
    st.dataframe(hist_df[display_cols].rename(columns={
        "time":"Time","inn":"Inn","team":"Batting","score":"Score",
        "overs":"Overs","target":"Target","prob_pct":"Win %","verdict":"Verdict"
    }), use_container_width=True, hide_index=True)

    # Win probability trend across logged predictions
    if len(st.session_state.history) >= 2:
        st.markdown("#### 📈 Win Probability Trend")
        trend_data = [
            {"Log #": i+1, "Batting": r["team"], "Win %": r["prob"]*100}
            for i, r in enumerate(st.session_state.history)
        ]
        fig_trend = px.line(
            pd.DataFrame(trend_data), x="Log #", y="Win %", color="Batting",
            markers=True, color_discrete_sequence=["#74c69d","#e63946","#f4a261","#4cc9f0"],
        )
        fig_trend.add_hline(y=50, line_dash="dot", line_color="#aaa")
        fig_trend.update_layout(
            height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", yaxis=dict(range=[0,105], gridcolor="rgba(255,255,255,0.07)"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ─────────────────────────── FOOTER ───────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#555;'>Cricket Win Predictor Pro • LSTM Neural Network • T20 format • "
    "Model predictions are probabilistic estimates, not guaranteed outcomes.</small>",
    unsafe_allow_html=True,
)