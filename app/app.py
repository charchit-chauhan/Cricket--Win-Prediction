import streamlit as st
import torch
import numpy as np
import os
import sys

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_path = os.path.join(root_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from model import LSTMWinPredictor

@st.cache_resource
def load_assets():
    # Try multiple common paths to ensure the model is found
    possible_paths = [
        os.path.join(root_dir, "models", "lstm_cricket_win_predictor.pt"),
        os.path.join(os.getcwd(), "models", "lstm_cricket_win_predictor.pt")
    ]
    
    path = next((p for p in possible_paths if os.path.exists(p)), None)
    if not path: return None

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    # Use 'feature_cols' to match our train.py update
    input_dim = len(ckpt.get('feature_cols', ckpt.get('features', [])))
    model = LSTMWinPredictor(input_dim=input_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['encoders'], ckpt['scaler']

def main():
    st.set_page_config(page_title="Cricket Predictor Pro", layout="centered")
    st.title("🏏 Cricket Win Predictor")
    
    assets = load_assets()
    if not assets:
        st.error("Model file not found! Please run 'python src/train.py' first.")
        return
    model, encoders, scaler = assets

    # --- SIDEBAR: MATCH CONTEXT ---
    with st.sidebar:
        st.header("Match Info")
        innings = st.radio("Innings", [1, 2])
        bat_team = st.selectbox("Batting Team", encoders['batting_team'].classes_)
        venue = st.selectbox("Venue", encoders['venue'].classes_)

    # --- MAIN INPUTS ---
    col1, col2 = st.columns(2)
    score = col1.number_input("Current Score", 0, 500, 100)
    wickets = col1.slider("Wickets Lost", 0, 10, 3)
    overs = col2.number_input("Overs (e.g. 15.2)", 0.0, 20.0, 15.0)
    target = col2.number_input("Target", 0, 500, 200) if innings == 2 else 0

    if st.button("Predict Probability"):
        
        whole_overs = int(overs)
        balls_in_over = int(round((overs - whole_overs) * 10))
        balls_bowled = (whole_overs * 6) + min(balls_in_over, 6)
        balls_left = max(0, 120 - balls_bowled)
        runs_req = max(0, target - score) if innings == 2 else 0
        
        crr = (score * 6 / balls_bowled) if balls_bowled > 0 else 0
        rrr = (runs_req * 6 / balls_left) if (innings == 2 and balls_left > 0) else 0

       
        
        raw = np.array([[
            float(innings), float(score), float(runs_req), float(balls_left), 
            float(10-wickets), float(crr), float(rrr), 40.0, 80.0,
            encoders['batting_team'].transform([bat_team])[0],
            encoders['venue'].transform([venue])[0]
        ]], dtype=np.float32)

        
        scaled = scaler.transform(raw)
        seq = np.repeat(scaled[np.newaxis, :, :], 12, axis=1) 
        
        with torch.no_grad():
            logit = model(torch.FloatTensor(seq)).item()
            
            prob = torch.sigmoid(torch.tensor(logit)).item()

        
        if innings == 2:
            if score >= target: prob = 1.0
            elif wickets >= 10: prob = 0.0
            elif balls_left <= 0 and score < target: prob = 0.0
            elif (target - score) > (balls_left * 6): prob = 0.0 

        
        st.subheader("Win Probability")
        st.metric(f"{bat_team} Win Chance", f"{prob:.1%}")
        st.progress(prob)
        
        if prob > 0.8: st.success(f"High confidence for {bat_team}!")
        elif prob < 0.2: st.error(f"Looking tough for {bat_team}...")

if __name__ == "__main__":
    main()