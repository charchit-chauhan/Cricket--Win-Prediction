import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Internal project imports
from data_pipeline import load_ball_by_ball, train_val_test_split_matches
from feature_engineering import add_basic_features, encode_categoricals
from model import LSTMWinPredictor

# --- Configuration ---
WINDOW_SIZE = 12 
BATCH_SIZE = 64  
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_cricket_win_predictor.pt")

def build_sequences_efficient(df, feature_cols, target_col, window=WINDOW_SIZE):
    """
    Creates overlapping sequences of length 'window'.
    Uses the final match outcome as the target for every ball in that match.
    """
    X, y = [], []
    
    # Group by match_id to prevent sequences from bridging different games
    for _, group in df.groupby('match_id'):
        # Ensure temporal order for LSTM
        group = group.sort_values(['innings', 'over', 'ball']) 
        
        if len(group) < window: 
            continue
        
        feat_vals = group[feature_cols].values.astype(np.float32)
        
        # Use the final result of the match as the label for all sequences in this match
        match_result = group[target_col].iloc[-1] 
        
        # Sliding window including the very last ball of the match
        for i in range(window, len(feat_vals) + 1):
            X.append(feat_vals[i-window:i])
            y.append(match_result)
            
    return np.array(X), np.array(y)

def train():
    # 1. Setup Environment
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 2. Load and Preprocess Data
    df = load_ball_by_ball()
    
    # Determine if the batting team on each ball eventually won
    df['result'] = (df['batting_team'] == df['winner']).astype(int)
    
    # Engineering cumulative scores, wickets, and run rates
    df = add_basic_features(df)
    
    # Clean data: Replace infinities from RRR calculations with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Label Encoding for Teams and Venues
    df, encoders = encode_categoricals(df, fit=True)
    
    # Define feature list - Order must be identical in app.py
    features = [
        "innings", "cumulative_score", "runs_required", "balls_remaining", 
        "wickets_remaining", "current_run_rate", "required_run_rate", 
        "recent_runs_5", "recent_runs_10", "batting_team", "venue"
    ]

    # 3. Scaling
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # 4. Split and Sequence Building
    train_df, _, _ = train_val_test_split_matches(df)
    X_train, y_train = build_sequences_efficient(train_df, features, "result")

    # Convert to Tensors for PyTorch
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Model Initialization
    model = LSTMWinPredictor(input_dim=len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss() 

    print(f"🚀 Training on {len(X_train)} sequences...")
    model.train()
    
    # 6. Training Loop
    for epoch in range(10):
        total_loss = 0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            outputs = model(b_x)
            
            # FIX: Cast b_y (Long) to float() to match model output type (Float)
            loss = criterion(outputs, b_y.float()) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/10 | Avg Loss: {total_loss/len(train_loader):.4f}")

    # 7. Save Checkpoint
    # 'feature_cols' key is used to satisfy evaluate.py and diagnose.py requirements
    torch.save({
        'model_state_dict': model.state_dict(), 
        'scaler': scaler, 
        'encoders': encoders, 
        'feature_cols': features, 
        'window_size': WINDOW_SIZE
    }, MODEL_PATH)
    
    print(f"✅ Success! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()