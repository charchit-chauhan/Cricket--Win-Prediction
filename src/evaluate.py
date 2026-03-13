import os
import torch
import numpy as np
from model import LSTMWinPredictor

# Use absolute path or relative path based on your project structure
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_cricket_win_predictor.pt")

def evaluate_model():
    print(f"🔍 Loading model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found. Please run train.py first.")
        return

    # 1. Load Checkpoint
    # weights_only=False is used because we are loading custom objects like Scalers/Encoders
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # 2. Extract Metadata
    features = checkpoint.get("feature_cols", [])
    if not features:
        # Fallback for older versions of your training script
        features = checkpoint.get("features", [])
        
    print(f"✅ Model loaded successfully.")
    print(f"📊 Features used during training: {len(features)}")
    print(f"📋 Feature List: {', '.join(features)}")

    # 3. Initialize and Load Model Architecture
    model = LSTMWinPredictor(input_dim=len(features))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("🧠 Model architecture initialized and weights loaded.")

    # 4. Perform a "Smoke Test" Prediction
    # Create a dummy sequence: (batch_size=1, sequence_length=12, input_dim)
    dummy_input = torch.randn(1, 12, len(features))
    
    with torch.no_grad():
        logits = model(dummy_input)
        # Apply sigmoid to convert logit to probability
        probability = torch.sigmoid(logits).item()
    
    print(f"\n🚀 Smoke Test Result:")
    print(f"   Input Shape: {dummy_input.shape}")
    print(f"   Win Probability: {probability:.2%}")
    print("\n✅ EVALUATION SUCCESS: The model is ready for app.py!")

if __name__ == "__main__":
    evaluate_model()

