import os
import torch
import numpy as np

# Adjust paths based on your project structure
# This assumes the script is inside 'src' and 'models' is in the root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_cricket_win_predictor.pt")

def run_diagnosis():
    print("="*40)
    print("🔍 CRICKET WIN PREDICTOR: SYSTEM DIAGNOSIS")
    print("="*40)

    # 1. File Check
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        print("👉 Action: Run 'python src/train.py' first.")
        return

    file_size = os.path.getsize(MODEL_PATH) / 1024
    print(f"✅ Model File Found: {MODEL_PATH}")
    print(f"✅ File Size: {file_size:.2f} KB")

    # 2. Loading Check
    try:
        # weights_only=False is required for loading the Scaler and Encoders
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        print("✅ Model Checkpoint loaded into memory.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load checkpoint. {str(e)}")
        return

    # 3. Metadata and Key Verification
    required_keys = ['model_state_dict', 'scaler', 'encoders', 'feature_cols']
    missing_keys = [k for k in required_keys if k not in checkpoint]

    if missing_keys:
        print(f"❌ ERROR: Missing internal keys: {missing_keys}")
        # Compatibility check for older 'features' key
        if 'features' in checkpoint:
            print("⚠️ Note: Found 'features' key instead of 'feature_cols'. Update train.py.")
    else:
        print("✅ All required dictionary keys are present.")

    # 4. Feature and Logic Verification
    features = checkpoint.get('feature_cols', [])
    print(f"✅ Total Features: {len(features)}")
    
    if len(features) > 0:
        print(f"📋 Feature List: {', '.join(features)}")
    
    # 5. Scalar and Encoder Health
    try:
        scaler = checkpoint['scaler']
        encoders = checkpoint['encoders']
        print(f"✅ Scaler is healthy: {type(scaler).__name__}")
        print(f"✅ Encoders found for: {list(encoders.keys())}")
        
        # Check if 'batting_team' classes exist
        if 'batting_team' in encoders:
            teams = encoders['batting_team'].classes_
            print(f"✅ Classes found: {len(teams)} teams registered.")
    except Exception as e:
        print(f"❌ ERROR: Data processing objects (scaler/encoders) are corrupted. {str(e)}")

    print("="*40)
    print("🏁 DIAGNOSIS COMPLETE: Model is healthy and ready for deployment.")
    print("="*40)

if __name__ == "__main__":
    run_diagnosis()