# Cricket Win Predictor

End-to-end deep learning project to predict win probability from ball-by-ball data using an LSTM on Cricsheet data. [file:1][web:2][web:5]

## Steps

1. Download Cricsheet data and convert to `data/processed/ball_by_ball.csv`.
2. Update column names in `data_pipeline.py` and `feature_engineering.py` if needed.
3. Train model:
   ```bash
   cd src
   python train.py

### Evaluate:

python evaluate.py



### Run Streamlit app:

cd ..
streamlit run app/streamlit_app.py



***

If you tell me which exact Cricsheet format you will use (YAML or which CSV variant), I can adjust the `load_ball_by_ball` + feature code to match those exact columns.[3][5][2]

