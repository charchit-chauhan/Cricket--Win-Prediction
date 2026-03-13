# src/feature_engineering.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.preprocessing import LabelEncoder


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True)

    # runs for this ball
    df["runs_this_ball"] = df["total_runs"].fillna(0)

    # cumulative score per innings
    df["cumulative_score"] = (
        df.groupby(["match_id", "innings"])["runs_this_ball"].cumsum()
    )

    # ball index within innings (0..119)
    df["ball_index"] = df["over"] * 6 + (df["ball"] - 1)
    df["balls_bowled_in_innings"] = df.groupby(
        ["match_id", "innings"]
    )["ball_index"].rank(method="first").astype(int)

    max_balls = 120  # T20
    df["balls_remaining"] = max_balls - df["balls_bowled_in_innings"]

    # cumulative wickets
    df["wicket"] = df["wicket"].fillna(0).astype(int)
    df["wickets_down"] = df.groupby(["match_id", "innings"])["wicket"].cumsum()
    df["wickets_remaining"] = 10 - df["wickets_down"]

    # innings 1 total
    inn1_score = (
        df[df["innings"] == 1]
        .groupby("match_id")["runs_this_ball"]
        .sum()
        .rename("innings1_total")
    )
    df = df.merge(inn1_score, on="match_id", how="left")
    df["target"] = df["innings1_total"] + 1

    df["runs_required"] = np.where(
        df["innings"] == 2,
        df["target"] - df["cumulative_score"],
        np.nan,
    )

    df["overs_faced"] = df["balls_bowled_in_innings"] / 6.0
    df["current_run_rate"] = np.where(
        df["overs_faced"] > 0,
        df["cumulative_score"] / df["overs_faced"],
        0.0,
    )

    df["required_run_rate"] = np.where(
        (df["innings"] == 2) & (df["balls_remaining"] > 0),
        df["runs_required"] / (df["balls_remaining"] / 6.0),
        np.nan,
    )

    # recent runs windows
    for n in [5, 10]:
        df[f"recent_runs_{n}"] = (
            df.groupby(["match_id", "innings"])["runs_this_ball"]
            .rolling(window=n, min_periods=1)
            .sum()
            .reset_index(level=[0, 1], drop=True)
        )

    return df


def add_result_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    result = 1 if second-innings batting team wins, else 0.
    """
    df = df.copy()

    final_score = (
        df.groupby(["match_id", "innings"])["cumulative_score"]
        .max()
        .reset_index()
        .rename(columns={"cumulative_score": "final_score"})
    )

    inn1 = final_score[final_score["innings"] == 1][["match_id", "final_score"]]
    inn2 = final_score[final_score["innings"] == 2][["match_id", "final_score"]]

    inn1 = inn1.rename(columns={"final_score": "innings1_score"})
    inn2 = inn2.rename(columns={"final_score": "innings2_score"})

    res = pd.merge(inn1, inn2, on="match_id", how="inner")

    res["result"] = (res["innings2_score"] >= res["innings1_score"] + 1).astype(int)
    df = df.merge(res[["match_id", "result"]], on="match_id", how="left")

    return df


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: Dict[str, LabelEncoder] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    if encoders is None:
        encoders = {}

    cat_cols = ["batting_team", "venue"]

    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
            
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # FIX: Handle unseen values safely
            df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    return df, encoders


def build_sequences(df, feature_cols, target_col, window_size=5):
    X_seqs = []
    y_labels = []

    # Group by match to ensure we don't mix balls from different games
    for match_id, group in df.groupby('match_id'):
        features = group[feature_cols].values
        target = group[target_col].iloc[-1] # Final result of the match

        # Only create a sequence if we have enough data points
        if len(features) >= window_size:
            # Taking the last 'window_size' balls to represent the current state
            X_seqs.append(features[-window_size:]) 
            y_labels.append(target)

    # --- THE FIX ---
    if not X_seqs:
        raise ValueError(f"No sequences could be built. Check if feature_cols {feature_cols} exist and if the dataframe is empty.")
    
    return np.array(X_seqs, dtype=np.float32), np.array(y_labels, dtype=np.float32)
