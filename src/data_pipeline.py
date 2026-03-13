

import os
import glob
import yaml
import pandas as pd
from typing import Tuple

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_YAML_DIR = os.path.join(BASE_DIR, "data", "raw", "yaml")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "ball_by_ball_from_yaml.csv")


def parse_yaml_match(path: str) -> pd.DataFrame:
    """
    Parse a single Cricsheet YAML file into a ball-by-ball DataFrame.
    Structure based on https://cricsheet.org/format/yaml/.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    info = data.get("info", {})
    match_id = info.get("match_id", os.path.basename(path))
    venue = info.get("venue", "")
    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    winner = outcome.get("winner", None)

    rows = []

    # innings is list like: [{"1st innings": {...}}, {"2nd innings": {...}}, ...]
    for innings_index, inn in enumerate(data.get("innings", []), start=1):
        innings_name = list(inn.keys())[0]
        inn_data = inn[innings_name]
        batting_team = inn_data.get("team", "")

        deliveries = inn_data.get("deliveries", [])
        for delivery in deliveries:
            ball_key = list(delivery.keys())[0]
            ball_data = delivery[ball_key]

            # ball_key is like 0.1, 19.6 etc.
            over_str, ball_str = str(ball_key).split(".")
            over = int(over_str)
            ball = int(ball_str)

            bowler = ball_data.get("bowler", "")
            batter = ball_data.get("batter", ball_data.get("batsman", ""))
            non_striker = ball_data.get("non_striker", "")

            runs = ball_data.get("runs", {})
            runs_batter = runs.get("batter", 0)
            extras = runs.get("extras", 0)
            total_runs = runs.get("total", runs_batter + extras)

            wicket_flag = 0
            if "wickets" in ball_data and ball_data["wickets"]:
                wicket_flag = 1

            row = {
                "match_id": match_id,
                "innings": innings_index,
                "innings_name": innings_name,
                "over": over,
                "ball": ball,
                "batting_team": batting_team,
                "bowler": bowler,
                "batter": batter,
                "non_striker": non_striker,
                "venue": venue,
                "runs_off_bat": runs_batter,
                "extras": extras,
                "total_runs": total_runs,
                "wicket": wicket_flag,
                "winner": winner,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_ball_by_ball_from_yaml() -> None:
    """
    Read all YAMLs from RAW_YAML_DIR and save combined CSV.
    """
    yaml_files = glob.glob(os.path.join(RAW_YAML_DIR, "*.yaml"))
    print("YAML files found:", len(yaml_files))

    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found in {RAW_YAML_DIR}")

    dfs = []
    for path in yaml_files:
        print("Parsing:", os.path.basename(path))
        dfs.append(parse_yaml_match(path))

    full_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    full_df.to_csv(PROCESSED_PATH, index=False)
    print("Saved combined CSV to:", PROCESSED_PATH)


def load_ball_by_ball(path: str = PROCESSED_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run build_ball_by_ball_from_yaml() first."
        )
    return pd.read_csv(path)


def train_val_test_split_matches(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by match_id.
    """
    match_col = "match_id"
    if match_col not in df.columns:
        raise KeyError(f"'{match_col}' not in columns: {df.columns.tolist()}")

    match_ids = df[match_col].drop_duplicates().sample(frac=1.0, random_state=random_state)
    n = len(match_ids)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_ids = match_ids.iloc[:n_train]
    val_ids = match_ids.iloc[n_train:n_train + n_val]
    test_ids = match_ids.iloc[n_train + n_val:]

    train_df = df[df[match_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[match_col].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[match_col].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    build_ball_by_ball_from_yaml()
