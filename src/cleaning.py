import kagglehub
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from features import (
    get_match_result,
    goal_diff,
    add_position_group
)

# Loading Datasets
path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
path2 = kagglehub.dataset_download("akshyakumarkc/fifa-25-player-ratings")
dataset_path = Path(path)
results = pd.read_csv(dataset_path / "results.csv")
former_names = pd.read_csv(dataset_path / "former_names.csv")
players = pd.read_csv(Path(path2) / "players_info.csv")
BASE_DIR = Path(__file__).resolve().parent.parent
wc2026_draw = pd.read_csv(BASE_DIR / "data" / "wc2026_draw.csv")
elo_ratings = pd.read_csv(BASE_DIR / "data" /"fifa_ranking-2026-04-01.csv")

# List of countries in WC2026
wc2026_countries = wc2026_draw["country"].unique().tolist()

# Country mapping to fix inconsistencies in country names between datasets
country_mapping = {
    "Holland": "Netherlands",
    "Côte d'Ivoire": "Ivory Coast",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde"
}

def clean_results(df):
    df = df.copy()
    df = df.drop_duplicates()

    # Converting date column from string to datetime format
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Removing rows that have NA values in critical columns
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

    # Converting home_score and away_score to integers
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Map former country names to current names
    former_names_mapping = dict(zip(former_names["former"], former_names["current"]))
    df["home_team"] = df["home_team"].replace(former_names_mapping)
    df["away_team"] = df["away_team"].replace(former_names_mapping)

    # Map country names to ensure consistency across datasets
    df["home_team"] = df["home_team"].replace(country_mapping)
    df["away_team"] = df["away_team"].replace(country_mapping)

    # Adding match result and goal difference features
    df["match_result"] = df.apply(get_match_result, axis=1)
    df["goal_diff"] = df.apply(goal_diff, axis=1)

    return df

def clean_players(df):
    df = df.copy()
    df = df.drop_duplicates()

    # Filter for male players only
    df = df[df["gender"] == "M"]

    # Map country names
    df["nationality"] = df["nationality"].replace(country_mapping)

    # Filter for players from WC2026 countries
    df = df[df["nationality"].isin(wc2026_countries)]

    # Add position group feature
    df = add_position_group(df)

    return df

def clean_wc2026_draw(df):
    df = df.copy()
    df = df.drop_duplicates()

    # Remove leading/trailing whitespace as I made this csv manually
    df["country"] = df["country"].str.strip() 
    df["group"] = df["group"].str.strip()

    return df

def clean_elo(df):
    df = df.copy()
    df = df.drop_duplicates()

    # Convert rank_date to datetime
    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")

    # Map country names
    df["country_full"] = df["country_full"].replace(country_mapping)

    # Filter for participating countries
    df = df[df["country_full"].isin(wc2026_countries)]

    return df

# Create and save cleaned datasets
results = clean_results(results)
players = clean_players(players)
wc2026_draw = clean_wc2026_draw(wc2026_draw)
elo = clean_elo(elo_ratings)
results.to_csv(BASE_DIR / "data" / "processed" / "results_clean.csv", index=False)
players.to_csv(BASE_DIR / "data" / "processed" / "players_clean.csv", index=False)
wc2026_draw.to_csv(BASE_DIR / "data" / "processed" / "wc2026_draw_clean.csv", index=False)
elo.to_csv(BASE_DIR / "data" / "processed" / "elo_clean.csv", index=False)