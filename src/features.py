import pandas as pd
import numpy as np

def get_match_result(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"

def goal_diff(row):
    return row["home_score"] - row["away_score"]

position_mapping = {
    "GK": "GK",
    "LB": "DEF", 
    "CB": "DEF", 
    "RB": "DEF",
    "CDM": "MID", 
    "CM": "MID",
    "CAM": "MID",
    "LM": "MID", 
    "RM": "MID",
    "LW": "FWD", 
    "RW": "FWD",
    "ST": "FWD"
}

def add_position_group(df):
    df = df.copy()

    df["position_group"] = (
        df["position"]
        .map(position_mapping)
    )

    return df

def squad_rating(players):
    
    squad_requirements = {
        "GK": 3,
        "DEF": 9,
        "MID": 9,
        "FWD": 5
    }

    team_ratings = []

    for country in players["nationality"].unique():
        country_players = players[players["nationality"] == country]
        selected_players = []

        for pos, n_players in squad_requirements.items():
            pos_players = country_players[country_players["position_group"] == pos]
            if len(pos_players) == 0:
                continue
            elif len(pos_players) != 0 and len(pos_players) < n_players:
                top_players = pos_players
            else:
                top_players = pos_players.nlargest(n_players, "ovr")
            selected_players.append(top_players)
        
        if len(selected_players) == 0:
            continue

        squad = pd.concat(selected_players) # combine the selected players for each position into a single df
        squad_rating = squad["ovr"].mean() # calculate average

        team_ratings.append({"country": country, "squad_rating": squad_rating, "squad_size": len(squad)})

    return pd.DataFrame(team_ratings)