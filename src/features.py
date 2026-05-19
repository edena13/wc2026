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

def team_rating(players, requirements, rating_name, fill_rating=50):
    
    total_requirements = sum(requirements.values()) 

    team_ratings = []

    countries = players["nationality"].unique()
    countries = np.append(countries, "Qatar") # Add Qatar as they are missing from the players dataset

    for country in countries:
        country_players = players[players["nationality"] == country]
        selected_players = []

        for pos, n_players in requirements.items():
            pos_players = country_players[country_players["position_group"] == pos]
            if len(pos_players) == 0:
                continue
            elif len(pos_players) < n_players:
                top_players = pos_players
            else:
                top_players = pos_players.nlargest(n_players, "ovr")
            selected_players.append(top_players)
        
        # Handling countries with zero rated players
        if len(selected_players) == 0:
            team_ratings.append({
                "country": country,
                "avg_rating": fill_rating,
                "team_size": 0,
                "missing_players": total_requirements
            })

            continue

        squad = pd.concat(selected_players) # combine the selected players for each position into a single df

        missing_players = total_requirements - len(squad)
        total_rating = squad["ovr"].sum() + (missing_players * fill_rating) # add fill_rating for missing players
        avg_rating = total_rating / total_requirements # calculate average

        team_ratings.append({"country": country, "avg_rating": avg_rating, "team_size": len(squad), "missing_players": missing_players})

    return pd.DataFrame(team_ratings)