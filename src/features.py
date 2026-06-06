import pandas as pd
import numpy as np

# Match Result -- Target Feature
def get_match_result(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"

# Goal Difference
def goal_diff(row):
    return row["home_score"] - row["away_score"]

# Dictionary of position mappings to group similar positions together
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

# Function to standardise and simplify position names
def add_position_group(df):
    df = df.copy()

    df["position_group"] = (
        df["position"]
        .map(position_mapping)
    )

    return df

####################################
## Squad Rating and Top 11 Rating ##
####################################

# Create Ratings
def team_rating(players, requirements, rating_name, fill_rating=50):
    
    total_requirements = sum(requirements.values()) 

    team_ratings = []

    countries = players["nationality"].unique()

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
                top_players = pos_players.nlargest(n_players, "overall")
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
        total_rating = squad["overall"].sum() + (missing_players * fill_rating) # add fill_rating for missing players
        avg_rating = total_rating / total_requirements # calculate average

        team_ratings.append({"country": country, "avg_rating": avg_rating, "team_size": len(squad), "missing_players": missing_players})

    return pd.DataFrame(team_ratings)

# Add Ratings
def add_ratings(df, squad_ratings, top11_ratings):
    df = df.merge(
    squad_ratings[["country", "squad_rating"]], 
    left_on="home_team", 
    right_on="country", 
    how="left").rename(columns={"squad_rating": "home_squad_rating"}).drop(columns=["country"])

    df = df.merge(
        squad_ratings[["country", "squad_rating"]], 
        left_on="away_team", 
        right_on="country", 
        how="left").rename(columns={"squad_rating": "away_squad_rating"}).drop(columns=["country"])

    df = df.merge(
        top11_ratings[["country", "top11_rating"]], 
        left_on="home_team", 
        right_on="country", 
        how="left").rename(columns={"top11_rating": "home_top11_rating"}).drop(columns=["country"])

    df = df.merge(
        top11_ratings[["country", "top11_rating"]], 
        left_on="away_team", 
        right_on="country", 
        how="left").rename(columns={"top11_rating": "away_top11_rating"}).drop(columns=["country"])

    return df

#######################
## Tournament Weight ##
#######################

# Create tournament weights
def assign_tournament_weight(tournament):
    if tournament == "FIFA World Cup":
        return 5
    elif tournament in ["UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup"]:
        return 4
    elif tournament == "FIFA World Cup qualification":
        return 3
    elif tournament == "Friendly":
        return 1
    else:
        return 2

# Add tournament weight
def add_tournament_weight(df):
    df["tournament_weight"] = df["tournament"].apply(assign_tournament_weight)
    return df

###########################
## Home Team Host Status ##
###########################
def add_host_advantage(df):

    df["home_is_host"] = (
        (df["home_team"] == df["host_country"]) |
        (df["away_team"] == df["host_country"])
    ).astype(int)

    return df

##################
## Elo Features ##
##################

def add_elo_features(df, elo_ratings_wc2026):
    # Adding home and away elo ratings to  dataframe
    df = df.merge(
        elo_ratings_wc2026[["country_full", "total_points"]],
        left_on="home_team",
        right_on="country_full",
        how="left"
    ).rename(columns={"total_points": "home_elo"}).drop(columns=["country_full"])

    df = df.merge(
        elo_ratings_wc2026[["country_full", "total_points"]],
        left_on="away_team",
        right_on="country_full",
        how="left"
    ).rename(columns={"total_points": "away_elo"}).drop(columns=["country_full"])

    # Calculating elo difference and absolute elo difference
    df["elo_diff"] = df["home_elo"] - df["away_elo"] 
    df["abs_elo_diff"] = df["elo_diff"].abs()

    # Remove home_elo and away_elo as they are not needed for the model
    df.drop(columns=["home_elo", "away_elo"], inplace=True) 

    return df

##########################
## Recent Form Features ##
##########################

def add_recent_stats(df, latest_team_features):
    # Merging the latest team features with the fixtures dataframe for home teams
    df = df.merge(
        latest_team_features,
        left_on="home_team",
        right_on="team",
        how="left"
    )

    # Renaming columns to be specific to home team and dropping redundant columns
    df = df.rename(columns={
        "recent_win_rate": "home_recent_win_rate",
        "recent_draw_rate": "home_recent_draw_rate",
        "avg_goals_last10": "home_avg_goals_last10",
        "avg_conceded_last10": "home_avg_conceded_last10"
    })
    df.drop(columns=["team", "date_y"], errors="ignore", inplace=True)

    # Merging the latest team features with the dataframe for away teams
    df = df.merge(
        latest_team_features,
        left_on="away_team",
        right_on="team",
        how="left"
    )

    # Renaming columns to be specific to away team and dropping redundant columns
    df = df.rename(columns={
        "recent_win_rate": "away_recent_win_rate",
        "recent_draw_rate": "away_recent_draw_rate",
        "avg_goals_last10": "away_avg_goals_last10",
        "avg_conceded_last10": "away_avg_conceded_last10"
    })
    df.drop(columns=["team", "date"], errors="ignore", inplace=True)

    # Rename date_x back to date
    df.rename(columns={"date_x": "date"}, inplace=True)
    
    # Add difference features
    df["recent_form_diff"] = df["home_recent_win_rate"] - df["away_recent_win_rate"]
    df["abs_recent_form_diff"] = df["recent_form_diff"].abs()
    df["recent_draw_diff"] = df["home_recent_draw_rate"] - df["away_recent_draw_rate"]
    df["abs_recent_draw_diff"] = df["recent_draw_diff"].abs()
    df["diff_in_avg_goals"] = df["home_avg_goals_last10"] - df["away_avg_goals_last10"]
    df["diff_in_avg_conceded"] = df["home_avg_conceded_last10"] - df["away_avg_conceded_last10"]

    return df

############
## Master ##
############

def create_match_features(df, squad_ratings, top11_ratings, elo_ratings_wc2026, latest_team_features):

    df = add_ratings(
        df,
        squad_ratings,
        top11_ratings
    )

    df = add_host_advantage(df)

    df = add_elo_features(
        df,
        elo_ratings_wc2026
    )

    df = add_recent_stats(
        df,
        latest_team_features
    )

    return df