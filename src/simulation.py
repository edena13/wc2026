# Import libraries and functions
import pandas as pd
import numpy as np
from .features import create_match_features

ROUND_ORDER = ["R32", "R16", "QF", "SF", "3P", "F"]


def simulate_score(result, home_attack, away_attack, rng=None):
    """Simulate a scoreline that is consistent with the sampled match result."""
    rng = rng or np.random.default_rng()
    home_attack = max(float(home_attack), 0.05)
    away_attack = max(float(away_attack), 0.05)

    while True:
        home_goals = rng.poisson(home_attack)
        away_goals = rng.poisson(away_attack)

        if result == "home_win" and home_goals > away_goals:
            return home_goals, away_goals
        if result == "away_win" and away_goals > home_goals:
            return home_goals, away_goals
        if result == "draw" and home_goals == away_goals:
            return home_goals, away_goals


def simulate_group_stage_matches(fixtures, model, feature_cols, adjust_probs_with_squad, rng=None):
    """Sample group-stage results and scorelines for one simulated tournament."""
    rng = rng or np.random.default_rng()
    fixtures = fixtures.copy()

    results = []
    home_goals = []
    away_goals = []

    for _, row in fixtures.iterrows():
        x = row[feature_cols].to_frame().T
        probs = model.predict_proba(x)[0]
        probs = adjust_probs_with_squad(
            probs,
            row["home_top11_rating"],
            row["away_top11_rating"],
        )

        result = rng.choice(["away_win", "draw", "home_win"], p=probs)
        hg, ag = simulate_score(
            result,
            row["home_avg_goals_last10"],
            row["away_avg_goals_last10"],
            rng=rng,
        )

        results.append(result)
        home_goals.append(hg)
        away_goals.append(ag)

    fixtures["sim_result"] = results
    fixtures["sim_home_goals"] = home_goals
    fixtures["sim_away_goals"] = away_goals

    return fixtures


def build_group_standings(fixtures, wc2026_draw, fifa_rank):
    """Create group standings from simulated fixtures."""
    standings = pd.DataFrame({
        "country": wc2026_draw["country"].unique(),
        "points": 0,
        "gf": 0,
        "ga": 0,
    })

    for _, row in fixtures.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hg = int(row["sim_home_goals"])
        ag = int(row["sim_away_goals"])

        standings.loc[standings["country"] == home, ["gf", "ga"]] += [hg, ag]
        standings.loc[standings["country"] == away, ["gf", "ga"]] += [ag, hg]

        if hg > ag:
            standings.loc[standings["country"] == home, "points"] += 3
        elif ag > hg:
            standings.loc[standings["country"] == away, "points"] += 3
        else:
            standings.loc[standings["country"].isin([home, away]), "points"] += 1

    standings["gd"] = standings["gf"] - standings["ga"]
    standings = standings.merge(wc2026_draw[["country", "group"]], on="country", how="left")
    standings["fifa_rank"] = standings["country"].map(fifa_rank)

    missing_rank = standings.loc[standings["fifa_rank"].isna(), "country"].tolist()
    if missing_rank:
        raise ValueError(f"Missing FIFA rankings for: {missing_rank}")

    # Lower FIFA rank is better, so fifa_rank sorts ascending.
    standings = standings.sort_values(
        ["group", "points", "gd", "gf", "fifa_rank"],
        ascending=[True, False, False, False, True],
    )
    standings["position"] = standings.groupby("group").cumcount() + 1
    standings["position_group"] = standings["position"].astype(str) + standings["group"]

    return standings[[
        "group",
        "country",
        "position",
        "position_group",
        "points",
        "gf",
        "ga",
        "gd",
        "fifa_rank",
    ]]


def get_qualifiers(standings):
    """Return group winner, runner-up, and best third-place dictionaries."""
    first_place = (
        standings[standings["position"] == 1]
        .set_index("group")["country"]
        .to_dict()
    )
    second_place = (
        standings[standings["position"] == 2]
        .set_index("group")["country"]
        .to_dict()
    )

    best_third = (
        standings[standings["position"] == 3]
        .sort_values(
            ["points", "gd", "gf", "fifa_rank"],
            ascending=[False, False, False, True],
        )
        .head(8)
    )
    third_place = best_third.set_index("group")["country"].to_dict()

    return first_place, second_place, third_place, best_third


def get_third_mapping(best_third, mapping):
    """Look up FIFA's R32 third-place assignment for the simulated groups."""
    combo_key = "".join(sorted(best_third["group"].values))
    matchup_row = mapping[mapping["combo"] == combo_key]

    if matchup_row.empty:
        raise ValueError(f"No round-of-32 mapping found for combo: {combo_key}")

    matchup_row = matchup_row.iloc[0]
    return {
        "THIRD_A": matchup_row["1A"],
        "THIRD_B": matchup_row["1B"],
        "THIRD_D": matchup_row["1D"],
        "THIRD_E": matchup_row["1E"],
        "THIRD_G": matchup_row["1G"],
        "THIRD_I": matchup_row["1I"],
        "THIRD_K": matchup_row["1K"],
        "THIRD_L": matchup_row["1L"],
    }


def resolve_team_source(
    source,
    first_place,
    second_place,
    third_place,
    third_mapping,
    winners,
    losers,
):
    """Resolve a bracket source like 1A, THIRD_A, W89, or L101 to a team."""
    source = str(source)

    if source.startswith("1"):
        return first_place[source[1:]]
    if source.startswith("2"):
        return second_place[source[1:]]
    if source.startswith("THIRD"):
        third_slot = third_mapping[source]
        group = third_slot[1]
        return third_place[group]
    if source.startswith("W"):
        return winners[int(source[1:])]
    if source.startswith("L"):
        return losers[int(source[1:])]

    raise ValueError(f"Cannot resolve bracket source: {source}")


def simulate_knockout_match(match_row, model, feature_cols, adjust_probs_with_squad, rng=None):
    """Simulate a knockout match; draws are settled by penalties."""
    rng = rng or np.random.default_rng()
    x = match_row[feature_cols].to_frame().T
    probs = model.predict_proba(x)[0]
    probs = adjust_probs_with_squad(
        probs,
        match_row["home_top11_rating"],
        match_row["away_top11_rating"],
    )

    result = rng.choice(["away_win", "draw", "home_win"], p=probs)

    if result == "home_win":
        return match_row["home_team"], match_row["away_team"]
    if result == "away_win":
        return match_row["away_team"], match_row["home_team"]

    winner = rng.choice([match_row["home_team"], match_row["away_team"]])
    loser = match_row["away_team"] if winner == match_row["home_team"] else match_row["home_team"]
    return winner, loser


def prepare_round_matches(
    bracket,
    round_name,
    first_place,
    second_place,
    third_place,
    third_mapping,
    winners,
    losers,
    squad_ratings,
    top11_ratings,
    elo_ratings_wc2026,
    latest_team_features,
):
    """Resolve teams for a bracket round and rebuild model features."""
    round_matches = bracket[bracket["round"] == round_name].copy()

    round_matches["home_team"] = round_matches["team1_source"].apply(
        lambda source: resolve_team_source(
            source,
            first_place,
            second_place,
            third_place,
            third_mapping,
            winners,
            losers,
        )
    )
    round_matches["away_team"] = round_matches["team2_source"].apply(
        lambda source: resolve_team_source(
            source,
            first_place,
            second_place,
            third_place,
            third_mapping,
            winners,
            losers,
        )
    )

    round_matches = create_match_features(
        round_matches,
        squad_ratings,
        top11_ratings,
        elo_ratings_wc2026,
        latest_team_features,
    )
    round_matches["tournament_weight"] = 5

    return round_matches


def simulate_one_tournament(
    fixtures,
    bracket,
    wc2026_draw,
    mapping,
    fifa_rank,
    model,
    feature_cols,
    adjust_probs_with_squad,
    squad_ratings,
    top11_ratings,
    elo_ratings_wc2026,
    latest_team_features,
    rng=None,
):
    """Simulate one full World Cup and return the main tournament outputs."""
    rng = rng or np.random.default_rng()

    simulated_fixtures = simulate_group_stage_matches(
        fixtures,
        model,
        feature_cols,
        adjust_probs_with_squad,
        rng=rng,
    )
    standings = build_group_standings(simulated_fixtures, wc2026_draw, fifa_rank)
    first_place, second_place, third_place, best_third = get_qualifiers(standings)
    third_mapping = get_third_mapping(best_third, mapping)

    winners = {}
    losers = {}
    round_results = {}

    for round_name in ROUND_ORDER:
        round_matches = prepare_round_matches(
            bracket,
            round_name,
            first_place,
            second_place,
            third_place,
            third_mapping,
            winners,
            losers,
            squad_ratings,
            top11_ratings,
            elo_ratings_wc2026,
            latest_team_features,
        )

        round_results[round_name] = round_matches.copy()

        for _, match in round_matches.iterrows():
            winner, loser = simulate_knockout_match(
                match,
                model,
                feature_cols,
                adjust_probs_with_squad,
                rng=rng,
            )
            winners[int(match["match_id"])] = winner
            losers[int(match["match_id"])] = loser

    champion = winners[104]
    runner_up = losers[104]
    third_place_winner = winners.get(103)
    fourth_place = losers.get(103)

    return {
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place_winner,
        "fourth_place": fourth_place,
        "standings": standings,
        "simulated_fixtures": simulated_fixtures,
        "best_third": best_third,
        "third_mapping": third_mapping,
        "winners": winners,
        "losers": losers,
        "round_results": round_results,
    }


def run_monte_carlo(
    n_simulations,
    fixtures,
    bracket,
    wc2026_draw,
    mapping,
    fifa_rank,
    model,
    feature_cols,
    adjust_probs_with_squad,
    squad_ratings,
    top11_ratings,
    elo_ratings_wc2026,
    latest_team_features,
    random_seed=None,
):
    """Run many tournament simulations and summarize stage probabilities."""
    rng = np.random.default_rng(random_seed)

    rows = []
    stage_counts = {
        "r32": {},
        "r16": {},
        "quarter_final": {},
        "semi_final": {},
        "final": {},
        "champion": {},
    }

    def add_count(stage, team):
        stage_counts[stage][team] = stage_counts[stage].get(team, 0) + 1

    for sim_id in range(1, n_simulations + 1):
        result = simulate_one_tournament(
            fixtures,
            bracket,
            wc2026_draw,
            mapping,
            fifa_rank,
            model,
            feature_cols,
            adjust_probs_with_squad,
            squad_ratings,
            top11_ratings,
            elo_ratings_wc2026,
            latest_team_features,
            rng=rng,
        )

        champion = result["champion"]
        runner_up = result["runner_up"]
        third_place_team = result["third_place"]
        fourth_place = result["fourth_place"]

        rows.append({
            "simulation": sim_id,
            "champion": champion,
            "runner_up": runner_up,
            "third_place": third_place_team,
            "fourth_place": fourth_place,
        })

        add_count("champion", champion)

        # Reached Round of 32
        for team in pd.concat([
            result["standings"][result["standings"]["position"].isin([1, 2])]["country"],
            result["best_third"]["country"],
        ]):
            add_count("r32", team)

        # Reached Round of 16
        for match_id in range(73, 89):
            add_count("r16", result["winners"][match_id])

        # Reached Quarter-Finals
        for match_id in range(89, 97):
            add_count("quarter_final", result["winners"][match_id])

        # Reached Semi-Finals
        for match_id in range(97, 101):
            add_count("semi_final", result["winners"][match_id])

        # Reached Final
        for match_id in [101, 102]:
            add_count("final", result["winners"][match_id])

    simulations = pd.DataFrame(rows)
    probabilities = build_probability_table(stage_counts, n_simulations)

    return {
        "simulations": simulations,
        "probabilities": probabilities,
        "stage_counts": stage_counts,
    }


def build_probability_table(stage_counts, n_simulations):
    """Convert stage counts into one team-level probability table."""
    countries = sorted({
        country
        for counts in stage_counts.values()
        for country in counts.keys()
    })

    rows = []
    for country in countries:
        row = {"country": country}
        for stage, counts in stage_counts.items():
            row[f"{stage}_count"] = counts.get(country, 0)
            row[f"{stage}_prob"] = counts.get(country, 0) / n_simulations
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("champion_prob", ascending=False)
        .reset_index(drop=True)
    )
