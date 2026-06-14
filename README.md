# FIFA World Cup 2026 Prediction

## Overview

This project predicts the FIFA World Cup 2026 using machine learning and Monte Carlo simulation.

Historical international football match results were used to train classification models (Logistic Regression, Random Forest, XGBoost) that estimate match outcome probabilities (Home Win, Draw, Away Win). These probabilities were then used to simulate the entire 2026 FIFA World Cup thousands of times to estimate each team's chances of advancing through the tournament and ultimately winning the World Cup.


## Objectives

* Predict match outcomes using historical football data
* Compare multiple machine learning models
* Simulate the FIFA World Cup 2026 tournament structure
* Estimate advancement and championship probabilities for all participating teams


## Repository Structure

```text
WC2026/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_world_cup_simulation.ipynb
│   └── 05_result_visualisation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── features.py
│   └── simulation.py
│
└── README.md
```

## Technologies Used

Main Libraries:
* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn

This project was developed with the assistance of several AI coding tools:

| Tool | Usage |
|--------|--------|
| ChatGPT | General debugging, feature engineering discussions, model development, and project design |
| Codex | Implementing and refactoring the tournament simulation pipeline |
| Claude | Data visualisation implementation, hyperparameter tuning |

All visualisations, modelling decisions, feature engineering, data preparation, and evaluation were reviewed and validated manually.


## Data Sources

- Historical Match Results: [International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?resource=download&select=results.csv)
- Player Ratings [FC 26 (FIFA 26) Player Data](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data/data) 
- Elo Ratings: [Elo Ratings 1992-2024](https://www.kaggle.com/datasets/cashncarry/fifaworldranking/discussion/681592)
- FIFA World Cup 2026 Draw: Self-created CSV based on FIFA's official fixtures
- Combinations of Round of 32 Matches: Adapted from [Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage)


## Data Cleaning

* Standardised Country Names 
* Removed Invalid Records


## Exploratory Data Analysis (EDA)

* Home ground advantage across all international matches
* Home ground advantage for the past five World Cup hosts
* Past World Cup performance by the 2026 World Cup hosts (Mexico, USA, Canada)
* Tournaments with the highest number of match records in the dataset
* Availability of FIFA 26 Player Ratings for qualified nations
* Distribution of FIFA 26 Player Ratings
* Distribution of Elo ratings by confederation
* Comparison of 2026 World Cup teams' Elo ratings by group


## Feature Engineering

The following features were created for each match:

### Elo-Based Features

* Home Elo Rating
* Away Elo Rating
* Elo Difference
* Absolute Elo Difference

### Recent Form Features

Based on each team's previous 10 matches:

* Recent Win Rate
* Recent Draw Rate
* Average Goals Scored
* Average Goals Conceded

### Difference Features

* Recent Form Difference
* Absolute Recent Form Difference
* Recent Draw Difference
* Absolute Recent Draw Difference
* Difference in Average Goals Scored
* Difference in Average Goals Conceded

### Host Advantage

* Whether the home (or away) team is one of the tournament hosts

### Tournament Weight

* A numerical variable representing the competitive importance of a tournament.

| Tournament Type | Weight |
|---------------|--------|
| FIFA World Cup | 5 |
| Continental Championships (UEFA Euro, Copa América, AFCON, AFC Asian Cup, Gold Cup) | 4 |
| FIFA World Cup Qualification | 3 |
| Other Competitive Tournaments | 2 |
| International Friendlies | 1 |


## Models Evaluated

Three classification models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

Evaluation metrics:

* Accuracy
* Macro F1 Score
* Weighted F1 Score


## Tournament Simulation

The FIFA World Cup 2026 was simulated using Monte Carlo methods.

For each match:

1. The Random Forest model generated outcome probabilities.
2. Probabilities were adjusted using the top 11 players' ratings.
3. A match result was sampled from the resulting probability distribution.
4. Match scores were generated using Poisson distributions.
5. Group standings were calculated.
6. Knockout rounds were simulated until a champion was determined.

The entire tournament was simulated 1,000 times and 10,000 times.

The simulation also estimates the probability of each team reaching:

* Round of 32
* Round of 16
* Quarter-Finals
* Semi-Finals
* Final
* Champion
