# filename: find_valid_matchups.py

import pandas as pd

try:
    df = pd.read_csv("basketball_matches_with_opponents.csv")

    # Count total games for each team
    team_game_counts = df['team'].value_counts()

    # Find teams that have at least 5 games (for the LSTM model)
    valid_teams = team_game_counts[team_game_counts >= 5].index.tolist()

    # Filter the dataframe to only include games between valid teams
    valid_games_df = df[df['team'].isin(valid_teams) & df['opponent_team'].isin(valid_teams)]

    # Find the first 5 head-to-head matchups that exist in this valid data
    good_matchups = valid_games_df[['team', 'opponent_team']].drop_duplicates().head(5)

    if not good_matchups.empty:
        print("\n--- Good Matchups Found ---")
        print("Try any of these pairs in your Streamlit app:\n")
        for index, row in good_matchups.iterrows():
            print(f"  - Home: {row['team']}, Away: {row['opponent_team']}")
        print("\n---------------------------\n")
    else:
        print("Could not find any matchups that meet the criteria.")

except FileNotFoundError:
    print("ERROR: 'basketball_matches_with_opponents.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")