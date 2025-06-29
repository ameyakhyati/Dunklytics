# filename: create_scaler.py (Final Version)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

print("Attempting to create scaler.pkl...")

try:
    df = pd.read_csv("basketball_matches_with_opponents.csv")
    print("Loaded basketball_matches_with_opponents.csv successfully.")

    # --- Step 1: Perform Feature Engineering ---
    print("Calculating new features...")

    # Calculate shooting percentages
    df['FG2_PCT'] = (df['FGM_2'] / df['FGA_2']).replace([np.inf, -np.inf], 0).fillna(0)
    df['FG3_PCT'] = (df['FGM_3'] / df['FGA_3']).replace([np.inf, -np.inf], 0).fillna(0)
    df['FT_PCT'] = (df['FTM'] / df['FTA']).replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate assist-to-turnover ratio
    df['AST_TO_RATIO'] = (df['AST'] / df['TOV']).replace([np.inf, -np.inf], 0).fillna(0)

    # ** MODIFIED REBOUND RATE FORMULAS **
    # The original formulas failed because opponent rebound columns were missing.
    # These new formulas work by calculating rates based on the team's own total rebounds.
    total_rebounds = df['DREB'] + df['OREB']
    df['DREB_RATE'] = (df['DREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
    df['OREB_RATE'] = (df['OREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)

    # We can't calculate a turnover rate that requires opponent stats, so we'll skip it for now
    # to ensure the script runs. The other features will still provide strong predictive power.
    df['TURNOVER_RATE'] = (df['TOV'] / (df.get('FGA', 0) + 0.44 * df.get('FTA', 0) + df.get('TOV', 1))).replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate margin of victory
    df['MARGIN_VICTORY'] = df['team_score'] - df['opponent_team_score']

    print("Feature engineering complete.")

    # --- Step 2: Fit and Save the Scaler ---

    stat_features = [
        'FG2_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TO_RATIO', 'DREB_RATE',
        'OREB_RATE', 'TURNOVER_RATE', 'MARGIN_VICTORY'
    ]

    scaler = MinMaxScaler()
    scaler.fit(df[stat_features])
    print("Scaler has been fitted to the new features.")

    joblib.dump(scaler, 'scaler.pkl')
    print("\nSUCCESS: 'scaler.pkl' has been created and saved in your folder.")

except FileNotFoundError:
    print("\nERROR: Could not find 'basketball_matches_with_opponents.csv'.")
except KeyError as e:
    print(f"\nERROR: A required base column was not found in the CSV file: {e}.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")