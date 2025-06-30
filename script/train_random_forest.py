# filename: train_random_forest.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

print("--- Starting Random Forest Model Training ---")

try:
    # Load the dataset
    df = pd.read_csv("basketball_matches_with_opponents.csv") # 
    print("Dataset loaded successfully.")

    # --- Feature Engineering ---
    print("Performing feature engineering...")
    df['FG2_PCT'] = (df['FGM_2'] / df['FGA_2']).replace([np.inf, -np.inf], 0).fillna(0) # 
    df['FG3_PCT'] = (df['FGM_3'] / df['FGA_3']).replace([np.inf, -np.inf], 0).fillna(0) # 
    df['FT_PCT'] = (df['FTM'] / df['FTA']).replace([np.inf, -np.inf], 0).fillna(0) # 
    df['AST_TO_RATIO'] = (df['AST'] / df['TOV']).replace([np.inf, -np.inf], 0).fillna(0) # 
    total_rebounds = df['DREB'] + df['OREB']
    df['DREB_RATE'] = (df['DREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
    df['OREB_RATE'] = (df['OREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
    df['TURNOVER_RATE'] = (df['TOV'] / (df.get('FGA', 0) + 0.44 * df.get('FTA', 0) + df.get('TOV', 1))).replace([np.inf, -np.inf], 0).fillna(0) # 
    df['MARGIN_VICTORY'] = df['team_score'] - df['opponent_team_score'] # 

    # --- Data Preprocessing ---
    print("Encoding team names...")
    team_encoder = LabelEncoder() # 
    df['team_encoded'] = team_encoder.fit_transform(df['team'])
    df['opponent_team_encoded'] = team_encoder.transform(df['opponent_team'])

    # Define the target variable 'Win'
    df['Win'] = (df['team_score'] > df['opponent_team_score']).astype(int) # 

    # --- Model Training ---
    # Define features and target
    features = [col for col in df.columns if 'PCT' in col or 'RATIO' in col or 'RATE' in col or 'MARGIN' in col or 'encoded' in col]
    X = df[features]
    y = df['Win']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # 
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42) # 
    rf_model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.2%}")

    # --- Save the outputs ---
    joblib.dump(rf_model, 'random_forest_model.pkl')
    print("SUCCESS: New 'random_forest_model.pkl' has been saved.")
    joblib.dump(team_encoder, 'team_encoder.pkl')
    print("SUCCESS: New 'team_encoder.pkl' has been saved.")

except FileNotFoundError:
    print("\nERROR: Could not find 'basketball_matches_with_opponents.csv'. Please make sure it's in the folder.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")