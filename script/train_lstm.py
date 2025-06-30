# filename: train_lstm.py

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

print("\n--- Starting LSTM Model Training ---")

try:
    # Load the dataset
    df = pd.read_csv("basketball_matches_with_opponents.csv")
    print("Dataset loaded successfully.")

    # --- Feature Engineering ---
    print("Performing feature engineering...")
    df['FG2_PCT'] = (df['FGM_2'] / df['FGA_2']).replace([np.inf, -np.inf], 0).fillna(0)
    df['FG3_PCT'] = (df['FGM_3'] / df['FGA_3']).replace([np.inf, -np.inf], 0).fillna(0)
    df['FT_PCT'] = (df['FTM'] / df['FTA']).replace([np.inf, -np.inf], 0).fillna(0)
    df['AST_TO_RATIO'] = (df['AST'] / df['TOV']).replace([np.inf, -np.inf], 0).fillna(0)
    total_rebounds = df['DREB'] + df['OREB']
    df['DREB_RATE'] = (df['DREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
    df['OREB_RATE'] = (df['OREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
    df['TURNOVER_RATE'] = (df['TOV'] / (df.get('FGA', 0) + 0.44 * df.get('FTA', 0) + df.get('TOV', 1))).replace([np.inf, -np.inf], 0).fillna(0)
    df['MARGIN_VICTORY'] = df['team_score'] - df['opponent_team_score']

    # --- Data Preparation for LSTM ---
    stat_features = [
        'FG2_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TO_RATIO', 'DREB_RATE',
        'OREB_RATE', 'TURNOVER_RATE', 'MARGIN_VICTORY'
    ]

    # Scale the data and save the scaler
    scaler = MinMaxScaler()
    df_scaled_values = scaler.fit_transform(df[stat_features])
    df_scaled = pd.DataFrame(df_scaled_values, columns=stat_features)

    joblib.dump(scaler, 'scaler.pkl')
    print("SUCCESS: New 'scaler.pkl' has been saved.")

    # Create sequences
    sequence_length = 5
    X_lstm, y_lstm = [], []
    for i in range(len(df_scaled) - sequence_length):
        X_lstm.append(df_scaled.iloc[i:i + sequence_length].values)
        y_lstm.append(df_scaled.iloc[i + sequence_length].values)

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    print(f"Created {len(X_lstm)} sequences for LSTM training.")

    # Split data
    split = int(0.8 * len(X_lstm))
    X_train, X_test = X_lstm[:split], X_lstm[split:]
    y_train, y_test = y_lstm[:split], y_lstm[split:]

    # --- Model Training ---
    print("Defining and training LSTM model (this may take a few minutes)...")
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, len(stat_features))),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(len(stat_features))
    ])

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    # verbose=0 makes the training cleaner by not printing progress for every epoch
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0) 
    print("Model training complete.")

    # --- Save the Model ---
    lstm_model.save('lstm_stat_predictor.h5')
    print("SUCCESS: New 'lstm_stat_predictor.h5' has been saved.")

except FileNotFoundError:
    print("\nERROR: Could not find 'basketball_matches_with_opponents.csv'. Please make sure it's in the folder.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")