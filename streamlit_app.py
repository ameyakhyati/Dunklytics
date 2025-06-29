# filename: app.py

import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Dunklytics Predictor",
    page_icon="🏀",
    layout="centered"
)

# --- Load Models and Data ---
@st.cache_resource
def load_models_and_data():
    """Load all necessary models, encoders, and data files only once."""
    try:
        # Load the main dataset
        df = pd.read_csv("basketball_matches_with_opponents.csv")

        # --- START: FEATURE ENGINEERING ---
        df['FG2_PCT'] = (df['FGM_2'] / df['FGA_2']).replace([np.inf, -np.inf], 0).fillna(0)
        df['FG3_PCT'] = (df['FGM_3'] / df['FGA_3']).replace([np.inf, -np.inf], 0).fillna(0)
        df['FT_PCT'] = (df['FTM'] / df['FTA']).replace([np.inf, -np.inf], 0).fillna(0)
        df['AST_TO_RATIO'] = (df['AST'] / df['TOV']).replace([np.inf, -np.inf], 0).fillna(0)
        total_rebounds = df['DREB'] + df['OREB']
        df['DREB_RATE'] = (df['DREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
        df['OREB_RATE'] = (df['OREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
        df['TURNOVER_RATE'] = (df['TOV'] / (df.get('FGA', 0) + 0.44 * df.get('FTA', 0) + df.get('TOV', 1))).replace([np.inf, -np.inf], 0).fillna(0)
        df['MARGIN_VICTORY'] = df['team_score'] - df['opponent_team_score']
        # --- END: FEATURE ENGINEERING ---

        # Load the trained models
        rf_model = joblib.load('random_forest_model.pkl')
        lstm_model = tf.keras.models.load_model('lstm_stat_predictor.h5')

        # Load the encoder and scaler
        team_encoder = joblib.load('team_encoder.pkl')
        scaler = joblib.load('scaler.pkl')

        rf_features = rf_model.feature_names_in_

        # Use the loaded encoder to create the encoded columns in the dataframe
        df['team_encoded'] = team_encoder.transform(df['team'])
        df['opponent_team_encoded'] = team_encoder.transform(df['opponent_team'])

        return df, rf_model, lstm_model, team_encoder, scaler, rf_features

    except FileNotFoundError as e:
        st.error(f"ERROR: A required file is missing -> {e}. Please ensure all model and data files are in your GitHub repository.")
        return None, None, None, None, None, None
    except KeyError as e:
        st.error(f"ERROR: A required column was not found in your CSV file: {e}. Please check that 'basketball_matches_with_opponents.csv' contains all necessary base columns.")
        return None, None, None, None, None, None

# Load all assets
df, rf_model, lstm_model, team_encoder, scaler, rf_features = load_models_and_data()

# --- Helper Functions from Project Documentation ---

def get_matchup_mean_stats(team_name, opponent_name, data_df, encoder):
    """Computes mean stats for a team from historical matches against a specific opponent."""
    team_encoded = encoder.transform([team_name])[0]
    opponent_encoded = encoder.transform([opponent_name])[0]
    
    matchup_data = data_df[(data_df['team_encoded'] == team_encoded) & (data_df['opponent_team_encoded'] == opponent_encoded)]
    
    if matchup_data.empty:
        return None
        
    stat_features = ['FG2_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TO_RATIO', 'DREB_RATE', 'OREB_RATE', 'TURNOVER_RATE', 'MARGIN_VICTORY']
    return matchup_data[stat_features].mean().to_dict()

def predict_future_stats(team_name, data_df, encoder, scaler_model, lstm_model, sequence_length=5):
    """Predicts future performance stats for a team using the trained LSTM model."""
    team_encoded = encoder.transform([team_name])[0]
    team_history = data_df[data_df['team_encoded'] == team_encoded].tail(sequence_length)
    
    if len(team_history) < sequence_length:
        return None  # Not enough data for a prediction
        
    stat_features = ['FG2_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TO_RATIO', 'DREB_RATE', 'OREB_RATE', 'TURNOVER_RATE', 'MARGIN_VICTORY']
    team_history_scaled = scaler_model.transform(team_history[stat_features])
    
    team_history_input = np.array([team_history_scaled])
    
    predicted_stats_scaled = lstm_model.predict(team_history_input)
    predicted_stats = scaler_model.inverse_transform(predicted_stats_scaled)
    
    return dict(zip(stat_features, predicted_stats[0]))

# --- Streamlit User Interface ---

st.title("🏀 Dunklytics: Basketball Match Predictor")
st.write("This app predicts basketball match outcomes using a hybrid model that blends historical head-to-head data with LSTM-forecasted performance.")

if df is not None:
    TEAMS = sorted(team_encoder.classes_)

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Select Home Team:", TEAMS, index=TEAMS.index("oklahoma_sooners"))
    with col2:
        away_team = st.selectbox("Select Away Team:", TEAMS, index=TEAMS.index("baylor_bears"))

    if home_team == away_team:
        st.error("Please select two different teams.")
    else:
        if st.button("Predict Win Probability", type="primary"):
            with st.spinner("Analyzing matchup and running models..."):
                
                team_mean_stats = get_matchup_mean_stats(home_team, away_team, df, team_encoder)
                opponent_mean_stats = get_matchup_mean_stats(away_team, home_team, df, team_encoder)

                team_predicted_stats = predict_future_stats(home_team, df, team_encoder, scaler, lstm_model)
                opponent_predicted_stats = predict_future_stats(away_team, df, team_encoder, scaler, lstm_model)

                if not all([team_mean_stats, opponent_mean_stats, team_predicted_stats, opponent_predicted_stats]):
                    st.error("Insufficient historical or recent data to make a prediction for this matchup.")
                else:
                    input_features = {}
                    stat_features = list(team_mean_stats.keys())
                    
                    for col in stat_features:
                        input_features[col] = (0.8 * team_mean_stats.get(col, 0)) + (0.2 * team_predicted_stats.get(col, 0))
                        input_features[f"opponent_{col}"] = (0.8 * opponent_mean_stats.get(col, 0)) + (0.2 * opponent_predicted_stats.get(col, 0))
                    
                    input_features["team_encoded"] = team_encoder.transform([home_team])[0]
                    input_features["opponent_team_encoded"] = team_encoder.transform([away_team])[0]
                    
                    input_df = pd.DataFrame([input_features])
                    input_df = input_df[rf_features]
                    
                    win_proba = rf_model.predict_proba(input_df)[:, 1][0]
                    
                    st.success("Prediction Complete!")
                    st.subheader(f"Predicted Win Probability for {home_team}")
                    st.progress(win_proba, text=f"{win_proba:.0%}")
                    st.write(f"The model predicts that the **{home_team}** have a **{win_proba:.0%}** chance of winning against the **{away_team}**.")
