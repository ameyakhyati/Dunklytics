# filename: app.py

import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import warnings

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
    """Load all necessary models, encoders, and data files."""
    try:
        # Load the main dataset required for stats calculation
        df = pd.read_csv("basketball_matches_with_opponents.csv")
        
        # Load the trained models
        rf_model = joblib.load('random_forest_model.pkl')
        lstm_model = tf.keras.models.load_model('lstm_stat_predictor.h5')
        
        # Load the team encoder and scaler used during training
        # These would have been saved in your training notebooks
        team_encoder = joblib.load('team_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Get the feature columns from the trained RF model
        rf_features = rf_model.feature_names_in_
        
        return df, rf_model, lstm_model, team_encoder, scaler, rf_features
    except FileNotFoundError as e:
        st.error(f"Error loading a required file: {e}. Please ensure all model and data files are in the repository.")
        return None, None, None, None, None, None

df, rf_model, lstm_model, team_encoder, scaler, rf_features = load_models_and_data()

# --- Helper Functions from Project Docs ---

def get_matchup_mean_stats(team_name, opponent_name, data_df, encoder): #
    """Computes mean stats for a team from historical matches against a specific opponent."""
    team_encoded = encoder.transform([team_name])[0] #
    opponent_encoded = encoder.transform([opponent_name])[0] #
    
    matchup_data = data_df[(data_df['team_encoded'] == team_encoded) & (data_df['opponent_team_encoded'] == opponent_encoded)] #
    if matchup_data.empty:
        return None
    stat_features = ['FG2_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TO_RATIO', 'DREB_RATE', 'OREB_RATE', 'TURNOVER_RATE', 'MARGIN_VICTORY']
    return matchup_data[stat_features].mean().to_dict() #

def predict_future_stats(team_name, data_df, encoder, scaler_model, lstm, sequence_length=5): #
    """Predicts future stats for a team using the LSTM model."""
    team_encoded = encoder.transform([team_name])[0]
    team_history = data_df[data_df['team_encoded'] == team_encoded].tail(sequence_length)
    
    if len(team_history) < sequence_length:
        return None # Not enough data
        
    stat_features = ['FG2_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TO_RATIO', 'DREB_RATE', 'OREB_RATE', 'TURNOVER_RATE', 'MARGIN_VICTORY']
    team_history_scaled = scaler_model.transform(team_history[stat_features])
    
    # Reshape for LSTM: (1, sequence_length, num_features)
    team_history_input = np.array([team_history_scaled])
    
    predicted_stats_scaled = lstm.predict(team_history_input)
    predicted_stats = scaler_model.inverse_transform(predicted_stats_scaled) #
    
    return dict(zip(stat_features, predicted_stats[0])) #


# --- Main App Interface ---
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
            with st.spinner("Running models... This may take a moment."):
                
                # 1. Get head-to-head historical stats
                team_mean_stats = get_matchup_mean_stats(home_team, away_team, df, team_encoder)
                opponent_mean_stats = get_matchup_mean_stats(away_team, home_team, df, team_encoder)

                # 2. Get LSTM predicted future stats
                team_predicted_stats = predict_future_stats(home_team, df, team_encoder, scaler, lstm_model)
                opponent_predicted_stats = predict_future_stats(away_team, df, team_encoder, scaler, lstm_model)

                if not all([team_mean_stats, opponent_mean_stats, team_predicted_stats, opponent_predicted_stats]):
                    st.error("Insufficient historical or recent data to make a prediction for this matchup.")
                else:
                    # 3. Combine stats using the 80/20 weighted average
                    input_features = {}
                    stat_features = list(team_mean_stats.keys())
                    
                    for col in stat_features:
                        input_features[col] = (0.8 * team_mean_stats.get(col, 0)) + (0.2 * team_predicted_stats.get(col, 0)) #
                        input_features[f"opponent_{col}"] = (0.8 * opponent_mean_stats.get(col, 0)) + (0.2 * opponent_predicted_stats.get(col, 0)) #
                    
                    input
