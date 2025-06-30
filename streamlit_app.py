# filename: app.py (Final Corrected Version for Organized Directory)

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
        # --- PATHS UPDATED FOR ORGANIZED FOLDERS ---
        df = pd.read_csv("data/basketball_matches_with_opponents.csv")
        rf_model = joblib.load('models/random_forest_model.pkl')
        lstm_model = tf.keras.models.load_model('models/lstm_stat_predictor.h5')
        team_encoder = joblib.load('models/team_encoder.pkl')
        scaler = joblib.load('models/scaler.pkl')
        # ---------------------------------------------

        # --- FEATURE ENGINEERING ---
        df['FG2_PCT'] = (df['FGM_2'] / df['FGA_2']).replace([np.inf, -np.inf], 0).fillna(0)
        df['FG3_PCT'] = (df['FGM_3'] / df['FGA_3']).replace([np.inf, -np.inf], 0).fillna(0)
        df['FT_PCT'] = (df['FTM'] / df['FTA']).replace([np.inf, -np.inf], 0).fillna(0)
        df['AST_TO_RATIO'] = (df['AST'] / df['TOV']).replace([np.inf, -np.inf], 0).fillna(0)
        total_rebounds = df['DREB'] + df['OREB']
        df['DREB_RATE'] = (df['DREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
        df['OREB_RATE'] = (df['OREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
        df['TURNOVER_RATE'] = (df['TOV'] / (df.get('FGA', 0) + 0.44 * df.get('FTA', 0) + df.get('TOV', 1))).replace([np.inf, -np.inf], 0).fillna(0)
        df['MARGIN_VICTORY'] = df['team_score'] - df['opponent_team_score']
        
        rf_features = rf_model.feature_names_in_

        # Use the loaded encoder to create the encoded columns in the dataframe
        df['team_encoded'] = team_encoder.transform(df['team'])
        df['opponent_team_encoded'] = team_encoder.transform(df['opponent_team'])

        return df, rf_model, lstm_model, team_encoder, scaler, rf_features

    except FileNotFoundError as e:
        st.error(f"ERROR: A required file is missing -> {e}. Please check your repository's file structure.")
        return None, None, None, None, None, None
    except KeyError as e:
        st.error(f"ERROR: A required column was not found in your CSV file: {e}. Please check that the data file contains all necessary base columns.")
        return None, None, None, None, None, None

# Load all assets
df, rf_model, lstm_model, team_encoder, scaler, rf_features = load_models_and_data()
stat_features = ['FG2_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TO_RATIO', 'DREB_RATE', 'OREB_RATE', 'TURNOVER_RATE', 'MARGIN_VICTORY']

# --- Helper Functions ---
def get_matchup_mean_stats(team_name, opponent_name, data_df, encoder):
    team_encoded = encoder.transform([team_name])[0]
    opponent_encoded = encoder.transform([opponent_name])[0]
    matchup_data = data_df[(data_df['team_encoded'] == team_encoded) & (data_df['opponent_team_encoded'] == opponent_encoded)]
    if matchup_data.empty: return None
    return matchup_data[stat_features].mean().to_dict()

def predict_future_stats(team_name, data_df, encoder, scaler_model, lstm_model, sequence_length=5):
    team_encoded = encoder.transform([team_name])[0]
    team_history = data_df[data_df['team_encoded'] == team_encoded].tail(sequence_length)
    if len(team_history) < sequence_length: return None
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
        # Set a default index that is within the bounds of the list
        home_index = min(10, len(TEAMS) - 1) if "oklahoma_sooners" not in TEAMS else TEAMS.index("oklahoma_sooners")
        home_team = st.selectbox("Select Home Team:", TEAMS, index=home_index)
    with col2:
        # Set a different default index
        away_index = min(25, len(TEAMS) - 1) if "baylor_bears" not in TEAMS else TEAMS.index("baylor_bears")
        away_team = st.selectbox("Select Away Team:", TEAMS, index=away_index)

    if home_team == away_team:
        st.error("Please select two different teams.")
    else:
        if st.button("Predict Win Probability", type="primary"):
            with st.spinner("Analyzing matchup and running models..."):
                team_h2h_stats = get_matchup_mean_stats(home_team, away_team, df, team_encoder)
                opponent_h2h_stats = get_matchup_mean_stats(away_team, home_team, df, team_encoder)
                team_predicted_stats = predict_future_stats(home_team, df, team_encoder, scaler, lstm_model)
                opponent_predicted_stats = predict_future_stats(away_team, df, team_encoder, scaler, lstm_model)
                team_overall_stats = df[df['team'] == home_team][stat_features].mean().to_dict()
                opponent_overall_stats = df[df['team'] == away_team][stat_features].mean().to_dict()

                prediction_made = False
                input_features = {}

                # TIER 1: Gold Standard (Head-to-head + LSTM)
                if all([team_h2h_stats, opponent_h2h_stats, team_predicted_stats, opponent_predicted_stats]):
                    st.info("Using Tier 1 Prediction: Head-to-Head History + LSTM Forecast")
                    for col in stat_features:
                        input_features[col] = (0.8 * team_h2h_stats.get(col, 0)) + (0.2 * team_predicted_stats.get(col, 0))
                        input_features[f"opponent_{col}"] = (0.8 * opponent_h2h_stats.get(col, 0)) + (0.2 * opponent_predicted_stats.get(col, 0))
                    prediction_made = True
                # TIER 2: Fallback (Overall Averages + LSTM)
                elif all([team_predicted_stats, opponent_predicted_stats]):
                    st.warning("Using Tier 2 Prediction: Overall Season Averages + LSTM Forecast (No head-to-head history found).")
                    for col in stat_features:
                        input_features[col] = (0.6 * team_overall_stats.get(col, 0)) + (0.4 * team_predicted_stats.get(col, 0))
                        input_features[f"opponent_{col}"] = (0.6 * opponent_overall_stats.get(col, 0)) + (0.4 * opponent_predicted_stats.get(col, 0))
                    prediction_made = True
                # TIER 3: Fallback (Head-to-head Only)
                elif all([team_h2h_stats, opponent_h2h_stats]):
                    st.warning("Using Tier 3 Prediction: Head-to-Head History Only (Not enough recent games for LSTM).")
                    for col in stat_features:
                        input_features[col] = team_h2h_stats.get(col, 0)
                        input_features[f"opponent_{col}"] = opponent_h2h_stats.get(col, 0)
                    prediction_made = True
                # TIER 4: Final Fallback (Overall Averages Only)
                else:
                    st.warning("Using Tier 4 Prediction: Overall Season Averages Only (Limited data available).")
                    for col in stat_features:
                        input_features[col] = team_overall_stats.get(col, 0)
                        input_features[f"opponent_{col}"] = opponent_overall_stats.get(col, 0)
                    prediction_made = True

                if prediction_made:
                    input_features["team_encoded"] = team_encoder.transform([home_team])[0]
                    input_features["opponent_team_encoded"] = team_encoder.transform([away_team])[0]
                    input_df = pd.DataFrame([input_features])[rf_features]
                    win_proba = rf_model.predict_proba(input_df)[:, 1][0]
                    
                    st.success("Prediction Complete!")
                    st.subheader(f"Predicted Win Probability for {home_team}")
                    st.progress(win_proba, text=f"{win_proba:.0%}")
                    st.write(f"The model predicts that the **{home_team}** have a **{win_proba:.0%}** chance of winning against the **{away_team}**.")
                else:
                    st.error("Could not generate a prediction. Not enough data even for the most basic model.")
