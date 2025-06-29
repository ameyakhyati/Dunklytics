import streamlit as st
import pandas as pd
import joblib
# We will add more imports like tensorflow later

# --- Page Configuration ---
st.set_page_config(
    page_title="Dunklytics Predictor",
    page_icon="🏀",
    layout="centered"
)

# --- Title and Description ---
st.title("🏀 Dunklytics: Basketball Match Predictor")
st.write("""
This app predicts basketball match outcomes using a hybrid model. 
Select two teams to see the predicted win probability.
""")

# --- Load Data and Models (Placeholder) ---
# This function will eventually contain your complex prediction logic.
# For now, it will just return a dummy value.
def get_prediction(team1, team2):
    """
    This is where your prediction logic will go.
    You will need to:
    1. Load your trained Random Forest model.
    2. Load your trained LSTM model.
    3. Load any necessary data scalers or encoders.
    4. Get historical stats for both teams.
    5. Use the LSTM model to predict future stats.
    6. Combine the stats into a single feature row.
    7. Use the Random Forest model to make the final prediction.
    """
    # For now, we'll return a placeholder probability
    st.info("Note: This is a demo. The backend prediction logic is not yet fully connected.")
    # In a real scenario, this would be the model's output, e.g., model.predict_proba(features)[0][1]
    dummy_probability = 0.67 
    return dummy_probability

# --- User Interface ---
try:
    # Load team names for the dropdown menus
    # This assumes you have run preprocess_data.py and have this file
    team_stats_df = pd.read_csv("teamwise_stats.csv")
    TEAMS = sorted(team_stats_df['team'].unique())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Select Home Team:", TEAMS, index=10) # Default to a random team
    with col2:
        away_team = st.selectbox("Select Away Team:", TEAMS, index=25) # Default to another team

    if home_team == away_team:
        st.error("Please select two different teams.")
    else:
        # Predict button
        if st.button("Predict Win Probability", type="primary"):
            with st.spinner("Analyzing matchup and running prediction..."):
                # Get the prediction
                win_probability = get_prediction(home_team, away_team)
                
                # Display the result
                st.success("Prediction Complete!")
                st.subheader(f"Predicted Win Probability for {home_team}")
                
                # Display as a progress bar
                st.progress(win_probability, text=f"{win_probability:.0%}")
                st.write(f"The model predicts that the **{home_team}** have a **{win_probability:.0%}** chance of winning against the **{away_team}**.")

except FileNotFoundError:
    st.error("Error: `teamwise_stats.csv` not found. Please run the `preprocess_data.py` script first.")