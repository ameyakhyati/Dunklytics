import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# List of the regional files you created in the last step
files_to_rank = [
    "teamwise_stats_north.csv",
    "teamwise_stats_south.csv",
    "teamwise_stats_east.csv",
    "teamwise_stats_west.csv"
]

print("Starting team ranking process...")

# Loop through each regional file to process and rank the teams
for file in files_to_rank:
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"\nWarning: Could not find '{file}'. Skipping.")
        continue

    # Create a more balanced ranking score
    df['Ranking_Score'] = df['Total_Wins'] * (df['Total_Wins'] / (df['Total_Matches'] + 1)) # 

    # Select the features the model will use for prediction
    features = ['Avg_AST', 'Avg_BLK', 'Avg_STL', 'Avg_TOV', 'Avg_TOV_team',
                'Avg_F_tech', 'Avg_F_personal', 'Avg_largest_lead',
                'Avg_FGP_2', 'Avg_FGP_3', 'Avg_FT_Percentage', 'Avg_Total_Rebounds', 'Total_Matches'] # [cite: 176, 177, 178]
    
    X = df[features]
    y = df['Ranking_Score']

    # Split data to train and test the model's performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # [cite: 183]

    # Scale data for better model performance
    scaler = StandardScaler() # [cite: 185]
    X_train_scaled = scaler.fit_transform(X_train) # [cite: 186, 187]
    X_test_scaled = scaler.transform(X_test) # [cite: 188]

    # Initialize and train the XGBoost Regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1) # [cite: 190]
    model.fit(X_train_scaled, y_train) # [cite: 190]

    # Use the trained model to predict scores for the entire dataset
    df['Predicted_Ranking_Score'] = model.predict(scaler.transform(X)) # [cite: 192]

    # Rank teams based on the predicted scores
    df = df.sort_values(by='Predicted_Ranking_Score', ascending=False) # [cite: 194]
    df['Rank'] = range(1, len(df) + 1) # [cite: 195]

    # Save the results for the region
    output_file = file.replace(".csv", "_ranking.csv") # [cite: 197]
    df.to_csv(output_file, index=False) # [cite: 197]
    print(f"Successfully created ranking file: {output_file}") # [cite: 198]

print("\nTeam ranking process complete.")