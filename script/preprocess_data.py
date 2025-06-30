# Import the pandas library for data manipulation
import pandas as pd

# --- Load the datasets ---
try:
    df = pd.read_csv("games_2022.csv")
    regions = pd.read_csv("Team Region Groups.csv")
except FileNotFoundError as e:
    print(f"\n--- ERROR ---")
    print(f"File not found: {e}. ")
    print("Please make sure 'games_2022.csv' and 'Team Region Groups.csv' are in the Dunklytics folder.")
    print("---------------\n")
    exit()

# Drop unnecessary columns
df.drop(columns=['travel_dist', 'home away NS', 'home_away', 'attendance', 'game_date',
                 'tz dif H E', 'prev_game_dist', 'rest_days', 'OT_length min_tot'],
        inplace=True, errors='ignore')

# Merge the dataframes
region_df = df.merge(regions, on="team", how="left")

# Fill missing region values
region_df['region'] = region_df['region'].fillna('East')

# --- Feature Engineering ---
# Note: Using .get(column, 0) to avoid errors if a column is missing
region_df['FGP_2'] = ((region_df.get('FGM_2', 0) / region_df.get('FGA_2', 1)) * 100).replace([float('inf'), -float('inf')], 0)
region_df['FGP_3'] = ((region_df.get('FGM_3', 0) / region_df.get('FGA_3', 1)) * 100).replace([float('inf'), -float('inf')], 0)
region_df['FT_Percentage'] = ((region_df.get('FTM', 0) / region_df.get('FTA', 1)) * 100).replace([float('inf'), -float('inf')], 0)
region_df['Total Rebounds'] = region_df.get('DREB', 0) + region_df.get('OREB', 0)

# ** THIS IS THE CORRECTED LINE FOR THE 'KeyError' **
# Using 'team_score' and 'opponent_team_score' with underscores
try:
    region_df['result'] = region_df.apply(lambda row: True if row['team_score'] > row['opponent_team_score'] else False, axis=1)
except KeyError:
    print("\n--- ERROR ---")
    print("A 'KeyError' occurred. The script expected 'team_score' and 'opponent_team_score' columns in 'games_2022.csv'.")
    print("Please check your CSV file for the correct column names.")
    print("---------------\n")
    exit()

region_df['result'] = region_df['result'].astype(bool)

# Drop columns that are no longer needed
region_df.drop(columns=['FGA_2', 'FGA_3', 'FGM_2', 'FGM_3', 'FTA', 'FTM',
                         'team_score', 'opponent_team_score', 'OREB', 'DREB',
                         'notDi_incomplete'], inplace=True, errors='ignore')

# Group data by team and aggregate stats
teamwise_stats = region_df.groupby('team').agg({
    'AST': 'mean', 'BLK': 'mean', 'STL': 'mean', 'TOV': 'mean', 'TOV_team': 'mean',
    'F_tech': 'mean', 'F_personal': 'mean', 'largest_lead': 'mean', 'FGP_2': 'mean',
    'FGP_3': 'mean', 'FT_Percentage': 'mean', 'Total Rebounds': 'mean',
    'result': ['sum', 'count'], 'region': 'first'
})

# Rename the aggregated columns
teamwise_stats.columns = ['Avg_AST', 'Avg_BLK', 'Avg_STL', 'Avg_TOV', 'Avg_TOV_team',
                          'Avg_F_tech', 'Avg_F_personal', 'Avg_largest_lead',
                          'Avg_FGP_2', 'Avg_FGP_3', 'Avg_FT_Percentage', 'Avg_Total_Rebounds',
                          'Total_Wins', 'Total_Matches', 'Region']

# Save the main aggregated statistics
teamwise_stats.to_csv('teamwise_stats.csv')
print(f"Successfully saved: teamwise_stats.csv")

# --- Split by region and save separate files ---
regions_list = ['East', 'West', 'North', 'South']
for region in regions_list:
    region_data = teamwise_stats[teamwise_stats['Region'] == region]
    filename = f'teamwise_stats_{region.lower()}.csv'
    region_data.to_csv(filename, index=True)
    print(f"Successfully saved: {filename}")