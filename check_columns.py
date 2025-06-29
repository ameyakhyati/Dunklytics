# filename: check_columns.py
import pandas as pd

try:
    df = pd.read_csv("basketball_matches_with_opponents.csv")
    print("\n--- Columns in your CSV file ---")
    # Print all column names, one per line, for easy reading
    for col in df.columns:
        print(col)
    print("--------------------------------\n")
except Exception as e:
    print(f"An error occurred: {e}")