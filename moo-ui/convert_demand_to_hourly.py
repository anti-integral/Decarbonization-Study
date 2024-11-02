import pandas as pd
import datetime

# Read the CSV file
df = pd.read_csv('data/demand_data_kbh.csv')

# Convert 'day' column to datetime
df['day'] = pd.to_datetime(df['day'])

# Create a list to store hourly data
hourly_data = []

# Process each row in the daily data
for _, row in df.iterrows():
    day = row['day']
    daily_demand = row['PF']
    hourly_demand = daily_demand / 24  # Divide daily demand by 24 for hourly demand
    
    # Create 24 entries for each day
    for hour in range(24):
        datetime = day + pd.Timedelta(hours=hour)
        hourly_data.append({
            'datetime': datetime,
            'PF': hourly_demand
        })

# Create a new DataFrame with hourly data
hourly_df = pd.DataFrame(hourly_data)

# Sort the DataFrame by datetime
hourly_df = hourly_df.sort_values('datetime')

# Reset the index
hourly_df = hourly_df.reset_index(drop=True)

# Save the hourly data to a new CSV file
hourly_df.to_csv('data/hourly_demand_data_kbh.csv', index=False)

print("Conversion complete. Hourly data saved to 'hourly_demand_data_kbh.csv'")

# Display the first few rows of the hourly data
print(hourly_df.head(25))