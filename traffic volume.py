import pandas as pd
from tqdm import tqdm

# Load the traffic data into a DataFrame
data = pd.read_csv("C:/Users/CASPER/MOS-24/logs/log-20240211-142419-berwyn/apps/output/simpleModified.csv")

# Convert 'Zaman' column to datetime type
data['Zaman'] = pd.to_datetime(data['Zaman'])

# Find the minimum and maximum timestamps in the dataset
min_time = data['Zaman'].min()
max_time = data['Zaman'].max()

# Determine the start and end times for the analysis
start_time = min_time
end_time = max_time.ceil('10T')  # Round up to the nearest 10 minutes

# Define the time interval for volume calculation (e.g., every 10 minutes)
interval = pd.Timedelta(minutes=10)

# Initialize lists to store volumes for each interval
volume_columns = []
volume_per_interval = []

# Iterate over each time interval
current_time = start_time
while current_time < end_time:
    # Filter data within the current time interval
    interval_data = data[(data['Zaman'] >= current_time) & (data['Zaman'] < current_time + interval)]

    # Initialize tqdm to track progress
    with tqdm(total=len(interval_data['Lane'].unique()), desc=f"Processing interval {current_time} to {current_time + interval}") as pbar:
        # Calculate the volume of each lane for the current time interval
        for lane in interval_data['Lane'].unique():
            volume = interval_data[(interval_data['Lane'] == lane) & ((interval_data['Plaka'].notnull()) | (interval_data['Dname'].notnull()))].shape[0]
            volume_per_interval.append(volume)
            pbar.update(1)

    # Append the volume list for the current time interval to the columns list
    volume_columns.append(f"Volume{len(volume_columns) + 1}")

    # Move to the next time interval
    current_time += interval

# Print the length of volume_per_interval for troubleshooting
print("Length of volume_per_interval:", len(volume_per_interval))

# Check if the volume data is empty
if volume_per_interval:
    # Add the volume columns to the DataFrame
    for i, column_name in enumerate(volume_columns):
        data[column_name] = volume_per_interval[i::len(volume_columns)]
else:
    print("No data available for the specified time intervals.")

# Save the updated data to a new CSV file
data.to_csv("C:/Users/CASPER/MOS-24/logs/log-20240211-142419-berwyn/apps/output/traffic_data_with_volume.csv", index=False)


