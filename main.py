import pandas as pd
from tqdm import tqdm
import string

file_path = 'C:/Users/CASPER/MOS-24/logs/log-20240211-142419-berwyn/apps/output/modified_data.csv'
output_path = 'C:/Users/CASPER/MOS-24/logs/log-20240211-142419-berwyn/apps/output/simpleModified.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Initialize an empty dictionary to store the log
renaming_log = {}

# Get the unique lane values
unique_lanes = df['Lane'].unique()

# Generate simplified lane names
simplified_names = [f"Lane{letter}{number}" for letter in string.ascii_uppercase for number in range(1, len(unique_lanes) + 1)]

# Create a dictionary to map original lane names to simplified names
lane_mapping = dict(zip(unique_lanes, simplified_names))

# Apply the mapping function to the 'Lane' column
tqdm.pandas(desc="Simplifying Lane")
df['Lane'] = df['Lane'].progress_apply(lambda x: lane_mapping[x])

# Write the renaming log to a file
log_file_path = "C:/Users/CASPER/Desktop/Kodlar/renaming_log.txt"
with open(log_file_path, 'w') as log_file:
    log_file.write("Renaming Log:\n")
    for original_name, simplified_name in lane_mapping.items():
        log_file.write(f"{original_name} -> {simplified_name}\n")

print(f"Renaming log saved to: {log_file_path}")

# Save the modified DataFrame to a new CSV file
df.to_csv(output_path, index=False)

print(f"Modified CSV file saved to: {output_path}")
