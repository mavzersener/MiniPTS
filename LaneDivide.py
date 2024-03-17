import pandas as pd
import os

# Assuming df is your DataFrame containing the vehicle perception data
# Load your data into a DataFrame
df = pd.read_csv("your_file.csv")

# Extract lane data between "<edge id=" and "from" part
df['Lane'] = df['Lane'].str.extract(r'<edge id=".*?".*?from="(.*?)".*?>')

# Display the DataFrame with updated Lane column
print(df)
# Create a directory for the output files if it doesn't exist
output_directory = 'Yollar1'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the CSV file into a DataFrame
df = pd.read_csv('C:/Users/CASPER/Desktop/yollar1.csv', delimiter=';')

# Get unique lane IDs
unique_lanes = df['Lane'].unique()

# Iterate over each unique lane ID and save the corresponding subset to a CSV file
for lane_id in unique_lanes:
    # Filter the DataFrame for the current lane ID
    lane_df = df[df['Lane'] == lane_id]

    # Define the output file path with lane ID
    output_file = os.path.join(output_directory, f'lane_{lane_id}.csv')

    # Save the subset DataFrame to a CSV file
    lane_df.to_csv(output_file, index=False)
