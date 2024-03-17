import sumolib
import os
import pandas as pd
import sys

# Check if SUMO_HOME is declared
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Parse the net with location information
net = sumolib.net.readNet('C:/Users/CASPER/MOS-24/scenarios/berwyn/sumo/berwyn.net.xml')

# Define the input and output CSV file paths
input_dir = "C:/Users/CASPER/MOS-24/logs/log-20240206-180440-berwyn/apps/output"
output_dir = "C:/Users/CASPER/MOS-24/logs/log-20240206-180440-berwyn/apps/output"

# Ensure the output directory exists, create it if necessary
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to store individual dataframes
dfs = []

# Iterate through all CSV files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_csv_path = os.path.join(input_dir, filename)

        # Read data from the current CSV file
        data = []
        with open(input_csv_path, 'r') as csv_file:
            header = next(csv_file)  # Skip the header
            for row in csv_file:
                fields = row.strip().split(',')
                Plaka, X, Y, Hiz, Yon, Zaman, SimTime, Dname = fields

                X, Y, Hiz = float(X), float(Y), float(Hiz)

                radius = 0.1
                edges = net.getNeighboringEdges(X, Y, radius)
                # pick the closest edge
                if len(edges) > 0:
                    distancesAndEdges = sorted([(dist, edge) for edge, dist in edges], key=lambda x: x[0])
                    dist, closestEdge = distancesAndEdges[0]
                data.append([Plaka, X, Y, Hiz, Yon, Zaman, SimTime, Dname, closestEdge])

        # Create a dataframe for the current CSV file
        df = pd.DataFrame(data,
                          columns=['Plaka', 'X', 'Y', 'Hiz', 'Yon', 'Zaman', 'SimTime', 'Dname', 'Lane'])
        dfs.append(df)

# Merge dataframes based on the "Zaman" column and sort by "Zaman"
merged_df = pd.concat(dfs).sort_values(by='Zaman')

# Save the merged dataframe to a new CSV file
merged_csv_path = os.path.join(output_dir, 'merged_output.csv')
merged_df.to_csv(merged_csv_path, index=False)

print(f"Merged CSV file created at: {merged_csv_path}")
