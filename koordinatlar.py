import os
import sys
import sumolib
import pyproj
import csv
import pandas as pd

# Check if SUMO_HOME is declared
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Parse the net with location information
net = sumolib.net.readNet('C:/Users/CASPER/MOS-24/scenarios/berwyn/sumo/berwyn.net.xml')

# Input and output directories
input_dir = 'C:/Users/CASPER/MOS-24/logs/log-20240206-180440-berwyn/apps/output/'
output_dir = 'C:/Users/CASPER/MOS-24/logs/log-20240206-180440-berwyn/apps/convertedOutput/'

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
        with open(input_csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)  # Skip the header
            data = []

            for row in csv_reader:
                Plaka, X, Y, Hiz, Yon, Zaman, SimTime, Dname = row

                X, Y, Hiz = float(X), float(Y), float(Hiz)

                # Offset values from location info
                offset_x, offset_y = -432967.79, -4632114.98

                # Convert UTM to Longitude and Latitude
                X -= offset_x
                Y -= offset_y
                reference_lon, reference_lat = -87.807349, 41.838091
                reference_x, reference_y = net.convertLonLat2XY(reference_lon, reference_lat)
                X -= reference_x
                Y -= reference_y
                proj_parameters = "+proj=utm +init=epsg:32616 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
                utm_to_lonlat = pyproj.Transformer.from_proj(proj_parameters, "EPSG:4326", always_xy=True)
                lon, lat = utm_to_lonlat.transform(X, Y)

                Eks = X
                Vey = Y

                # Find the nearest lane
                radius = 0.1
                edges = net.getNeighboringEdges(Vey, Eks, radius)
                if edges:
                    edge = edges[0]
                    lane = net.getNeighboringLanes(Vey, Eks, radius)[0][0]
                    lanePos, dist = sumolib.geomhelper.polygonOffsetAndDistanceToPoint((Vey, Eks), lane.getShape())
                    lane_id = lane.getID()
                else:
                    lane_id = "No Lane Found"

                data.append([Plaka, lon, lat, Hiz, Yon, Zaman, SimTime, Dname, lane_id])

        # Create a dataframe for the current CSV file
        df = pd.DataFrame(data, columns=['Plaka', 'Lon', 'Lat', 'Hiz', 'Yon', 'Zaman', 'SimTime', 'Dname', 'Lane'])
        dfs.append(df)

# Merge dataframes based on the "Zaman" column and sort by "Zaman"
merged_df = pd.concat(dfs).sort_values(by='Zaman')

# Save the merged dataframe to a new CSV file
merged_csv_path = os.path.join(output_dir, 'merged_output.csv')
merged_df.to_csv(merged_csv_path, index=False)

print(f"Merged CSV file created at: {merged_csv_path}")
