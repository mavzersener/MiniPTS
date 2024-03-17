import pandas as pd

# Read the CSV data into a DataFrame
df = pd.read_csv("C:/Users/CASPER/Desktop/VerilerTrafik/simpleModified.csv", delimiter=";")

# Convert SimTime to datetime format
df['SimTime'] = pd.to_datetime(df['SimTime'], unit='s')

# Save the resulting DataFrame to a new CSV file
df.to_csv("C:/Users/CASPER/Desktop/VerilerTrafik/grouped_data.csv", index=False)
