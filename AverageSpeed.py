import pandas as pd
import os


def fix_date_and_save(average_speed_df, input_file_path, output_dir):
    # Set a fixed start time (assuming January 1, 2024)
    # start_time = pd.Timestamp('2024-01-01')

    # Convert SimTime to datetime format by adding seconds to the start time
    # average_speed_df['SimTime'] = start_time + pd.to_timedelta(average_speed_df['SimTime'], unit='s')

    # Convert SimTime to the final format of YYYY-MM-DD HH:MM:SS
    # average_speed_df['SimTime'] = average_speed_df['SimTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the modified DataFrame back to a CSV file
    output_file_path = os.path.join(output_dir, 'fixed_' + os.path.basename(input_file_path))
    average_speed_df.to_csv(output_file_path, index=False)

    # Save the time series data with average speed to a new CSV file
    output_file_path = os.path.join(output_dir, 'average_speed.csv')
    average_speed_df.to_csv(output_file_path, index=False)

    df = pd.read_csv(output_file_path, delimiter=',')
    check_continuous_time(df)

    return output_file_path


def process_and_save_time_series(input_file_path, output_dir):
    """
    Read the CSV data, fix the date format, drop specified columns, calculate average speed,
    simplify average speed values, convert datetime format, and save to a new CSV file.
    """
    df = pd.read_csv(input_file_path, delimiter=',')

    # List of column names to drop
    columns_to_drop = ['Yon', 'Dname', 'Lane']

    # Drop the columns
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Group the data by SimTime and calculate the average speed for each SimTime
    average_speed_df = df.groupby('SimTime')['Hiz'].mean().reset_index()

    # Rename the columns
    average_speed_df.columns = ['SimTime', 'AvSpeed']

    # Simplify the values in the "AvSpeed" column to two decimal places, except for zero values
    average_speed_df['AvSpeed'] = average_speed_df['AvSpeed'].apply(lambda x: round(x, 0) if x != 0 else 0)

    return fix_date_and_save(average_speed_df, input_file_path, output_dir)

def check_continuous_time(df):
    # Convert 'SimTime' to pandas datetime format
    df['SimTime'] = pd.to_datetime(df['SimTime'])

    # Calculate time differences between consecutive timestamps
    time_diff = df['SimTime'].diff().dropna()

    # Check if time differences are consistent (i.e., all differences are equal)
    if time_diff.nunique() == 1:
        print("SimTime consists of continuous numbers.")
    else:
        print("SimTime does not consist of continuous numbers.")
        print("Please verify the time sequence in your dataset.")


# Example usage:
# input_file_path = 'C:/Users/CASPER/Desktop/Kodlar/Yollar1/lane_id=24101060_261279742_261174989.csv'



# Example usage
input_file_path = 'C:/Users/CASPER/Desktop/Kodlar/Yollar1/lane_id=24101060_261279742_261174989.csv'
output_dir = 'C:/Users/CASPER/Desktop/Kodlar/fixed_files'

process_and_save_time_series(input_file_path, output_dir)
