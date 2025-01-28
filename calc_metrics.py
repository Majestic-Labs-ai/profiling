import pandas as pd
import argparse
import os

def calc_metrics(input_csv_path, output_csv_path):
    if not os.path.exists(input_csv_path):
        print(f"Input CSV not found: {input_csv_path}. Cannot calculate additional metrics.")
        return

    # Load the CSV file
    data = pd.read_csv(input_csv_path)

    # Check for required columns
    required_columns = ['SM Frequency', 'SM Active Cycles']
    for col in required_columns:
        if col not in data.columns:
            print(f"Missing required column in the input CSV: {col}")
            return

    # Calculate Frequency [Hz]
    data['Frequency [Hz]'] = data['SM Frequency'].apply(lambda x: x * 1e9 if x < 100 else x * 1e6)

    # Calculate SM Time [Sec]
    data['SM Time [Sec]'] = data['SM Active Cycles'] / data['Frequency [Hz]']

    #Calculate Total Time [Sec]
    data['Total Kernel Time [Sec]'] = data['Elapsed Cycles'] / data['Frequency [Hz]']

    #Calculate Kernel Utilization
    data['Kernel Utilization'] = data['SM Time [Sec]'] / data['Total Kernel Time [Sec]']

    #Calculate compute utilization
    data['Compute Utilization'] = data['Compute (SM) Throughput'] * data['Duration']

    sum_duration = data['Duration'].sum()
    sum_compute = data['Compute Utilization'].sum()
    average_util = sum_compute/sum_duration if sum_duration != 0 else 0
    data['Average Utilization'] = average_util

    # Save the updated CSV
    data.to_csv(output_csv_path, index=False)
    print(f"CSV with additional metrics saved to: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate additional metrics for profiling CSV.")
    parser.add_argument("input_csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_csv_path", type=str, help="Path to save the output CSV with additional metrics.")

    args = parser.parse_args()

    # Run the metrics calculation
    calc_metrics(args.input_csv_path, args.output_csv_path)

if __name__ == "__main__":
    main()
