import os
import pandas as pd
import argparse

# Step 1: Run the NVIDIA Nsight Compute profiler
def run_ncu_profiler(profiler_output_path, python_script_path, script_args, start, count):
    # Build the command with optional start and count arguments
    launch_skip = f"--launch-skip {start}" if start is not None else ""
    launch_count = f"--launch-count {count}" if count is not None else ""
    command = f"ncu --call-stack --set roofline {launch_skip} {launch_count} -o {profiler_output_path} python {python_script_path} {script_args}"
    os.system(command)

# Step 2: Export the raw data as a CSV
def export_raw_csv(ncu_report_path, raw_csv_output_path):
    command = f"ncu --import {ncu_report_path} --csv --section SpeedOfLight > {raw_csv_output_path}"
    os.system(command)

# Step 3: Transform raw CSV into user-friendly format
def transform_csv(raw_csv_path, output_csv_path):
    if not os.path.exists(raw_csv_path) or os.path.getsize(raw_csv_path) == 0:
        print("No data available in the raw CSV. Skipping transformation.")
        return
    # Read the raw CSV
    data = pd.read_csv(raw_csv_path)

    # Restructure the data
    restructured_data = data.pivot_table(
        index=["ID", "Kernel Name", "Device", "CC"],
        columns="Metric Name",
        values="Metric Value",
        aggfunc="first"  # Handle duplicates by taking the first value
    ).reset_index()

    # Flatten the columns
    restructured_data.columns.name = None  # Remove the column group name
    restructured_data.columns = [str(col) for col in restructured_data.columns]

    # Save the transformed data
    restructured_data.to_csv(output_csv_path, index=False)
    print(f"Transformed CSV saved to: {output_csv_path}")

# Step 4: Calculate additional metrics and add columns
def calc_metrics(transformed_csv_path, output_with_metrics_path):
    if not os.path.exists(transformed_csv_path):
        print("Transformed CSV not found. Cannot calculate additional metrics.")
        return

    # Load the CSV file
    data = pd.read_csv(transformed_csv_path)

    # Calculate Frequency [Hz]
    data['Frequency [Hz]'] = data['SM Frequency'].apply(lambda x: x * 1e9 if x < 100 else x * 1e6)

    # Calculate SM Time [Sec]
    data['SM Time [Sec]'] = data['SM Active Cycles'] / data['Frequency [Hz]']

    #Calculate Total Time [Sec]
    data['Total Kernel Time [Sec]'] = data['Elapsed Cycles'] / data['Frequency [Hz]']

    #Calculate
    data['Kernel Utilization'] = data['SM Time [Sec]' / data['Total Kernel Time [Sec]']

    # Save the updated CSV
    data.to_csv(output_with_metrics_path, index=False)
    print(f"CSV with additional metrics saved to: {output_with_metrics_path}")

# Main function to integrate the steps
def main():
    parser = argparse.ArgumentParser(description="Automate NVIDIA Nsight Compute profiling and CSV transformation.")
    parser.add_argument("python_script_path", type=str, help="Path to the Python script to profile.")
    parser.add_argument("output_csv_path", type=str, help="Path to save the final user-friendly CSV file.")
    parser.add_argument("--start", type=int, default=None, help="Number of kernels to skip before starting profiling (default: start from the beginning).")
    parser.add_argument("--count", type=int, default=None, help="Number of kernels to profile (default: profile all kernels).")
    parser.add_argument("--script-params", type=str, nargs=argparse.REMAINDER, help="Additional parameters for the Python script being profiled. Include flags such as --compile, --max_new_tokens, etc.")

    args = parser.parse_args()

    profiler_output_path = os.path.splitext(args.output_csv_path)[0]  # Remove the .csv extension
    ncu_report_path = profiler_output_path + ".ncu-rep"
    raw_csv_output_path = profiler_output_path + "_raw.csv"
    output_with_metrics_path = profiler_output_path + "_with_metrics.csv"

    # Build additional parameters string for the script
    script_params = " ".join(args.script_params) if args.script_params else ""

    # Step 1: Run the profiler
    print("Running NVIDIA Nsight Compute profiler...")
    run_ncu_profiler(profiler_output_path, args.python_script_path, script_params, args.start, args.count)

    # Step 2: Export the raw CSV
    print("Exporting raw CSV from profiler report...")
    export_raw_csv(ncu_report_path, raw_csv_output_path)

    # Step 3: Transform to user-friendly CSV
    print("Transforming raw CSV into user-friendly format...")
    transform_csv(raw_csv_output_path, args.output_csv_path)

    # Step 4: Calculate additional metrics
    print("Calculating additional metrics and saving to new CSV...")
    calc_metrics(args.output_csv_path, output_with_metrics_path)

if __name__ == "__main__":
    main()
