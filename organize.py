import os
import pandas as pd

# Step: Transform raw CSV into user-friendly format
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

# Example usage
if __name__ == "__main__":
    # Replace 'raw_csv_path' and 'output_csv_path' with actual file paths
    raw_csv_path = "./vllm_output_test.csv"
    output_csv_path = "./output_test_3.csv"

    print("Transforming raw CSV into user-friendly format...")
    transform_csv(raw_csv_path, output_csv_path)
