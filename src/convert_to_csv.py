import pandas as pd
import os

# Define file paths
input_file = os.path.join('data', 'household_power_consumption.txt')
output_file = os.path.join('data', 'household_power_consumption.csv')

# Read the txt file and convert to CSV
print(f"Reading {input_file}...")
df = pd.read_csv(input_file, sep=';', low_memory=False)

print(f"Shape of data: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Save as CSV
print(f"\nSaving to {output_file}...")
df.to_csv(output_file, index=False)

print(f"✓ Successfully converted to CSV format!")
print(f"File saved: {output_file}")
