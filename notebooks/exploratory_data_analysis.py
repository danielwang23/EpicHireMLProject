import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Step 1: Checking current working directory:", os.getcwd())

# Change the working directory to the Data-Science-Team folder
project_dir = '/Users/danielwang/EpicHire/Data-Science-Team'
os.chdir(project_dir)

print("Step 2: Changed new working directory to:", os.getcwd())

# Load the dataset
print("Reading CSV file...")
df = pd.read_csv('data/job_descriptions.csv')
print("CSV file read successfully. Number of records:", len(df))


print("Columns in the dataset:", df.columns)


# Extracting the first 100 job postings
print("Extracting the first 100 job postings...")
df_top_100 = df.head(100)
print(f"Extracted {len(df_top_100)} job postings.")


# Select key columns to include in the table
print("Step 5: Selecting key columns...")
key_columns = ['Job Title', 'Experience', 'Qualifications', 'skills', 'location', 'Company']

# Check if all key columns are present in the dataset
available_columns = [col for col in key_columns if col in df.columns]
print(f"Step 5: Available columns selected: {available_columns}")


# Create a new DataFrame with only the key columns
# df_filtered = df_top_100[available_columns].dropna()

# Instead of dropping rows with any NaN, drop only those missing critical columns
print("Step 6: Filtering rows (keeping those with Job Title and Experience)...")
df_filtered = df_top_100[available_columns]
# df_filtered = df_top_100[available_columns].dropna(subset=['Job Title', 'Experience'])
print(f"Step 6: Number of rows after filtering: {len(df_filtered)}")

# Display the resulting table (first 100 rows)
print("Key descriptions for the first 100 job postings:")
print(df_filtered)

# Optional: Save the table to a CSV file
output_file = 'data/adjusted_top_100_job_descriptions.csv'
print(f"Step 8: Saving the filtered DataFrame to {output_file}...")
df_filtered.to_csv(output_file, index=False)
print("File saved successfully.")

