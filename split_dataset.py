"""
Usage:
    python3 -m split_dataset PATH_TO_FILE COLUMN_NAME TARGET_DIRECTORY

Example:
    python3 -m split_dataset city traffic_data.csv city_name traffic_data
"""

import sys
import os
import pandas as pd


def ensure_dir_and_gitignore(dir_path):
	"""
	Ensure a directory exists, and add it to .gitignore if not already listed.
	Parameters:
		dir_path (str): The directory path to create/ignore (e.g. "splits").
		gitignore_path (str): Path to the .gitignore file (default: ".gitignore").
	"""
	# 1. Create directory if it doesn't exist
	if not os.path.exists(dir_path):
		os.makedirs(dir_path, exist_ok=True)
		print(f"Created directory: {dir_path}")
	else:
		print(f"Directory already exists: {dir_path}")

	# 2. Update .gitignore
	if not os.path.exists(".gitignore"):
		open(".gitignore", "w").close()  # create empty .gitignore if missing

	with open(".gitignore", "r+") as f:
		lines = [line.strip() for line in f.readlines()]
		if dir_path not in lines and f"{dir_path}/" not in lines:
			f.write(f"{dir_path}/\n")
			print(f"Added '{dir_path}/' to .gitignore")
		else:
			print(f"'{dir_path}/' is already ignored in .gitignore")


def main():
	if len(sys.argv) < 4:
		print("Error: You must specify the column name to split on, the path to the dataset and the target directory")
		print(__doc__)
		sys.exit(1)

	target_dir = sys.argv[3]
	colname = sys.argv[2]
	data_set = sys.argv[1]

	# Optional arguments
	input_file = data_set
	output_dir = target_dir
	ensure_dir_and_gitignore(output_dir)

	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)

	# Load dataset
	print(f"Loading {input_file} ...")
	df = pd.read_csv(input_file)

	if colname not in df.columns:
		print(f"Error: Column '{colname}' not found in {input_file}.")
		sys.exit(1)

	# Split by unique values
	unique_vals = df[colname].dropna().unique()
	print(f"Splitting dataset into {len(unique_vals)} parts based on '{colname}' ...")

	for val in unique_vals:
		subset = df[df[colname] == val]
		safe_val = str(val).replace(" ", "_").replace("/", "-")
		outfile = os.path.join(output_dir, f"{safe_val}_{input_file.rsplit('/', 1)[-1]}")
		subset.to_csv(outfile, index=False)
		print(f"  Saved {len(subset)} rows to {outfile}")

	print("Done.")


if __name__ == "__main__":
	main()
