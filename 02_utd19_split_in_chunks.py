"""
Usage:
    python split_dataset.py PATH_TO_FILE COLUMN_NAME TARGET_DIRECTORY

Example:
    python split_dataset.py utd19_u.csv city utd19_splits
"""

import sys
import os
import pandas as pd


def ensure_dir_and_gitignore(dir_path):
    """
    Ensure a directory exists, and add it to .gitignore if not already listed.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

    if not os.path.exists(".gitignore"):
        open(".gitignore", "w").close()

    with open(".gitignore", "r+") as f:
        lines = [line.strip() for line in f.readlines()]
        if dir_path not in lines and f"{dir_path}/" not in lines:
            f.write(f"{dir_path}/\n")
            print(f"Added '{dir_path}/' to .gitignore")
        else:
            print(f"'{dir_path}/' is already ignored in .gitignore")


def main():
    if len(sys.argv) < 4:
        print("Error: You must specify: PATH_TO_FILE COLUMN_NAME TARGET_DIRECTORY")
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    colname = sys.argv[2]
    output_dir = sys.argv[3]

    ensure_dir_and_gitignore(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {input_file} in chunks...")

    written_files = {}  # track which files already got headers

    # Read dataset in chunks of 100k rows
    for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=100000)):
        if colname not in chunk.columns:
            print(f"Error: Column '{colname}' not found in {input_file}.")
            print("Available columns:", list(chunk.columns))
            sys.exit(1)

        # Split this chunk by city
        for val, subset in chunk.groupby(colname):
            safe_val = str(val).replace(" ", "_").replace("/", "-")
            outfile = os.path.join(output_dir, f"{safe_val}_{os.path.basename(input_file)}")

            # Append rows if file exists, otherwise create new file with header
            subset.to_csv(outfile,
                          mode='a',
                          index=False,
                          header=not os.path.exists(outfile))

            if outfile not in written_files:
                written_files[outfile] = True
                print(f"Started file {outfile}")

        if (chunk_idx + 1) % 10 == 0:
            print(f"  Processed {chunk_idx+1} chunks...")

    print("Done splitting dataset.")


if __name__ == "__main__":
    main()
