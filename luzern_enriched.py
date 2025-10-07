import pandas as pd

# File paths (adjust to your paths)
luzern_file = r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern_utd19_u.csv"
detectors_file = r"E:\00Studies\Aalto\2ndYear\ML_Project\detectors_public.csv"

# Load both CSVs
print("Loading data...")
df_luzern = pd.read_csv(luzern_file)
df_detectors = pd.read_csv(detectors_file)

# Drop unwanted columns from Luzern file
df_luzern = df_luzern.drop(columns=["error", "city", "speed"], errors="ignore")

# Select only the needed columns from detectors
df_detectors = df_detectors[["detid", "length", "pos", "road", "linkid", "long", "lat"]]

# Merge on detid
df_merged = df_luzern.merge(df_detectors, on="detid", how="left")

# Save result to new CSV
output_file = r"E:\00Studies\Aalto\2ndYear\ML_Project\luzern_enriched.csv"
df_merged.to_csv(output_file, index=False)

print(f"Done! Saved enriched file to {output_file}")
print("Shape of final file:", df_merged.shape)
