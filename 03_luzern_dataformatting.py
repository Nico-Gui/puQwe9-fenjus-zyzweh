import pandas as pd

# File paths
luzern_file = r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern_utd19_u.csv" #Address for city.csv file
detectors_file = r"E:\00Studies\Aalto\2ndYear\ML_Project\detectors_public.csv" #Address for detectors_public.csv

# Load CSVs
print("Loading Luzern dataset...")
df_luzern = pd.read_csv(luzern_file)

print("Loading detectors dataset...")
df_detectors = pd.read_csv(detectors_file)

# 1. Cleaned Luzern (drop columns)
df_luzern_clean = df_luzern.drop(columns=["error", "city", "speed"], errors="ignore")

clean_output = r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern.csv"
df_luzern_clean.to_csv(clean_output, index=False)
print(f"Saved cleaned Luzern file to {clean_output}")

# 2. Detector metadata for only the detids in Luzern
luzern_detids = df_luzern["detid"].unique()
df_detectors_sub = df_detectors[df_detectors["detid"].isin(luzern_detids)]

# Select only needed columns
df_detectors_sub = df_detectors_sub[["detid", "length", "pos", "road", "linkid", "long", "lat"]]

det_output = r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern_detectors_meta.csv"
df_detectors_sub.to_csv(det_output, index=False)
print(f"Saved detector metadata file to {det_output}")
