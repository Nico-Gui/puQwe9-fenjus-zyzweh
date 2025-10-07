import os

# Path to the folder where you saved per-city CSVs
split_folder = "utd19_splits"

# Get all .csv files in that folder
files = [f for f in os.listdir(split_folder) if f.endswith(".csv")]

# Extract city names (remove "_utd19_u.csv")
cities = [f.replace("_utd19_u.csv", "") for f in files]

# Sort alphabetically
cities = sorted(cities)

# Save to file
with open("unique_cities.txt", "w", encoding="utf-8") as f:
    for city in cities:
        f.write(city + "\n")

print(f"Found {len(cities)} unique cities.")
print("Sample:", cities[:20])
