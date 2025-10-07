filename = "E:/00Studies/Aalto/2ndYear/ML_Project/utd19_u.csv"

with open(filename, "r", encoding="utf-8", errors="ignore") as f:
   row_count = sum(1 for _ in f)

print(f"Total rows (including header): {row_count}")
print(f"Data rows (excluding header): {row_count - 1}")

filename = "E:/00Studies/Aalto/2ndYear/ML_Project/utd19_u.csv"

with open(filename, "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline().strip()

print("Headers are:")
print(header.split(","))