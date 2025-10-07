import pandas as pd
import matplotlib.pyplot as plt



# Load the file
print("Loading csv")
df = pd.read_csv(r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern.csv")


# Choose one detector ID
my_detid = "ig11FD208_D4"   # The detid you want
print("Sensor ID is ", my_detid)
df = df[df["detid"] == my_detid]
print(df)

# --- 1. Monthly traffic histogram ---
df["day"] = pd.to_datetime(df["day"])
df["month"] = df["day"].dt.to_period("M")

monthly_flow = df.groupby("month")["flow"].sum()

plt.figure(figsize=(10,5))
monthly_flow.plot(kind="bar")
plt.title(f"Total Traffic Flow per Month (detid={my_detid})")
plt.xlabel("Month")
plt.ylabel("Total Flow")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
