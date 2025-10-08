import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
data = pd.read_csv(r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern.csv")

# --- Select one detector ID and one date ---
sensor_id = "ig11FD208_D4"  # Change to your detid
selected_day = "2015-05-15"  # Change to the date you want (format YYYY-MM-DD)

# --- Filter data for that sensor and day ---
data['day'] = pd.to_datetime(data['day'])
data = data[data['detid'] == sensor_id]
data_day = data[data['day'] == selected_day]

# --- Convert 'interval' (in seconds) to hour of day ---
data_day['hour'] = (data_day['interval'] // 3600) % 24

# --- Group by hour and compute total or average flow ---
hourly_flow = data_day.groupby('hour')['flow'].mean().reset_index()

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(hourly_flow['hour'], hourly_flow['flow'], marker='o', linestyle='-', color='b')
plt.title(f"Hourly Traffic Flow on {selected_day} for {sensor_id}")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Flow")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.show()