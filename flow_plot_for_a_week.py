import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
data = pd.read_csv(r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern.csv")

# --- Select detector and week range ---
sensor_id = "ig11FD208_D4"  # Change as needed
start_day = "2015-05-21"     # Wednesday
end_day   = "2015-05-24"     # Sunday

# --- Preprocess data ---
data['day'] = pd.to_datetime(data['day'])
data = data[data['detid'] == sensor_id]

# --- Filter for selected week ---
mask = (data['day'] >= start_day) & (data['day'] <= end_day)
data_week = data.loc[mask].copy()

# --- Convert interval (seconds from midnight) to hour ---
data_week['hour'] = (data_week['interval'] // 3600) % 24

# --- Group by day and hour ---
hourly_flow = data_week.groupby(['day', 'hour'])['flow'].mean().reset_index()

# --- Plot ---
plt.figure(figsize=(10, 6))

# Loop through each day and plot its flow curve
for day in hourly_flow['day'].unique():
    daily_data = hourly_flow[hourly_flow['day'] == day]
    plt.plot(daily_data['hour'], daily_data['flow'], marker='o', label=day.strftime('%a %d-%b'))

plt.title(f"Hourly Traffic Flow for {sensor_id} ({start_day} to {end_day})")
plt.xlabel("Hour of Day")
plt.ylabel("Average Flow")
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend(title="Day")
plt.tight_layout()
plt.show()
