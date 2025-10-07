import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Loading CSV
print("Reading csv")
data = pd.read_csv(r"C:\Users\ziaulm1\OneDrive - Aalto University\Documents\luzern.csv")

# --- Feature Engineering ---
# Filter data for one specific sensor
sensor_id = "ig11FD208_D4"  # <-- change to your desired detid
data = data[data['detid'] == sensor_id]

# Droping unnecessary columns
data = data[['interval', 'day', 'flow', 'occ']]

# Converting 'date' to datetime and extract useful features
print("Converting date to day, month and year")
data['date'] = pd.to_datetime(data['day'], errors='coerce')
data['weekday'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Defining features and labels
# Inputs (X): interval, day, month, year
# Outputs (y): flow and occupancy
X = data[['interval', 'weekday', 'month', 'year']]
y = data[['flow', 'occ']]

# Splitting data into training and test sets
print("Spitting for training")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initializing and training model
print("Training the model")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # use all CPU cores
)
model.fit(X_train, y_train)

# Predicting and evaluating
y_pred = model.predict(X_test)

# Converting to DataFrame for comparison
pred_df = pd.DataFrame(y_pred, columns=['flow_pred', 'occupancy_pred'])

# Evaluating performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model Performance for Sensor ID {sensor_id}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Saving predictions (optional)
result = X_test.copy()
result[['flow_actual', 'occupancy_actual']] = y_test.values
result[['flow_pred', 'occupancy_pred']] = pred_df.values
result.to_csv(f"predictions_sensor_{sensor_id}.csv", index=False)

print(f"\nPredictions saved to predictions_sensor_{sensor_id}.csv")