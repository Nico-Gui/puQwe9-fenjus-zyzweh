import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -------------------------------
# 1. Load and filter data
# -------------------------------
print("Reading csv...")
data = pd.read_csv(r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern.csv")

# Choose one detector
sensor_id = "ig11FD208_D4"
data = data[data["detid"] == sensor_id].copy()

# Convert date column
print("Parsing dates...")
data["date"] = pd.to_datetime(data["day"], errors="coerce")
data = data.dropna(subset=["date"])

# -------------------------------
# 2. Feature engineering
# -------------------------------
print("Creating time features...")

# Extract time-of-day features from 'interval' (assuming seconds since midnight)
data["hour"] = (data["interval"] // 3600).astype(int)
data["minute"] = ((data["interval"] % 3600) // 60).astype(int)
data["time_in_hours"] = data["hour"] + data["minute"]/60.0

# Periodic (cyclic) encoding for time of day (24h cycle)
data["time_sin"] = np.sin(2 * np.pi * data["time_in_hours"] / 24)
data["time_cos"] = np.cos(2 * np.pi * data["time_in_hours"] / 24)

# Day-of-week and month
data["weekday"] = data["date"].dt.dayofweek  # 0=Monday
data["month"] = data["date"].dt.month
data["year"] = data["date"].dt.year

# Cyclic encoding for weekdays (7-day cycle)
data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)

# -------------------------------
# 3. Select features and target
# -------------------------------
feature_cols = [
    "time_sin", "time_cos",
    "weekday_sin", "weekday_cos",
    "month"
]
X = data[feature_cols]
y = data["flow"]

# -------------------------------
# 4. Time-based split: train first 10 months, test on 11th
# -------------------------------
print("Splitting train/test by date...")
train_data = data[data["date"].dt.month <= 11]
test_data = data[data["date"].dt.month == 12]

X_train = train_data[feature_cols]
y_train = train_data["flow"]
X_test = test_data[feature_cols]
y_test = test_data["flow"]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# -------------------------------
# 5. Train model
# -------------------------------
print("Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate model
# -------------------------------
print("Evaluating model...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance for Sensor {sensor_id}")
print(f"R² Score:        {r2:.3f}")
print(f"RMSE:            {rmse:.2f}")
print(f"MAE:             {mae:.2f}")

# -------------------------------
# 7. Feature importance
# -------------------------------
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importances:")
print(importances)

# -------------------------------
# 8. Save predictions (optional)
# -------------------------------
results = test_data[["date", "interval", "flow"]].copy()
results["flow_pred"] = y_pred
results.to_csv(f"predictions_sensor_{sensor_id}.csv", index=False)

print(f"\nPredictions saved to predictions_sensor_{sensor_id}.csv")
