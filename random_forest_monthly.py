import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Load data
# -------------------------------
print("Reading CSV...")
data = pd.read_csv(r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern.csv")

sensor_id = "ig11FD208_D4"
data = data[data["detid"] == sensor_id].copy()

# Convert to datetime
data["date"] = pd.to_datetime(data["day"], errors="coerce")
data.dropna(subset=["date"], inplace=True)

# Extract useful features
data["hour"] = (data["interval"] // 3600).astype(int)
data["minute"] = ((data["interval"] % 3600) // 60).astype(int)
data["time_in_hours"] = data["hour"] + data["minute"] / 60.0
data["weekday"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month
data["day_num"] = data["date"].dt.day

# Cyclic time/weekday encoding
data["time_sin"] = np.sin(2 * np.pi * data["time_in_hours"] / 24)
data["time_cos"] = np.cos(2 * np.pi * data["time_in_hours"] / 24)
data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)

# Define feature columns
feature_cols = ["time_sin", "time_cos", "weekday_sin", "weekday_cos"]
target_col = "flow"

# -------------------------------
# Train/test by month (1–23 train, rest test)
# -------------------------------
all_predictions = []   # store predictions for all months
metrics = []           # store performance for each month

print("\nStarting month-by-month training and prediction...\n")

for month in sorted(data["month"].unique()):
    month_data = data[data["month"] == month].copy()
    
    if month_data["day_num"].max() <= 23:
        continue
    
    # Split by day
    train = month_data[month_data["day_num"] <= 23]
    test = month_data[month_data["day_num"] > 23]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    # Train a Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict on the test days
    y_pred = model.predict(X_test)
    
    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics.append({"month": month, "R2": r2, "RMSE": rmse, "MAE": mae})
    
    # Store predictions for this month
    pred_df = test.copy()
    pred_df["flow_pred"] = y_pred
    all_predictions.append(pred_df)
    
    print(f"Month {month:02d}: R²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

# -------------------------------
# Combine and save predictions
# -------------------------------
print("\nCombining all months’ predictions...")
all_predictions_df = pd.concat(all_predictions, ignore_index=True)

# Sort by date & interval
all_predictions_df.sort_values(["date", "interval"], inplace=True)

# Save full dataset of actual vs predicted flows
output_path = f"monthly_predictions_sensor_{sensor_id}.csv"
all_predictions_df.to_csv(output_path, index=False)

print(f"\n All months’ predictions saved to: {output_path}")

# -------------------------------
# Show summary metrics
# -------------------------------
metrics_df = pd.DataFrame(metrics)
print("\n=== Performance Summary per Month ===")
print(metrics_df)
print("\n=== Average Across Months ===")
print(metrics_df[["R2", "RMSE", "MAE"]].mean())

# Optionally, print part of combined predictions
#print("\nSample of combined predictions:")
#print(all_predictions_df.head(10))
