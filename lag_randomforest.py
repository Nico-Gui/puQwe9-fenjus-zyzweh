import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load and preprocess data
# -------------------------------
print("Reading CSV...")
data = pd.read_csv(r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern.csv")

sensor_id = "ig11FD208_D4"
data = data[data["detid"] == sensor_id].copy()

# Convert date
data["date"] = pd.to_datetime(data["day"], errors="coerce")
data.dropna(subset=["date"], inplace=True)

# Extract date parts
data["hour"] = (data["interval"] // 3600).astype(int)
data["minute"] = ((data["interval"] % 3600) // 60).astype(int)
data["time_in_hours"] = data["hour"] + data["minute"] / 60.0
data["weekday"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month
data["day_num"] = data["date"].dt.day

# -------------------------------
# Feature engineering: cyclic time encodings
# -------------------------------
data["time_sin"] = np.sin(2 * np.pi * data["time_in_hours"] / 24)
data["time_cos"] = np.cos(2 * np.pi * data["time_in_hours"] / 24)
data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)
data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

# -------------------------------
# Add lag and rolling features (within each month)
# -------------------------------
print("Adding lag and rolling features...")

lags = [1, 2, 3, 6, 12]  # each = 3min interval * lag → up to 36 min
for lag in lags:
    data[f"flow_lag{lag}"] = data.groupby("month")["flow"].shift(lag)

# Rolling mean & std (6 intervals = 18 minutes window)
data["flow_roll_mean_6"] = data.groupby("month")["flow"].rolling(window=6, min_periods=1).mean().reset_index(level=0, drop=True)
data["flow_roll_std_6"] = data.groupby("month")["flow"].rolling(window=6, min_periods=1).std().reset_index(level=0, drop=True)

# Drop rows with NaN from lag features (start of each month)
data.dropna(inplace=True)

# -------------------------------
# Train-test split: use days 1–23 for training, rest for testing (all months together)
# -------------------------------
train = data[data["day_num"] <= 23]
test = data[data["day_num"] > 23]

feature_cols = [
    "time_sin", "time_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
] + [f"flow_lag{lag}" for lag in lags] + ["flow_roll_mean_6", "flow_roll_std_6"]

X_train = train[feature_cols]
y_train = train["flow"]
X_test = test[feature_cols]
y_test = test["flow"]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -------------------------------
# Train Random Forest
# -------------------------------
print("Training model...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------
# Predict and evaluate
# -------------------------------
print("Evaluating model...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nOverall Model Performance")
print(f"R² Score: {r2:.3f}")
print(f"RMSE:     {rmse:.2f}")
print(f"MAE:      {mae:.2f}")

# -------------------------------
# Month-wise evaluation
# -------------------------------
print("\nMonth-wise performance:")
month_metrics = []
for month in sorted(test["month"].unique()):
    mask = test["month"] == month
    
    r2_m = r2_score(y_test[mask], y_pred[mask])
    rmse_m = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
    mae_m = mean_absolute_error(y_test[mask], y_pred[mask])
    month_metrics.append({"month": month, "R2": r2_m, "RMSE": rmse_m, "MAE": mae_m})
    print(f"Month {month:02d}: R²={r2_m:.3f}, RMSE={rmse_m:.2f}, MAE={mae_m:.2f}")

month_metrics_df = pd.DataFrame(month_metrics)

# -------------------------------
# Save predictions
# -------------------------------
results = test[["date", "interval", "flow", "month"]].copy()
results["flow_pred"] = y_pred
results.to_csv(f"predictions_with_lags_{sensor_id}.csv", index=False)
print(f"\nPredictions saved to predictions_with_lags_{sensor_id}.csv")


# Select only numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
corr_matrix = numeric_data.corr()

"""# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="PuBuGn", linewidths=0.5)
plt.title("Feature Correlation Heatmap for Traffic Flow Data", fontsize=14)
plt.tight_layout()
plt.show()"""