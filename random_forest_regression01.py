import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
df = pd.read_csv(r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern.csv")

# --- Feature Engineering ---
# Convert 'day' to datetime
df['day'] = pd.to_datetime(df['day'], errors='coerce')

# Extract time-based features
df['year'] = df['day'].dt.year
df['month'] = df['day'].dt.month
df['dayofweek'] = df['day'].dt.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Convert interval (seconds from midnight) -> hour of day
df['hour'] = (df['interval'] // 3600).astype(int)

# Select features (X) and targets (y)
X = df[['detid', 'year', 'month', 'dayofweek', 'is_weekend', 'hour']]
y = df[['flow', 'occ']]

# Encode categorical detid (one-hot encoding)
X = pd.get_dummies(X, columns=['detid'], drop_first=True)

# --- Train/Test Split (time-based) ---
# Sort by date
df_sorted = df.sort_values('day')

# Use 80% earliest dates for training, 20% latest for testing
split_date = df_sorted['day'].quantile(0.8)
train_idx = df_sorted['day'] <= split_date
test_idx = df_sorted['day'] > split_date

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

# --- Model ---
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Evaluation ---
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)
