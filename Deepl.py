import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ================================
# Load Data
# ================================
# Load the paths with barrier info
df = pd.read_csv('simulated_gbm_paths_with_barrier.csv')

# Separate paths and barrier hits
paths = df.iloc[:, :-1].values  # All columns except the last one
barrier_hits = df['Barrier_Hit'].values  # The last column

# ================================
# Prepare Features and Labels
# ================================
X = []
y = []

for i in range(1, paths.shape[1]):
    spot_prices = paths[:, i - 1]
    next_prices = paths[:, i]
    time_to_maturity = paths.shape[1] - i

    deltas = np.maximum(100 - next_prices, 0)  # Payoff for a put option

    # Fix the dimension mismatch
    X.extend(np.column_stack([spot_prices, np.full_like(spot_prices, time_to_maturity)]))
    y.extend(deltas)

X = np.array(X)
y = np.array(y)

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Convert Training Data to CUDA
# ================================
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

# ================================
# Train XGBoost Regressor (GPU Enabled)
# ================================
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    tree_method="hist",
    device="cuda"  # Enable GPU usage
)

xgb.fit(X_train, y_train)

# ================================
# Evaluate the Model
# ================================
y_pred = xgb.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# ================================
# Plot Predictions vs. Actual Values
# ================================
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.title("Predicted vs. Actual Delta Values")
plt.xlabel("Actual Delta")
plt.ylabel("Predicted Delta")
plt.show()
