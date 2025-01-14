import yfinance as yf
import numpy as np
import pandas as pd

# Step 1: Download historical data for AAPL
ticker = "AAPL"
data = yf.download(ticker, start="2018-01-01", end="2023-01-01")

# Step 2: Calculate daily returns and realized volatility (using a rolling window of 20 days)
data['LogReturn'] = np.log(data['Close']).diff()
data['RealizedVol'] = data['LogReturn'].rolling(window=20).std() * np.sqrt(252)  # annualized vol

# Step 3: Generate synthetic GBM paths based on historical volatility
def generate_gbm_paths(S0, r, sigma, T, N, M, B):
    """
    Generate M paths using Geometric Brownian Motion with barrier check.
    Returns:
      - paths: (M, N+1) simulated paths
      - barrier_hit: (M,) boolean array indicating if the barrier was hit
    """
    dt = T / N
    paths = np.zeros((M, N+1))
    paths[:, 0] = S0
    barrier_hit = np.zeros(M, dtype=bool)

    for t in range(1, N+1):
        Z = np.random.normal(size=M)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Use np.logical_or to avoid type mismatches
        barrier_hit = np.logical_or(barrier_hit, paths[:, t] <= B)

    return paths, barrier_hit

# Step 4: Generate paths with realistic parameters
S0 = float(data['Close'].iloc[-1])  # Ensure S0 is a float
r = 0.01  # risk-free rate
T = 1  # 1 year to maturity
N = 252  # 252 trading days
M = 10000  # number of paths
B = S0 * 0.9  # barrier set at 90% of the current spot price
sigma = data['RealizedVol'].mean()  # use the average realized volatility

paths, barrier_hit = generate_gbm_paths(S0, r, sigma, T, N, M, B)

# Convert to DataFrame for easier visualization and saving
paths_df = pd.DataFrame(paths)
paths_df['Barrier_Hit'] = barrier_hit

# Save the generated paths
paths_df.to_csv('simulated_gbm_paths_with_barrier.csv', index=False)

# Output the number of paths that hit the barrier
print(f"Number of paths that hit the barrier: {barrier_hit.sum()}")
