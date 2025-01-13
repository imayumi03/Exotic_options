import numpy as np
import math
from scipy.stats import norm

# ✅ Fonction utilitaire (ancienne fonction dans utils.py)
def put_vanilla_bs_price(S0, K, r, sigma, T):
    """
    Calcule le prix d'une option Put vanille selon le modèle de Black-Scholes.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price

# ✅ Fonction principale de simulation Monte Carlo
def simulate_paths_mc_di_put(S0, K, B, r, sigma, T, M=10000, N=100):
    """
    Simule le prix d'une option Put Down-and-In via Monte Carlo.
    """
    dt = T / N
    Z = np.random.normal(size=(M, N))
    payoffs = np.zeros(M)
    
    for i in range(M):
        S = S0
        barrier_hit = False
        for j in range(N):
            S = S * math.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z[i, j])
            if S <= B:
                barrier_hit = True
        
        if barrier_hit:
            payoffs[i] = max(K - S, 0)
    
    price = math.exp(-r * T) * np.mean(payoffs)
    return price

# ✅ Fonction pour choisir la méthode de pricing
def price_put_down_and_in(S0, K, B, r, sigma, T, method="mc"):
    """
    Calcule le prix d'une option Put Down-and-In.
    """
    if method == "mc":
        return simulate_paths_mc_di_put(S0, K, B, r, sigma, T)
    else:
        raise NotImplementedError("Seule la méthode Monte Carlo est implémentée.")
