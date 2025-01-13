import matplotlib.pyplot as plt
import torch

def plot_pnl_histogram(pnl_bs, pnl_dh, bins=50):
    """
    Compare deux distributions de PnL (BS vs Deep Hedging).
    """
    plt.figure(figsize=(8,5))
    plt.hist(pnl_bs, bins=bins, alpha=0.5, label='Black-Scholes PnL')
    plt.hist(pnl_dh, bins=bins, alpha=0.5, label='Deep Hedging PnL')
    plt.title("Comparaison des distributions de PnL")
    plt.xlabel("PNL")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.show()
