import torch
import numpy as np
import math
import streamlit as st
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Vérification GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Appareil utilisé : {device}")

# =========================
# Partie 1 : Black-Scholes
# =========================

def price_put_down_and_in(S0, K, B, r, sigma, T, M=10000, N=100):
    """
    On simule M trajectoires pour un put down-and-in.
    On renvoie :
      - Le prix simulé
      - Un vecteur de taille M contenant les PnL
      - Un vecteur de taille N contenant la MOYENNE des deltas à chaque pas de temps
    """
    dt = T / N

    payoffs = np.zeros(M)
    pnl_values = []
    sum_deltas_per_step = np.zeros(N)

    for i in range(M):
        S = S0
        barrier_hit = False
        pnl = 0.0

        for j in range(N):
            S_prev = S
            Z = np.random.normal()
            S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            if S <= B:
                barrier_hit = True

            remaining_time = T - j * dt
            delta_put = delta_put_black_scholes(S_prev, K, r, sigma, remaining_time)

            sum_deltas_per_step[j] += delta_put
            pnl += delta_put * (S - S_prev)

        payoff = max(K - S, 0) if barrier_hit else 0
        pnl -= payoff
        pnl_values.append(pnl)
        payoffs[i] = payoff

    discounted_price = np.exp(-r * T) * np.mean(payoffs)
    avg_deltas = sum_deltas_per_step / M

    return discounted_price, pnl_values, avg_deltas

# Fonction pour le calcul du Delta pour un put sous Black-Scholes
def delta_put_black_scholes(S, K, r, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta_call = np.exp(-r * T) * 0.5 * (1 + math.erf(d1 / np.sqrt(2)))
    delta_put = delta_call - np.exp(-r * T)
    return delta_put

# =========================
# Partie 2 : Heston Deep Hedging
# =========================

def generate_paths_heston(S0, r, T, nb_paths=10000, nb_steps=50, v0=0.04, kappa=2.0, theta=0.04, xi=0.1, rho=-0.7):
    dt = T / nb_steps
    dt_t = torch.tensor(dt, device=device).float()

    paths = torch.zeros(nb_paths, nb_steps + 1, device=device).float()
    volatilities = torch.zeros(nb_paths, nb_steps + 1, device=device).float()

    paths[:, 0] = S0
    volatilities[:, 0] = v0

    for t in range(nb_steps):
        Z1 = torch.randn(nb_paths, device=device)
        Z2 = torch.randn(nb_paths, device=device)
        W1 = Z1
        W2 = rho * Z1 + torch.sqrt(torch.tensor(1 - rho**2, device=device)) * Z2

        vt = volatilities[:, t].clamp(min=0)
        volatilities[:, t + 1] = volatilities[:, t] + kappa * (theta - vt) * dt_t + xi * torch.sqrt(vt) * torch.sqrt(dt_t) * W2
        paths[:, t + 1] = paths[:, t] * torch.exp((r - 0.5 * vt) * dt_t + torch.sqrt(vt) * torch.sqrt(dt_t) * W1)

    return paths

class HedgingStrategy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.cost = nn.Parameter(torch.tensor(7.0))

    def forward(self, x):
        return self.net(x)

def loss_fn(PnL):
    return torch.mean(torch.relu(-PnL))

def train_deep_hedge(S0, K, B, r, T, nb_paths=10000, nb_steps=50, epochs=200, lr=1e-3):
    paths = generate_paths_heston(S0, r, T, nb_paths=nb_paths, nb_steps=nb_steps).to(device)
    model = HedgingStrategy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    sum_delta_at_step = torch.zeros(nb_steps, device=device, requires_grad=False)

    for epoch in range(epochs):
        optimizer.zero_grad()
        cash = torch.zeros(nb_paths, device=device)

        for t in range(nb_steps):
            St = paths[:, t].unsqueeze(-1)
            deltas = model(St).squeeze(-1)
            dS = (paths[:, t + 1] - paths[:, t])

            cash += deltas * dS
            if epoch == epochs - 1:
                sum_delta_at_step[t] += deltas.mean().detach()

        final_S = paths[:, -1]
        payoff = torch.relu(K - final_S)
        barrier_hit = (torch.min(paths, dim=1).values <= B)
        payoff = torch.where(barrier_hit, payoff, torch.zeros_like(payoff))

        PnL = model.cost + cash - payoff
        loss = loss_fn(PnL)

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    final_cost = model.cost.item()
    final_PnL = PnL.detach().cpu().numpy()
    avg_deltas_deep_hedge = sum_delta_at_step.cpu().numpy()

    return final_cost, final_PnL, avg_deltas_deep_hedge, loss_history

def plot_pnl_histogram(pnl_bs_values, pnl_dh_values):
    bins = np.linspace(min(min(pnl_bs_values), min(pnl_dh_values)), max(max(pnl_bs_values), max(pnl_dh_values)), 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pnl_bs_values, bins=bins, alpha=0.7, color='red', label="Black-Scholes PnL (Benchmark)")
    ax.hist(pnl_dh_values, bins=bins, alpha=0.7, color='blue', label="Deep-Hedging PnL")
    ax.set_title("Profit and Loss (PnL) Histogram (Adjusted)")
    ax.set_xlabel("Profit and Loss (PnL)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

def plot_delta_curve(delta_bs_values, delta_dh_values):
    min_length = min(len(delta_bs_values), len(delta_dh_values))
    delta_bs_values = delta_bs_values[:min_length]
    delta_dh_values = delta_dh_values[:min_length]
    steps = np.arange(min_length)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, delta_bs_values, 'r-', label="Black-Scholes Delta (Benchmark)")
    ax.plot(steps, delta_dh_values, 'b-', label="Deep-Hedging Delta")
    ax.set_title("Hedging Strategy: Delta")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Delta")
    ax.legend()
    st.pyplot(fig)

def plot_loss_function(loss_history, price_bs):
    epochs = np.arange(len(loss_history))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(epochs, loss_history, 'b-', label="Deep-Hedging Loss")
    ax.axhline(y=price_bs, color='red', linestyle='--', label="Black-Scholes Loss (Benchmark)")
    ax.set_title("Loss Function")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss Value")
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Comparaison : Modèle Classique vs Réseaux de Neurones (Put Down-and-In)")
    st.sidebar.header("Paramètres Option")
    S0 = st.sidebar.slider("Spot initial (S0)", min_value=50, max_value=200, value=100, step=1)
    K = st.sidebar.slider("Strike (K)", min_value=50, max_value=200, value=100, step=1)
    B = st.sidebar.slider("Barrière (B)", min_value=10, max_value=180, value=90, step=1)
    r = st.sidebar.slider("Taux sans risque (r)", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    T = st.sidebar.slider("Maturité (T, années)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    st.write("## 1) Pricing via Monte Carlo (Black-Scholes)")
    if st.button("Calculer le prix Black-Scholes"):
        price_bs, pnl_bs_values, delta_bs_values = price_put_down_and_in(S0, K, B, r, sigma=0.2, T=T)
        st.write(f"**Prix Put Down-and-In (Monte Carlo) :** {price_bs:.4f}")
        st.session_state['price_bs'] = price_bs
        st.session_state['pnl_bs_values'] = pnl_bs_values
        st.session_state['delta_bs_values'] = delta_bs_values

    st.write("## 2) Deep Hedging")
    if st.button("Calculer le prix Deep Hedging"):
        dh_price, pnl_dh_values, delta_dh_values, loss_history = train_deep_hedge(S0, K, B, r, T)
        st.write(f"**Prix Put Down-and-In (Deep Hedging) :** {dh_price:.4f}")
        st.session_state['price_dh'] = dh_price
        st.session_state['pnl_dh_values'] = pnl_dh_values
        st.session_state['delta_dh_values'] = delta_dh_values
        st.session_state['loss_history'] = loss_history

    if 'price_dh' in st.session_state and 'price_bs' in st.session_state:
        st.write("### Distribution du PnL (Profit and Loss)")
        plot_pnl_histogram(st.session_state['pnl_bs_values'], st.session_state['pnl_dh_values'])
        st.write("### Courbe du Delta")
        plot_delta_curve(st.session_state['delta_bs_values'], st.session_state['delta_dh_values'])
        st.write("### Fonction de perte au fil des itérations")
        plot_loss_function(st.session_state['loss_history'], st.session_state['price_bs'])

if __name__ == "__main__":
    main()