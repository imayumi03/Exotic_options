import torch
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os

# Fix for duplicate library error in Streamlit
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Appareil utilisé : {device}")

# =========================
# Black-Scholes Pricing Function
# =========================

def price_put_down_and_in(S0, K, B, r, sigma, T, M=10000, N=100):
    """
    Simulate M paths for a put down-and-in option using Monte Carlo.
    Returns:
      - Simulated price
      - PnL values for each path
      - Average deltas at each step
    """
    dt = torch.tensor(T / N, device=device)
    payoffs = torch.zeros(M, device=device)
    pnl_values = []
    sum_deltas_per_step = torch.zeros(N, device=device)

    for i in range(M):
        S = torch.tensor([S0], device=device)  # Ensure S is a tensor with shape [1]
        barrier_hit = False
        pnl = torch.tensor([0.0], device=device)  # Initialize pnl as a tensor with shape [1]

        for j in range(N):
            S_prev = S.clone()
            Z = torch.randn(1, device=device)
            S = S * torch.exp((r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * Z)

            if S <= B:
                barrier_hit = True

            # Calculate Delta for a European put
            remaining_time = torch.tensor(T - j * dt.item(), device=device)
            delta_put = delta_put_black_scholes(S_prev, K, r, sigma, remaining_time)

            sum_deltas_per_step[j] += delta_put
            pnl += delta_put * (S - S_prev)  # Ensure consistent tensor shapes

        # Payoff if barrier was hit
        payoff = torch.relu(K - S) if barrier_hit else torch.tensor([0.0], device=device)
        pnl -= payoff
        pnl_values.append(pnl.item())
        payoffs[i] = payoff

    discounted_price = torch.exp(-r * T) * torch.mean(payoffs)
    avg_deltas = sum_deltas_per_step / M

    return discounted_price.item(), pnl_values, avg_deltas.cpu().numpy()

# =========================
# Black-Scholes Delta Function
# =========================

def delta_put_black_scholes(S, K, r, sigma, T):
    S = torch.tensor(S, device=device)
    K = torch.tensor(K, device=device)
    r = torch.tensor(r, device=device)
    sigma = torch.tensor(sigma, device=device)
    T = torch.tensor(T, device=device)

    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    delta_call = torch.exp(-r * T) * 0.5 * (1 + torch.erf(d1 / torch.sqrt(torch.tensor(2.0, device=device))))
    delta_put = delta_call - torch.exp(-r * T)
    return delta_put

# =========================
# Deep Hedging Model
# =========================

class DeepHedgingLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(DeepHedgingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        delta = self.fc(lstm_out)
        return delta.squeeze(-1)

# =========================
# Training Deep Hedging Model
# =========================

def train_deep_hedge_lstm(S0, K, B, r, T, nb_paths=10000, nb_steps=50, epochs=200, lr=1e-3):
    dt = torch.tensor(T / nb_steps, device=device)
    paths = torch.zeros(nb_paths, nb_steps + 1, device=device)
    paths[:, 0] = torch.tensor(S0, device=device)

    for i in range(nb_steps):
        Z = torch.randn(nb_paths, device=device)
        paths[:, i + 1] = paths[:, i] * torch.exp((r - 0.5 * 0.2 ** 2) * dt + 0.2 * torch.sqrt(dt) * Z)

    paths = paths.unsqueeze(-1)
    model = DeepHedgingLSTM(input_dim=1, hidden_dim=64, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        cash = torch.zeros(nb_paths, device=device)

        for t in range(1, paths.shape[1]):
            St_history = paths[:, :t].to(device)
            delta = model(St_history)[:, -1]
            dS = (paths[:, t, 0] - paths[:, t - 1, 0])
            cash += delta * dS

        final_S = paths[:, -1, 0]
        payoff = torch.relu(K - final_S)
        barrier_hit = (torch.min(paths[:, :, 0], dim=1).values <= B)
        payoff = torch.where(barrier_hit, payoff, torch.zeros_like(payoff))

        PnL = cash - payoff
        loss = torch.mean(torch.relu(-PnL))

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return model, loss_history, cash.detach().cpu().numpy()

# =========================
# Streamlit Interface
# =========================

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

    st.write("## 2) Deep Hedging")
    if st.button("Calculer le prix Deep Hedging"):
        model, loss_history, pnl_dh_values = train_deep_hedge_lstm(S0, K, B, r, T)
        st.write(f"**Prix Put Down-and-In (Deep Hedging) :** {model.cost.item():.4f}")

if __name__ == "__main__":
    main()