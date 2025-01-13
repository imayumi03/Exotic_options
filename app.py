from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

# Vérification de l'appareil (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Appareil utilisé : {device}")

# =========================
# Partie 1 : Black-Scholes
# =========================
def price_put_down_and_in(S0, K, B, r, sigma, T, M=10000, N=100):
    dt = T / N
    payoffs = np.zeros(M)
    for i in range(M):
        S = S0
        barrier_hit = False
        for j in range(N):
            Z = np.random.normal()
            S = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            if S <= B:
                barrier_hit = True
        if barrier_hit:
            payoffs[i] = max(K - S, 0)
    return np.exp(-r * T) * np.mean(payoffs)

# =========================
# Partie 2 : Deep Hedging
# =========================
def generate_paths_gbm(S0, r, sigma, T, nb_paths=20000, nb_steps=100):
    dt = T / nb_steps
    paths = torch.zeros(nb_paths, nb_steps + 1, device=device)
    paths[:, 0] = S0
    for t in range(nb_steps):
        Z = torch.randn(nb_paths, device=device)
        paths[:, t + 1] = paths[:, t] * torch.exp((r - 0.5 * sigma ** 2) * dt + sigma * torch.sqrt(torch.tensor(dt, device=device)) * Z)
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

def train_deep_hedge(S0, K, B, r, sigma, T, nb_paths=20000, nb_steps=100, epochs=200, lr=1e-3):
    paths = generate_paths_gbm(S0, r, sigma, T, nb_paths, nb_steps).to(device)
    model = HedgingStrategy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        cash = torch.zeros(nb_paths, device=device)

        for t in range(nb_steps):
            St = paths[:, t].unsqueeze(-1).float().to(device)
            deltas = model(St).squeeze(-1)
            dS = (paths[:, t + 1] - paths[:, t]).to(device)
            cash += deltas * dS

        payoff = torch.relu(K - paths[:, -1]).to(device)
        barrier_hit = (torch.min(paths, dim=1).values <= B).to(device)
        payoff = torch.where(barrier_hit, payoff, torch.zeros_like(payoff, device=device))

        PnL = model.cost + cash - payoff
        loss = loss_fn(PnL)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss={loss.item():.4f}, MeanPnL={PnL.mean().item():.4f}, Cost={model.cost.item():.4f}")

    return model.cost.item()

# =========================
# Partie 3 : Interface Streamlit
# =========================
def main():
    st.title("Comparaison : Black-Scholes vs Deep Hedging (Put Down-and-In)")

    st.sidebar.header("Paramètres Option")
    S0 = st.sidebar.slider("Spot initial (S0)", min_value=50, max_value=200, value=100, step=1)
    K = st.sidebar.slider("Strike (K)", min_value=50, max_value=200, value=100, step=1)
    B = st.sidebar.slider("Barrière (B)", min_value=10, max_value=180, value=90, step=1)
    r = st.sidebar.slider("Taux sans risque (r)", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    sigma = st.sidebar.slider("Volatilité (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    T = st.sidebar.slider("Maturité (T, années)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    st.write("## 1) Pricing via Monte Carlo (Black-Scholes)")
    if st.button("Calculer le prix Black-Scholes"):
        price_bs = price_put_down_and_in(S0, K, B, r, sigma, T)
        st.write(f"**Prix Put Down-and-In (Monte Carlo) :** {price_bs:.4f}")
        st.session_state['price_bs'] = price_bs

    st.write("---")

    st.write("## 2) Deep Hedging")
    if st.button("Calculer le prix Deep Hedging"):
        dh_price = train_deep_hedge(S0, K, B, r, sigma, T)
        st.write(f"**Prix Put Down-and-In (Deep Hedging) :** {dh_price:.4f}")
        st.session_state['price_dh'] = dh_price

    if 'price_bs' in st.session_state and 'price_dh' in st.session_state:
        st.write("### Comparaison des résultats :")
        fig, ax = plt.subplots()
        ax.bar(["Black-Scholes", "Deep Hedging"], [st.session_state['price_bs'], st.session_state['price_dh']])
        ax.set_ylabel("Prix")
        ax.set_title("Comparaison Black-Scholes vs Deep Hedging")
        st.pyplot(fig)

if __name__ == "__main__":
    main()