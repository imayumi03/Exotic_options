import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =========================
# Partie Black-Scholes
# =========================

def price_put_down_and_in(S0, K, B, r, sigma, T, M=10000, N=100):
    dt = T / N
    payoffs = np.zeros(M)
    for i in range(M):
        S = S0
        barrier_hit = False
        for j in range(N):
            Z = np.random.normal()
            S = S * math.exp((r - 0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*Z)
            if S <= B:
                barrier_hit = True
        if barrier_hit:
            payoffs[i] = max(K - S, 0)
    return math.exp(-r*T) * np.mean(payoffs)

# =========================
# Partie Deep Hedging
# =========================

def generate_paths_gbm(S0, r, sigma, T, nb_paths=10000, nb_steps=50):
    dt = T / nb_steps
    paths = torch.zeros(nb_paths, nb_steps+1)
    paths[:, 0] = S0
    for t in range(nb_steps):
        Z = torch.randn(nb_paths)
        paths[:, t+1] = paths[:, t] * torch.exp((r - 0.5*sigma**2)*dt + sigma*torch.sqrt(dt)*Z)
    return paths

class HedgingStrategy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: shape (batch_size, 1)
        return self.net(x)

def train_deep_hedge(S0, K, B, r, sigma, T, nb_paths=10000, nb_steps=50, epochs=10, lr=1e-3):
    paths = generate_paths_gbm(S0, r, sigma, T, nb_paths, nb_steps)
    model = HedgingStrategy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        cash = torch.zeros(nb_paths)
        # On calcule deltas sur tous les pas de temps
        for t in range(nb_steps):
            # input shape: (nb_paths,)
            input_t = paths[:, t].unsqueeze(-1).float()  # shape (nb_paths, 1)
            deltas = model(input_t).squeeze(-1)          # shape (nb_paths,)
            dS = paths[:, t+1] - paths[:, t]
            cash += deltas * dS

        payoff = torch.relu(K - paths[:, -1])
        barrier_hit = torch.min(paths, dim=1).values <= B
        payoff = torch.where(barrier_hit, payoff, torch.zeros_like(payoff))

        PnL = cash - payoff
        loss = loss_fn(PnL, torch.zeros_like(PnL))

        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}, PnL={PnL.mean().item():.4f}")

    # On peut stocker le PnL final
    model.final_pnl = PnL.mean().item()
    return model
