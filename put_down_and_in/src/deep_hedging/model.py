import torch
import torch.nn as nn
import torch.optim as optim

from src.deep_hedging.utils import generate_paths_gbm

# Réseau neuronal simple pour Deep Hedging
class HedgingStrategy(nn.Module):
    def __init__(self):
        super(HedgingStrategy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),   # La taille d'entrée est 1 (le prix de l'actif à chaque étape)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # La taille de sortie est 1 (le Delta prédit)
        )

    def forward(self, x):
        # x doit être un tensor 2D de taille (batch_size, 1)
        return self.net(x.unsqueeze(-1))


# Fonction d'entraînement du modèle Deep Hedging
def train_deep_hedge(S0, K, B, r, sigma, T, nb_paths=5000, nb_steps=50, epochs=10, lr=1e-3):
    # Générer les chemins GBM
    paths = generate_paths_gbm(S0, r, sigma, T, nb_paths, nb_steps)
    model = HedgingStrategy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Liste pour stocker les valeurs de PnL
    pnl_list = []

    # Boucle d'entraînement
    for epoch in range(epochs):
        optimizer.zero_grad()
        deltas = model(paths[:, :-1])
        cash = torch.zeros(nb_paths)

        # Calcul du cashflow et des variations de prix
        for t in range(nb_steps):
            dS = paths[:, t+1] - paths[:, t]
            deltas = model(paths[:, t].unsqueeze(-1))  # Ajuster la forme de l'entrée
            cash += deltas.squeeze(-1) * dS


        # Calcul du payoff
        payoff = torch.relu(K - paths[:, -1])
        barrier_hit = torch.min(paths, dim=1).values <= B
        payoff = torch.where(barrier_hit, payoff, torch.zeros_like(payoff))

        # Calcul du PnL final
        PnL = cash - payoff
        loss = loss_fn(PnL, torch.zeros_like(PnL))

        loss.backward()
        optimizer.step()

        # Ajouter la moyenne du PnL à la liste
        pnl_list.append(PnL.mean().item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, PnL: {PnL.mean().item():.4f}")

    # Ajouter le PnL final au modèle
    model.final_pnl = torch.tensor(pnl_list[-1])
    return model

def test_hedging_strategy():
    model = HedgingStrategy()

    # Créer un tensor de test
    test_input = torch.tensor([[100.0]], dtype=torch.float32)  # Entrée de type float32
    output = model(test_input)

    # Vérifier la sortie
    print("Input :", test_input)
    print("Output :", output)

# Exécuter le test
if __name__ == "__main__":
    test_hedging_strategy()