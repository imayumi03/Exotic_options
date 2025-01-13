import torch

def generate_paths_gbm(S0, r, sigma, T, nb_paths=10000, nb_steps=50, seed=None):
    """
    Génère des scénarios de sous-jacent (Geometric Brownian Motion).
    Retour: un tenseur shape (nb_paths, nb_steps+1)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dt = T / nb_steps
    S = torch.zeros(nb_paths, nb_steps+1)
    S[:, 0] = S0
    
    for t in range(nb_steps):
        Z = torch.randn(nb_paths)
        S[:, t+1] = S[:, t] * torch.exp((r - 0.5*sigma**2)*dt + sigma*torch.sqrt(torch.tensor(dt))*Z)
        
    return S
