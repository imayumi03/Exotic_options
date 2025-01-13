import pytest
import torch
from src.deep_hedging.model import train_deep_hedge

def test_train_deep_hedge():
    model = train_deep_hedge(nb_paths=100, nb_steps=10, epochs=1)
    # Juste un test basique : vérifier que le modèle a bien été entraîné
    assert model is not None, "Le modèle doit être instancié"
    # On peut vérifier par ex. qu'un forward pass ne plante pas
    S_test = torch.ones(10, 10) * 100.0
    deltas = model(S_test)
    assert deltas.shape == (10, 10), "La sortie doit être (batch_size, nb_steps)"
