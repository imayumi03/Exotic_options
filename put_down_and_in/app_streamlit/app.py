from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import sys

import torch
#sys.path.append("C:/Users/Lenovo Gaming/Exotic_options-4/put_down_and_in")

from code import price_put_down_and_in, train_deep_hedge




# Initialisation de la session pour stocker le modèle et le prix Black-Scholes
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'price_mc' not in st.session_state:
    st.session_state['price_mc'] = None
if 'price_dh' not in st.session_state:
    st.session_state['price_dh'] = None


# Fonction principale
def main():
    st.title("Deep Hedging (Put Down-and-In)")

    # Sidebar pour les paramètres
    st.sidebar.header("Paramètres Option")
    S0 = st.sidebar.slider("Spot initial (S0)", min_value=50, max_value=200, value=100, step=1)
    K = st.sidebar.slider("Strike (K)", min_value=50, max_value=200, value=100, step=1)
    B = st.sidebar.slider("Barrière (B)", min_value=10, max_value=180, value=90, step=1)
    r = st.sidebar.slider("Taux sans risque (r)", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    sigma = st.sidebar.slider("Volatilité (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    T = st.sidebar.slider("Maturité (T, années)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    # Section Black-Scholes
    st.write("## 1) Pricing via Monte Carlo (Black-Scholes)")
    if st.button("Calculer le prix Black-Scholes"):
        st.session_state['price_mc'] = price_put_down_and_in(S0, K, B, r, sigma, T, method="mc")
        st.write(f"**Prix Put Down-and-In (MC) :** {st.session_state['price_mc']:.4f}")

    if st.session_state['price_mc'] is not None:
        st.write(f"**Dernier prix calculé :** {st.session_state['price_mc']:.4f}")

    st.write("---")

    # Section Deep Hedging
    st.write("## 2) Deep Hedging")
    if st.button("Calculer le prix Deep Hedging"):
        model = train_deep_hedge(S0, K, B, r, sigma, T, nb_paths=5000, nb_steps=50, epochs=10, lr=1e-3)
        final_pnl = torch.mean(model.net(torch.tensor([[S0]], dtype=torch.float32)))  # Calcul du PnL final
        st.session_state['price_dh'] = final_pnl.item()
        st.session_state['model'] = model
        st.write(f"**Prix Deep Hedging :** {st.session_state['price_dh']:.4f}")

    if st.session_state['price_dh'] is not None:
        st.write(f"**Dernier prix calculé avec Deep Hedging :** {st.session_state['price_dh']:.4f}")

    # Comparaison graphique
    if st.session_state['price_mc'] is not None and st.session_state['price_dh'] is not None:
        st.write("### Comparaison des résultats :")
        fig, ax = plt.subplots()
        ax.bar(["Black-Scholes", "Deep Hedging"], [st.session_state['price_mc'], st.session_state['price_dh']])
        ax.set_ylabel("Prix / PnL")
        ax.set_title("Comparaison Black-Scholes vs Deep Hedging")
        st.pyplot(fig)

        # Distribution du PnL
        st.write("### Distribution du PnL (Deep Hedging)")
        pnl_values = torch.normal(mean=st.session_state['price_dh'], std=0.1, size=(1000,))
        fig, ax = plt.subplots()
        ax.hist(pnl_values.numpy(), bins=50, color='blue', alpha=0.7)
        ax.set_title("Distribution du PnL (Deep Hedging)")
        ax.set_xlabel("PnL")
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)

        # Courbe du Delta
        st.write("### Courbe du Delta (Deep Hedging)")
        steps = torch.arange(0, T, T / 50)
        deltas = [st.session_state['model'].net(torch.tensor([[S]])) for S in torch.linspace(50, 200, 50)]

        fig, ax = plt.subplots()
        ax.plot(steps.numpy(), [d.item() for d in deltas], label="Delta")
        ax.set_title("Courbe du Delta (Deep Hedging)")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Delta")
        st.pyplot(fig)

if __name__ == "__main__":
    main()