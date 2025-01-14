import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
#                  1) CHECK DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Appareil utilisé : {device}")

# =========================================================
#          2) HELPER: GET REAL DATA & FEATURE ENG
# =========================================================
def get_real_data_yfinance(ticker, start_date, end_date, features=("Close",)):
    """
    Fetch daily data from yfinance for a given ticker and date range.
    Return a DataFrame with additional features, e.g., realized volatility.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)

    # Basic features
    data["LogReturn"] = np.log(data["Close"]).diff()
    data["RealizedVol"] = data["LogReturn"].rolling(window=10).std()  # 10-day rolling volatility

    # Fill forward / drop NaNs
    data.fillna(method='bfill', inplace=True)

    # You can add more features, e.g.:
    # - high-low range
    # - volume features
    # - macro data

    # Drop any residual NaNs (first rows)
    data.dropna(inplace=True)
    return data

# =========================================================
#     3) BLACK-SCHOLES MONTE CARLO FOR DOWN-AND-IN PUT
# =========================================================
def price_put_down_and_in(S0, K, B, r, sigma, T, M=10000, N=100):
    """
    Monte Carlo pricing for a Put Down-and-In under Black-Scholes.
    Returns:
      - discounted MC price
      - vector of PnL (for naive delta hedge)
      - average deltas at each step
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

            # Delta for a European put
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

def delta_put_black_scholes(S, K, r, sigma, T):
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    # Delta_call under BS
    delta_call = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0))) * math.exp(-r * T)
    # Put-Call parity => delta_put = delta_call - e^{-rT}
    delta_put = delta_call - math.exp(-r * T)
    return delta_put

# =========================================================
#          4) LSTM MODEL FOR DEEP HEDGING
# =========================================================
class LSTMDeepHedging(nn.Module):
    """
    A more advanced LSTM that takes in a sequence of features
    and outputs a hedge ratio at each step.
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # output = hedge ratio
        # We'll also learn an initial cost offset
        self.cost = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        Return: hedge_ratios: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_len, hidden_dim)
        hedge_ratios = self.fc(out).squeeze(-1)  # -> (batch_size, seq_len)
        return hedge_ratios

# =========================================================
#    5) MULTI-STEP TRAINING ON REAL-WORLD DATA (LSTM)
# =========================================================

def train_deep_hedge_lstm(
    df,
    K, B, r, T,
    seq_len=20,
    epochs=100,
    lr=1e-3,
    device=torch.device("cuda")
):
    
    """
    Train an LSTM deep hedge strategy on real-world data for a Put Down-and-In.

    df: DataFrame with columns like: ["Close", "LogReturn", "RealizedVol", ...]
    K, B, r, T: option parameters
    seq_len: number of lookback days for the LSTM
    epochs: number of training epochs
    lr: learning rate
    device: 'cpu' or 'cuda'

    Returns:
       model, loss_history, final_pnls (numpy), average_deltas (numpy)
    """
    # 1) Feature engineering (example)
    df["NormPrice"] = df["Close"] / df["Close"].iloc[0]
    df["RealizedVol"] = df["RealizedVol"].fillna(method='bfill').fillna(0.0)

    data_array = df[["NormPrice", "RealizedVol"]].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)

    X_seqs = []
    Y_seqs = []  # will store close prices over the sequence + 1 step
    for i in range(len(df) - seq_len):
        X_seqs.append(data_array[i : i + seq_len])       # shape (seq_len, 2)
        Y_seqs.append(prices[i : i + seq_len + 1])       # shape (seq_len+1,)

    X_seqs = np.array(X_seqs, dtype=np.float32)  # (N, seq_len, 2)
    Y_seqs = np.array(Y_seqs, dtype=np.float32)  # (N, seq_len+1)

    X_tensor = torch.from_numpy(X_seqs).to(device)  # (N, seq_len, 2)
    Y_tensor = torch.from_numpy(Y_seqs).to(device)  # (N, seq_len+1)

    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 2) Model & optimizer
    model = LSTMDeepHedging(input_dim=2, hidden_dim=64, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    # 3) Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for X_batch, Y_batch in loader:
            # X_batch shape => (batch_size, seq_len, 2)
            # Y_batch shape => (batch_size, seq_len+1)

            hedge_ratios = model(X_batch)  # (batch_size, seq_len)
            batch_size, seq_len_curr = hedge_ratios.shape

            # PnLs shape => (batch_size,)
            PnLs = torch.zeros(batch_size, device=device)

            # day-by-day rebalancing
            for t in range(seq_len_curr):
                # S_t shape => (batch_size,)
                # S_next shape => (batch_size,)
                S_t    = Y_batch[:, t]
                S_next = Y_batch[:, t + 1]

                # Ensure shape is (batch_size,) if needed
                if S_t.ndim == 2:
                    S_t = S_t.squeeze(-1)
                if S_next.ndim == 2:
                    S_next = S_next.squeeze(-1)

                delta_t = hedge_ratios[:, t]  # shape (batch_size,)

                dS = S_next - S_t            # shape (batch_size,)
                PnLs += delta_t * dS

            # Barrier check: min over the path
            min_price, _ = torch.min(Y_batch, dim=1)  # shape (batch_size,)
            barrier_hit = (min_price <= B)

            S_final = Y_batch[:, -1]                 # shape (batch_size,)
            payoff = torch.relu(K - S_final) * barrier_hit.float()

            net_pnl = model.cost + PnLs - payoff

            # Example loss: penalize negative final PnL (shortfall risk)
            loss = torch.mean(torch.relu(-net_pnl))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        avg_loss = total_loss / len(loader.dataset)
        loss_history.append(avg_loss)
        # You can print or log the loss
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    # 4) Evaluate final PnLs on entire dataset (optional)
    model.eval()
    with torch.no_grad():
        hedge_ratios_all = model(X_tensor)  # (N, seq_len)
        final_pnls = []
        for i in range(len(X_tensor)):
            hr = hedge_ratios_all[i]      # (seq_len,)
            S_path = Y_tensor[i]          # (seq_len+1,)

            pnl_ = 0.0
            for t in range(seq_len):
                S_t    = S_path[t]
                S_next = S_path[t+1]
                pnl_  += float(hr[t] * (S_next - S_t))

            # barrier
            if float(torch.min(S_path)) <= B:
                payoff_ = max(K - float(S_path[-1]), 0.0)
            else:
                payoff_ = 0.0

            net_pnl_ = float(model.cost) + pnl_ - payoff_
            final_pnls.append(net_pnl_)

        final_pnls = np.array(final_pnls)
        avg_deltas = hedge_ratios_all.mean(dim=0).cpu().numpy()

    return model, loss_history, final_pnls, avg_deltas


# =========================================================
#            6) STREAMLIT VISUALIZATIONS
# =========================================================

def plot_pnl_histogram(pnl_bs_values, pnl_dh_values):
    bins = np.linspace(min(pnl_bs_values.min(), pnl_dh_values.min()),
                       max(pnl_bs_values.max(), pnl_dh_values.max()), 50)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pnl_bs_values, bins=bins, alpha=0.7, color='red', label="Black-Scholes PnL (Benchmark)")
    ax.hist(pnl_dh_values, bins=bins, alpha=0.7, color='blue', label="Deep-Hedging PnL")
    ax.set_title("Profit and Loss (PnL) Histogram (Real Data)")
    ax.set_xlabel("Profit and Loss (PnL)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

def plot_delta_curve(delta_bs_values, delta_dh_values):
    min_length = min(len(delta_bs_values), len(delta_dh_values))
    steps = np.arange(min_length)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, delta_bs_values[:min_length], 'r-', label="Black-Scholes Delta (Benchmark)")
    ax.plot(steps, delta_dh_values[:min_length], 'b-', label="Deep-Hedging Delta")
    ax.set_title("Comparison of Delta Curves")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Delta")
    ax.legend()
    st.pyplot(fig)

def plot_loss_function(loss_history, title="Deep-Hedging Loss"):
    epochs = np.arange(len(loss_history))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, loss_history, 'b-', label="Deep-Hedging Loss")
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss Value")
    ax.legend()
    st.pyplot(fig)

# =========================================================
#            7) STREAMLIT MAIN FUNCTION
# =========================================================

def main():
    st.title("Comparaison : Modèle Classique vs LSTM Deep Hedging (Put Down-and-In)")

    # Sidebar - Paramètres Option
    st.sidebar.header("Paramètres Option")
    S0 = st.sidebar.slider("Spot initial (S0)", min_value=50, max_value=200, value=100, step=1)
    K = st.sidebar.slider("Strike (K)", min_value=50, max_value=200, value=100, step=1)
    B = st.sidebar.slider("Barrière (B)", min_value=10, max_value=180, value=90, step=1)
    r = st.sidebar.slider("Taux sans risque (r)", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    T = st.sidebar.slider("Maturité (T, années)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    # Side bar - LSTM training
    st.sidebar.header("Entraînement LSTM")
    ticker_input = st.sidebar.text_input("Ticker (yfinance)", value="AAPL")
    start_date = st.sidebar.text_input("Date de début (YYYY-MM-DD)", value="2018-01-01")
    end_date   = st.sidebar.text_input("Date de fin   (YYYY-MM-DD)", value="2023-01-01")
    epochs     = st.sidebar.slider("Époques (Deep Hedging)", min_value=1, max_value=50, value=5, step=1)
    seq_len    = st.sidebar.slider("Sequence length (jours)", min_value=5, max_value=60, value=20, step=5)

    # 1) Compute Black-Scholes Price
    st.write("## 1) Pricing via Monte Carlo (Black-Scholes)")
    if st.button("Calculer le prix Black-Scholes"):
        sigma_default = 0.2
        price_bs, pnl_bs_values, delta_bs_values = price_put_down_and_in(
            S0, K, B, r, sigma=sigma_default, T=T
        )
        st.write(f"**Prix Put Down-and-In (Monte Carlo BS) :** {price_bs:.4f}")

        # Sauvegarde dans la session
        st.session_state['price_bs'] = price_bs
        st.session_state['pnl_bs_values'] = np.array(pnl_bs_values)
        st.session_state['delta_bs_values'] = np.array(delta_bs_values)

    # 2) Deep Hedging with LSTM on Real Data
    st.write("## 2) LSTM Deep Hedging (données réelles)")
    if st.button("Entraîner le modèle LSTM Deep Hedging"):
        # a) Fetch real data
        df_data = get_real_data_yfinance(ticker_input, start_date, end_date)
        if len(df_data) < seq_len + 1:
            st.warning("Pas assez de données pour cette plage. Veuillez choisir une autre période.")
        else:
            # b) Train LSTM
            model, loss_hist, final_pnls, avg_deltas = train_deep_hedge_lstm(
                df=df_data,
                K=K, B=B, r=r, T=T,
                seq_len=seq_len,
                epochs=epochs,
                lr=1e-3,
                device=device
            )
            st.write(f"**Deep Hedging terminé. Taille dataset =** {len(df_data)}")

            # Sauvegarde dans la session
            st.session_state['price_dh'] = float(model.cost.item())
            st.session_state['pnl_dh_values'] = final_pnls
            st.session_state['delta_dh_values'] = avg_deltas
            st.session_state['loss_history'] = loss_hist

    # 3) Plots + Comparaison si on a les deux modèles
    if 'price_dh' in st.session_state and 'price_bs' in st.session_state:
        st.write("### Distribution du PnL (Profit and Loss)")
        plot_pnl_histogram(st.session_state['pnl_bs_values'], st.session_state['pnl_dh_values'])

        st.write("### Courbe du Delta")
        plot_delta_curve(st.session_state['delta_bs_values'], st.session_state['delta_dh_values'])

        st.write("### Fonction de perte au fil des itérations")
        plot_loss_function(st.session_state['loss_history'], title="Deep-Hedging Loss")

        # === NOUVEAU : TABLEAU DE COMPARAISON ===

        # 1) Calcul des metrics pour chaque modèle
        # ---------------------------------------
        price_bs = st.session_state['price_bs']
        price_dh = st.session_state['price_dh']

        # On peut prendre la moyenne (et stdev) des PnLs comme indicateur de performance
        pnl_bs = st.session_state['pnl_bs_values']
        pnl_dh = st.session_state['pnl_dh_values']

        pnl_bs_mean = np.mean(pnl_bs)
        pnl_bs_std  = np.std(pnl_bs)

        pnl_dh_mean = np.mean(pnl_dh)
        pnl_dh_std  = np.std(pnl_dh)

        # Moyenne des deltas (ou la dernière valeur, selon ce qui vous intéresse)
        delta_bs_mean = np.mean(st.session_state['delta_bs_values'])
        delta_dh_mean = np.mean(st.session_state['delta_dh_values'])

        # 2) Construction du DataFrame de comparaison
        # -------------------------------------------
        df_compare = pd.DataFrame({
            'Modèle':        ['Black-Scholes', 'Deep Hedging'],
            'Prix estimé':   [price_bs, price_dh],
            'Delta moyen':   [delta_bs_mean, delta_dh_mean],
            'PnL moyen':     [pnl_bs_mean, pnl_dh_mean],
            'PnL écart-type':[pnl_bs_std,  pnl_dh_std]
        })

        st.write("### Tableau comparatif")
        st.table(df_compare)

        # Vous pouvez aussi afficher sous forme de dataframe éditable
        # st.dataframe(df_compare)

if __name__ == "__main__":
    main()