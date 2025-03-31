import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from indicators.indicator_utils import add_indicators
from rl_env.trading_env import TradingEnv

# === Charger les données et indicateurs ===
df = add_indicators("data/BTC-USD_stock_data.csv")

# === Paramètres de la simulation ===
window_size = 30
start_index = 200  # Tu peux le changer pour explorer d'autres fenêtres

# === Environnement de simulation ===
env = DummyVecEnv([lambda: TradingEnv(
    df,
    window_size=window_size,
    signal_features=["close", "rsi", "macd", "ema"],
    custom_actions=True
)])

# === Charger le modèle DQN entraîné ===
model_path = Path("models/dqn_btc_advanced.zip")
model = DQN.load(model_path.as_posix(), env=env)

# === Fonction de simulation d’actions ===
def simulate_actions(df, start, window):
    state = df.loc[start - window:start - 1, ["close", "rsi", "macd", "ema"]].values
    state = state.reshape(1, window, -1)
    action_list = []

    for i in range(start, start + 100):
        obs = df.loc[i - window:i - 1, ["close", "rsi", "macd", "ema"]].values
        obs = obs.reshape(1, window, -1)
        action, _ = model.predict(obs, deterministic=True)
        action_list.append(int(action))

    return action_list

# === Lancer la simulation ===
actions = simulate_actions(df, start_index, window_size)
df_window = df.iloc[start_index:start_index + len(actions)].copy()
df_window["Action"] = actions

# === Mapping des actions ===
action_map = {
    0: "BUY",
    1: "SELL",
    2: "HOLD",
    3: "SCALP_BUY",
    4: "SCALP_SELL",
    5: "SWAP"
}
df_window["Action"] = df_window["Action"].map(action_map)

# === Affichage Streamlit ===
st.title("Simulation DQN - Fenêtre de Trading")
st.markdown("Fenêtre actuelle du marché avec actions simulées par l’agent :")
st.dataframe(df_window[["date", "close", "Action"]].reset_index(drop=True))

# === Graphique ===
fig, ax = plt.subplots()
ax.plot(df_window["close"].values, label="Prix BTC", color="black")

# Ajouter marqueurs colorés pour chaque action
colors = {
    "BUY": "green", "SELL": "red", "HOLD": "blue",
    "SCALP_BUY": "orange", "SCALP_SELL": "purple", "SWAP": "brown"
}
markers = {
    "BUY": "o", "SELL": "v", "HOLD": "s",
    "SCALP_BUY": ">", "SCALP_SELL": "<", "SWAP": "x"
}

for i, action in enumerate(df_window["Action"]):
    ax.scatter(i, df_window["close"].values[i], color=colors[action], marker=markers[action], label=action if i == 0 else "", s=80)

ax.set_xlabel("Pas de temps")
ax.set_ylabel("Prix (USD)")
ax.set_title("Actions de Trading simulées sur prix BTC")
ax.legend(loc="upper left")
st.pyplot(fig)
