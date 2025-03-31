import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

# Pour s'assurer qu'on peut importer localement
sys.path.append(os.path.abspath("."))

from indicators.indicator_utils import add_indicators
from rl_env.trading_env import TradingEnv

def simulate_ppo(model_path="models/ppo_gold_advanced.zip",
                 csv_path="data/Gold_Daily.csv",
                 window_size=30,
                 signal_features=["price", "rsi", "macd", "ema"],
                 max_steps=3000):
    """
    - Charge le modèle PPO et exécute un certain nombre de pas (max_steps).
    - Enregistre à chaque pas : l'action, la récompense, le prix.
    - Affiche ensuite un graphe pour visualiser le prix + actions.

    Paramètres:
    -----------
    model_path : str
        Chemin vers le modèle PPO sauvegardé (.zip).
    csv_path : str
        Chemin vers le CSV de données.
    window_size : int
        Taille de la fenêtre d’observation.
    signal_features : list[str]
        Nom des colonnes de features dans df, ex: ["price", "rsi", "macd", "ema"].
    max_steps : int
        Nombre max de steps (pas de temps) à simuler.

    Retour:
    -------
    None (mais génère une figure matplotlib).
    """

    # Charger les données
    df = pd.read_csv(csv_path)
    df = add_indicators(df)  # Ajoute columns: price, rsi, macd, ema
    # S'assurer qu'il y a assez de données pour window_size + max_steps
    if len(df) < (window_size + max_steps):
        print(f"Attention: le CSV n'a que {len(df)} lignes, max_steps={max_steps} sera limité.")
        max_steps = len(df) - window_size - 1

    # Créer l’environnement
    env = TradingEnv(df, window_size=window_size, signal_features=signal_features)

    # Charger le modèle PPO
    model = PPO.load(model_path, env=None)  # On va donner l'env nous-même en manuel

    # On va reset l'env manuellement
    obs = env.reset()

    # Pour la simulation
    prices = []
    actions_list = []
    rewards_list = []

    done = False
    step_count = 0

    while not done and step_count < max_steps:
        # On prédit l’action
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs

        # Stocker les infos
        current_price = df["price"].iloc[env.current_step]  # le prix actuel
        prices.append(current_price)
        actions_list.append(action)
        rewards_list.append(reward)

        step_count += 1

        if done:
            print(f"Simulation terminée au step {step_count} (fin de l’épisode).")

    # === Visualisation : on trace le prix et on marque les actions
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Prix du marché", color="black", alpha=0.7)

    # On colorie les points selon l’action
    # S'assurer qu'on mappe la même énum qu'on a dans le code
    action_names = [
        "SCALP_BUY", "SCALP_SELL", "STAKING",
        "SWAP", "STOP_LOSS", "TAKE_PROFIT"
    ]
    # Couleurs associées par action index
    # (à adapter à ta guise)
    colors = [
        "green",    # SCALP_BUY
        "red",      # SCALP_SELL
        "blue",     # STAKING
        "orange",   # SWAP
        "magenta",  # STOP_LOSS
        "purple"    # TAKE_PROFIT
    ]

    # Tracer un scatter pour chaque action
    x_idx = np.arange(len(prices))  # Indice x
    for a_idx in range(len(action_names)):
        # Indices où l'action prise = a_idx
        indices = [i for i, act in enumerate(actions_list) if act == a_idx]
        plt.scatter(indices,
                    [prices[i] for i in indices],
                    color=colors[a_idx],
                    label=action_names[a_idx],
                    alpha=0.6)

    plt.title("Simulation PPO - Actions prises sur prix")
    plt.xlabel("Step")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Afficher un résumé
    total_reward = sum(rewards_list)
    print(f"Nombre de steps simulés : {step_count}")
    print(f"Reward total accumulé : {total_reward:.2f}")

# Pour exécuter directement
if __name__ == "__main__":
    simulate_ppo(
        model_path="models/ppo_gold_6_actions.zip",
        csv_path="data/gold_prices.csv",
        window_size=30,
        signal_features=["price", "rsi", "macd", "ema"],
        max_steps=3000
    )
