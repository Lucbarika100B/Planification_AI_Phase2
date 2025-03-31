import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# === Importations ===
import numpy as np
from rl_env.trading_env import TradingEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rl_env.trading_env import TradingEnv
from indicators.indicator_utils import add_indicators
from custom_dqn_policy import CustomDQNPolicy
import os

# === S'assurer que le répertoire de sauvegarde existe ===
os.makedirs("models", exist_ok=True)
os.makedirs("tensorboard/dqn_advanced", exist_ok=True)

# === Charger les données + indicateurs ===
df = add_indicators('data/Gold_Daily.csv')  # CSV doit contenir 'Close'

# === Créer l'environnement avec indicateurs et actions enrichies ===
env = DummyVecEnv([
    lambda: TradingEnv(
        df,
        window_size=30,
        trading_fee=0.001,
        signal_features=["close", "rsi", "macd", "ema"],  # indicateurs utilisés
        custom_actions=True  # active les actions SCALP, SWAP, STAKE, TP, SL
    )
])

# === Initialiser modèle DQN avec Custom Policy ===
model = DQN(
    policy=CustomDQNPolicy,
    env=env,
    learning_rate=0.0003,
    buffer_size=10000,
    learning_starts=500,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=1,
    target_update_interval=250,
    verbose=1,
    tensorboard_log="./tensorboard/dqn_advanced/"
)

# === Lancement de l'apprentissage ===
print("Entraînement du modèle DQN en cours...")
model.learn(total_timesteps=50000)

# === Sauvegarde du modèle ===
model.save("models/dqn_btc_advanced")
print("Modèle sauvegardé dans models/dqn_btc_advanced.zip")

# === Évaluation rapide ===
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Moyenne des rewards obtenus sur 5 épisodes d’évaluation : {mean_reward}")
