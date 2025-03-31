from collections import Counter
from stable_baselines3 import DQN
from rl_env.trading_env import TradingEnv
from indicators.indicator_utils import add_indicators
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

# Charger modèle
model = DQN.load("models/dqn_btc_advanced")

# Charger données BTC avec indicateurs
data = add_indicators("data/BTC-USD_stock_data.csv")
env = DummyVecEnv([lambda: TradingEnv(data, window_size=30)])

obs = env.reset()
action_counter = Counter()

done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    action_counter[int(action)] += 1
    obs, reward, done, info = env.step(action)

# Associer les labels d’actions
action_labels = ["BUY", "SELL", "HOLD", "SCALP_BUY", "SCALP_SELL", "SWAP"]
frequencies = [action_counter[i] for i in range(len(action_labels))]

# Sauvegarde dans fichier CSV
df = pd.DataFrame({
    "Action": action_labels,
    "Frequency": frequencies
})
df.to_csv("results/dqn_action_distribution.csv", index=False)

print("Distribution sauvegardée dans results/dqn_action_distribution.csv")
