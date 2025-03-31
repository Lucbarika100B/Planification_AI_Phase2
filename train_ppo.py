import sys
import os

sys.path.append(os.path.abspath("."))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

from indicators.indicator_utils import add_indicators
from rl_env.trading_env import TradingEnv

df = pd.read_csv("data/gold_prices.csv")
df = add_indicators(df)

window_size = 30
features = ["price", "rsi", "macd", "ema"]

env = DummyVecEnv([
    lambda: TradingEnv(df, window_size=window_size, signal_features=features)
])
env = VecMonitor(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(50000)
model.save("models/ppo_gold_6_actions")

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
print("Reward moyen:", mean_reward)
