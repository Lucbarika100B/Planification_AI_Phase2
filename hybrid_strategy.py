from stable_baselines3 import PPO, DQN
from rl_env.trading_env import TradingEnv
from src.utils import load_data

class MetaAgent:
    def __init__(self, dqn_agent, ppo_agent):
        self.dqn = dqn_agent
        self.ppo = ppo_agent

    def select_action(self, obj):
        if self.is_scalping_moment(obs):
            return self.dqn.predict(obs)
        else:
            return self.ppo.predict(obs)
    
    def is_scalping_moment(self, obs):
        #Si RSI + MACD indique une tendance court terme haussière ou baissière
        return obs["rsi"] < 30 or obs["macd_cross"] == 1

