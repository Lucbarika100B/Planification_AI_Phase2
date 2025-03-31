import gym
import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box, Discrete
from enum import Enum

class Actions(Enum):
    SCALP_BUY = 0
    SCALP_SELL = 1
    STAKING = 2
    SWAP = 3
    STOP_LOSS = 4
    TAKE_PROFIT = 5

class TradingEnv(Env):
    def __init__(self, df, window_size, signal_features):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.signal_features = signal_features

        self.action_space = Discrete(len(Actions))  # 6 actions
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.signal_features)),
            dtype=np.float32
        )

        self.current_step = self.window_size
        self.balance = 1000
        self.holdings = 0

    def reset(self):
        self.current_step = self.window_size
        self.balance = 1000
        self.holdings = 0
        obs = self._get_observation()
        return obs

    def _get_observation(self):
        return self.df[self.signal_features].iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0

        price = self.df['price'].iloc[self.current_step]
        # Exécuter l’action
        if action == Actions.SCALP_BUY.value:
            self.holdings += 1
            self.balance -= price
        elif action == Actions.SCALP_SELL.value and self.holdings > 0:
            self.holdings -= 1
            self.balance += price
            reward = 1
        elif action == Actions.STAKING.value:
            reward = 0.05
        elif action == Actions.SWAP.value:
            reward = np.random.choice([0.1, -0.1])
        elif action == Actions.STOP_LOSS.value and self.holdings > 0:
            self.holdings = 0
            self.balance += price * 0.9
            reward = -1
        elif action == Actions.TAKE_PROFIT.value and self.holdings > 0:
            self.holdings = 0
            self.balance += price * 1.1
            reward = 2

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_observation()
        return obs, reward, done, {}
