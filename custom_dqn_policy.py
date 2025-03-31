import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class CustomDQNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super(CustomDQNFeatureExtractor, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[1]  # ex: 5 indicateurs (close, rsi, macd, ema, etc.)
        self.conv1 = nn.Conv1d(in_channels=n_input_channels, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        # Calcul de la taille du flatten après les Conv1D (dépend de window_size)
        conv_output_size = self._get_conv_output(observation_space.shape)

        # MLP après les Conv1D
        self.fc1 = nn.Linear(conv_output_size, features_dim)

    def _get_conv_output(self, shape):
        # shape = (window_size, n_features)
        o = torch.zeros(1, shape[1], shape[0])
        o = self.conv1(o)
        o = self.conv2(o)
        return int(np.prod(o.size()))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # reshape: (batch_size, time_steps, features) -> (batch, features, time)
        x = observations.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        return F.relu(self.fc1(x))


class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        # Remplacer le backbone par notre extracteur custom
        kwargs['features_extractor_class'] = CustomDQNFeatureExtractor
        kwargs['features_extractor_kwargs'] = dict(features_dim=128)
        super(CustomDQNPolicy, self).__init__(*args, **kwargs)
