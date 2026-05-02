"""Feature extractor com duas streams CNN para o env CPP.

local_map (3,7,7) e global_map (2,8,8) entram em CNNs paralelas; coverage
e frontier (4 escalares) entram num MLP curto. Tudo concatenado em 128
features para a policy/value head do PPO.
"""
from __future__ import annotations

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Flatten(),
    )


class CPPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        local_space = observation_space.spaces["local_map"]
        global_space = observation_space.spaces["global_map"]

        self.local_cnn = _conv_block(local_space.shape[0], 32)
        self.global_cnn = _conv_block(global_space.shape[0], 32)

        with th.no_grad():
            local_sample = th.as_tensor(local_space.sample()[None]).float()
            global_sample = th.as_tensor(global_space.sample()[None]).float()
            local_flat = self.local_cnn(local_sample).shape[1]
            global_flat = self.global_cnn(global_sample).shape[1]

        scalar_dim = 16
        # Split the latent budget evenly between local and global streams.
        spatial_dim = features_dim - scalar_dim
        local_dim = spatial_dim // 2
        global_dim = spatial_dim - local_dim

        self.local_proj = nn.Sequential(
            nn.Linear(local_flat, local_dim),
            nn.ReLU(inplace=True),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_flat, global_dim),
            nn.ReLU(inplace=True),
        )
        # 4 scalar inputs: coverage (1) + frontier (3 = dx, dy, dist).
        self.scalar_proj = nn.Sequential(
            nn.Linear(4, scalar_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: dict) -> th.Tensor:
        local_feat = self.local_proj(self.local_cnn(obs["local_map"]))
        global_feat = self.global_proj(self.global_cnn(obs["global_map"]))
        scalars = th.cat([obs["coverage"], obs["frontier"]], dim=1)
        scalar_feat = self.scalar_proj(scalars)
        return th.cat([local_feat, global_feat, scalar_feat], dim=1)
