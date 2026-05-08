"""Dual-stream CNN feature extractor para o env CPP."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    # GroupNorm(1, C) em vez de BatchNorm: PPO coleta 1 amostra/env por step.
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(1, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(1, out_ch),
        nn.ReLU(inplace=True),
        nn.Flatten(),
    )


class CPPFeatureExtractor(BaseFeaturesExtractor):
    """Dual-stream CNN + scalar projection.

    Streams:
      - local_cnn  : 3 canais one-hot na janela KxK egocêntrica
      - global_cnn : 2 canais (visitadas pooleadas + posição) em FxF fixo
      - scalar_proj: coverage + frontier(3) + progress + trail(L*2) flatten

    Concat final -> features_dim (128 default; teste 256 via kwarg).
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        local_space = observation_space.spaces["local_map"]
        global_space = observation_space.spaces["global_map"]
        trail_space = observation_space.spaces["trail"]

        self.local_cnn = _conv_block(local_space.shape[0], 32)
        self.global_cnn = _conv_block(global_space.shape[0], 32)

        with th.no_grad():
            local_sample = th.as_tensor(local_space.sample()[None]).float()
            global_sample = th.as_tensor(global_space.sample()[None]).float()
            local_flat = self.local_cnn(local_sample).shape[1]
            global_flat = self.global_cnn(global_sample).shape[1]

        # Escalares concatenados: coverage(1) + frontier(3) + progress(1) + trail(L*2)
        trail_flat = int(np.prod(trail_space.shape))
        scalar_in = 1 + 3 + 1 + trail_flat

        scalar_dim = 16
        spatial_dim = features_dim - scalar_dim
        local_dim = spatial_dim // 2
        global_dim = spatial_dim - local_dim

        self.local_proj = nn.Sequential(
            nn.Linear(local_flat, local_dim),
            nn.LayerNorm(local_dim),
            nn.ReLU(inplace=True),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_flat, global_dim),
            nn.LayerNorm(global_dim),
            nn.ReLU(inplace=True),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_in, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: dict) -> th.Tensor:
        local_feat = self.local_proj(self.local_cnn(obs["local_map"]))
        global_feat = self.global_proj(self.global_cnn(obs["global_map"]))
        trail_flat = obs["trail"].flatten(start_dim=1)
        scalars = th.cat(
            [obs["coverage"], obs["frontier"], obs["progress"], trail_flat],
            dim=1,
        )
        scalar_feat = self.scalar_proj(scalars)
        return th.cat([local_feat, global_feat, scalar_feat], dim=1)
