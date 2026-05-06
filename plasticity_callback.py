"""Plasticity diagnostics: dormant ratio, weight norm, stable rank."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.callbacks import BaseCallback


class PlasticityCallback(BaseCallback):

    def __init__(self, log_freq: int = 5, tau: float = 0.025,
                 sample_size: int = 256, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.tau = tau
        self.sample_size = sample_size
        self._rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def _blocks(self) -> List[Tuple[str, nn.Module]]:
        fe = self.model.policy.features_extractor
        mlp = self.model.policy.mlp_extractor
        return [
            ("local_cnn", fe.local_cnn),
            ("global_cnn", fe.global_cnn),
            ("local_proj", fe.local_proj),
            ("global_proj", fe.global_proj),
            ("scalar_proj", fe.scalar_proj),
            ("policy_net", mlp.policy_net),
            ("value_net", mlp.value_net),
            ("action_net", self.model.policy.action_net),
            ("value_head", self.model.policy.value_net),
        ]

    def _sample_batch(self) -> dict | None:
        rb = self.model.rollout_buffer
        n_steps = rb.buffer_size
        n_envs = rb.n_envs
        total = n_steps * n_envs
        if total <= 0 or rb.full is False and rb.pos == 0:
            return None
        usable = total if rb.full else rb.pos * n_envs
        k = min(self.sample_size, usable)
        if k <= 0:
            return None
        idx = np.random.choice(usable, size=k, replace=False)
        if rb.full:
            step_idx, env_idx = np.unravel_index(idx, (n_steps, n_envs))
        else:
            step_idx, env_idx = np.unravel_index(idx, (rb.pos, n_envs))
        if isinstance(rb.observations, dict):
            return {
                k_: th.as_tensor(v[step_idx, env_idx]).float().to(self.model.device)
                for k_, v in rb.observations.items()
            }
        return th.as_tensor(rb.observations[step_idx, env_idx]).float().to(self.model.device)

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.log_freq != 0:
            return

        batch = self._sample_batch()
        if batch is None:
            return

        for name, block in self._blocks():
            sq = 0.0
            for p in block.parameters():
                if p.requires_grad:
                    sq += float(p.detach().pow(2).sum().item())
            self.logger.record(f"plasticity/wnorm_{name}", sq ** 0.5)

        activations: dict[str, th.Tensor] = {}
        handles = []
        fe = self.model.policy.features_extractor
        relu_targets = [
            ("local_cnn", fe.local_cnn),
            ("global_cnn", fe.global_cnn),
            ("local_proj", fe.local_proj),
            ("global_proj", fe.global_proj),
            ("scalar_proj", fe.scalar_proj),
        ]
        for parent_name, parent in relu_targets:
            for sub_name, sub in parent.named_modules():
                if isinstance(sub, nn.ReLU):
                    key = f"{parent_name}.{sub_name}" if sub_name else parent_name
                    def make_hook(k):
                        def hook(_m, _inp, out):
                            activations[k] = out.detach()
                        return hook
                    handles.append(sub.register_forward_hook(make_hook(key)))

        try:
            with th.no_grad():
                features = self.model.policy.features_extractor(batch)
        finally:
            for h in handles:
                h.remove()

        total_units = 0
        dormant_units = 0
        for key, act in activations.items():
            if act.dim() == 4:
                per_unit = act.abs().mean(dim=(0, 2, 3))
            else:
                per_unit = act.abs().mean(dim=0)
            mean_mag = per_unit.mean().clamp(min=1e-8)
            dormant_mask = (per_unit / mean_mag < self.tau)
            n_dormant = int(dormant_mask.sum().item())
            n_units = int(per_unit.numel())
            self.logger.record(f"plasticity/dormant_{key}", n_dormant / max(1, n_units))
            total_units += n_units
            dormant_units += n_dormant
        if total_units > 0:
            self.logger.record("plasticity/dormant_total", dormant_units / total_units)

        # stable rank = ||X||_F² / σ_max(X)²
        try:
            with th.no_grad():
                fro_sq = float(features.pow(2).sum().item())
                s = th.linalg.svdvals(features.float())
                op_sq = float((s.max() ** 2).item())
            if op_sq > 1e-12:
                self.logger.record("plasticity/stable_rank", fro_sq / op_sq)
        except Exception:
            pass
