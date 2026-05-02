"""Treino do agente PPO para o ambiente CPP.

Subcomandos:
  train       — treina um único estágio (com --init opcional para transfer)
  curriculum  — pipeline 5x5 → 10x10 → 20x20 com transfer entre estágios
  test        — avalia um modelo (n episódios)
  run         — renderiza um episódio
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from gymnasium_env.cpp_policy import CPPFeatureExtractor


ENV_ID = "gymnasium_env/GridWorldCPP-v0"
DATA_DIR = "data"
LOG_DIR = "log"


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def _register_env() -> None:
    try:
        gym.register(id=ENV_ID, entry_point=GridWorldCPPEnv)
    except gym.error.Error:
        pass


# --- Per-size defaults --------------------------------------------------
# The obstacle ratio is held roughly constant (~12%) and max_steps grows
# proportionally to the number of free cells (~4x area).
SIZE_DEFAULTS = {
    5:  {"obs_quantity": 3,  "max_steps": 100},
    10: {"obs_quantity": 12, "max_steps": 600},
    20: {"obs_quantity": 50, "max_steps": 2400},
}


def _config_for_size(size: int) -> dict:
    if size in SIZE_DEFAULTS:
        return SIZE_DEFAULTS[size]
    # Generic fallback: ~12% obstacles, ~4 visits per free cell budget
    free = size * size
    return {
        "obs_quantity": max(1, int(0.12 * free)),
        "max_steps": int(4 * free),
    }


def make_env(size: int, n_envs: int = 8, seed: int = 0):
    cfg = _config_for_size(size)
    env_kwargs = dict(
        size=size,
        obs_quantity=cfg["obs_quantity"],
        max_steps=cfg["max_steps"],
        render_mode="rgb_array",
    )

    # Build a vectorized env. Use SubprocVecEnv when n_envs>1 so steps
    # actually run in parallel processes (CPU-bound rollouts).
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    def make_one(rank: int):
        def _init():
            env = GridWorldCPPEnv(**env_kwargs)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    return vec_cls([make_one(i) for i in range(n_envs)])


def _ppo_kwargs() -> dict:
    return dict(
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,        # held constant inside one stage; reduce per stage
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        policy_kwargs=dict(
            features_extractor_class=CPPFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )


def _model_tag(size: int, total_steps: int, suffix: str = "") -> str:
    cfg = _config_for_size(size)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"ppo_cpp_{size}_{cfg['obs_quantity']}_{cfg['max_steps']}_{total_steps}_{ts}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def train_stage(
    size: int,
    total_steps: int,
    init_model: str | None = None,
    n_envs: int = 8,
    ent_coef: float | None = None,
    seed: int = 0,
    suffix: str = "",
) -> str:
    """Train (or fine-tune) for one curriculum stage. Returns model path."""
    _ensure_dirs()
    _register_env()

    print(f"\n=== Training stage: size={size}, steps={total_steps:,}, "
          f"init={init_model or 'scratch'} ===")
    env = make_env(size, n_envs=n_envs, seed=seed)

    kwargs = _ppo_kwargs()
    if ent_coef is not None:
        kwargs["ent_coef"] = ent_coef

    if init_model is None:
        model = PPO("MultiInputPolicy", env, verbose=1, **kwargs)
    else:
        # Continue training from a previous checkpoint. Because the obs
        # space is size-invariant, the policy architecture is identical
        # and the weights load directly.
        print(f"Loading weights from {init_model}")
        # We have to drop unsupported kwargs that load() doesn't accept.
        model = PPO.load(
            init_model,
            env=env,
            device="cpu",
            custom_objects={"learning_rate": kwargs["learning_rate"],
                            "clip_range": kwargs["clip_range"]},
        )
        if ent_coef is not None:
            model.ent_coef = ent_coef

    tag = _model_tag(size, total_steps, suffix=suffix)
    log_path = os.path.join(LOG_DIR, tag)
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=total_steps, progress_bar=False)

    model_path = os.path.join(DATA_DIR, f"{tag}.zip")
    model.save(model_path)
    env.close()
    print(f"Saved model -> {model_path}")
    print(f"Logs        -> {log_path}")
    return model_path


def cmd_train(args):
    train_stage(
        size=args.size,
        total_steps=args.steps,
        init_model=args.init,
        n_envs=args.n_envs,
        ent_coef=args.ent_coef,
        seed=args.seed,
    )


def cmd_curriculum(args):
    """5x5 -> 10x10 -> 20x20 with weight transfer between stages.

    Budgets target ~100% full coverage at each stage. Empirically, with
    the dual observation + potential-based shaping, 1M / 4M / 8M timesteps
    are enough; smaller multipliers (--total-multiplier 0.5) reduce these
    proportionally for fast iteration.
    """
    m = args.total_multiplier
    seed = args.seed

    p1 = train_stage(size=5,  total_steps=int(1_000_000 * m), n_envs=args.n_envs,
                     ent_coef=0.04, seed=seed,        suffix="stage1")
    p2 = train_stage(size=10, total_steps=int(4_000_000 * m), n_envs=args.n_envs,
                     init_model=p1, ent_coef=0.02, seed=seed + 1, suffix="stage2")
    p3 = train_stage(size=20, total_steps=int(8_000_000 * m), n_envs=args.n_envs,
                     init_model=p2, ent_coef=0.015, seed=seed + 2, suffix="stage3")

    print("\n=== Curriculum complete ===")
    print(f"Stage 1 (5x5)   : {p1}")
    print(f"Stage 2 (10x10) : {p2}")
    print(f"Stage 3 (20x20) : {p3}")


def _resolve_model_path(model_arg: str) -> str:
    if os.path.isfile(model_arg):
        return model_arg
    cand = os.path.join(DATA_DIR, model_arg)
    if os.path.isfile(cand):
        return cand
    cand_zip = cand + ".zip" if not cand.endswith(".zip") else cand
    if os.path.isfile(cand_zip):
        return cand_zip
    raise FileNotFoundError(f"Could not locate model: {model_arg}")


def cmd_test(args):
    _register_env()
    model_path = _resolve_model_path(args.model)
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    cfg = _config_for_size(args.size)
    env = GridWorldCPPEnv(
        size=args.size,
        obs_quantity=cfg["obs_quantity"],
        max_steps=cfg["max_steps"],
        render_mode="rgb_array",
    )

    full_count = 0
    coverages = []
    steps_list = []
    rng_seed = args.seed

    for i in range(args.episodes):
        obs, info = env.reset(seed=rng_seed + i)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, terminated, truncated, info = env.step(int(action))
            steps += 1

        coverages.append(info["coverage"])
        steps_list.append(steps)
        if terminated and not truncated:
            full_count += 1

    full_rate = 100 * full_count / args.episodes
    print(f"\n--- Test (size={args.size}, {args.episodes} eps) ---")
    print(f"Full coverage rate : {full_rate:.2f}% ({full_count}/{args.episodes})")
    print(f"Average coverage   : {100*np.mean(coverages):.2f}% "
          f"(std {100*np.std(coverages):.2f}%, "
          f"min {100*np.min(coverages):.2f}%, max {100*np.max(coverages):.2f}%)")
    print(f"Average steps      : {np.mean(steps_list):.1f} "
          f"(std {np.std(steps_list):.1f}, min {np.min(steps_list)}, "
          f"max {np.max(steps_list)})")


def cmd_run(args):
    _register_env()
    model_path = _resolve_model_path(args.model)
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    cfg = _config_for_size(args.size)
    env = GridWorldCPPEnv(
        size=args.size,
        obs_quantity=cfg["obs_quantity"],
        max_steps=cfg["max_steps"],
        render_mode="human",
    )

    obs, info = env.reset(seed=args.seed)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
    print(f"Run finished: total_reward={total_reward:.2f}, "
          f"coverage={info['coverage']:.1%}, terminated={terminated}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train a single stage")
    pt.add_argument("--size", type=int, default=5)
    pt.add_argument("--steps", type=int, default=500_000)
    pt.add_argument("--init", type=str, default=None,
                    help="Path to a checkpoint to load before training.")
    pt.add_argument("--n-envs", type=int, default=8)
    pt.add_argument("--ent-coef", type=float, default=None)
    pt.add_argument("--seed", type=int, default=0)
    pt.set_defaults(func=cmd_train)

    pc = sub.add_parser("curriculum",
                        help="Run the full 5x5 -> 10x10 -> 20x20 pipeline")
    pc.add_argument("--total-multiplier", type=float, default=1.0,
                    help="Scale total timesteps for all stages (use <1 for smoke)")
    pc.add_argument("--n-envs", type=int, default=8)
    pc.add_argument("--seed", type=int, default=0)
    pc.set_defaults(func=cmd_curriculum)

    pe = sub.add_parser("test", help="Evaluate a trained model")
    pe.add_argument("--size", type=int, default=5)
    pe.add_argument("--model", type=str, required=True)
    pe.add_argument("--episodes", type=int, default=100)
    pe.add_argument("--deterministic", action="store_true")
    pe.add_argument("--seed", type=int, default=1000)
    pe.set_defaults(func=cmd_test)

    pr = sub.add_parser("run", help="Render a single episode with a model")
    pr.add_argument("--size", type=int, default=5)
    pr.add_argument("--model", type=str, required=True)
    pr.add_argument("--deterministic", action="store_true")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(func=cmd_run)

    return p


def _legacy_compat(argv: list[str]) -> list[str]:
    """Allow the old ``python train_grid_world_cpp.py train`` form to keep
    working with sensible defaults."""
    if len(argv) == 1 and argv[0] in {"train", "test", "run", "curriculum"}:
        return [argv[0]]
    return argv


if __name__ == "__main__":
    argv = _legacy_compat(sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
