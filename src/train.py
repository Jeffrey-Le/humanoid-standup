"""
train.py
--------
Train a PPO agent on HumanoidStandup-v5 with arbitrary initial poses.

Run 2 changes vs Run 1:
  - ent_coef 0.002 → 0.01      (prevent entropy collapse)
  - lr 3e-4 → 1e-4             (slower, more stable updates)
  - n_epochs 10 → 5            (less over-optimization per rollout)
  - n_steps 1024 → 2048        (larger rollout buffer)
  - Reward: staged shaping with milestone bonuses (replaces linear bonus)
  - Curriculum: pose_weights start face_up-heavy, flatten over training

Usage
-----
    cd src/
    python train.py                             # defaults (Run 2 settings)
    python train.py --timesteps 1_000_000       # quick smoke run
    python train.py --resume ../models/best_model.zip
    python train.py --no-curriculum             # uniform pose sampling
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

if sys.platform == "darwin":
    mp.set_start_method("spawn", force=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(__file__))
from wrapper import make_standup_env, POSE_NAMES


# ---------------------------------------------------------------------------
# Curriculum callback — gradually flattens pose weights over training
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Starts with face_up heavily weighted, transitions to uniform by
    curriculum_end_steps. The logic: learn to stand at all first,
    then generalise to harder orientations.

    Initial weights : [4, 1, 1, 1]  → 57% face_up
    Final weights   : [1, 1, 1, 1]  → 25% each (uniform)
    """

    def __init__(self, vec_env, curriculum_end_steps: int = 3_000_000, verbose: int = 0):
        super().__init__(verbose)
        self._vec_env = vec_env
        self._end = curriculum_end_steps

    def _on_step(self) -> bool:
        t = self.num_timesteps
        frac = min(t / self._end, 1.0)

        # Interpolate: face_up weight 4→1, others 1→1
        face_up_w = 4.0 - frac * 3.0
        weights = np.array([face_up_w, 1.0, 1.0, 1.0])
        probs = weights / weights.sum()

        # Push new weights into every env's ArbitraryPoseWrapper
        def _set_weights(env):
            # Walk wrapper stack to find ArbitraryPoseWrapper
            e = env
            while hasattr(e, "env"):
                from wrapper import ArbitraryPoseWrapper
                if isinstance(e, ArbitraryPoseWrapper):
                    e._probs = probs
                    break
                e = e.env

        self._vec_env.env_method("_set_curriculum_weights", probs,
                                 indices=None) if hasattr(self._vec_env, "env_method") else None
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on HumanoidStandup from arbitrary poses")
    p.add_argument("--n-envs",           type=int,   default=8)
    p.add_argument("--timesteps",        type=int,   default=10_000_000)
    p.add_argument("--n-steps",          type=int,   default=2048,  help="Steps per env per rollout")
    p.add_argument("--batch-size",       type=int,   default=256)
    p.add_argument("--n-epochs",         type=int,   default=5,     help="PPO gradient epochs per rollout")
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--ent-coef",         type=float, default=0.01)
    p.add_argument("--no-upright-bonus", action="store_true")
    p.add_argument("--no-curriculum",    action="store_true",       help="Uniform pose sampling from step 0")
    p.add_argument("--curriculum-end",   type=int,   default=3_000_000, help="Step at which curriculum flattens to uniform")
    p.add_argument("--standing-prob",    type=float, default=0.3,   help="Fraction of episodes that start already standing (balance training)")
    p.add_argument("--log-dir",          type=str,   default="../logs")
    p.add_argument("--model-dir",        type=str,   default="../models")
    p.add_argument("--resume",           type=str,   default=None)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--eval-episodes",    type=int,   default=20)
    p.add_argument("--no-normalize",     action="store_true")
    p.add_argument("--run-name",         type=str, default="ppo_standup_run2",
                   help="TensorBoard run name (default: ppo_standup_run2)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.log_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    use_bonus = not args.no_upright_bonus

    # Initial pose weights: face_up heavy if curriculum enabled
    init_weights = None if args.no_curriculum else [4.0, 1.0, 1.0, 1.0]

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    print(f"[train] Spawning {args.n_envs} parallel environments …")

    def _make(rank: int):
        return lambda: make_standup_env(
            rank=rank,
            seed=args.seed,
            use_upright_bonus=use_bonus,
            pose_weights=init_weights,
            standing_prob=args.standing_prob,
        )

    train_env = SubprocVecEnv([_make(i) for i in range(args.n_envs)])

    stats_path = os.path.join(args.model_dir, "vec_normalize.pkl")
    if not args.no_normalize:
        if args.resume and os.path.exists(stats_path):
            print(f"[train] Loading VecNormalize stats from {stats_path}")
            train_env = VecNormalize.load(stats_path, train_env)
        else:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_raw = SubprocVecEnv([
        (lambda r: lambda: make_standup_env(rank=r, seed=args.seed + 100))(i)
        for i in range(4)
    ])
    if not args.no_normalize:
        eval_env = VecNormalize(eval_raw, training=False, norm_reward=False)
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
    else:
        eval_env = eval_raw

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    policy_kwargs = dict(net_arch=[256, 256])

    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env,
                         tensorboard_log=args.log_dir, verbose=1)
        model.num_timesteps = 0
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            learning_rate=args.lr,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.log_dir,
            verbose=1,
            seed=args.seed,
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[train] Policy parameters : {n_params:,}")
    print(f"[train] Rollout buffer     : {args.n_steps * args.n_envs:,} steps")
    print(f"[train] Target timesteps   : {args.timesteps:,}")
    print(f"[train] Curriculum         : {'off' if args.no_curriculum else f'face_up→uniform over {args.curriculum_end:,} steps'}")
    print(f"[train] Reward shaping     : {'off' if not use_bonus else 'staged (linear + quadratic + milestones)'}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    save_freq = max(500_000 // args.n_envs, 1)

    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=args.model_dir,
            name_prefix="humanoid_standup",
            save_vecnormalize=True,
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=args.model_dir,
            log_path=args.log_dir,
            eval_freq=max(100_000 // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\n[train] Monitor with:  tensorboard --logdir {os.path.abspath(args.log_dir)}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=args.run_name,
        reset_num_timesteps=not bool(args.resume),
        progress_bar=True,
    )

    final_path = os.path.join(args.model_dir, "final_model")
    model.save(final_path)
    print(f"[train] Saved → {final_path}.zip")

    if not args.no_normalize:
        train_env.save(stats_path)
        print(f"[train] Saved VecNormalize → {stats_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()