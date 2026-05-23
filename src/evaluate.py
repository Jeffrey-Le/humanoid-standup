"""
evaluate.py
-----------
Post-training evaluation of the humanoid standup policy.

Produces:
  1. Bar chart  — success rate + mean max z-height per pose class.
  2. Line chart — z-height over time for one episode per pose class.
  3. MP4 video  — rendered rollout for each pose class (optional, slow).
  4. JSON summary written to --output-dir.

Usage
-----
    cd src/
    python evaluate.py --model ../models/final_model.zip \\
                       --stats ../models/vec_normalize.pkl \\
                       --episodes 100 \\
                       --render-video

Success criterion: torso z-height exceeds Z_SUCCESS threshold for at
least Z_HOLD_FRAMES consecutive timesteps (default: z > 1.0 m for 20 frames).
This mirrors the intuition that "standing" is a sustained posture, not a
momentary peak.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import gymnasium as gym
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wrapper import POSE_NAMES, POSE_QUATERNIONS

# Success thresholds
Z_SUCCESS    = 1.0   # metres — torso must reach this height
Z_HOLD_FRAMES = 20   # consecutive steps at or above Z_SUCCESS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained humanoid standup policy")
    p.add_argument("--model",       required=True,                 help="Path to .zip model file")
    p.add_argument("--stats",       default=None,                  help="Path to vec_normalize.pkl (if used during training)")
    p.add_argument("--episodes",    type=int,   default=50,        help="Evaluation episodes per pose class")
    p.add_argument("--max-steps",   type=int,   default=1000,      help="Max steps per episode")
    p.add_argument("--output-dir",  type=str,   default="../logs/eval", help="Where to save plots + JSON")
    p.add_argument("--render-video",action="store_true",           help="Render MP4 video for each pose class (slow)")
    p.add_argument("--seed",        type=int,   default=0,         help="RNG seed")
    p.add_argument("--z-success",   type=float, default=Z_SUCCESS, help="Z-height threshold for success")
    p.add_argument("--hold-frames", type=int,   default=Z_HOLD_FRAMES)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-pose evaluation environment
# ---------------------------------------------------------------------------

class FixedPoseEnv(gym.Wrapper):
    """Reset always to a specific pose class (+ small noise)."""

    def __init__(self, env: gym.Env, pose_name: str, quat_noise: float = 0.04):
        super().__init__(env)
        self._pose_name = pose_name
        self._quat_noise = quat_noise

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        base = self.env.unwrapped
        quat = POSE_QUATERNIONS[self._pose_name].copy()
        if self._quat_noise:
            quat += np.random.randn(4) * self._quat_noise
            quat /= np.linalg.norm(quat)
        base.data.qpos[3:7] = quat
        mujoco.mj_forward(base.model, base.data)
        obs = base._get_obs()
        info["pose_class"] = self._pose_name
        return obs, info


# ---------------------------------------------------------------------------
# Normalisation helper (mirrors VecNormalize without vectorisation)
# ---------------------------------------------------------------------------

class ObsNormaliser:
    """Apply stored VecNormalize obs_rms without a full vectorised env."""

    def __init__(self, stats_path: str):
        import pickle
        with open(stats_path, "rb") as f:
            vec_env = pickle.load(f)
        self.mean  = vec_env.obs_rms.mean
        self.var   = vec_env.obs_rms.var
        self.clip  = 10.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(obs, -self.clip, self.clip)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_pose(
    model,
    pose_name: str,
    n_episodes: int,
    max_steps: int,
    z_success: float,
    hold_frames: int,
    normaliser=None,
    record_trajectory: bool = False,
    seed: int = 0,
) -> dict:
    env = gym.make("HumanoidStandup-v5")
    env = FixedPoseEnv(env, pose_name)

    rewards, max_heights, success_flags = [], [], []
    traj_z: list[float] | None = [] if record_trajectory else None

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_r = 0.0
        max_z   = float(obs[0])
        hold    = 0
        success = False

        for _ in range(max_steps):
            obs_in = normaliser(obs) if normaliser else obs
            action, _ = model.predict(obs_in[np.newaxis], deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action[0])

            total_r += float(reward)
            z = float(obs[0])
            max_z = max(max_z, z)

            if z >= z_success:
                hold += 1
                if hold >= hold_frames:
                    success = True
            else:
                hold = 0

            if ep == 0 and traj_z is not None:
                traj_z.append(z)

            if terminated or truncated:
                break

        rewards.append(total_r)
        max_heights.append(max_z)
        success_flags.append(int(success))

    env.close()
    return {
        "pose_class":       pose_name,
        "n_episodes":       n_episodes,
        "success_rate":     float(np.mean(success_flags)),
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "mean_max_z":       float(np.mean(max_heights)),
        "std_max_z":        float(np.std(max_heights)),
        "trajectory_z":     traj_z,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: list[dict], output_dir: str, z_success: float) -> None:
    import matplotlib.pyplot as plt

    poses        = [r["pose_class"]   for r in results]
    success_rates = [r["success_rate"] for r in results]
    mean_z       = [r["mean_max_z"]   for r in results]
    std_z        = [r["std_max_z"]    for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Humanoid Standup — Evaluation by Initial Pose", fontsize=14, fontweight="bold")

    # Bar 1: success rate
    ax = axes[0]
    colors = ["#1d9e75", "#D85A30", "#7F77DD", "#D4537E"]
    bars = ax.bar(poses, success_rates, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title(f"Success rate (z > {z_success} m for ≥20 frames)")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10)

    # Bar 2: mean max height
    ax = axes[1]
    ax.bar(poses, mean_z, yerr=std_z, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.8, capsize=5)
    ax.axhline(y=z_success, color="crimson", linestyle="--", linewidth=1, label=f"Success threshold ({z_success} m)")
    ax.set_ylabel("Mean max torso z-height (m)")
    ax.set_title("Max torso height reached per episode")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "eval_by_pose.png")
    plt.savefig(path, dpi=150)
    print(f"[eval] Saved bar chart → {path}")
    plt.close()

    # Trajectory plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Torso z-height over time (one episode per pose class)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Torso z (m)")
    ax.axhline(y=z_success, color="crimson", linestyle="--", linewidth=1,
               label=f"Success threshold ({z_success} m)", alpha=0.7)

    for r, col in zip(results, colors):
        if r["trajectory_z"]:
            ax.plot(r["trajectory_z"], label=r["pose_class"], color=col, linewidth=1.5)

    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "trajectory_z.png")
    plt.savefig(path, dpi=150)
    print(f"[eval] Saved trajectory chart → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

def render_video(model, pose_name: str, output_dir: str, normaliser=None,
                 max_steps: int = 1000, fps: int = 30) -> None:
    try:
        import imageio
    except ImportError:
        print("[eval] imageio not installed — skipping video render.")
        return

    env = gym.make("HumanoidStandup-v5", render_mode="rgb_array")
    env = FixedPoseEnv(env, pose_name)

    frames = []
    obs, _ = env.reset(seed=999)
    for _ in range(max_steps):
        obs_in = normaliser(obs) if normaliser else obs
        action, _ = model.predict(obs_in[np.newaxis], deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action[0])
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    env.close()

    path = os.path.join(output_dir, f"standup_{pose_name}.mp4")
    imageio.mimwrite(path, frames, fps=fps, quality=8)
    print(f"[eval] Saved video → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Load model
    from stable_baselines3 import PPO
    print(f"[eval] Loading model from {args.model}")
    model = PPO.load(args.model)

    # Load normaliser if stats provided
    normaliser = None
    if args.stats and os.path.exists(args.stats):
        print(f"[eval] Loading VecNormalize stats from {args.stats}")
        normaliser = ObsNormaliser(args.stats)

    # Evaluate each pose class
    results = []
    for pose_name in POSE_NAMES:
        print(f"\n[eval] Evaluating pose: {pose_name} ({args.episodes} episodes) …")
        r = evaluate_pose(
            model=model,
            pose_name=pose_name,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            z_success=args.z_success,
            hold_frames=args.hold_frames,
            normaliser=normaliser,
            record_trajectory=True,
            seed=args.seed,
        )
        print(f"  success_rate={r['success_rate']:.1%}  "
              f"mean_max_z={r['mean_max_z']:.3f} ± {r['std_max_z']:.3f}  "
              f"mean_reward={r['mean_reward']:.1f}")
        results.append(r)

    # Save JSON summary (drop trajectory for brevity)
    summary = [{k: v for k, v in r.items() if k != "trajectory_z"} for r in results]
    json_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] Saved JSON summary → {json_path}")

    # Plots
    plot_results(results, args.output_dir, args.z_success)

    # Videos
    if args.render_video:
        for pose_name in POSE_NAMES:
            print(f"[eval] Rendering video for {pose_name} …")
            render_video(model, pose_name, args.output_dir,
                         normaliser=normaliser, max_steps=args.max_steps)

    # Overall summary line
    overall_success = float(np.mean([r["success_rate"] for r in results]))
    print(f"\n{'='*50}")
    print(f"Overall success rate (all poses): {overall_success:.1%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
