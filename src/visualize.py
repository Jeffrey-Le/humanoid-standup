"""
visualize.py
------------
Watch the trained policy control the humanoid in real time, or record a video.

macOS note
----------
mujoco.viewer.launch_passive requires `mjpython` on macOS and is NOT used here.
Instead, this script uses Gymnasium's built-in render_mode="human" (OpenGL via
glfw) for live viewing, and render_mode="rgb_array" + matplotlib for a
no-dependency fallback or recording.

Usage
-----
    # Live window (default)
    python visualize.py --model ../models/final_model.zip \\
                        --stats ../models/vec_normalize.pkl

    # Specific pose
    python visualize.py --model ../models/final_model.zip --pose face_down

    # All 4 poses in sequence
    python visualize.py --model ../models/final_model.zip --pose all

    # Slow down playback (0.5 = half speed)
    python visualize.py --model ../models/final_model.zip --slow 0.5

    # Record MP4s (no window)
    python visualize.py --model ../models/final_model.zip --pose all --record

    # Sanity check before training is done (random actions)
    python visualize.py --random-policy --pose all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))
from wrapper import POSE_NAMES, POSE_QUATERNIONS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize humanoid standup policy")
    p.add_argument("--model",         default=None,
                   help="Path to .zip model (omit for --random-policy)")
    p.add_argument("--stats",         default=None,
                   help="Path to vec_normalize.pkl")
    p.add_argument("--pose",          default="random",
                   choices=POSE_NAMES + ["random", "all"],
                   help="Starting pose class (default: random each episode)")
    p.add_argument("--episodes",      type=int, default=1,
                   help="Episodes per pose when --pose all")
    p.add_argument("--max-steps",     type=int, default=1000)
    p.add_argument("--record",        action="store_true",
                   help="Save MP4 instead of opening a window")
    p.add_argument("--output-dir",    default="../logs/videos")
    p.add_argument("--fps",           type=int, default=30)
    p.add_argument("--random-policy", action="store_true",
                   help="Use random actions (no model needed)")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--slow",          type=float, default=1.0,
                   help="Playback speed multiplier (0.5 = half speed, live only)")
    p.add_argument("--matplotlib",    action="store_true",
                   help="Force matplotlib renderer instead of OpenGL window")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Obs normaliser
# ---------------------------------------------------------------------------

class ObsNormaliser:
    def __init__(self, stats_path: str):
        import pickle
        with open(stats_path, "rb") as f:
            vec_env = pickle.load(f)
        self.mean = vec_env.obs_rms.mean
        self.var  = vec_env.obs_rms.var
        self.clip = 10.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(obs, -self.clip, self.clip)


# ---------------------------------------------------------------------------
# Pose injection
# ---------------------------------------------------------------------------

def inject_pose(env: gym.Env, pose_name: str, noise: float = 0.04) -> np.ndarray:
    base = env.unwrapped
    quat = POSE_QUATERNIONS[pose_name].copy()
    quat += np.random.randn(4) * noise
    quat /= np.linalg.norm(quat)
    base.data.qpos[3:7] = quat
    mujoco.mj_forward(base.model, base.data)
    return base._get_obs()


# ---------------------------------------------------------------------------
# Action helper
# ---------------------------------------------------------------------------

def get_action(model, normaliser, obs: np.ndarray, env: gym.Env) -> np.ndarray:
    if model is None:
        return env.action_space.sample()
    obs_in = normaliser(obs) if normaliser else obs
    action, _ = model.predict(obs_in[np.newaxis], deterministic=True)
    return action[0]


# ---------------------------------------------------------------------------
# Live viewer — Gymnasium render_mode="human" (works on macOS without mjpython)
# ---------------------------------------------------------------------------

def run_live_gymnasium(
    model,
    normaliser,
    pose_name: str,
    max_steps: int,
    slow: float,
    seed: int,
) -> None:
    """
    Uses Gymnasium's built-in OpenGL window (glfw). Works on macOS natively.
    No mjpython needed.
    """
    env = gym.make("HumanoidStandup-v5", render_mode="human")
    obs, _ = env.reset(seed=seed)
    obs = inject_pose(env, pose_name)

    print(f"\n  Pose   : {pose_name}")
    print(f"  Policy : {'random actions' if model is None else 'trained model'}")
    print(f"  Steps  : {max_steps}  |  Speed: {slow}x")
    print("  Close the window or Ctrl-C to stop.\n")

    step_dt  = env.unwrapped.model.opt.timestep
    sleep_dt = step_dt / max(slow, 0.01)

    try:
        for step in range(max_steps):
            action = get_action(model, normaliser, obs, env)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()   # draws to the glfw window

            z = float(obs[0])
            if step % 50 == 0:
                print(f"  step={step:4d}  z={z:.3f}  reward={reward:+.1f}", end="\r")

            time.sleep(sleep_dt)

            if terminated or truncated:
                print(f"\n  Episode done at step {step}  final z={z:.3f}")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        env.close()
    print()


# ---------------------------------------------------------------------------
# Live viewer — matplotlib (fallback, no OpenGL required)
# ---------------------------------------------------------------------------

def run_live_matplotlib(
    model,
    normaliser,
    pose_name: str,
    max_steps: int,
    slow: float,
    seed: int,
) -> None:
    """
    Renders into a matplotlib window using rgb_array frames.
    Slower than OpenGL but works in any environment (SSH, headless, etc).
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as _  # noqa — ensure backend loaded

    env = gym.make("HumanoidStandup-v5", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    obs = inject_pose(env, pose_name)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f"HumanoidStandup — pose: {pose_name}", fontsize=11)
    ax.axis("off")

    frame = env.render()
    im = ax.imshow(frame)
    step_text = ax.text(0.01, 0.97, "", transform=ax.transAxes,
                        color="white", fontsize=9, va="top",
                        bbox=dict(facecolor="black", alpha=0.4, pad=2))

    print(f"\n  Pose: {pose_name}  |  Close the window to stop.\n")

    step_dt  = env.unwrapped.model.opt.timestep
    sleep_dt = step_dt / max(slow, 0.01)

    try:
        for step in range(max_steps):
            if not plt.fignum_exists(fig.number):
                break

            action = get_action(model, normaliser, obs, env)
            obs, reward, terminated, truncated, _ = env.step(action)
            z = float(obs[0])

            frame = env.render()
            im.set_data(frame)
            step_text.set_text(f"step={step}  z={z:.3f}  r={reward:+.1f}")
            fig.canvas.draw()
            fig.canvas.flush_events()

            if step % 50 == 0:
                print(f"  step={step:4d}  z={z:.3f}  reward={reward:+.1f}", end="\r")

            time.sleep(sleep_dt)

            if terminated or truncated:
                print(f"\n  Episode done at step {step}  final z={z:.3f}")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        plt.ioff()
        plt.close(fig)
        env.close()
    print()


# ---------------------------------------------------------------------------
# Video recorder
# ---------------------------------------------------------------------------

def record_episode(
    model,
    normaliser,
    pose_name: str,
    max_steps: int,
    output_path: str,
    fps: int,
    seed: int,
) -> None:
    try:
        import imageio
    except ImportError:
        print("[viz] imageio not installed — run: pip install imageio imageio-ffmpeg")
        return

    env = gym.make("HumanoidStandup-v5", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    obs = inject_pose(env, pose_name)

    frames      = []
    total_reward = 0.0

    for step in range(max_steps):
        action = get_action(model, normaliser, obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        z = float(obs[0])

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if step % 100 == 0:
            print(f"  [{pose_name}] step={step:4d}  z={z:.3f}", end="\r")

        if terminated or truncated:
            break

    env.close()
    print(f"\n  [{pose_name}] {len(frames)} frames  total_reward={total_reward:.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # Determine pose list
    if args.pose == "all":
        poses = POSE_NAMES * max(args.episodes, 1)
    elif args.pose == "random":
        poses = [np.random.choice(POSE_NAMES)
                 for _ in range(max(args.episodes, 1))]
    else:
        poses = [args.pose] * max(args.episodes, 1)

    # Load model
    model = None
    if not args.random_policy:
        if args.model is None:
            print("[viz] Provide --model or pass --random-policy.")
            sys.exit(1)
        from stable_baselines3 import PPO
        print(f"[viz] Loading model: {args.model}")
        model = PPO.load(args.model)

    # Load normaliser
    normaliser = None
    if args.stats and os.path.exists(args.stats):
        print(f"[viz] Loading normaliser: {args.stats}")
        normaliser = ObsNormaliser(args.stats)

    os.makedirs(args.output_dir, exist_ok=True)

    for i, pose_name in enumerate(poses):
        print(f"\n── Episode {i+1}/{len(poses)}  pose={pose_name} ──")

        if args.record:
            out_path = os.path.join(
                args.output_dir, f"standup_{pose_name}_{i:02d}.mp4"
            )
            record_episode(
                model=model,
                normaliser=normaliser,
                pose_name=pose_name,
                max_steps=args.max_steps,
                output_path=out_path,
                fps=args.fps,
                seed=args.seed + i,
            )

        elif args.matplotlib:
            run_live_matplotlib(
                model=model,
                normaliser=normaliser,
                pose_name=pose_name,
                max_steps=args.max_steps,
                slow=args.slow,
                seed=args.seed + i,
            )

        else:
            # Default: Gymnasium native OpenGL window (no mjpython needed)
            run_live_gymnasium(
                model=model,
                normaliser=normaliser,
                pose_name=pose_name,
                max_steps=args.max_steps,
                slow=args.slow,
                seed=args.seed + i,
            )

    print("\n[viz] Done.")


if __name__ == "__main__":
    main()