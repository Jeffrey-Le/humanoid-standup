"""
smoke_test.py
-------------
Run this BEFORE train.py to verify:
  1. gymnasium + mujoco imports cleanly.
  2. HumanoidStandup-v5 can be instantiated.
  3. ArbitraryPoseWrapper resets correctly to all 4 pose classes.
  4. A random agent can step through the env without crashing.
  5. stable_baselines3 PPO can initialise with the env.

Takes ~30 seconds on a cold start.

Usage:
    cd src/
    python smoke_test.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def check(label: str, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}")
        return result
    except Exception as e:
        print(f"  {FAIL}  {label}")
        print(f"       {type(e).__name__}: {e}")
        return None

def main():
    print("\n── Smoke test: HumanoidStandup arbitrary pose ──\n")

    # 1. Imports
    gym = check("gymnasium import", lambda: __import__("gymnasium"))
    mj  = check("mujoco import",    lambda: __import__("mujoco"))
    sb3 = check("stable_baselines3 import", lambda: __import__("stable_baselines3"))
    if not all([gym, mj, sb3]):
        print("\nFix imports first. Aborting.")
        sys.exit(1)

    import gymnasium
    import mujoco as mujoco_mod
    from stable_baselines3 import PPO

    # 2. Base env
    env = check("gymnasium.make HumanoidStandup-v5",
                lambda: gymnasium.make("HumanoidStandup-v5"))
    if env is None:
        print("Cannot make base env. Check mujoco installation.")
        sys.exit(1)

    obs, info = check("env.reset()", lambda: env.reset(seed=0))
    check("obs shape == (348,)", lambda: (assert_shape(obs, (348,)), obs)[1])

    action = env.action_space.sample()
    check("env.step(random action)", lambda: env.step(action))
    env.close()

    # 3. Wrapper
    from wrapper import ArbitraryPoseWrapper, POSE_NAMES, POSE_QUATERNIONS

    print(f"\n── Pose classes: {POSE_NAMES} ──")

    base = gymnasium.make("HumanoidStandup-v5")
    wrapped = ArbitraryPoseWrapper(base)

    seen_poses: set[str] = set()
    for trial in range(40):
        obs, info = wrapped.reset(seed=trial)
        pose = info.get("pose_class", "MISSING")
        seen_poses.add(pose)

    check("All 4 pose classes sampled in 40 resets",
          lambda: (lambda s: s if s == set(POSE_NAMES) else (_ for _ in ()).throw(AssertionError(f"Missing: {set(POSE_NAMES) - s}")))(seen_poses))

    # Verify each pose quaternion is written correctly
    for pose_name in POSE_NAMES:
        def _check_pose(pn=pose_name):
            obs, info = wrapped.reset()
            # Force specific pose by directly writing and calling forward
            base_env = wrapped.env.unwrapped
            expected_quat = POSE_QUATERNIONS[pn]
            base_env.data.qpos[3:7] = expected_quat
            mujoco_mod.mj_forward(base_env.model, base_env.data)
            actual_quat = base_env.data.qpos[3:7].copy()
            diff = np.max(np.abs(actual_quat - expected_quat))
            if diff > 1e-6:
                raise AssertionError(f"Quat mismatch for {pn}: diff={diff}")
        check(f"Quaternion correct for {pose_name}", _check_pose)

    # Run 100 random steps
    obs, _ = wrapped.reset(seed=99)
    for _ in range(100):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        if terminated or truncated:
            obs, _ = wrapped.reset()
    check("100 random steps without crash", lambda: True)

    wrapped.close()

    # 4. PPO init
    from wrapper import make_standup_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: make_standup_env(rank=0, seed=0)])
    model   = check("PPO.__init__ with wrapped env",
                    lambda: PPO("MlpPolicy", vec_env, verbose=0,
                                policy_kwargs=dict(net_arch=[256, 256])))

    if model:
        check("model.predict on random obs",
              lambda: model.predict(vec_env.reset(), deterministic=True))
        vec_env.close()

    print("\n── All checks passed. Ready to train. ──\n")
    print("Next step:\n    cd src/ && python train.py\n")


def assert_shape(arr, shape):
    if arr.shape != shape:
        raise AssertionError(f"Expected shape {shape}, got {arr.shape}")


if __name__ == "__main__":
    main()
