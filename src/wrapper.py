"""
wrapper.py
----------
ArbitraryPoseWrapper: overrides reset() to start the humanoid in a randomly
sampled lying orientation (face-up, face-down, side-left, side-right).

UprightBonusWrapper: staged reward shaping with linear ramp, quadratic boost
above standing height, and one-time milestone bonuses.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

try:
    import mujoco
except ImportError as e:
    raise ImportError(
        "mujoco Python package not found. Install with: pip install mujoco"
    ) from e


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle_rad / 2.0)
    return np.array([np.cos(angle_rad / 2.0), *(s * axis)])


_X = np.array([1.0, 0.0, 0.0])

POSE_QUATERNIONS: dict[str, np.ndarray] = {
    "face_up":    _axis_angle_to_quat(_X,  0.0),
    "face_down":  _axis_angle_to_quat(_X,  np.pi),
    "side_left":  _axis_angle_to_quat(_X,  np.pi / 2),
    "side_right": _axis_angle_to_quat(_X, -np.pi / 2),
}

POSE_NAMES = list(POSE_QUATERNIONS.keys())


# ---------------------------------------------------------------------------
# Pose randomisation wrapper
# ---------------------------------------------------------------------------

class ArbitraryPoseWrapper(gym.Wrapper):
    """
    Randomises the humanoid's initial lying orientation at every reset.

    Parameters
    ----------
    pose_weights : list[float] | None
        Sampling weight for [face_up, face_down, side_left, side_right].
        Defaults to uniform. Pass e.g. [4,1,1,1] for curriculum weighting.
    quat_noise : float
        Std-dev of Gaussian noise added to the base quaternion.
    standing_prob : float
        Probability [0, 1] of resetting to an upright standing pose instead
        of a lying pose. Used to give the policy direct balance training signal.
        0.0 = always lying (original behaviour), 0.3 = 30% standing starts.
    """

    metadata = {"pose_names": POSE_NAMES}

    def __init__(
        self,
        env: gym.Env,
        pose_weights: list[float] | None = None,
        quat_noise: float = 0.04,
        standing_prob: float = 0.0,
    ):
        super().__init__(env)
        weights = np.ones(len(POSE_NAMES)) if pose_weights is None else np.array(pose_weights, dtype=np.float64)
        self._probs = weights / weights.sum()
        self._quat_noise = float(quat_noise)
        self._standing_prob = float(standing_prob)
        self.last_pose: str | None = None

    def _set_standing_start(self, base, data, model) -> np.ndarray:
        """
        Place the humanoid in a perturbed upright pose.
        Uses identity quaternion (upright) + small noise on all joints
        so the policy sees varied balance challenges, not a single fixed pose.
        """
        # Upright torso quaternion with small perturbation
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat += np.random.randn(4) * 0.05
        quat /= np.linalg.norm(quat)
        data.qpos[3:7] = quat

        # Lift torso to standing height (~1.3m) — read from model default
        # HumanoidStandup default standing z is around 1.4m
        data.qpos[2] = 1.35 + np.random.uniform(-0.05, 0.05)

        # Small random joint perturbations so balance is non-trivial
        data.qpos[7:] += np.random.randn(len(data.qpos) - 7) * 0.05

        # Zero velocities — start from near-static
        data.qvel[:] = np.random.randn(len(data.qvel)) * 0.02

        mujoco.mj_forward(model, data)
        return base._get_obs()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        base  = self.env.unwrapped
        model: mujoco.MjModel = base.model
        data:  mujoco.MjData  = base.data

        # Decide: standing start or lying start
        if np.random.random() < self._standing_prob:
            obs = self._set_standing_start(base, data, model)
            pose_name = "standing"
        else:
            pose_name = np.random.choice(POSE_NAMES, p=self._probs)
            quat = POSE_QUATERNIONS[pose_name].copy()
            if self._quat_noise > 0.0:
                quat += np.random.randn(4) * self._quat_noise
                quat /= np.linalg.norm(quat)
            data.qpos[3:7] = quat
            mujoco.mj_forward(model, data)
            obs = base._get_obs()

        self.last_pose = pose_name
        info["pose_class"] = pose_name
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["pose_class"] = self.last_pose
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Staged reward shaping wrapper
# ---------------------------------------------------------------------------

class UprightBonusWrapper(gym.Wrapper):
    """
    Staged reward shaping — now with sustained balance incentive.

    Four components:

    1. LINEAR ramp         — continuous bonus above z_threshold (0.3m).
                             Rewards any upward progress from the floor.

    2. QUADRATIC boost     — extra weight above stand_threshold (1.0m).
                             Makes full standing far more valuable than kneeling.

    3. MILESTONE bonuses   — large one-time rewards on first crossing
                             z = 0.5, 0.9, 1.1, 1.3 m.

    4. BALANCE bonus       — per-step bonus that grows with consecutive steps
                             spent above balance_height (1.1m). Resets to zero
                             if the agent falls below that threshold.
                             Directly incentivises sustained standing stability.

                             bonus = w_balance * min(steps_standing, cap) / cap

                             Caps at balance_cap steps so the reward stays bounded.

    5. TORSO VELOCITY penalty — small penalty on torso angular velocity while
                                standing. Discourages wobbling in place.
    """

    MILESTONES        = [0.5,  0.9,   1.1,   1.3  ]  # metres
    MILESTONE_REWARDS = [50.0, 200.0, 500.0, 1000.0]  # one-time bonuses

    def __init__(
        self,
        env: gym.Env,
        w_linear: float        = 1.0,
        w_quadratic: float     = 5.0,
        z_threshold: float     = 0.3,
        stand_threshold: float = 1.0,
        w_balance: float       = 3.0,   # max per-step balance bonus
        balance_height: float  = 1.1,   # z must exceed this to count
        balance_cap: int       = 200,   # steps at which bonus maxes out
        w_vel_penalty: float   = 0.05,   # torso angular velocity penalty weight (lowered from 0.2)
    ):
        super().__init__(env)
        self._w_lin      = float(w_linear)
        self._w_quad     = float(w_quadratic)
        self._z_thr      = float(z_threshold)
        self._s_thr      = float(stand_threshold)
        self._w_bal      = float(w_balance)
        self._bal_height = float(balance_height)
        self._bal_cap    = int(balance_cap)
        self._w_vel      = float(w_vel_penalty)
        self._milestones_hit: set[int] = set()
        self._steps_standing: int = 0

    def reset(self, **kwargs):
        self._milestones_hit  = set()
        self._steps_standing  = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        z = float(obs[0])

        # 1. Linear ramp from floor
        linear = self._w_lin * max(0.0, z - self._z_thr)

        # 2. Quadratic boost past standing height
        quad = self._w_quad * max(0.0, z - self._s_thr) ** 2

        # 3. One-time milestone bonuses
        milestone_bonus = 0.0
        for idx, (z_ms, r_ms) in enumerate(
            zip(self.MILESTONES, self.MILESTONE_REWARDS)
        ):
            if idx not in self._milestones_hit and z >= z_ms:
                milestone_bonus += r_ms
                self._milestones_hit.add(idx)

        # 4. Sustained balance bonus
        if z >= self._bal_height:
            self._steps_standing += 1
        else:
            self._steps_standing = 0  # reset streak on fall

        balance_bonus = self._w_bal * min(self._steps_standing, self._bal_cap) / self._bal_cap

        # 5. Torso angular velocity penalty (obs[22:25] = torso ang vel in HumanoidStandup)
        # Only penalise while standing — we don't want to discourage rolling over
        vel_penalty = 0.0
        if z >= self._bal_height and len(obs) > 24:
            torso_angvel = obs[22:25]
            vel_penalty = self._w_vel * float(np.sum(torso_angvel ** 2))

        total_bonus = linear + quad + milestone_bonus + balance_bonus - vel_penalty
        info["upright_bonus"]    = total_bonus
        info["balance_bonus"]    = balance_bonus
        info["steps_standing"]   = self._steps_standing
        info["milestone_bonus"] = milestone_bonus
        info["z_torso"]         = z

        return obs, reward + total_bonus, terminated, truncated, info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_standup_env(
    rank: int = 0,
    seed: int = 0,
    use_upright_bonus: bool = True,
    pose_weights: list[float] | None = None,
    standing_prob: float = 0.3,
    render_mode: str | None = None,
) -> gym.Env:
    """Return a fully-wrapped HumanoidStandup-v5 ready for training."""
    from stable_baselines3.common.monitor import Monitor

    env = gym.make("HumanoidStandup-v5", render_mode=render_mode)
    env = ArbitraryPoseWrapper(env, pose_weights=pose_weights, standing_prob=standing_prob)
    if use_upright_bonus:
        env = UprightBonusWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed + rank)
    return env