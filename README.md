# Humanoid Standup from Arbitrary Pose

**Tesla Controls Engineering Take-Home — Problem 8**

A single PPO policy that stands up a MuJoCo humanoid from any initial lying orientation — face-up, face-down, or either side — and maintains balance.

**99.8% success rate across 400 episodes (100 per pose class)**

---

## Results

| Pose | Success rate | Mean max z | Std z | Mean reward |
|---|---|---|---|---|
| face_up | 100% | 1.228 m | ± 0.012 | 385,410 |
| face_down | 100% | 1.227 m | ± 0.006 | 374,848 |
| side_left | 100% | 1.230 m | ± 0.013 | 340,809 |
| side_right | 99% | 1.224 m | ± 0.029 | 373,039 |
| **Overall** | **99.8%** | — | — | **368,527** |

Success criterion: torso z-height > 1.0 m sustained for ≥ 20 consecutive timesteps.

Video demos: [Google Drive](https://drive.google.com/drive/folders/1Kf01WvBmyqSABHDOyPdUb0ZlmH7-A-EL?usp=share_link)

---

## Approach

**PPO (Stable-Baselines3) with domain randomisation and staged reward shaping.**

The standard `HumanoidStandup-v5` only resets to a face-up pose. The core engineering contribution is:

1. **`ArbitraryPoseWrapper`** — overrides `reset()` to inject one of four lying quaternions (face-up, face-down, side-left, side-right) via `mj_forward`, with 15% of episodes starting from a perturbed upright pose for balance training.

2. **`UprightBonusWrapper`** — staged reward shaping with five components: linear ramp from the floor, quadratic boost above 1.0 m, one-time milestone bonuses at 0.5/0.9/1.1/1.3 m, a streak bonus that compounds while z ≥ 1.1 m, and a torso angular velocity penalty while standing.

The key insight: the existing `uph_cost` reward (z_torso / dt) incentivises height but creates a kneeling local optimum at ~0.8 m. The milestone bonuses break this by making the risky push to full standing clearly worth attempting.

---

## Requirements

- macOS Apple Silicon (M3 / M2 / M1) or Linux x86-64
- Python 3.11
- ~8 GB RAM

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd humanoid-standup

# 2. Create and activate conda environment
conda env create -f environment.yml
conda activate humanoid-standup

# 3. Verify MuJoCo
python -c "import mujoco; print(mujoco.__version__)"
```

---

## Running the policy

```bash
cd src/

# Smoke test — verify all dependencies
python smoke_test.py

# Live viewer — watch the policy on all 4 pose classes
python visualize.py \
  --model ../models/best_model.zip \
  --stats ../models/vec_normalize.pkl \
  --pose all

# Watch a specific pose
python visualize.py \
  --model ../models/best_model.zip \
  --stats ../models/vec_normalize.pkl \
  --pose face_down

# Record MP4 videos
python visualize.py \
  --model ../models/best_model.zip \
  --stats ../models/vec_normalize.pkl \
  --pose all --record
```

---

## Evaluation

```bash
cd src/
python evaluate.py \
  --model ../models/best_model.zip \
  --stats ../models/vec_normalize.pkl \
  --episodes 100
```

Outputs to `logs/eval/`:
- `eval_summary.json` — success rate, mean reward, mean max height per pose
- `eval_by_pose.png` — bar charts
- `trajectory_z.png` — torso z-height over time per pose class

---

## Training from scratch

```bash
cd src/
python train.py \
  --timesteps 12_000_000 \
  --no-curriculum \
  --standing-prob 0.15 \
  --run-name my_run

# Monitor
tensorboard --logdir ../logs
```

Approximate training time: **5–6 hours** on Apple M3 Max (8 parallel envs).

### Resuming from a checkpoint

```bash
python train.py \
  --resume ../models/best_model.zip \
  --timesteps 5_000_000 \
  --lr 5e-5 \
  --ent-coef 0.008 \
  --run-name my_run_resume
```

### Key hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| n_envs | 8 |
| n_steps | 2048 |
| batch_size | 256 |
| n_epochs | 5 |
| lr | 1e-4 |
| ent_coef | 0.01 |
| Policy network | MLP [256, 256] |
| Obs normalisation | VecNormalize |
| Standing starts | 15% of episodes |

---

## Model files

Two files are required to run the policy:

| File | Purpose |
|---|---|
| `models/best_model.zip` | Trained PPO policy weights |
| `models/vec_normalize.pkl` | Observation normalisation statistics (required) |

The policy was trained with `VecNormalize`. The network weights were learned assuming normalised inputs — `vec_normalize.pkl` is a required part of the model, not optional.

---

## Project layout

```
humanoid-standup/
├── environment.yml          # Conda environment spec
├── requirements.txt         # pip dependencies
├── README.md
├── src/
│   ├── wrapper.py           # ArbitraryPoseWrapper, UprightBonusWrapper
│   ├── train.py             # PPO training entry point
│   ├── evaluate.py          # Per-pose evaluation + plots + video
│   ├── visualize.py         # Live viewer + video recorder
│   └── smoke_test.py        # Dependency / sanity checks
├── models/
│   ├── best_model.zip       # Trained policy weights
│   └── vec_normalize.pkl    # Obs normalisation stats
└── logs/
    └── eval/
        ├── eval_summary.json
        ├── eval_by_pose.png
        └── trajectory_z.png
```

---

## Limitations

- **No memory**: MLP policy cannot recover from unexpected mid-episode disturbances (e.g. a push while standing) since it has no trajectory context.
- **Diagonal orientations**: poses at ~45° between canonical classes were not explicitly sampled; Gaussian noise covers small deviations only.
- **Torque saturation**: ±0.4 N·m limit is tight; the kneeling-to-stand transition occasionally stalls.
- **Sim-to-real gap**: contact dynamics are MuJoCo-specific; domain randomisation over physics parameters would be needed before hardware deployment.
- **Posture**: the standing posture is slightly crouched and forward-leaning; a posture regularisation term or longer training would improve this.

---

## Tools used

- Gymnasium / MuJoCo — simulation
- Stable-Baselines3 — PPO implementation
- PyTorch — deep learning backend
- TensorBoard — training monitoring
