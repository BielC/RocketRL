# RocketRL

A compact implementation of **Proximal Policy Optimization (PPO)** for
training a Rocket League agent with [RLGym](https://github.com/AechPro/rlgym).
It was built as a hands-on reinforcement-learning experiment: the policy and
value networks live in [`ppo.py`](ppo.py), while [`train.py`](train.py) creates
the RLGym environment, defines rewards, and runs the training loop.

> This is an archived learning project, not a maintained Rocket League bot.
> It targets the 2021-era RLGym API and may need adaptation for current RLGym
> releases.

## What it includes

- Gaussian continuous-action actor and value networks in PyTorch
- PPO clipping objective and generalized advantage estimation (GAE)
- TensorBoard metrics for rewards, losses, entropy, and action distributions
- A custom RLGym setup with ball- and goal-oriented rewards
- Periodic model checkpoints under `Saves/`

## Requirements

- Python 3.8 or 3.9
- A working local [RLGym](https://github.com/AechPro/rlgym) installation and
  the Rocket League/RLBot setup required by that version
- PyTorch compatible with your hardware

## Setup

Create an isolated Python environment and install the small dependency set:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

RLGym has changed significantly since this project was written. If its legacy
`0.4.1` release does not install in your environment, follow the RLGym setup
guidance for a compatible Python, Rocket League, and RLBot configuration.

## Train

```bash
python train.py
```

The default run is deliberately long (`100,000` games). Adjust the constants
near the top of [`train.py`](train.py)—especially `N_GAMES`, `N`, and
`BATCH_SIZE`—before starting an experiment.

During training:

- TensorBoard logs are written to `runs/`.
- Checkpoints are written to `Saves/`.

View metrics with:

```bash
tensorboard --logdir runs
```

## Project structure

```text
.
├── ppo.py            # PPO memory, actor/critic networks, and update loop
├── train.py          # RLGym environment, reward design, and training run
└── requirements.txt  # Runtime dependencies
```

## Notes

- The reward function currently combines player-to-ball, ball-to-goal, and
  goal rewards; experiment with the weighting in `train.py`.
- The repository intentionally excludes generated checkpoints and TensorBoard
  logs.
- No license has been selected for this repository yet. Contact the author
  before reusing the code outside personal experimentation.
