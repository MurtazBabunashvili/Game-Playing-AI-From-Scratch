# AI - Learn Playing From Scratch

A reinforcement learning project that implements game-playing agents from the ground up.  
The goal of the project is to understand how agents learn through interaction, starting from classic tabular RL algorithms and moving toward deep reinforcement learning methods such as DQN, REINFORCE, Actor-Critic, and PPO.

## What This Project Covers

- Tabular reinforcement learning on GridWorld
- Deep Q-Learning with replay buffer and target network
- Policy-gradient methods
- Actor-Critic with eligibility traces
- Proximal Policy Optimization
- Custom Gymnasium-style environments
- Training logs and reward-curve visualization
- Playable/rendered visualizers for trained agents

## Implemented Algorithms

### Tabular RL

Located in `tabular/`.

- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Value Iteration
- Monte Carlo Prediction
- Monte Carlo Control
- TD(0) Prediction
- SARSA
- Q-Learning

These algorithms are mainly demonstrated on the custom `GridWorld` environment.

### Deep Q-Network

Located in `dqn/`.

The DQN implementation includes:

- Q-network
- Target network
- Experience replay buffer
- Epsilon-greedy exploration
- Model saving and loading
- Training loop compatible with Gymnasium environments and the custom Snake environment

### Policy Gradient

Located in `policy_gradient/`.

Implemented methods:

- REINFORCE
- REINFORCE with value baseline
- Actor-Critic
- Actor-Critic with eligibility traces

### PPO

Located in `ppo/`.

The PPO implementation includes:

- Actor network
- Critic network
- Clipped surrogate objective
- Generalized Advantage Estimation
- Entropy regularization
- Mini-batch updates
- Model saving and loading

## Environments

Located in `environments/`.

### GridWorld

A deterministic grid environment used for studying classic RL concepts such as value functions, policies, Bellman updates, and temporal-difference learning.

### Snake

A custom Snake environment with a Gymnasium-style API.

The observation contains danger, direction, and food-position features.  
The action space contains:

- go straight
- turn right
- turn left

Rewards:

- positive reward for eating food
- negative reward for dying
- neutral reward for normal movement

### Pong

A custom 1v1 Pong environment designed for PPO and self-play experiments.

The observation contains normalized paddle and ball information.  
The action space contains:

- stay
- move up
- move down

## Project Structure

```text
AI - Learn playing from scratch/
│
├── dqn/
│   ├── agent.py
│   ├── model.py
│   ├── replay_buffer.py
│   └── train.py
│
├── environments/
│   ├── gridworld.py
│   ├── pong.py
│   └── snake.py
│
├── experiments/
│   ├── run_actor_critic.py
│   ├── run_cartpole.py
│   ├── run_gridworld.py
│   ├── run_pong.py
│   ├── run_ppo.py
│   ├── run_reinforce.py
│   └── run_snake.py
│
├── policy_gradient/
│   ├── actor_critic.py
│   ├── reinforce.py
│   └── train.py
│
├── ppo/
│   ├── agent.py
│   ├── model.py
│   └── train.py
│
├── tabular/
│   ├── monte_carlo.py
│   ├── policy_evaluation.py
│   ├── policy_improvement.py
│   ├── policy_iteration.py
│   ├── q_learning.py
│   ├── sarsa.py
│   ├── td_learning.py
│   ├── utils.py
│   └── value_iteration.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   └── plotting.py
│
└── visualization/
    ├── dqn/
    │   └── snake_game.py
    └── ppo/
        ├── pong_ppo.py
        └── ppo_cartpole.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy matplotlib torch gymnasium pygame
```

## How to Run

Run commands from the project root directory.

### GridWorld Experiments

```bash
python -m experiments.run_gridworld
```

This runs multiple tabular RL algorithms and visualizes value functions, policies, and learning curves.

### Train DQN on CartPole

```bash
python -m experiments.run_cartpole
```

This trains a DQN agent on `CartPole-v1`.

### Train DQN on Snake

```bash
python -m experiments.run_snake
```

This trains a DQN agent on the custom Snake environment.

### Train REINFORCE on CartPole

```bash
python -m experiments.run_reinforce
```

### Train Actor-Critic

```bash
python -m experiments.run_actor_critic
```

The current script is configured for `Acrobot-v1`.

### Train PPO on CartPole

```bash
python -m experiments.run_ppo
```

### Train PPO on Pong

```bash
python -m experiments.run_pong
```

This script trains a PPO agent in the custom Pong environment using shaped rewards and mixed opponent strategies.

## Visualization

### Visualize Snake

```bash
python -m visualization.dqn.snake_game
```

This loads a saved Snake DQN model if available.  
Without a trained model, you can call the visualizer function manually and run random actions.

### Visualize PPO CartPole

```bash
python -m visualization.ppo.ppo_cartpole
```

### Visualize PPO Pong Models

```bash
python -m visualization.ppo.pong_ppo
```

This searches for saved Pong PPO model files and tests them in a rendered Pong window.

## Logs and Saved Models

Training logs are written to the `runs/` directory as CSV files.

Saved model files usually use the `.pth` extension, for example:

```text
cartpole_dqn.pth
ppo_cartpole.pth
snake_10x10_dqn.pth
```

These files are generated after training when a `save_path` is provided.

## Configuration

Reusable configuration classes are located in `utils/config.py`.

Available config classes include:

- `DQNConfig`
- `SnakeConfig`
- `REINFORCEConfig`
- `ActorCriticConfig`
- `PPOConfig`

Example:

```python
from utils.config import DQNConfig

cfg = DQNConfig(
    env_id="CartPole-v1",
    n_episodes=600,
    save_path="cartpole_dqn.pth"
)

print(cfg)
```

## Learning Goal

This project is designed as a practical reinforcement learning playground.  
Instead of only using ready-made libraries, the core agent logic is implemented manually so that the important ideas are visible in code:

- how agents select actions
- how rewards are collected
- how value functions are updated
- how neural networks approximate policies or Q-values
- how exploration changes over time
- how training performance can be logged and visualized

## Author

Murtaz Babunashvili
