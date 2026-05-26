import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.pong import PongEnv
from ppo.agent import PPOAgent
from utils.plotting import plot_training_curve


def make_agent(
    state_dim,
    action_dim,
    hidden_dim,
    actor_lr,
    critic_lr,
    gamma,
    clip_epsilon,
    n_epochs,
    batch_size,
    entropy_coef,
    gae_lambda,
    device,
):
    return PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        n_epochs=n_epochs,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        gae_lambda=gae_lambda,
        device=device,
    )


def copy_agent_weights(source, target):
    target.actor.load_state_dict(deepcopy(source.actor.state_dict()))
    target.critic.load_state_dict(deepcopy(source.critic.state_dict()))


def rule_based_action(obs):
    paddle_y = obs[0]
    ball_y = obs[3]

    if ball_y < paddle_y - 0.03:
        return 1
    elif ball_y > paddle_y + 0.03:
        return 2
    return 0


def random_action(action_dim):
    return np.random.randint(action_dim)


def shaped_reward(env, raw_reward, old_distance):
    new_distance = abs(env.player_y - env.ball_y)

    shaping = 0.0

    if env.ball_vx < 0:
        shaping += 0.002 * (old_distance - new_distance)

    shaping = float(np.clip(shaping, -0.01, 0.01))

    return raw_reward + shaping


def evaluate_agent(agent, opponent_type="rule", n_games=100, device="cpu"):
    env = PongEnv(max_steps=2000)
    wins = 0
    total_reward = 0.0

    for _ in range(n_games):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        done = False
        episode_reward = 0.0

        while not done:
            action, _, _ = agent.select_action(state)

            opponent_obs = env.get_obs_opponent()

            if opponent_type == "random":
                opponent_action = random_action(env.action_space.n)
            elif opponent_type == "rule":
                opponent_action = rule_based_action(opponent_obs)
            else:
                opponent_action = 0

            next_state, reward, terminated, truncated, _ = env.step(
                (action, opponent_action)
            )

            done = terminated or truncated
            episode_reward += reward
            state = np.array(next_state, dtype=np.float32)

        if episode_reward > 0:
            wins += 1

        total_reward += episode_reward

    env.close()

    return {
        "win_rate": wins / n_games * 100.0,
        "avg_reward": total_reward / n_games,
    }


def train_pong_self_play(
    n_episodes=8000,
    hidden_dim=128,
    actor_lr=1e-4,
    critic_lr=5e-4,
    gamma=0.99,
    clip_epsilon=0.10,
    n_epochs=4,
    batch_size=128,
    entropy_coef=0.02,
    gae_lambda=0.95,
    update_interval=4096,
    opponent_sync_every=500,
    print_every=50,
    eval_every=500,
    save_path=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = PongEnv(max_steps=2000)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = make_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        n_epochs=n_epochs,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        gae_lambda=gae_lambda,
        device=device,
    )

    opponent = make_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        n_epochs=n_epochs,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        gae_lambda=gae_lambda,
        device=device,
    )

    copy_agent_weights(agent, opponent)

    rewards = []
    wins = []

    timesteps_since_update = 0
    last_policy_loss = 0.0
    last_value_loss = 0.0

    best_selfplay_win_rate = -1.0
    best_rule_win_rate = -1.0

    save_path = Path(save_path) if save_path is not None else None

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        best_path = save_path.with_name("best_ppo_pong_selfplay.pth")
        last_path = save_path
    else:
        best_path = None
        last_path = None

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        episode_reward = 0.0
        episode_raw_reward = 0.0
        done = False

        while not done:
            player_action, player_log_prob, player_value = agent.select_action(state)

            opponent_state = env.get_obs_opponent()
            opponent_action, _, _ = opponent.select_action(opponent_state)

            old_distance = abs(env.player_y - env.ball_y)

            next_state, raw_reward, terminated, truncated, _ = env.step(
                (player_action, opponent_action)
            )

            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            reward = shaped_reward(env, raw_reward, old_distance)

            agent.store_transition(
                state=state,
                action=player_action,
                log_prob=player_log_prob,
                reward=reward,
                done=done,
                value=player_value,
            )

            episode_reward += reward
            episode_raw_reward += raw_reward

            state = next_state
            timesteps_since_update += 1

            if timesteps_since_update >= update_interval:
                last_policy_loss, last_value_loss = agent.update(next_state, done)
                timesteps_since_update = 0

        rewards.append(episode_raw_reward)
        wins.append(1 if episode_raw_reward > 0 else 0)

        if episode % opponent_sync_every == 0:
            copy_agent_weights(agent, opponent)
            print(f"Opponent updated at episode {episode}")

        if episode % print_every == 0:
            avg_reward = np.mean(rewards[-print_every:])
            win_rate = np.mean(wins[-print_every:]) * 100.0

            print(
                f"Episode {episode:5d} | "
                f"Avg reward: {avg_reward:7.3f} | "
                f"Win rate: {win_rate:6.1f}% | "
                f"Policy loss: {last_policy_loss:8.4f} | "
                f"Value loss: {last_value_loss:8.4f}"
            )

            if win_rate > best_selfplay_win_rate:
                best_selfplay_win_rate = win_rate

                if best_path is not None:
                    agent.save(str(best_path))
                    print(
                        f"Best self-play model saved | "
                        f"Win rate: {best_selfplay_win_rate:.1f}%"
                    )

        if episode % eval_every == 0:
            random_eval = evaluate_agent(
                agent,
                opponent_type="random",
                n_games=100,
                device=device,
            )

            rule_eval = evaluate_agent(
                agent,
                opponent_type="rule",
                n_games=100,
                device=device,
            )

            print(
                f"\nEvaluation at episode {episode}:"
                f"\n  vs random | Win rate: {random_eval['win_rate']:6.1f}% | "
                f"Avg reward: {random_eval['avg_reward']:7.3f}"
                f"\n  vs rule   | Win rate: {rule_eval['win_rate']:6.1f}% | "
                f"Avg reward: {rule_eval['avg_reward']:7.3f}\n"
            )

            if rule_eval["win_rate"] > best_rule_win_rate:
                best_rule_win_rate = rule_eval["win_rate"]

                if best_path is not None:
                    agent.save(str(best_path))
                    print(
                        f"Best evaluation model saved | "
                        f"Rule win rate: {best_rule_win_rate:.1f}%"
                    )

    if len(agent.buffer_states) > 0:
        last_policy_loss, last_value_loss = agent.update(state, done)

    env.close()

    if last_path is not None:
        agent.save(str(last_path))
        print(f"\nLast model saved to: {last_path}")

    if best_path is not None:
        print(f"Best model path: {best_path}")

    return rewards


if __name__ == "__main__":
    save_path = ROOT_DIR / "experiments" / "ppo_pong_selfplay.pth"

    rewards = train_pong_self_play(
        n_episodes=8000,

        hidden_dim=128,

        actor_lr=1e-4,
        critic_lr=5e-4,

        gamma=0.99,
        gae_lambda=0.95,

        clip_epsilon=0.10,
        n_epochs=4,
        batch_size=128,

        entropy_coef=0.02,

        update_interval=4096,
        opponent_sync_every=500,

        print_every=50,
        eval_every=500,

        save_path=save_path,
    )

    plot_training_curve(
        rewards,
        title="Self-play PPO on Pong",
        window=50,
    )