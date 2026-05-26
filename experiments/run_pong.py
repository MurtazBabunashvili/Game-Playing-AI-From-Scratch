
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


def greedy_action(agent, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        logits = agent.actor(state_tensor)
        return int(torch.argmax(logits, dim=-1).item())


def copy_agent_weights(source, target):
    target.actor.load_state_dict(deepcopy(source.actor.state_dict()))
    target.critic.load_state_dict(deepcopy(source.critic.state_dict()))


def random_action(action_dim):
    return np.random.randint(action_dim)


def rule_based_action(obs):
    paddle_y = obs[0]
    ball_y = obs[3]

    if ball_y < paddle_y - 0.03:
        return 1
    if ball_y > paddle_y + 0.03:
        return 2
    return 0


def get_phase(episode, warmup_idle, warmup_random, warmup_rule):
    random_end = warmup_idle + warmup_random
    rule_end = warmup_idle + warmup_random + warmup_rule

    if episode <= warmup_idle:
        return "idle"
    if episode <= random_end:
        return "random"
    if episode <= rule_end:
        return "rule"
    return "mixed"


def pick_opponent_type(
    episode,
    warmup_idle,
    warmup_random,
    warmup_rule,
    mix_selfplay,
    mix_random,
):
    phase = get_phase(episode, warmup_idle, warmup_random, warmup_rule)

    if phase == "idle":
        return "idle"
    if phase == "random":
        return "random"
    if phase == "rule":
        return "rule"

    r = np.random.random()

    if r < mix_selfplay:
        return "self_play"
    if r < mix_selfplay + mix_random:
        return "random"
    return "rule"


def agent_has_buffer(agent):
    return hasattr(agent, "buffer_states") and len(agent.buffer_states) > 0



def shaped_reward(
    env,
    raw_reward,
    prev_ball_vx,
    terminated,
    hit_reward=0.20,
    alignment_coef=0.05,
    score_bonus=0.30,
):
    shaping = 0.0

    # Extra reward for actually scoring, so mixed phase does not become only defensive.
    if raw_reward > 0:
        shaping += score_bonus

    # Successful left paddle hit: ball was moving left, now moving right.
    if not terminated and prev_ball_vx < 0 and env.ball_vx > 0:
        shaping += hit_reward

    # Reward being aligned with the incoming ball.
    if not terminated and env.ball_vx < 0:
        gap = abs(env.player_y - env.ball_y)
        alignment = max(0.0, 1.0 - gap / max(env.paddle_h, 1.0))
        shaping += alignment_coef * alignment

    shaping = float(np.clip(shaping, -0.03, 0.35))

    return raw_reward + shaping


def evaluate_agent(agent, opponent_type="rule", n_games=100):
    env = PongEnv(max_steps=2000)

    wins = 0
    total_reward = 0.0
    total_hits = 0
    total_steps = 0

    action_counts = {0: 0, 1: 0, 2: 0}

    for _ in range(n_games):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        done = False
        episode_reward = 0.0

        while not done:
            action = greedy_action(agent, state)
            action_counts[action] += 1
            total_steps += 1

            opponent_obs = env.get_obs_opponent()

            if opponent_type == "idle":
                opponent_action = 0
            elif opponent_type == "random":
                opponent_action = random_action(env.action_space.n)
            elif opponent_type == "rule":
                opponent_action = rule_based_action(opponent_obs)
            else:
                opponent_action = 0

            prev_ball_vx = env.ball_vx

            next_state, reward, terminated, truncated, _ = env.step(
                (action, opponent_action)
            )

            if not terminated and prev_ball_vx < 0 and env.ball_vx > 0:
                total_hits += 1

            done = terminated or truncated
            episode_reward += reward
            state = np.array(next_state, dtype=np.float32)

        if episode_reward > 0:
            wins += 1

        total_reward += episode_reward

    env.close()

    total_steps = max(1, total_steps)

    return {
        "win_rate": wins / n_games * 100.0,
        "avg_reward": total_reward / n_games,
        "hit_rate": total_hits / n_games,
        "action_dist": {
            0: action_counts[0] / total_steps,
            1: action_counts[1] / total_steps,
            2: action_counts[2] / total_steps,
        },
    }


def fmt_eval(label, result):
    d = result["action_dist"]

    return (
        f"  vs {label:<8} | "
        f"win {result['win_rate']:5.1f}% | "
        f"avg_r {result['avg_reward']:+6.3f} | "
        f"hits/ep {result['hit_rate']:5.1f} | "
        f"stay {d[0]:.0%}  up {d[1]:.0%}  dn {d[2]:.0%}"
    )


def print_diagnosis(rule_eval):
    d = rule_eval["action_dist"]

    if d[0] > 0.70 and rule_eval["hit_rate"] < 3:
        print("  [diag] FREEZE: high stay and low hits — policy is not tracking.")

    elif (
        d[0] > 0.70
        and rule_eval["hit_rate"] >= 20
        and rule_eval["avg_reward"] >= -0.05
    ):
        print("  [diag] STABLE DEFENSE: high stay, high hits, not losing.")

    elif max(d[1], d[2]) > 0.70:
        dominant = "up" if d[1] > d[2] else "down"
        print(f"  [diag] STUCK: {dominant} > 70% — paddle may hug wall.")

    elif abs(d[1] - d[2]) < 0.07 and d[0] < 0.15:
        print("  [diag] JITTER: up ≈ down and low stay — oscillating.")

    else:
        print("  [diag] OK: action mix looks reasonable.")


def train_pong_self_play(
    n_episodes=4000,
    hidden_dim=128,
    actor_lr=1e-4,
    critic_lr=5e-4,
    gamma=0.99,
    clip_epsilon=0.10,
    n_epochs=4,
    batch_size=128,
    entropy_coef=0.02,
    gae_lambda=0.95,
    update_interval=1024,
    opponent_sync_every=300,

    # Curriculum:
    warmup_idle=300,
    warmup_random=500,
    warmup_rule=1700,

    # Mixed phase after curriculum:
    mix_selfplay=0.60,
    mix_random=0.20,

    # Reward shaping:
    hit_reward=0.20,
    alignment_coef=0.05,
    action_switch_penalty=0.003,
    score_bonus=0.30,

    # Save/stop:
    checkpoint_every=250,
    early_stop_after_rule_solved=False,
    rule_solved_avg_reward=-0.02,
    rule_solved_hits=25.0,
    rule_solved_misses=0.05,
    rule_solved_patience=2,

    print_every=50,
    eval_every=250,
    save_path=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    warmup_total = warmup_idle + warmup_random + warmup_rule
    mix_rule = 1.0 - mix_selfplay - mix_random

    print(f"Device: {device}")
    print(
        f"Phase 1 — curriculum: "
        f"idle={warmup_idle}, random={warmup_random}, rule={warmup_rule} "
        f"(total={warmup_total})"
    )
    print(
        f"Phase 2 — mixed: "
        f"self={mix_selfplay:.0%}, random={mix_random:.0%}, rule={mix_rule:.0%}"
    )
    print(
        f"Reward shaping: hit={hit_reward}, alignment={alignment_coef}, "
        f"switch_penalty={action_switch_penalty}, score_bonus={score_bonus}"
    )
    print()

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

    window_hits = []
    window_misses = []

    timesteps_since_update = 0
    last_policy_loss = 0.0
    last_value_loss = 0.0

    best_mixed_wr = -1.0
    best_rule_wr = -1.0
    best_rule_window_score = -1e9
    best_rule_eval_score = -1e9
    solved_rule_windows = 0
    solved_rule_saved = False

    save_path = Path(save_path) if save_path is not None else None

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        last_path = save_path
        latest_checkpoint_path = save_path.with_name("latest_checkpoint_ppo_pong.pth")
        best_mixed_path = save_path.with_name("best_mixed_ppo_pong.pth")
        best_eval_path = save_path.with_name("best_eval_ppo_pong.pth")
        best_rule_window_path = save_path.with_name("best_rule_window_ppo_pong.pth")
        best_rule_survival_path = save_path.with_name("best_rule_survival_ppo_pong.pth")
        solved_rule_path = save_path.with_name("solved_rule_defense_ppo_pong.pth")
    else:
        last_path = None
        latest_checkpoint_path = None
        best_mixed_path = None
        best_eval_path = None
        best_rule_window_path = None
        best_rule_survival_path = None
        solved_rule_path = None

    final_state = None
    final_done = True

    for episode in range(1, n_episodes + 1):
        phase = get_phase(episode, warmup_idle, warmup_random, warmup_rule)
        in_warmup = episode <= warmup_total

        opponent_type = pick_opponent_type(
            episode=episode,
            warmup_idle=warmup_idle,
            warmup_random=warmup_random,
            warmup_rule=warmup_rule,
            mix_selfplay=mix_selfplay,
            mix_random=mix_random,
        )

        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        episode_raw_reward = 0.0
        episode_hits = 0
        episode_misses = 0

        prev_player_action = 0
        done = False

        while not done:
            player_action, player_log_prob, player_value = agent.select_action(state)

            opponent_obs = env.get_obs_opponent()

            if opponent_type == "idle":
                opponent_action = 0
            elif opponent_type == "random":
                opponent_action = random_action(action_dim)
            elif opponent_type == "rule":
                opponent_action = rule_based_action(opponent_obs)
            elif opponent_type == "self_play":
                opponent_action, _, _ = opponent.select_action(opponent_obs)
            else:
                opponent_action = 0

            prev_ball_vx = env.ball_vx

            next_state, raw_reward, terminated, truncated, _ = env.step(
                (player_action, opponent_action)
            )

            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            if not terminated and prev_ball_vx < 0 and env.ball_vx > 0:
                episode_hits += 1

            if terminated and raw_reward < 0:
                episode_misses += 1

            reward = shaped_reward(
                env=env,
                raw_reward=raw_reward,
                prev_ball_vx=prev_ball_vx,
                terminated=terminated,
                hit_reward=hit_reward,
                alignment_coef=alignment_coef,
                score_bonus=score_bonus,
            )

            # Anti-jitter penalty.
            if player_action != prev_player_action:
                reward -= action_switch_penalty

            prev_player_action = player_action

            agent.store_transition(
                state=state,
                action=player_action,
                log_prob=player_log_prob,
                reward=reward,
                done=done,
                value=player_value,
            )

            episode_raw_reward += raw_reward
            state = next_state
            final_state = next_state
            final_done = done

            timesteps_since_update += 1

            if timesteps_since_update >= update_interval:
                last_policy_loss, last_value_loss = agent.update(next_state, done)
                timesteps_since_update = 0

        rewards.append(episode_raw_reward)
        wins.append(1 if episode_raw_reward > 0 else 0)

        window_hits.append(episode_hits)
        window_misses.append(episode_misses)

        if episode in {
            warmup_idle,
            warmup_idle + warmup_random,
            warmup_total,
        }:
            if agent_has_buffer(agent):
                last_policy_loss, last_value_loss = agent.update(state, done)
                timesteps_since_update = 0

            if episode == warmup_idle:
                print("\n" + "=" * 65)
                print(f"Idle phase complete at episode {episode}. Switching to random.")
                print("=" * 65 + "\n")

            elif episode == warmup_idle + warmup_random:
                print("\n" + "=" * 65)
                print(f"Random phase complete at episode {episode}. Switching to rule.")
                print("=" * 65 + "\n")

            elif episode == warmup_total:
                copy_agent_weights(agent, opponent)
                print("\n" + "=" * 65)
                print(f"Rule phase complete at episode {episode}. Switching to mixed.")
                print("=" * 65 + "\n")

        if not in_warmup and (episode - warmup_total) % opponent_sync_every == 0:
            copy_agent_weights(agent, opponent)
            print(f"  [sync] opponent updated at episode {episode}")

        if (
            latest_checkpoint_path is not None
            and episode % checkpoint_every == 0
        ):
            agent.save(str(latest_checkpoint_path))
            print(f"  [save] latest checkpoint → {latest_checkpoint_path.name}")

        if episode % print_every == 0:
            avg_reward = np.mean(rewards[-print_every:])
            win_rate = np.mean(wins[-print_every:]) * 100.0
            avg_hits = np.mean(window_hits[-print_every:])
            avg_misses = np.mean(window_misses[-print_every:])

            print(
                f"Ep {episode:5d} [{phase:<6}] | "
                f"wr {win_rate:5.1f}% | "
                f"avg_r {avg_reward:+6.3f} | "
                f"hits/ep {avg_hits:4.1f} | "
                f"miss/ep {avg_misses:3.1f} | "
                f"π {last_policy_loss:7.4f} | "
                f"V {last_value_loss:7.4f}"
            )

            # Save best rule-phase survival from print windows.
            # This catches strong models immediately, without waiting for eval.
            if phase == "rule":
                rule_window_score = avg_reward + 0.01 * avg_hits - 0.25 * avg_misses

                if rule_window_score > best_rule_window_score:
                    best_rule_window_score = rule_window_score

                    if best_rule_window_path is not None:
                        agent.save(str(best_rule_window_path))
                        print(
                            f"  [save] best rule window → {best_rule_window_path.name} "
                            f"score={best_rule_window_score:+.3f} "
                            f"avg_r={avg_reward:+.3f} hits={avg_hits:.1f}"
                        )

                # Solved defense detector.
                if (
                    avg_reward >= rule_solved_avg_reward
                    and avg_hits >= rule_solved_hits
                    and avg_misses <= rule_solved_misses
                ):
                    solved_rule_windows += 1
                else:
                    solved_rule_windows = 0

                if solved_rule_windows >= rule_solved_patience and not solved_rule_saved:
                    solved_rule_saved = True

                    if solved_rule_path is not None:
                        agent.save(str(solved_rule_path))
                        print(
                            f"  [save] solved rule defense → {solved_rule_path.name} "
                            f"avg_r={avg_reward:+.3f} hits={avg_hits:.1f}"
                        )

                    if early_stop_after_rule_solved:
                        print("\nEarly stop: rule defense solved.")
                        break

            # Save best mixed-phase model by win rate.
            if not in_warmup and win_rate > best_mixed_wr:
                best_mixed_wr = win_rate

                if best_mixed_path is not None:
                    agent.save(str(best_mixed_path))
                    print(
                        f"  [save] best mixed → {best_mixed_path.name} "
                        f"wr={best_mixed_wr:.1f}%"
                    )

        if episode % eval_every == 0:
            idle_eval = evaluate_agent(agent, opponent_type="idle", n_games=50)
            random_eval = evaluate_agent(agent, opponent_type="random", n_games=100)
            rule_eval = evaluate_agent(agent, opponent_type="rule", n_games=100)

            print(f"\n  ── Evaluation at episode {episode} ──────────────────")
            print(fmt_eval("idle", idle_eval))
            print(fmt_eval("random", random_eval))
            print(fmt_eval("rule", rule_eval))
            print()
            print_diagnosis(rule_eval)
            print()

            # Save by rule survival, not only by win rate.
            rule_eval_score = rule_eval["avg_reward"] + 0.01 * rule_eval["hit_rate"]

            if rule_eval_score > best_rule_eval_score:
                best_rule_eval_score = rule_eval_score

                if best_rule_survival_path is not None:
                    agent.save(str(best_rule_survival_path))
                    print(
                        f"  [save] best rule survival → {best_rule_survival_path.name} "
                        f"score={best_rule_eval_score:+.3f} "
                        f"avg_r={rule_eval['avg_reward']:+.3f} "
                        f"hits={rule_eval['hit_rate']:.1f}\n"
                    )

            # Also save if it ever actually wins against rule.
            if rule_eval["win_rate"] > best_rule_wr and rule_eval["win_rate"] > 0:
                best_rule_wr = rule_eval["win_rate"]

                if best_eval_path is not None:
                    agent.save(str(best_eval_path))
                    print(
                        f"  [save] best eval win-rate → {best_eval_path.name} "
                        f"rule_wr={best_rule_wr:.1f}%\n"
                    )

    if agent_has_buffer(agent) and final_state is not None:
        agent.update(final_state, final_done)

    env.close()

    if last_path is not None:
        agent.save(str(last_path))
        print(f"\nLast model → {last_path}")

    if latest_checkpoint_path is not None:
        print(f"Latest checkpoint → {latest_checkpoint_path}")

    if best_rule_window_path is not None:
        print(f"Best rule window → {best_rule_window_path}")

    if best_rule_survival_path is not None:
        print(f"Best rule survival → {best_rule_survival_path}")

    if solved_rule_path is not None:
        print(f"Solved rule defense → {solved_rule_path}")

    if best_mixed_path is not None:
        print(f"Best mixed model → {best_mixed_path}")

    if best_eval_path is not None:
        print(f"Best eval win-rate model → {best_eval_path}")

    return rewards


if __name__ == "__main__":
    save_path = ROOT_DIR / "experiments" / "ppo_pong_selfplay.pth"

    rewards = train_pong_self_play(
        n_episodes=4000,

        hidden_dim=128,

        actor_lr=1e-4,
        critic_lr=5e-4,

        gamma=0.99,
        gae_lambda=0.95,

        clip_epsilon=0.10,
        n_epochs=4,
        batch_size=128,

        entropy_coef=0.02,

        update_interval=1024,
        opponent_sync_every=300,

        warmup_idle=300,
        warmup_random=500,
        warmup_rule=1700,

        mix_selfplay=0.60,
        mix_random=0.20,

        hit_reward=0.20,
        alignment_coef=0.05,
        action_switch_penalty=0.003,
        score_bonus=0.30,

        checkpoint_every=250,

        # True = training stops once rule-defense is solved (avg_r = +0).
        early_stop_after_rule_solved=True,

        rule_solved_avg_reward=-0.02,
        rule_solved_hits=25.0,
        rule_solved_misses=0.05,
        rule_solved_patience=2,

        print_every=50,
        eval_every=250,

        save_path=save_path,
    )

    plot_training_curve(
        rewards,
        title="Curriculum + Mixed PPO on Pong",
        window=50,
    )
