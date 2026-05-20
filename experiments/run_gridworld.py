"""
Runs all tabular RL methods on GridWorld and visualizes results.
Each section is independent — comment out any you don't want to run.

Methods demonstrated:
    1. Policy Evaluation       — estimate V under random policy
    2. Policy Iteration        — find optimal V and π via alternating eval/improve
    3. Value Iteration         — find optimal V and π via single sweep max
    4. Monte Carlo Control     — ε-greedy MC control
    5. TD(0) Prediction        — estimate V under random policy
    6. SARSA                   — on-policy TD control
    7. Q-Learning              — off-policy TD control
    8. SARSA vs Q-Learning     — side by side learning curve comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from environments.gridworld import GridWorld
from tabular.policy_evaluation import policy_evaluation, make_random_policy
from tabular.policy_iteration import policy_iteration
from tabular.value_iteration import value_iteration
from tabular.monte_carlo import mc_control_epsilon_greedy
from tabular.td_learning import td_prediction
from tabular.sarsa import sarsa
from tabular.q_learning import q_learning



def plot_value_function(env, V, title="State-Value Function V(s)"):
    grid = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            grid[r, c] = V.get((r, c), 0.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")
    plt.colorbar(im, ax=ax, label="V(s)")

    for r in range(env.rows):
        for c in range(env.cols):
            ax.text(c, r, f"{grid[r,c]:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="black")

    for (r, c), label in [(env.state_A, "A"), (env.state_B, "B"),
                           (env.state_A_, "A'"), (env.state_B_, "B'")]:
        ax.text(c, r - 0.35, label, ha="center", va="center",
                fontsize=8, color="navy")

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_policy(env, policy, title="Policy π"):
    arrow_map = {
        env.UP:    (0, -0.3),
        env.DOWN:  (0,  0.3),
        env.LEFT:  (-0.3, 0),
        env.RIGHT: (0.3,  0)
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_aspect("equal")

    for r in range(env.rows):
        for c in range(env.cols):
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                      linewidth=1, edgecolor="gray", facecolor="white")
            ax.add_patch(rect)

    for s, action_probs in policy.items():
        r, c = s
        best_action = max(action_probs, key=lambda a: action_probs[a])
        dx, dy = arrow_map[best_action]
        ax.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=2))

    for (r, c), label in [(env.state_A, "A"), (env.state_B, "B"),
                           (env.state_A_, "A'"), (env.state_B_, "B'")]:
        ax.text(c, r, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="darkred")

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(history, title="Learning Curve", label="reward", color="steelblue", window=20):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, alpha=0.3, color=color, label="raw")

    if len(history) >= window:
        smoothed = np.convolve(history, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(history)), smoothed,
                color=color, linewidth=2, label=f"{label} (avg {window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison(history1, history2, label1="SARSA", label2="Q-Learning",
                    title="SARSA vs Q-Learning", window=20):
    fig, ax = plt.subplots(figsize=(8, 4))

    for history, label, color in [(history1, label1, "steelblue"),
                                   (history2, label2, "tomato")]:
        ax.plot(history, alpha=0.2, color=color)
        if len(history) >= window:
            smoothed = np.convolve(history, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(history)), smoothed,
                    color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()


def run_policy_evaluation():
    print("\n" + "="*60)
    print("1. Policy Evaluation — Random Policy")
    print("="*60)
    env = GridWorld()
    policy = make_random_policy(env)
    V, history = policy_evaluation(env, policy, theta=1e-6)

    print(f"Converged in {len(history)-1} sweeps")
    print(f"V(state_A)  = {V[env.state_A]:.2f}  (expect ~8.8)")
    print(f"V(state_B)  = {V[env.state_B]:.2f}  (expect ~5.3)")

    plot_value_function(env, V, title="Policy Evaluation — Random Policy V(s)")


def run_policy_iteration():
    print("\n" + "="*60)
    print("2. Policy Iteration")
    print("="*60)
    env = GridWorld()
    V, policy, history = policy_iteration(env, theta=1e-6)

    print(f"Converged in {len(history)} iterations")
    print(f"V(state_A)  = {V[env.state_A]:.2f}")
    print(f"V(state_B)  = {V[env.state_B]:.2f}")

    plot_value_function(env, V, title="Policy Iteration — Optimal V*(s)")
    plot_policy(env, policy, title="Policy Iteration — Optimal Policy π*")


def run_value_iteration():
    print("\n" + "="*60)
    print("3. Value Iteration")
    print("="*60)
    env = GridWorld()
    V, policy, history = value_iteration(env, theta=1e-6)

    print(f"Converged in {len(history)-1} sweeps")
    print(f"V(state_A)  = {V[env.state_A]:.2f}")
    print(f"V(state_B)  = {V[env.state_B]:.2f}")

    plot_value_function(env, V, title="Value Iteration — Optimal V*(s)")
    plot_policy(env, policy, title="Value Iteration — Optimal Policy π*")


def run_monte_carlo():
    print("\n" + "="*60)
    print("4. Monte Carlo Control — ε-greedy")
    print("="*60)
    env = GridWorld()
    Q, policy, _ = mc_control_epsilon_greedy(env, n_episodes=10000, gamma=0.9, epsilon=0.1)

    V = {s: max(Q.get((s, a), 0.0) for a in env.actions) for s in env.get_all_states()}

    print(f"V(state_A)  = {V[env.state_A]:.2f}")
    print(f"V(state_B)  = {V[env.state_B]:.2f}")

    plot_value_function(env, V, title="MC Control — V(s) = max_a Q(s,a)")
    plot_policy(env, policy, title="MC Control — Learned Policy")


def run_td_prediction():
    print("\n" + "="*60)
    print("5. TD(0) Prediction — Random Policy")
    print("="*60)
    env = GridWorld()
    policy = make_random_policy(env)
    V, history = td_prediction(env, policy, n_episodes=2000, alpha=0.1, gamma=0.9)

    print(f"Ran {len(history)-1} episodes")
    print(f"V(state_A)  = {V[env.state_A]:.2f}")
    print(f"V(state_B)  = {V[env.state_B]:.2f}")

    plot_value_function(env, V, title="TD(0) Prediction — Random Policy V(s)")


def run_sarsa():
    print("\n" + "="*60)
    print("6. SARSA — On-Policy TD Control")
    print("="*60)
    env = GridWorld()
    Q, policy, history = sarsa(env, n_episodes=3000, alpha=0.5, gamma=0.9, epsilon=0.1)

    V = {s: max(Q.get((s, a), 0.0) for a in env.actions) for s in env.get_all_states()}

    print(f"V(state_A)  = {V[env.state_A]:.2f}")
    print(f"V(state_B)  = {V[env.state_B]:.2f}")
    print(f"Avg reward last 100 episodes: {np.mean(history[-100:]):.2f}")

    plot_value_function(env, V, title="SARSA — V(s) = max_a Q(s,a)")
    plot_policy(env, policy, title="SARSA — Learned Policy")
    plot_learning_curve(history, title="SARSA — Learning Curve", label="SARSA", color="steelblue")


def run_q_learning():
    print("\n" + "="*60)
    print("7. Q-Learning — Off-Policy TD Control")
    print("="*60)
    env = GridWorld()
    Q, policy, history = q_learning(env, n_episodes=3000, alpha=0.5, gamma=0.9, epsilon=0.1)

    V = {s: max(Q.get((s, a), 0.0) for a in env.actions) for s in env.get_all_states()}

    print(f"V(state_A)  = {V[env.state_A]:.2f}")
    print(f"V(state_B)  = {V[env.state_B]:.2f}")
    print(f"Avg reward last 100 episodes: {np.mean(history[-100:]):.2f}")

    plot_value_function(env, V, title="Q-Learning — V(s) = max_a Q(s,a)")
    plot_policy(env, policy, title="Q-Learning — Learned Policy")
    plot_learning_curve(history, title="Q-Learning — Learning Curve", label="Q-Learning", color="tomato")


def run_sarsa_vs_qlearning():
    print("\n" + "="*60)
    print("8. SARSA vs Q-Learning — Comparison")
    print("="*60)
    env1 = GridWorld()
    env2 = GridWorld()

    _, _, history_sarsa = sarsa(env1, n_episodes=3000, alpha=0.5, gamma=0.9, epsilon=0.1)
    _, _, history_ql    = q_learning(env2, n_episodes=3000, alpha=0.5, gamma=0.9, epsilon=0.1)

    print(f"SARSA      avg last 100: {np.mean(history_sarsa[-100:]):.2f}")
    print(f"Q-Learning avg last 100: {np.mean(history_ql[-100:]):.2f}")

    plot_comparison(history_sarsa, history_ql,
                    label1="SARSA", label2="Q-Learning",
                    title="SARSA vs Q-Learning — GridWorld")



if __name__ == "__main__":
    run_policy_evaluation()
    run_policy_iteration()
    run_value_iteration()
    run_monte_carlo()
    run_td_prediction()
    run_sarsa()
    run_q_learning()
    run_sarsa_vs_qlearning()
