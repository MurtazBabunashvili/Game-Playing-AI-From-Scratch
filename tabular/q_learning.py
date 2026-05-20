from collections import defaultdict
from tabular.utils import epsilon_greedy, extract_policy_from_q


def q_learning(env, n_episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    """
    Q-Learning: Off-Policy TD Control

    Directly approximates q* and update formula is following:
            Q(S, A) ← Q(S, A) + α[R + γ max_a Q(S', a) - Q(S, A)]

    Parameters:
        env : GridWorld instance
        n_episodes : int. Number of episodes to run
        alpha : float. Step-size learning rate parameter
        gamma : float. Discount rate
        epsilon : float. For epsilon greedy policy

    Returns:
        Q : dict {(State, action): float}
            estimated optimal action-value function
        policy : dict {state: {action: probability}}
            Final greedy policy derived from Q
        history : list of floats
    """

    Q = defaultdict(float)
    history = []

    for episode_i in range(n_episodes):
        state = env.reset()

        episode_reward = 0.0
        max_steps = 1000

        for _ in range(max_steps):
            #Choose A from S using epsilon greedy behavior
            action = epsilon_greedy(env, Q, state, epsilon)

            #Take action A, observe R and S'
            next_state, reward, done = env.step(action)
            episode_reward += reward

            #Q-Learning main formula:
            best_next = max(Q[(next_state, a)] for a in env.actions)
            td_error = reward + gamma * best_next - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state

            if done:
                break
        history.append(episode_reward)
    policy = extract_policy_from_q(env, Q, epsilon=0.0)
    return dict(Q), policy, history