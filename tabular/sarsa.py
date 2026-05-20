from collections import defaultdict
from tabular.utils import epsilon_greedy, extract_policy_from_q

def sarsa(env, n_episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    """
    SARSA learns optimal action-value function Q ≈ q*.
    Both the behavior policy and the target policy are the same ε-greedy policy derived from Q.

    Update rule:
        Q(S, A) ← Q(S, A) + α[R + γQ(S', A') - Q(S, A)]

    Parameters:
        env : GridWorld instance
        n_episodes : int. Number of episodes to run
        alpha : float. Step-size parameter (learning rate)
        gamma : float. Discount rate
        epsilon : float. ε  for ε-greedy action selection

    Returns:
        Q : dict {(state, action): float}
            estimated optimal action-value function
        policy : dict {state: {action: probability}}
            final ε-greedy policy
        history: list of floats
    """

    Q = defaultdict(float)
    history = []

    for episode_i in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(env, Q, state, epsilon)

        episode_reward = 0.0
        max_steps = 1000

        for _ in range(max_steps):
            #Take action A, observe R and S'
            next_state, reward, done = env.step(action)
            episode_reward += reward

            #Choose A' from S' using same ε-greedy policy
            next_action = epsilon_greedy(env, Q, next_state, epsilon)

            #SARSA update:
            td_error = reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state
            action = next_action

            if done:
                break
        history.append(episode_reward)

    policy = extract_policy_from_q(env, Q, epsilon)
    return dict(Q), policy, history