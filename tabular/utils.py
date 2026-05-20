import numpy as np

def epsilon_greedy(env, Q, state, epsilon):
    """
    ε-greedy action selection:
        with probability ε pick random action
        with probability 1 - ε pick argmax_a Q(s, a)

    Parameters:
        env : GridWorld instance
        Q : dict {(state, action): float}
        state : current state (row, col)
        epsilon : float. Exploration rate

    Returns:
        action : int
    """

    if np.random.random() < epsilon:
        return env.actions[np.random.randint(len(env.actions))]
    #Greedy pick with highest Q value
    q_values = [Q[(state, a)] for a in env.actions]
    best_value = max(q_values)
    best_actions = [a for a, q in zip(env.actions, q_values) if q == best_value]
    return best_actions[np.random.randint(len(best_actions))]


def extract_policy_from_q(env, Q, epsilon=0.0):
    """
    Extract ε-greedy policy from Q table

    Parameters:
         env : GridWorld instance
         Q : dict {(state, action): float}
         epsilon : float. If 0.0 returns greedy policy, else ε-greedy

    Returns:
        policy : dict {state: probability}}
    """

    policy = {}
    n_actions = len(env.actions)

    for s in env.get_all_states():
        q_values = {a: Q[(s, a)] for a in env.actions}
        best_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == best_value]

        policy[s] = {}
        for a in env.actions:
            if a in best_actions:
                policy[s][a] = (1.0 - epsilon)/len(best_actions) + epsilon / n_actions
            else:
                policy[s][a] = epsilon / n_actions
    return policy