import numpy as np

def policy_improvement(env, V):
    """
    Policy Improvement:

    Given  v_π (the value function for current policy π), we construct
    a strictly better policy π' by acting greedily:
            π'(s) = argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]

    Stopping condition:
    If π' == π (no state changed its action), then we have
    reached the Bellman optimality equation, that is, π is optimal.

    Parameters:
        env : GridWorld instance
        V : dict {state: float}

    Returns
        new_policy : dict {state: {action: probability}}
                    The improved greedy policy π'
        policy_stable : bool
                        True -> π' == π, policy did not change -> optimal
                        False -> π' != π, improvement happened
    """

    new_policy = {}
    policy_stable = True

    for s in env.get_all_states():
        #Compute q(s, a) for every action
        action_values = {}

        for action in env.actions:
            q_value = 0.0

            # Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
            for next_state, reward, trans_prob in env.get_transition(s, action):
                q_value += trans_prob * (reward + env.gamma * V[next_state])

            action_values[action] = q_value
        #Greedy Algorithm
        best_action = max(action_values, key=lambda a: action_values[a])
        best_value = action_values[best_action]

        #Check if multiple actions achieved maximum (stochastic policy improvement)
        best_actions = [a for a,v in action_values.items() if abs(v - best_value) < 1e-9]

        prob = 1.0 / len(best_actions)
        new_policy[s] = {a: (prob if a in best_actions else 0.0) for a in env.actions}

    return new_policy, policy_stable

def get_greedy_action(env, V, state):
    """
    Return the single best action at one state given value function V.
        a* = argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]

    Parameters:
        env : GridWorld instance
        V : dict {state: float}
        state : (row, col)

    Returns
    best_action : int (UP=0, DOWN=1, LEFT=2, RIGHT=3)
    """

    action_values = {}
    for action in env.actions:
        q = sum(prob * (r + env.gamma * V[s_]) for s_, r, prob in env.get_transition(state, action))
        action_values[action] = q
    return max(action_values, key = lambda a: action_values[a])

