def value_iteration(env, theta=1e-6, max_iters = 10000):

    """
    Value iteration:

    Instead of waiting for full policy evaluation convergence, we combine
    policy evaluation and policy improvement into single sweep.
    At each state we immediately take the max over all actions:
             V_{k+1}(s) = max_a Σ_{s',r} p(s',r|s,a) [r + γ V_k(s')]

    Parameters:
        env : GridWorld instance
        theta : float. Stopping threshold
        max_iters : int. Maximum numbers of iteration in loop.

    Returns:
        V : dict { state: float}
            the converged optimal value function V ≈ v*.
        policy : dict {state: {action: probability}}
                optimal greedy policy π*
        history : list of dicts
    """

    #Initialize
    V = {s: 0.0 for s in env.get_all_states()}
    history = [dict(V)]

    for k in range(max_iters):
        delta = 0.0

        for s in env.get_all_states():
            v_old = V[s]

            #Compute q(s, a) for every action and take the max
            action_values = []
            for action in env.actions:
                q_value = 0.0

                # Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
                for next_state, reward, trans_prob in env.get_transition(s, action):
                    q_value += trans_prob * (reward + env.gamma * V[next_state])

                action_values.append(q_value)

            V[s] = max(action_values)
            delta = max(delta, abs(v_old - V[s]))
        history.append(dict(V))

        if delta < theta:
            print(f"Value iteration converged after {k+1} sweeps with delta = {delta:.2e}")
            break
    else:
        print(f"Reached maximum numbers of iterations = {max_iters} without convergence. Final delta = {delta:.2e}")

    policy = extract_policy(env, V)
    return V, policy, history

def extract_policy(env, V):
    """
    Extract deterministic greedy policy from optimal value function V*.
            π*(s) = argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
    If multiple actions tie for the maximum, probability is split equally.

    Parameters:
        env : GridWorld isntance
        V : dict {state: float} - converged value function

    Returns:
        policy: dict { state: {action: probability}}
    """

    policy = {}

    for s in env.get_all_states():
        action_values = {}

        for action in env.actions:
            q_value = 0.0

            for next_state, reward, trans_prob in env.get_transition(s, action):
                q_value += trans_prob * (reward + env.gamma * V[next_state])

            action_values[action] = q_value

        best_value = max(action_values.values())
        best_actions = [a for a, v in action_values.items() if abs(v - best_value) < 1e-9]

        prob = 1.0 / len(best_actions)
        policy[s] = {a: (prob if a in best_actions else 0.0) for a in env.actions}

    return policy
