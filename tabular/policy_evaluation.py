import numpy as np


def policy_evaluation(env, policy, theta=1e-6, max_iters = 10000):
    """
    Iterative Policy Evaluation:

    Given fixed policy π we want to compute v_π(s) for every state s.
    v_π(s) is the expected return starting from s and always following π

    We can not solve Bellman equation analytically instead we iterate
            V_{k+1}(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V_k(s')]


    Parameters:
        env : GridWorld instance
        policy : dict {state: {action: probability}}
                     π(a|s) = probability of taking action a in state s.
                     Example — equiprobable random policy on 5×5 grid:
                     {
                       (0,0): {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
                       (0,1): {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
                       ...
                     }

        theta : float. Stopping threshold. Algorithm stops when
                             max_s |V_{k+1}(s) - V_k(s)| < θ.
        max_iters : int.
                    Safety cap on sweeps. Prevents infinite loops.
    Returns:
        V : dict {state: float}
            The converged value function V ≈ v_π.
            V[(r,c)] = expected return starting from cell (r,c)
        history : list of dicts
                V snapshot after every sweep
    """

    #Initialize V(s) = 0 for all s
    V = {s: 0.0 for s in env.get_all_states()}
    history = [dict(V)]

    #Sweep loop
    for k in range(max_iters):
        delta = 0.0

        #Outer sum Σ_a π(a|s) × [...]
        for s in env.get_all_states():
            v_old = V[s]
            new_value = 0.0

            # Inner sum Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
            for action, action_prob in policy[s].items():
                if action_prob == 0.0:
                    continue

                for next_state, reward, trans_prob in env.get_transition(s, action):
                    bellman_target = reward + env.gamma * V[next_state] # [r + γ V(s')]
                    new_value += action_prob * trans_prob * bellman_target # π(a|s) × p(s',r|s,a)

            V[s] = new_value #In-place update
            delta = max(delta, abs(v_old - V[s]))
        history.append(dict(V))

        if delta < theta:
            print(f"Policy evaluation converged after {k+1} sweeps with value = {delta:.2e}")
            break
    else:
        print(f"Reached max_iters = {max_iters} without convergence. Final value = {delta:.2e}")

    return V, history

#Helper make_random_policy. Builds equiprobable random policy π(a|s) = 1/|A| for all s, a

def make_random_policy(env):
    """
    Build equiprobable random policy
    π(a|s) = 1/|A| for all states s and actions a
    Every action is equally likely regardless of state

    Returns:
         policy : dict {state: {action: probability}}
    """

    n = len(env.actions)
    prob = 1.0 / n
    return {s: {a: prob for a in env.actions} for s in env.get_all_states()}
