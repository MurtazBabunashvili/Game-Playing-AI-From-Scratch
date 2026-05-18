from tabular.policy_evaluation import policy_evaluation, make_random_policy
from tabular.policy_improvement import policy_improvement, get_greedy_action

def policy_iteration(env, theta=1e-6, max_iters=1000):
    """
    Policy iteration alternates between two steps until convergence:
            π_0 →[E]→ v_π0 →[I]→ π_1 →[E]→ v_π1 →[I]→ π_2 → ... → π* → v*

    E = Policy evaluation
    I = Policy improvement

    Stop when policy improvement produces no change at any state

    Parameters:
        env : GridWorld instance
        theta : float - stopping threshold for policy evaluation
        max_iters : int - number of maximum iterations for loop

    Returns:
        V : dict {state: float} - Optimal value function v*
        policy : dict {state: {action: probability}} - optimal policy π*
        history : list of dicts, each containing:
                 {
                   'iteration': int,
                   'V':         dict {state: float},
                   'policy':    dict {state: {action: probability}},
                   'n_sweeps':  int  (how many evaluation sweeps this iteration)
                 }
    """

    #Initialization
    policy = make_random_policy(env)
    V = {s: 0.0 for s in env.get_all_states()}
    history = []

    for iteration in range(max_iters):
        # Policy Evaluation
        V, V_history = policy_evaluation(env, policy, theta=theta)
        n_sweeps = len(V_history) - 1 #How many sweeps needed

        #Policy Improvement
        old_policy = {s: dict(a_probs) for s, a_probs in policy.items()}
        new_policy, _ = policy_improvement(env, V)

        #policy stable checks if policy changed
        policy_stable = True
        n_changed = 0

        for s in env.get_all_states():
            old_best = max(old_policy[s], key=lambda a: old_policy[s][a])
            new_best = max(new_policy[s], key=lambda a: new_policy[s][a])
            if old_best != new_best:
                policy_stable = False
                n_changed += 1
        policy = new_policy
        history.append({
            'iteration': iteration + 1,
            'V': dict(V),
            'policy': {s: dict(a_probs) for s, a_probs in policy.items()},
            'n_sweeps': n_sweeps
        })

        if policy_stable:
            print(f"Policy iteration converged after {iteration + 1} iterations.")
            break
    else:
        print(f"Reached max_iters={max_iters}")
    return V, policy, history

