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


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from environments.gridworld import GridWorld

    env = GridWorld(rows=5, cols=5, gamma=0.9)

    # ── Run policy iteration ──────────────────────────────────────────────────
    V_star, pi_star, history = policy_iteration(env, theta=1e-6)

    # ── Print optimal value function ──────────────────────────────────────────
    print("\n--- Optimal Value Function V* ---")
    for r in range(env.rows):
        row = "  ".join(f"{V_star[(r, c)]:6.2f}" for c in range(env.cols))
        print("  " + row)

    # ── Print optimal policy as arrows ────────────────────────────────────────
    arrow = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    print("\n--- Optimal Policy π* ---")
    for r in range(env.rows):
        row_str = "  "
        for c in range(env.cols):
            s = (r, c)
            best_actions = [a for a, p in pi_star[s].items() if p > 0]
            if len(best_actions) == 1:
                row_str += f" {arrow[best_actions[0]]} "
            else:
                row_str += " * "
        print(row_str)

    # ── Spot-check against Figure 3.8b ───────────────────────────────────────
    print("\nSpot-check against Figure 3.8b:")
    print(f"  V* at A  (0,1) = {V_star[(0, 1)]:.2f}  (book: ~22.0)")
    print(f"  V* at B  (0,3) = {V_star[(0, 3)]:.2f}  (book: ~24.4)")
    print(f"  V* at    (0,0) = {V_star[(0, 0)]:.2f}  (book: ~22.0)")
    print(f"  V* at    (4,4) = {V_star[(4, 4)]:.2f}  (book: ~10.9)")

    # ── Show convergence speed per iteration ─────────────────────────────────
    print("\nConvergence summary:")
    for h in history:
        print(f"  Iteration {h['iteration']}: "
              f"{h['n_sweeps']} evaluation sweeps, "
              f"policy stable = {h['iteration'] == len(history)}")

    # ── Plot optimal value function ───────────────────────────────────────────
    env.plot_value_function(V_star, title="V* — Optimal Value Function (Policy Iteration)")

    # ── Plot optimal policy ───────────────────────────────────────────────────
    deterministic = {
        s: get_greedy_action(env, V_star, s)
        for s in env.get_all_states()
    }
    env.plot_policy(deterministic, title="π* — Optimal Policy (Policy Iteration)")