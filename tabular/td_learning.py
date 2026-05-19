import numpy as np

def td_prediction(env, policy, n_episodes=500, alpha=0.1, gamma=0.9):
    """
    TD(0) Prediction:

    Estimate v_π(s) by updating after every single step, not after full episode.
            TD target = R_{t+1} + γV(S_{t+1})
    Update rule:
        V(S) ← V(S) + α[R + γV(S') - V(S)]

    Parameters:
        env : GridWorld instance
        policy : dict {state: {action: probability}}
        n_episodes : int. Number of episodes to run
        alpha : float. Step-size parameter (Learning rate)
        gamma : float. Discount rate
    Returns:
        V : dict {state: float}
            Estimated state-value function
        history : list of dicts
    """

    V = {s: 0.0 for s in env.get_all_states()}
    history = [dict(V)]

    for episode_i in range(n_episodes):
        state = env.reset()
        max_steps = 1000
        for _ in range(max_steps):
            #Sample action from policy
            action_probs = policy[state]
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action = actions[np.random.choice(len(actions), p=probs)]

            #Take action
            next_state, reward, done = env.step(action)

            #TD(0) update rule
            td_error = reward + gamma * V[next_state] - V[state]
            V[state] = V[state] + alpha * td_error

            if done:
                break
            state =next_state
        history.append(dict(V))
    return V, history

