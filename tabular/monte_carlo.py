import numpy as np
from collections import defaultdict

def mc_prediction_first_visit(env, policy, n_episodes=10000, gamma=0.9):
    """
    First-Visit Monte Carlo Predictions:

    Estimate v_π(s) by averaging returns observed after the first visit to each state s across many episodes

    V(s) <- average all G's observed after first visit to s

    Parameters:
        env : GridWorld instance
        policy : dict {state: {action: probability}}
                     π(a|s) = probability of taking action a in state s
        n_episodes : int. Number of episodes to generate
        gamma : float. Discount rate

    Returns:
        V : dict {state:  float}
            Estimated state-value function V ≈ v_π
        returns : dict {state: list of floats}
                    All returns observed after first visits per state
    """

    V = {s: 0.0 for s in env.get_all_states()}
    returns_count = defaultdict(int)

    for episode_i in range(n_episodes):
        #Generate one full episode following π
        episode = generate_episode(env, policy)

        #Compute returns and update V for first visits only
        visited_states = set()
        G = 0.0

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G

            if s not in visited_states:
                visited_states.add(s)
                returns_count[s] += 1
                V[s] += (G - V[s]) / returns_count[s]
    return V, dict(returns_count)

def mc_control_exploring_starts(env, n_episodes=50000, gamma=0.9):
    """
    Monte Carlo control with exploring starts

    Parameters:
        env: GridWorld instance
        n_episodes : int. Number of episodes to generate
        gamma : float. Discount rate

    Returns:
        Q : dict {(state, action): float}
            Estimated optimal action-value function Q ≈ q*
        policy : dict {state: int}
                optimal deterministic policy π*(s) = argmax_a Q(s,a)
        returns: dict {(state, action)}: list of floats}
                All returns observed per state-action pair
    """

    Q = defaultdict(float)
    returns_count = defaultdict(int)

    policy = {s: env.actions[0] for s in env.get_all_states()}

    for episode_i in range(n_episodes):

        #Exploring starts: pick random (s0, a0) to start each episode
        all_states = env.get_all_states()
        s0 = all_states[np.random.randint(len(all_states))]
        a0 = env.actions[np.random.randint(len(env.actions))]

        #Generate episode starting from (s0, a0)
        episode = generate_episode_es(env, policy, s0, a0)

        visited_pairs = set()
        G = 0.0

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma*G

            if (s, a) not in visited_pairs:
                visited_pairs.add((s, a))
                returns_count[(s, a)] += 1
                Q[(s, a)] += (G - Q[(s, a)]) / returns_count[(s, a)]

                # Greedy policy improvement immediately after Q update
                # π(s) = argmax_a Q(s, a)
                policy[s] = max(env.actions, key = lambda act: Q[(s, act)])
    return dict(Q), policy, dict(returns_count)

def mc_control_epsilon_greedy(env, n_episodes=50000, gamma=0.9, epsilon=0.1):
    """
    On-policy first-visit mc control for ε-soft policies:

    ε-greedy action selection:
    π(a*|s) = 1 - ε + ε/|A(s)|     (greedy action)
    π(a |s) = ε/|A(s)|              (all other actions)

    Parameters:
        env : GridWorld instance
        n_episodes : int. Number of episodes to generate
        gamma : float. Discount rate
        epsilon : float. epsilon is in interval (0, 1]. Controls exploration vs exploitation
    """

    Q = defaultdict(float)
    returns_count = defaultdict(int)

    n_actions = len(env.actions)

    policy = {s: {a: 1.0/n_actions for a in env.actions} for s in env.get_all_states()}

    for episode_i in range(n_episodes):
        #Generate episode using current ε-soft policy
        episode = generate_episode(env, policy)

        #compute returns and update for first visits
        visited_pairs = set()
        G = 0.0

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            if (s, a) not in visited_pairs:
                visited_pairs.add((s, a))
                returns_count[(s, a)] += 1
                Q[(s, a)] += (G - Q[(s, a)]) / returns_count[(s, a)]

                #ε-greedy policy update for state s
                # a* = argmax_a Q(s,a)
                a_star = max(env.actions, key=lambda act: Q[(s, act)])

                for act in env.actions:
                    if act == a_star:
                        policy[s][act] = 1.0 - epsilon + epsilon/n_actions
                    else:
                        policy[s][act] = epsilon/n_actions
    return dict(Q), policy, dict(returns_count)

def generate_episode(env, policy):
    """
    Generate one full episode by following stochastic probability π

    Returns:
        episode : list of (state, action, reward) tuples.
                i.e. [(s0,a0,r1), (s1,a1,r2), ..., (sT-1,aT-1,rT)]
    """
    episode = []
    state = env.reset()
    max_steps = 200

    for _ in range(max_steps):
        action_probs = policy[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = actions[np.random.choice(len(actions), p=probs)]

        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))

        if done:
            break
        state = next_state
    return episode


def generate_episode_es(env, policy, s0, a0):
    """
    Generate one full episode with exploring starts.
    First step is forced: start at s0 and take a0.

    Returns:
        episode : list of (state, action, reward) tuples
    """

    episode =[]
    env.reset(start_pos = s0)

    next_state, reward, done = env.step(a0)
    episode.append((s0, a0, reward))

    if done:
        return episode
    state = next_state
    max_steps = 1000

    for _ in range(max_steps):
        action = policy[state]
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))

        if done:
            break
        state = next_state
    return episode
