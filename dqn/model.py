import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    """
    Q-Network: Neural network function approximator for Q(s, a; θ)

    Architecture:
        Input -> state vector
        Hidden -> fully connected layers with ReLU activations
        Output -> Q value for every action simultaneously

    Parameters:
        state_dim : int. Dimensionality of the state vector
        action_dim : int. Number of possible actions
        hidden_dim : int. Number of units in each hidden layer
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, state, device):
        # Greedy action selection a* = argmax_a Q(s, a; θ)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.forward(state_tensor)

        return q_values.argmax(dim=1).item()



