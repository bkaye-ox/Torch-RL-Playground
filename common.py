import torch.optim
import torch
import torch.nn as nn
import math

class FastGELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Policy():    
    opt: torch.optim.Optimizer
    N: int
    M: int

    def pi(self, state):
        raise NotImplementedError

    def act(self, state):
        return self.pi(state).sample()

    def learn(self, states, actions, rewards):
        raise NotImplementedError

class Mlp(nn.Module):
    def __init__(self, input_dim: int, output_n: int, hidden_dim: int, n_layers: int, act=FastGELU) -> None:
        super().__init__()

        act = act()
        layers = [nn.Linear(input_dim, hidden_dim), act] + [nn.Linear(
            hidden_dim, hidden_dim), act]*n_layers + [nn.Linear(hidden_dim, output_n)]
        self.model = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.model(x)

