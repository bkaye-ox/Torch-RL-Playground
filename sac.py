import torch
import torch.nn as nn
from torch import Tensor
import math
import gymnasium as gym
import copy

import collections as col
import numpy as np

from dataclasses import dataclass

from common import Policy, FastGELU, Mlp


class LunarActor(Policy):
    def __init__(self, input_dim: int, output_n: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.model = Mlp(input_dim, output_n, hidden_dim, n_layers)

        self.N = input_dim
        self.M = output_n
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        self.transform = torch.distributions.transforms.TanhTransform(dim=-1)

    def pi(self, state):
        # res = self.model(torch.tensor(state))

        res = self.model(torch.tensor(state))
        mu, logvar = res.chunk(2, dim=-1)
        scale = torch.exp(0.5*logvar)
        gauss = torch.distributions.Normal(mu, scale)

        dist = torch.distributions.TransformedDistribution(
            gauss, self.transform) if self.transform is not None else gauss

        # dist = torch.distributions.Categorical(logits=res)
        return dist


@dataclass
class Experience:
    state: Tensor
    action: Tensor
    reward: Tensor
    done: Tensor
    next_state: Tensor

class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int):
        """
        Args:
            capacity: maximum number of experiences that can be stored in the buffer
        """
        self.buffer = col.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a batch of experiences from the buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )

        return (
            torch.tensor(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(dones, dtype=torch.bool),
            torch.tensor(next_states),
        )


class Sac(nn.Module):
    Q: list[nn.Module]
    target_Q: list[nn.Module]
    replay: ReplayBuffer
    actor: Policy
    updates: int
    batch_size: int

    policy_optim: torch.optim.Optimizer
    q_optim: list[torch.optim.Optimizer]

    discount: float
    entropy_gain: float
    target_momentum: float

    def __init__(self, obs_dim, act_dim) -> None:
        super().__init__()

        self.Q = [Mlp(input_dim=obs_dim, output_n=1,
                      hidden_dim=256, n_layers=2) for _ in range(2)]
        self.target_Q = [copy.deepcopy(q) for q in self.Q]
        self.actor = LunarActor(
            input_dim=obs_dim, output_n=act_dim, hidden_dim=256, n_layers=2)
        self.configure_optimizers()
        self.replay = ReplayBuffer(int(1e4))

    def training_step(self):
        experience_batch = self.replay.sample(self.batch_size)

        for _ in range(self.updates):
            self.update_Q(experience_batch)
            self.update_policy(experience_batch)
            self.update_Q_targets()

    def configure_optimizers(self):
        self.policy_optim = torch.optim.Adam(self.actor.parameters())
        self.q_optim = [torch.optim.Adam(q.parameters())
                        for q in self.Q]

    def update_Q(self, experience_batch):
        states, actions, rewards, dones, next_states = experience_batch

        with torch.no_grad():
            next_actions = self.actor.act(next_states)

            Q_min = torch.min(*[q(next_states, next_actions)
                              for q in self.target_Q])

            next_action_logprob = self.actor.pi(states).log_prob(next_actions)

            targets = rewards + self.discount * (1 - dones) * (Q_min - self.entropy_gain *
                                                               next_action_logprob)

        for q in self.Q:
            self.policy_optim.zero_grad()
            Q_loss = torch.mean(torch.pow(q(states, actions) - targets, 2))
            Q_loss.backward()
            self.policy_optim.step()

    def update_policy(self, experience_batch):
        states = experience_batch[0]

        with torch.no_grad():
            actions = self.actor.act(states)
            action_logprob = self.actor.pi(states).log_prob(actions)
            Q_min = torch.min(*[q(states, actions) for q in self.Q])

        self.policy_optim.zero_grad()
        policy_loss = torch.mean(self.entropy_gain * action_logprob - Q_min)
        policy_loss.backward()
        self.policy_optim.step()

    def update_Q_targets(self):
        with torch.no_grad():
            self.target_Q = [self.target_momentum * q_target +
                             (1-self.target_momentum)*q for q, q_target in zip(self.Q, self.target_Q)]


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(0)
    torch.manual_seed(0)

    sac = Sac()

    for i in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = sac.actor.act(state)
            next_state, reward, done, _ = env.step(action)
            sac.replay.append(Experience(
                state, action, reward, done, next_state))
            state = next_state

        sac.training_step()

    env.close()
