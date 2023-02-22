import pytorch_lightning.loggers.tensorboard as tb_logger
import torch
import torch.nn as nn
# import torch.nn.functional as F
import math
import gym
import copy
import numpy as np
import gymnasium as gym


class Policy():
    opt: torch.optim.Optimizer
    N: int
    M: int

    def pi(self, state):
        raise NotImplementedError

    def act(self, state):
        return self.pi(state).sample()

    def learn(self, states, actions, rewards):
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        log_probs = self.pi(states).log_prob(actions.to(torch.int))
        loss = torch.mean(-log_probs.squeeze() * rewards)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class FastGELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ContPolicy(Policy):
    def __init__(self, input_dim: int, output_n: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        act = FastGELU()

        layers = [nn.Linear(input_dim, hidden_dim), act] + [nn.Linear(
            hidden_dim, hidden_dim), act]*n_layers + [nn.Linear(hidden_dim, 2*output_n)]
        self.model = nn.Sequential(
            *layers
        )

        self.N = input_dim
        self.M = output_n
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def pi(self, state):
        res = self.model(torch.tensor(state))
        mu, logvar = res.chunk(2, dim=-1)
        scale = torch.exp(0.5*logvar)
        dist = torch.distributions.Normal(mu, scale)
        return dist


class DiscretePolicy(Policy):
    def __init__(self, input_dim: int, output_n: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        act = FastGELU()

        self.model = nn.Linear(input_dim, output_n)

        self.N = input_dim
        self.M = output_n
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-1)

    def pi(self, state):
        logits = self.model(torch.tensor(state))
        logits = logits - torch.maximum(logits[...,0],logits[...,1]).unsqueeze(-1)

        dist = torch.distributions.Categorical(logits=logits)
        return dist


class GymEnvWrapper(gym.Wrapper):
    def __init__(self, env, num_envs=1):
        super().__init__(env)
        self.num_envs = num_envs
        self.envs = [copy.deepcopy(env) for _ in range(num_envs)]

    def reset(self):
        return np.asarray([env.reset()[0] for env in self.envs], dtype=np.float32)

    def step(self, actions):
        next_states, rewards, terminates = [], [], []
        for env, action in zip(self.envs, actions):
            observation, reward, terminated, truncated, info = env.step(action)
            finished = terminated or truncated

            reward = reward if not finished else 0

            next_states.append(observation)
            rewards.append(reward)
            terminates.append(finished)
        return np.asarray(next_states, dtype=np.float32,), np.asarray(rewards, dtype=np.float32,), np.asarray(terminates, dtype=np.float32,)


def calculate_returns(rewards, dones, gamma):
    result = np.empty_like(rewards)
    result[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1):
        result[t] = rewards[t] + gamma*(1-dones[t])*result[t+1]
    return result


def REINFORCE(env, policy, num_epochs=50, max_T=1000, gamma=0.99, torch_if=True, reward_augment: callable = None):
    states = np.empty((max_T, env.num_envs,  agent.N), dtype=np.float32)
    actions = np.empty((max_T, env.num_envs, agent.M), dtype=np.float32) if not isinstance(env.action_space, gym.spaces.Discrete) else np.empty(
        (max_T, env.num_envs), dtype=np.float32)
    rewards = np.empty((max_T, env.num_envs), dtype=np.float32)
    dones = np.empty((max_T, env.num_envs), dtype=np.float32)

    # totals = []

    for epoch in range(num_epochs):
        s_t = env.reset()

        B = s_t.shape[0]
        for t in range(max_T):
            a_t = agent.act(s_t)

            if not torch_if:
                a_t = a_t.detach().numpy()

            s_tp1, r_t, d_t = env.step(a_t)

            if reward_augment:
                r_t = reward_augment(s_tp1, r_t)

            if all(d_t):
                break

            states[t] = s_t
            actions[t] = a_t
            rewards[t] = r_t
            dones[t] = d_t

            s_t = s_tp1
        returns = calculate_returns(rewards, dones, gamma)
        agent.learn(states, actions, returns)

        log = logger.experiment
        log.add_scalar('returns', np.sum(rewards)/B, epoch)

    return agent
#


def pole_augment(obs, reward):
    # return reward + np.exp(-np.abs(obs[:,0]))
    # loss_shape = np.exp(-np.abs(obs))
    loss_shape = np.exp(-np.power(obs,2))
    return np.array([1, 1e-1, 1, 1e-1]).reshape(1,-1).dot(loss_shape.T) #+ reward
    return np.exp(-np.abs(obs[:,2])) +  np.exp(-np.abs(obs[:,0]))

logger = tb_logger.TensorBoardLogger('logs/')


if __name__ == '__main__':
    # env = GymEnvWrapper(gym.make("Pendulum-v1"), num_envs=64)
    env = GymEnvWrapper(gym.make('CartPole-v1'), num_envs=16)
    # agent = ContPolicy(
    #     env.observation_space.shape[0], env.action_space.shape[0], 256, 1)
    # agent = DiscretePolicy(
    #     env.observation_space.shape[0], 2, 256, 1)
    agent = torch.load('cp2.pt')
    agent = REINFORCE(env, agent, 120, 500, torch_if=False,
                      reward_augment=pole_augment)
    input('Press Enter to continue...')
