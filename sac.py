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

import pytorch_lightning.loggers.tensorboard as pltb


class LunarActor(Policy):
    def __init__(self, input_dim: int, output_n: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.model = Mlp(input_dim, 2*output_n, hidden_dim, n_layers)

        self.N = input_dim
        self.M = output_n
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        self.transform = torch.distributions.transforms.TanhTransform()

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

    def parameters(self):
        return self.model.parameters()


# @dataclass
# class Experience:
#     state: Tensor
#     action: Tensor
#     reward: Tensor
#     done: Tensor
#     next_state: Tensor

Experience = col.namedtuple(
    'Experience', ['state', 'action', 'reward', 'done', 'next_state'])

# def


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

        nan = any(math.isnan(x)
                  for e in experience if isinstance(e, np.ndarray) for x in e) or any(math.isnan(e) for e in experience if isinstance(e, float))
        if not nan:
            # if not any(math.isnan(e) for e in experience if isinstance(e, float) else math.isnan(x) for x in e if e is isinstance(e, np.ndarray)):

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
    # lr: float

    def __init__(self, obs_dim, act_dim, batch_size, updates, discount=0.99, entropy_gain=0.2, target_momentum=0.995, lr=1e-4, weight_decay=1e-5) -> None:
        super().__init__()

        self.Q = [Mlp(input_dim=obs_dim + act_dim, output_n=1,
                      hidden_dim=256, n_layers=1) for _ in range(2)]
        self.target_Q = [copy.deepcopy(q) for q in self.Q]
        self.actor = LunarActor(
            input_dim=obs_dim, output_n=act_dim, hidden_dim=128, n_layers=1)
        self.configure_optimizers(lr, weight_decay)
        self.replay = ReplayBuffer(int(1e4))
        self.batch_size = batch_size
        self.updates = updates

        self.discount = discount
        self.entropy_gain = entropy_gain
        self.target_momentum = target_momentum

    def training_step(self):
        experience_batch = self.replay.sample(self.batch_size)

        for _ in range(self.updates):
            self.update_Q(experience_batch)
            self.update_policy(experience_batch)
            self.ma_Q_targets()

    def configure_optimizers(self, lr, wd):

        # optim_c

        self.policy_optim = torch.optim.SGD(
            self.actor.parameters(), lr=lr, weight_decay=wd)
        self.q_optim = [torch.optim.AdamW(q.parameters(), lr=lr, weight_decay=wd)
                        for q in self.Q]

    def update_Q(self, experience_batch):
        states, actions, rewards, dones, next_states = experience_batch

        with torch.no_grad():
            next_actions = self.actor.act(next_states)

            q_arg = torch.cat([next_states, next_actions], dim=-1)

            Q_min = torch.min(*[target_q(q_arg)
                              for target_q in self.target_Q])

            next_action_logprob = torch.sum(self.actor.pi(
                states).log_prob(next_actions), dim=-1)

            targets = rewards + self.discount * (1 - dones.to(torch.int)) * (Q_min - self.entropy_gain *
                                                                             next_action_logprob)

        for q, optim in zip(self.Q, self.q_optim):           

            q_arg = torch.cat([states, actions], dim=-1)
            Q_loss = torch.mean(torch.pow(q(q_arg) - targets, 2))
            Q_loss.backward()

            nn.utils.clip_grad_norm_(q.parameters(), 1)

            optim.step()
            optim.zero_grad()

    def update_policy(self, experience_batch):
        states = experience_batch[0]

    # with torch.no_grad():
        actions = self.actor.act(states)

        # torch.clamp_(actions, -0.999,0.999)

        # actions[torch.abs(actions) > 0.9999] *= 0.99

        action_logprob = torch.sum(self.actor.pi(
            states).log_prob(actions), dim=-1)

        q_arg = torch.cat([states, actions], dim=-1)
        with torch.no_grad():   
            Q_min = torch.min(*[q(q_arg) for q in self.Q])

        

        loss_per = self.entropy_gain * action_logprob - Q_min

        policy_loss = torch.mean(
            loss_per[~torch.any(loss_per.isnan(), dim=1)])  # drop NaN rows

        if math.isnan(policy_loss):
            return
        else:
            policy_loss.backward()

        nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 1)

        # nan_in_grad = any(math.isnan(x) for p in self.actor.parameters() for x in p.grad.flatten())

        self.policy_optim.step()
        self.policy_optim.zero_grad()

        # with torch.no_grad():
        #     norm = np.mean([p.norm()/(p.shape[0]*p.shape[1]) for p in self.actor.parameters() if len(p.shape) > 1])
        #     logger.experiment.add_scalar('weights_norm', norm, episode_counter)

    def ma_Q_targets(self):
        '''update target Q with exponential moving average'''
        with torch.no_grad():
            _ = [interp_models_(
                q_target, q, self.target_momentum) for q, q_target in zip(self.Q, self.target_Q)]


def interp_models_(model_a, model_b, weight):
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    state_w = {key: weight * state_a[key] + (1-weight) * state_b[key]
               for key in state_a.keys()}
    model_a.load_state_dict(state_w)


global logger
global episode_counter
if __name__ == "__main__":

    env_info = {
        'name':'Pendulum-v1',
        'input_dim':3,
        'output_dim':1,
    }
    env_info = {
        'name':'LunarLanderContinuous-v2',
        'input_dim':8,
        'output_dim':2,
    }


    env = gym.make(env_info['name'])
    # env.seed(0)
    # torch.manual_seed(0)

    # torch.autograd.set_detect_anomaly(True)

    logger = pltb.TensorBoardLogger('logs/')

    batch_size = 512

    # updates = 5

    update_freq = 50

    lr = 1e-4

    sac = Sac(env_info['input_dim'], env_info['output_dim'], batch_size, update_freq, lr=lr)

    start_training = min(int(1e3), batch_size*10)

    # raise Exception('Note: update Q interpolation not working yet.')

    show_n = 10

    render = False

    for episode_counter in range(int(1e5)):
        if render:
            if episode_counter % 100 == 0 and episode_counter > 0:
                env.close()
                env = gym.make(env_info['name'], render_mode='human')
            if episode_counter % 100 == show_n:
                env.close()
                env = gym.make(env_info['name'])

        state = env.reset()[0]
        episode_end = False

        episode_reward = 0

        while not episode_end:
            if episode_counter > start_training:
                with torch.no_grad():
                    action = sac.actor.act(state).detach().numpy()
            else:
                action = torch.distributions.Uniform(-1,
                                                     1).sample((2,)).numpy()

            action = np.multiply(env.action_space.high, action)
            next_state, reward, terminal, truncated, info = env.step(action)

            episode_reward += reward

            episode_end = terminal or truncated

            sac.replay.append(Experience(
                state, action, reward, terminal, next_state))
            state = next_state

        logger.experiment.add_scalar('reward', episode_reward, episode_counter)

        if episode_counter > start_training and episode_counter % update_freq == 0:
            print(f'{episode_counter:d} SAC_train_step')
            sac.training_step()
    env.close()
