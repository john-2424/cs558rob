import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal


class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128, init_std=0.2):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )

        self.log_std = nn.Parameter(torch.ones(act_dim) * torch.log(torch.tensor(init_std)))

    def forward(self, obs):
        mean = self.mean_net(obs).squeeze(0)
        std = torch.exp(self.log_std)
        std = torch.clamp(std, min=1e-3)
        cov = torch.diag(std ** 2)
        return MultivariateNormal(mean, covariance_matrix=cov)