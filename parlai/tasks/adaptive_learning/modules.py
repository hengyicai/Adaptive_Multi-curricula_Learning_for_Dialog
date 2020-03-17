import torch.nn as nn
import torch.nn.functional as F
import torch
from parlai.agents.hy_lib.common_modules import FeedForward


class PolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy = FeedForward(state_dim, action_dim, hidden_sizes=(128, 64))

    def forward(self, state):
        action_score = self.policy(state)
        action_prob = F.softmax(action_score, dim=-1)

        return action_prob


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.critic = FeedForward(state_dim + action_dim, 1, hidden_sizes=(128, 64))

    def forward(self, state_actions):
        val = self.critic(state_actions)
        return val
