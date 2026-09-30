import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class Qnet(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(Qnet, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.model = nn.Sequential(
            nn.Linear(input_layer, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, output_layer)
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def sample_action_list(self, obs, epsilon):
        out = self.forward(obs)
        return out

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            action = random.randint(0, self.output_layer-1)
            return action
        else:
            return out.argmax().item()

    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        return out.argmax().item(), out