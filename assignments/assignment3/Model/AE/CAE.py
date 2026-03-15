import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

mse_loss = nn.MSELoss()

# class Encoder(nn.Module):
# 	def __init__(self):
# 		super(Encoder, self).__init__()
# 		print('using deep encoder')
# 		self.encoder = nn.Sequential(nn.Linear(2800, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 128),nn.PReLU(),nn.Linear(128, 28))
# 	def forward(self, x):
# 		x = self.encoder(x)
# 		return x

class Encoder(nn.Module):
    def __init__(self, input_size=2800):
        super(Encoder, self).__init__()
        print(f'using deep encoder, input_size={input_size}')
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 28)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x