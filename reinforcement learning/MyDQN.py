# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as func
import numpy as np
import gym

class MyDQN:
    def __init__(self,type):
        self.type = type
        self.batch_size = 32
        self.learning_rate = 0.01
        self.epsilon = 0.9
        self.gamma = 0.9
        self.target_replace_iter = 100
        self.memory_capacity = 2000
        self.env = gym.make(type)
        self.env = self.env.unwrapped
        self.action_len = self.env.action_space.n
        self.state_len = self.env.observation_space.high.shape[0]

class Net(nn.Module):
    def __init__(self, type):
        super(Net, self).__init__()
        mydqn = MyDQN(type)
        self.function = nn.Linear(mydqn.state_len, 10)
        self.function.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, mydqn.action_len)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.function(x)
        x = func.relu(x)
        action_value = self.out(x)
        return action_value
class DQN:
    def __int__(self, type):
        mydqn = MyDQN(type)
        self.e_net, self.t_net = Net(type), Net(type)
        self.learn_step_count = 0
        self.memory_count = 0
        









if __name__ == '__main__':
    pass