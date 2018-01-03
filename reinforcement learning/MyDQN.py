# -*- coding: utf-8 -*-
"""
Created on 17-12-27 下午5:10

@author: dmyan
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as func
import numpy as np
import gym
from gym import wrappers
type = 'Acrobot-v1'

import matplotlib.pyplot as plt

# CartPole-v0
# self.batch_size = 128
# self.learning_rate = 0.001
# self.epsilon = 0.1
# self.epsilon_decay = 0.95
# self.gamma = 0.99
# self.episodes = 800

# self.batch_size = 128
# self.learning_rate = 0.0025
# self.epsilon = 0.05
# self.epsilon_min = 0.01
# self.epsilon_decay = 0.95
# self.gamma = 0.99

class MyDQN:
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.0025
        self.epsilon = 0.05
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.gamma = 0.99
        self.target_replace_iter = 100
        self.memory_capacity = 2000
        self.env = gym.make(type)
        self.env = self.env.unwrapped
        self.action_len = self.env.action_space.n
        self.state_len = self.env.observation_space.high.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        mydqn = MyDQN()
        self.input = nn.Linear(mydqn.state_len, 128)
        self.input.weight.data.normal_(0, 0.1)
        self.middle = nn.Linear(128, 128)
        self.middle.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, mydqn.action_len)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.input(x)
        x = func.relu(x)
        x = self.middle(x)
        x = func.relu(x)
        action_value = self.out(x)
        return action_value


class DQN:
    def __init__(self):
        self.mydqn = MyDQN()
        self.q_net = Net()
        self.memory_count = 0
        self.memory = np.zeros((self.mydqn.memory_capacity, self.mydqn.state_len*2+2))
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=self.mydqn.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, obv, t):
        self.mydqn.epsilon = self.mydqn.epsilon*np.power(self.mydqn.epsilon_decay, t)
        epsilon = max(0.01, self.mydqn.epsilon)
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.mydqn.action_len)
        else:
            x = Variable(torch.unsqueeze(torch.FloatTensor(obv), 0))
            actions_value = self.q_net.forward(x)
            action = int(torch.max(actions_value, 1)[1].data.numpy())
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_count % self.mydqn.memory_capacity
        self.memory[index, :] = transition
        self.memory_count += 1

    def learn(self):
        sample_index = np.random.choice(self.mydqn.memory_capacity, self.mydqn.batch_size)
        sample_memory = self.memory[sample_index, :]
        sample_s = Variable(torch.FloatTensor(sample_memory[:, :self.mydqn.state_len]))
        sample_a = Variable(torch.LongTensor(sample_memory[:, self.mydqn.state_len:self.mydqn.state_len+1].astype(int)))
        sample_r = Variable(torch.FloatTensor(sample_memory[:, self.mydqn.state_len+1:self.mydqn.state_len+2]))
        sample_s_ = Variable(torch.FloatTensor(sample_memory[:, -self.mydqn.state_len:]))

        q_eval = self.q_net.forward(sample_s).gather(1, sample_a)   #获取当初在s状态下选择a动作的价值(32,1)
        q_next = self.q_net.forward(sample_s_).detach()
        q_max = torch.unsqueeze(q_next.max(1)[0], dim=1)
        y = sample_r + (self.mydqn.gamma * q_max)

        loss = self.loss_func(q_eval, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.data.numpy()


class TrainAndTest:
    def __init__(self):
        self.dqn = DQN()
        self.mydqn = MyDQN()
        self.episodes = 800
        self.max_step = 2000

    def train(self):

        losses = []
        x_loss = []
        rewards = []
        x_reward = []
        for i in range(self.episodes):
            s = self.mydqn.env.reset()
            reward = 0
            loss = []
            for t in range(self.max_step):
                a = self.dqn.choose_action(s, i)
                s_, r, done, info = self.mydqn.env.step(a)
                # CartPole-v0 reward
                # x, x_, theta, theta_ = s_
                # r1 = (self.mydqn.env.x_threshold - abs(x)) / self.mydqn.env.x_threshold - 0.8
                # r2 = (self.mydqn.env.theta_threshold_radians - abs(theta)) / self.mydqn.env.theta_threshold_radians - 0.5
                # r = r1 + r2
                # if x > 4 or x < -4:
                #     r = r - 0.05

                # MountainCar-v0 reward
                # position, velocity = s_
                # r = np.abs(position-(-0.5))

                # Acrobot-v1 reward
                # x1, _, x2, _, _, _ = s_
                # r = 1 - x1 + x2
                # if done and t < 500 :
                #     if t < 200:
                #         r += 1000
                #     if t < 100:
                #         r += 10000
                #     r += 500

                if type == 'CartPole-v0':  # CartPole-v0 reward
                    self.max_step = 20000
                    x, x_, theta, theta_ = s_
                    r1 = (self.mydqn.env.x_threshold - abs(x)) / self.mydqn.env.x_threshold - 0.8
                    r2 = (self.mydqn.env.theta_threshold_radians - abs(theta)) / self.mydqn.env.theta_threshold_radians - 0.5
                    r = r1 + r2
                    if x > 4 or x < -4:
                        r = r - 0.05
                elif type == 'MountainCar-v0 ':    # MountainCar-v0 reward
                    position, velocity = s_
                    r = np.abs(position-(-0.5))
                elif type == 'Acrobot-v1':                 # Acrobot-v1 reward
                    x1, _, x2, _, _, _ = s_
                    r = 1 - x1 + x2
                    if done and t <500 :
                        if t < 200:
                            r += 1000
                        r += 500
                self.dqn.store_transition(s, a, r, s_)
                reward += r
                if self.dqn.memory_count > self.mydqn.memory_capacity:
                    loss.append(self.dqn.learn())  # 记忆库满了就进行学习
                if done:
                    print("Episode %d finished after %f time steps" % (i, t))
                    break
                s = s_
            rewards.append(reward)
            x_reward.append(len(x_reward))
            if len(loss) == 0:
                losses.append(sum(loss))
                x_loss.append(len(x_loss))
            else:
                losses.append(sum(loss)/len(loss))
                x_loss.append(len(x_loss))
        plt.figure()
        plt.plot(x_loss, losses)
        plt.xlabel('Training episodes')
        plt.ylabel('Loss average')
        plt.savefig('./loss.png')
        plt.figure()
        plt.plot(x_reward, rewards)
        plt.xlabel('Training episodes')
        plt.ylabel('The sum of reawrd')
        plt.savefig('./reward.png')

    def test(self):
        print('----------------train---------------------')
        self.train()
        env = self.mydqn.env
        #env = wrappers.Monitor(env, './MyDQN/cartpole-vo', force=True)
        rewards = []
        print('----------------test----------------------')
        for i in range(100):
            obv = env.reset()
            r = 0
            done = False
            step = 0
            for t in range(self.max_step):
                action = self.dqn.q_net.forward(Variable(torch.FloatTensor(obv))).data.numpy()
                action = np.argmax(action)
                obv, reward, done, info = env.step(action)
                r += reward
                step += 1
                if done:
                    print("Episode {} finished after {} time steps ".format(i, t))
                    break
            if not done:
                print("Episode {} finished after {} time steps ".format(i, step))
            rewards.append(r)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print("average_reward: {},std_reward: {}".format(avg_reward, std_reward))


if __name__ == '__main__':
    np.random.seed(0)
    test = TrainAndTest()
    test.test()

