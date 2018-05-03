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
type = 'CartPole-v0'

import matplotlib.pyplot as plt

# CartPole-v0
# self.batch_size = 32
# self.learning_rate = 0.0025
# self.epsilon = 0.1
# self.epsilon_min = 0.01
# self.epsilon_decay = 0.995
# self.gamma = 0.9


# self.batch_size = 64
# self.learning_rate = 0.0035
# self.epsilon = 0.1
# self.epsilon_min = 0.01
# self.epsilon_decay = 0.995
# self.gamma = 0.9

class MyDQN:
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 0.0035
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.9
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
        #self.mydqn.epsilon = max(self.mydqn.epsilon_min, np.power(self.mydqn.epsilon_decay, t)*self.mydqn.epsilon)
        self.mydqn.epsilon = self.mydqn.epsilon*np.power(self.mydqn.epsilon_decay, t)
        epsilon = max(0.025, self.mydqn.epsilon)
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
        self.episodes = 500
        self.max_step = 20000

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

                x, x_, theta, theta_ = s_
                r1 = (self.mydqn.env.x_threshold - abs(x)) / self.mydqn.env.x_threshold - 0.8
                r2 = (self.mydqn.env.theta_threshold_radians - abs(theta)) / self.mydqn.env.theta_threshold_radians - 0.5
                r = r1 + r2
                if x > 4 or x < -4:
                    r = r - 0.05
                # r = 0
                # if type == 'CartPole-v0':
                #     x, x_, theta, theta_ = s_
                #     r1 = (self.mydqn.env.x_threshold - abs(x)) / self.mydqn.env.x_threshold - 0.8
                #     r2 = (self.mydqn.env.theta_threshold_radians - abs(theta)) / self.mydqn.env.theta_threshold_radians - 0.5
                #     r = r1 + r2
                #     if x > 4 or x < -4:
                #         r = r - 0.08
                # elif type == 'MountainCar-v0':
                # position, velocity = s_
                # r = np.abs(position-(-0.5))
                # if position > 0 and velocity > 0:
                #     r += 2
                # elif position < 0 and velocity < 0:
                #     r += 2
                # else:
                #     r = -2


                reward += r
                self.dqn.store_transition(s, a, r, s_)
                reward += r
                if self.dqn.memory_count > self.mydqn.memory_capacity:
                    loss.append(self.dqn.learn())  # 记忆库满了就进行学习
                #step += 1
                if done:  # 如果回合结束, 进入下回合
                    #print(t)
                    # rewards.append(reward)
                    # x_reward.append(len(x_reward))
                    # if len(loss) >0 :
                    #     losses.append(np.mean(loss))
                    #     x_loss.append(len(x_loss))
                    #print("Episode %d finished after %f time steps " % (i, step))
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
        plt.savefig('./MyDQN/1_1_loss.png')
        plt.figure()
        plt.plot(x_reward, rewards)
        plt.xlabel('Training episodes')
        plt.ylabel('The sum of reawrd')
        plt.savefig('./MyDQN/1_1_reward.png')

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
            for t in range(20000):
                #env.render()
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

        avg_reward = np.mean(rewards)   #sum(rewards) / len(rewards)  # 均值
        std_reward = np.std(rewards)
        print("average_reward: {},std_reward: {}".format(avg_reward, std_reward))


if __name__ == '__main__':
    np.random.seed(0)
    test = TrainAndTest()
    test.test()

