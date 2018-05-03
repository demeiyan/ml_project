# -*- coding: utf-8 -*-
"""
Created on 17-12-27 下午3:42

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
np.random.seed(0)


class MyDQN:
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.01
        self.epsilon = 0.1
        self.gamma = 0.9
        self.target_replace_iter = 100
        self.memory_capacity = 2000
        self.env = gym.make(type)
        self.env = self.env.unwrapped
        self.action_len = self.env.action_space.n
        self.state_len = self.env.observation_space.high.shape[0]+1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        mydqn = MyDQN()
        self.input = nn.Linear(mydqn.state_len - 1, 10)
        self.input.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, mydqn.action_len)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.input(x)
        #x = func.relu(x)
        x = func.tanh(x)
        action_value = self.out(x)
        return action_value


class DQN:
    def __init__(self):
        self.mydqn = MyDQN()
        self.e_net, self.t_net = Net(), Net()
        self.learn_step_count = 0
        self.memory_count = 0
        self.memory = np.zeros((self.mydqn.memory_capacity, (self.mydqn.state_len - 1)*2+2+1))
        self.optim = torch.optim.Adam(self.e_net.parameters(), lr=self.mydqn.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, obv):
        if np.random.random() < self.mydqn.epsilon:
            action = np.random.randint(0, self.mydqn.action_len)
        else:
            x = Variable(torch.unsqueeze(torch.FloatTensor(obv), 0))
            actions_value = self.e_net.forward(x)
            action = int(torch.max(actions_value, 1)[1].data.numpy())
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_count % self.mydqn.memory_capacity
        self.memory[index, :] = transition
        self.memory_count += 1

    def learn(self):
        if self.learn_step_count % self.mydqn.target_replace_iter == 0:
            self.t_net.load_state_dict(self.e_net.state_dict())
        self.learn_step_count += 1
        sample_index = np.random.choice(self.mydqn.memory_capacity, self.mydqn.batch_size)
        sample_memory = self.memory[sample_index, :]
        sample_s = Variable(torch.FloatTensor(sample_memory[:, :self.mydqn.state_len-1]))
        sample_a = Variable(torch.LongTensor(sample_memory[:, self.mydqn.state_len-1:self.mydqn.state_len].astype(int)))
        sample_r = Variable(torch.FloatTensor(sample_memory[:, self.mydqn.state_len:self.mydqn.state_len+1]))
        sample_s_ = Variable(torch.FloatTensor(sample_memory[:, -self.mydqn.state_len:]))

        terminal = sample_s_[:, self.mydqn.state_len-1]
        terminal = terminal.data.numpy().astype(int)
        index = np.where(terminal == 1) #终止状态的索引
        sample_s_ = sample_s_[:, :self.mydqn.state_len-1]

        q_eval = self.e_net.forward(sample_s).gather(1, sample_a)   #获取当初在s状态下选择a动作的价值(32,1)
        q_next = self.t_net.forward(sample_s_).detach()
        q_max = torch.unsqueeze(q_next.max(1)[0], dim=1)
        y = sample_r + (self.mydqn.gamma * q_max)
        y[index] = sample_r[index]
        #
        # print(y)
        loss = self.loss_func(q_eval, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        print(loss.data.numpy())


class TrainAndTest:
    def __init__(self):
        self.dqn = DQN()
        self.mydqn = MyDQN()
        self.episodes = 400
        self.max_step = 20000

    def train(self):
        for i in range(self.episodes):
            s = self.mydqn.env.reset()
            #step = 0
            #while True:
            for t in range(self.max_step):
                a = self.dqn.choose_action(s)
                s_, reward, done, info = self.mydqn.env.step(a)
                r = 0
                if type == 'CartPole-v0':
                    x, x_, theta, theta_ = s_
                    r1 = (self.mydqn.env.x_threshold - abs(x)) / self.mydqn.env.x_threshold - 0.8
                    r2 = (self.mydqn.env.theta_threshold_radians - abs(theta)) / self.mydqn.env.theta_threshold_radians - 0.5
                    r = r1 + r2
                    if x > 4 or x < -4:
                        r = r - 0.05
                elif type == 'MountainCar-v0':
                    position, velocity = s_
                    r = np.abs(position-(-0.07))
                    # if position > 0 and velocity > 0:
                    #     r += 2
                    # elif position < 0 and velocity < 0:
                    #     r += 2
                    # else:
                    #     r = -1
                #添加标志位0为非终止状态，1为终止状态
                if done:
                    state1 = np.insert(s_, self.mydqn.state_len-1, values=0)
                    break
                else:
                    state1 = np.insert(s_, self.mydqn.state_len - 1, values=1)
                self.dqn.store_transition(s, a, r, state1)

                if self.dqn.memory_count > self.mydqn.memory_capacity:
                    self.dqn.learn()  # 记忆库满了就进行学习
                #step += 1
                s = s_

    def test(self):
        print('----------------train---------------------')
        self.train()
        env = self.mydqn.env
        #env = wrappers.Monitor(env, './MyDQN_test/cartpole-vo', force=True)
        rewards = []
        std_reward = 0
        print('----------------test----------------------')
        for i in range(10):
            obv = env.reset()
            r = 0
            for t in range(self.max_step):
                #env.render()
                action = self.dqn.t_net.forward(Variable(torch.FloatTensor(obv))).data.numpy()
                action = np.argmax(action)
                obv, reward, done, info = env.step(action)
                r += reward
                if done:
                    print("Episode %d finished after %f time steps " % (i, t))
                    break
                elif t == self.max_step - 1:
                    print("Episode %d finished after %f time steps " % (i, t))
            rewards.append(r)
        avg_reward = sum(rewards) / len(rewards)  # 均值
        for i in range(len(rewards)):
            std_reward += np.square(rewards[i] - avg_reward)
        std_reward = np.sqrt(std_reward / len(rewards))  # 标准差
        print("average_reward: %.2f,std_reward: %.2f" % (avg_reward, std_reward))


class Test:
    def __init__(self):
        self.env = gym.make(type)
        self.env = self.env.unwrapped
        self.env = wrappers.Monitor(self.env, './MyDQN/cartpole-vo', force=True)

    def test(self):
        obv = self.env.reset()
        while True:
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            if done:
                break


if __name__ == '__main__':
    # test = Test()
    # test.test()
    test = TrainAndTest()
    test.test()


