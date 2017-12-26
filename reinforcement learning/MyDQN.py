# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as func
import numpy as np
import gym

type = 'MountainCar-v0'
class MyDQN:
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.01
        self.epsilon = 0.1
        self.gamma = 0.9
        self.target_replace_iter = 100
        self.memory_capacity = 2000
        self.env = gym.make('MountainCar-v0')
        self.env = self.env.unwrapped
        self.action_len = self.env.action_space.n
        self.state_len = self.env.observation_space.high.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        mydqn = MyDQN()
        self.input = nn.Linear(mydqn.state_len, 10)
        self.input.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, mydqn.action_len)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.input(x)
        x = func.relu(x)
        action_value = self.out(x)
        return action_value


class DQN:
    def __init__(self):
        self.mydqn = MyDQN()
        self.e_net, self.t_net = Net(), Net()
        self.learn_step_count = 0
        self.memory_count = 0
        self.memory = np.zeros((self.mydqn.memory_capacity, self.mydqn.state_len*2+2))
        self.optim = torch.optim.Adam(self.e_net.parameters(), lr=self.mydqn.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, obv):
        x = Variable(torch.unsqueeze(torch.FloatTensor(obv), 0))

        if np.random.uniform() < self.mydqn.epsilon:
            action = np.random.randint(0,self.mydqn.action_len)
        else:
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
        sample_s = Variable(torch.FloatTensor(sample_memory[:, :self.mydqn.state_len]))
        sample_a = Variable(torch.LongTensor(sample_memory[:, self.mydqn.state_len:self.mydqn.state_len+1].astype(int)))
        sample_r = Variable(torch.FloatTensor(sample_memory[:, self.mydqn.state_len+1:self.mydqn.state_len+2]))
        sample_s_ = Variable(torch.FloatTensor(sample_memory[:, -self.mydqn.state_len:]))

        q_eval = self.e_net.forward(sample_s).gather(1, sample_a)   #获取当初在s状态下选择a动作的价值(32,1)
        q_next = self.t_net.forward(sample_s_).detach()
        q_max = torch.unsqueeze(q_next.max(1)[0], dim=1)
        q_t = sample_r + (self.mydqn.gamma * q_max)
        loss = self.loss_func(q_eval, q_t)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


class TrainAndTest:
    def __init__(self):
        self.dqn = DQN()
        self.mydqn = MyDQN()
        self.episodes = 400
        self.max_step = 20000

    def train(self):
        for i in range(self.episodes):
            s = self.mydqn.env.reset()
            step = 0
            while True:
                a = self.dqn.choose_action(s)

                s_, reward, done, info = self.mydqn.env.step(a)
                r = reward
                if type == 'CartPole-v0':
                    x, x_, theta, theta_ = s_
                    r1 = (self.mydqn.env.x_threshold - abs(x)) / self.mydqn.env.x_threshold - 0.8
                    r2 = (self.mydqn.env.theta_threshold_radians - abs(theta)) / self.mydqn.env.theta_threshold_radians - 0.5
                    r = r1 + r2
                elif type == 'MountainCar-v0':
                    r = reward
                self.dqn.store_transition(s, a, r, s_)

                if self.dqn.memory_count > self.mydqn.memory_capacity:
                    self.dqn.learn()  # 记忆库满了就进行学习
                step += 1
                if done:  # 如果回合结束, 进入下回合
                    print("Episode %d finished after %f time steps " % (i, step))
                    break
                s = s_

    def test(self):
        print('------------train--------------')
        self.train()
        env = self.mydqn.env
        r = 0
        rewards = []
        std_reward = 0
        print('-------------test------------------')
        for i in range(100):
            obv = env.reset()
            for t in range(self.max_step):
                #env.render()
                action = self.dqn.t_net.forward(Variable(torch.FloatTensor(obv))).data.numpy()
                action = np.argmax(action)
                #print(action, np.argmax(action))

                obv, reward, done, info = env.step(action)
                r += reward
                if done:
                    print("Episode %d finished after %f time steps " % (i, t))
                    break
            rewards.append(r)
        avg_reward = sum(rewards) / len(rewards)  # 均值
        for i in range(len(rewards)):
            std_reward += np.square(rewards[i] - avg_reward)
        std_reward = np.sqrt(std_reward / len(rewards))  # 标准差
        print("average_reward: %.2f,std_reward: %.2f" % (avg_reward, std_reward))








if __name__ == '__main__':
    test = TrainAndTest()
    test.test()
    # mydqn = MyDQN()
    # dqn = DQN()
    # env = gym.make('CartPole-v0')
    # env = env.unwrapped
    # obv = env.reset()
    # obvs = []
    # action = env.action_space.sample()
    # actions = []
    # for i in range(10):
    #     obv = env.reset()
    #     action = env.action_space.sample()
    #     actions.append(action)
    #     obvs.append(obv)
    # print('actions', actions)
    # dqn.test(Variable(torch.FloatTensor(obvs)), Variable(torch.LongTensor(actions)))
    #torch.FloatTensor([[1, 2, 3], [4, 5, 6]])


