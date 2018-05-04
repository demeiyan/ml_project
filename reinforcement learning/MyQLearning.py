# -*- coding: utf-8 -*-
"""
Created on 2017/12/21 21:18
@author: dmyan
"""
import gym
import math
import numpy as np
import random
import sys
from gym import wrappers

# cartpole-v0
# self.epsilon = 0.2
# self.alpha = 0.08
# self.gamma = 0.9
# self.episodes = 2000
# self.epsilon_decay = 0.9

# self.epsilon = 0.1
# self.alpha = 0.08
# self.gamma = 0.9
# self.episodes = 2000
# self.epsilon_decay = 0.9


class QLearning:
    def __init__(self, name):
        """
        初始化
        :param name:
        """
        random.seed(0)
        self.type = name
        if name == 'CartPole-v0':
            self.env = gym.make('CartPole-v0')
            self.env = self.env.unwrapped
            self.state_len = (1, 1, 6, 5)
            self.action_len = self.env.action_space.n
            self.state_scopes = list(zip(self.env.observation_space.low, self.env.observation_space.high))
            self.state_scopes[1] = [-0.5, 0.5]
            self.state_scopes[3] = [-math.radians(50), math.radians(50)]
            self.max_steps = 20000
        elif name == 'MountainCar-v0':
            self.env = gym.make('MountainCar-v0')
            self.env = self.env.unwrapped
            self.state_len = (20, 20)
            self.action_len = self.env.action_space.n
            self.state_scopes = list(zip(self.env.observation_space.low, self.env.observation_space.high))
            self.max_steps = 2000
        else:
            self.env = gym.make('Acrobot-v1')
            self.env = self.env.unwrapped
            self.state_len = (10, 10, 10, 10, 5, 5)
            self.action_len = self.env.action_space.n
            self.state_scopes = list(zip(self.env.observation_space.low, self.env.observation_space.high))
            self.max_steps = 2000
        self.q_table = np.zeros(self.state_len + (self.action_len,))

        # CartPole-V0 parameters
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.95
        self.episodes = 2000
        self.epsilon_decay = 25
        # Acrobot-v1 parameters
        # self.epsilon = 0.1
        # self.alpha = 0.1
        # self.gamma = 0.95
        # self.episodes = 2000
        # self.epsilon = 0.1
        # self.alpha = 0.1
        # self.gamma = 0.99
        # self.episodes = 2000

        # MountainCar-v0 parameters
        # self.epsilon = 0.1
        # self.alpha = 0.1
        # self.gamma = 0.99
        # self.episodes = 2000

        # self.epsilon = 0.025
        # self.alpha = 0.05
        # self.gamma = 0.95
        # self.episodes = 2000

    def discrete(self, obv):
        """
        连续状态空间离散化
        :param obv:
        :return:
        """
        state = []
        for i in range(len(obv)):
            if obv[i] <= self.state_scopes[i][0]:
                state_index = 0
            elif obv[i] >= self.state_scopes[i][1]:
                state_index = self.state_len[i] - 1
            else:
                length = self.state_scopes[i][1] - self.state_scopes[i][0]
                state_index = int(round((self.state_len[i] - 1) * (obv[i] - self.state_scopes[i][0]) / length))
            state.append(state_index)
        return tuple(state)

    def choose_action(self, s, t):
        """
        根据epsilon greedy选择动作
        :param s:
        :param t:
        :return:
        """
        epsilon = max(0.01, min(1, 1.0 - math.log10((t + 1) / self.epsilon_decay)))
        if random.random() < epsilon:
            action =self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[s])
        return action

    def get_learning_rate(self, t):
        """
        更新learning_rate
        :param t:
        :return:
        """
        return max(0.1, min(0.5, 1.0 - math.log10((t + 1) / self.epsilon_decay)))

    def get_reward(self, obv, reward):
        """
        根据具体场景返回reward
        :param obv:
        :param reward:
        :return:
        """
        if self.type == 'CartPole-v0':
            x, x_, theta, theta_ = obv
            r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
            r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
            r = r1 + r2
            if x > 4 or x < -4:
                r = r - 0.05
            return r
        elif self.type == 'MountainCar-v0':
            return reward
        else:
            return reward

    def q_learining(self):
        """
        训练
        :return:
        """
        learning_rate = self.get_learning_rate(0)
        print('----- train -----')
        for episode in range(self.episodes):
            obv = self.env.reset()
            s = self.discrete(obv)
            for t in range(self.max_steps):
                action = self.choose_action(s, episode)
                obv, reward, done, info = self.env.step(action)
                s_ = self.discrete(obv)
                if self.type == 'CartPole-v0':  # CartPole-v0 reward
                    pass
                    # x, x_, theta, theta_ = obv
                    # r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                    # r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                    # r = r1 + r2
                    # if x > 4 or x < -4:
                    #     r = r - 0.05
                    # reward = r
                elif self.type == 'MountainCar-v0 ':    # MountainCar-v0 reward
                    position, velocity = s_
                    reward = np.abs(position-(-0.5))
                elif self.type == 'Acrobot-v1':                 # Acrobot-v1 reward
                    x1, _, x2, _, _, _ = s_
                    r = 1 - x1 + x2
                    if done and t <500 :
                        if t < 200:
                            r += 1000
                        r += 500
                    reward = r
                qmax = np.max(self.q_table[s_])
                self.q_table[s + (action,)] += learning_rate * (reward + self.gamma * qmax - self.q_table[s + (action,)])
                s = s_
                if done:
                    # print(t)
                    # print('q_table', self.q_table)
                    print("Episode %d finished after %f time steps" % (episode, t))
                    break
                learning_rate = self.get_learning_rate(episode)

    def run(self):
        """
        测试学习的效果，计算均值和方差
        :return:
        """
        rewards = []
        print('----- run -----')
        #self.env = wrappers.Monitor(self.env, './MyQLearning/'+self.type, force=True)
        for episode in range(100):
            obv = self.env.reset()
            s = self.discrete(obv)
            r = 0
            done = False
            step = 0
            for t in range(self.max_steps):
                action = np.argmax(self.q_table[s])
                obv, reward, done, info = self.env.step(action)
                s_ = self.discrete(obv)
                s = s_
                r += reward
                step += 1
                if done:
                    break
            print("Episode %d finished after %f time steps " % (episode, step))
            rewards.append(r)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print("average_reward: {},std_reward: {}".format(avg_reward, std_reward))


if __name__ =='__main__':
    q_learning = QLearning('CartPole-v0')
    q_learning.q_learining()
    q_learning.run()