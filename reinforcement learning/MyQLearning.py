# -*- coding: utf-8 -*-
"""
Created on 2017/12/21 21:18

@author: dmyan
"""
import gym
import math
import numpy as np
import random
from gym import wrappers


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
            self.state_len = (2, 2, 6, 3)
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
            self.max_steps = 200
        else:
            self.env = gym.make('Acrobot-v1')
            self.env = self.env.unwrapped
            self.state_len = (10, 10, 10, 10, 5, 5)
            self.action_len = self.env.action_space.n
            self.state_scopes = list(zip(self.env.observation_space.low, self.env.observation_space.high))
            self.max_steps = 200
        self.q_table = np.zeros(self.state_len + (self.action_len,))

        self.epsilon = 0.1
        self.alpha = 0.015
        self.gamma = 0.95
        self.episodes = 2000

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
        epsilon = max(self.epsilon, min(1, 1.0 - math.log10((t + 1) / 30)))
        #print(epsilon)
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
        return max(self.alpha, min(0.5, 1.0 - math.log10((t + 1) / 25)))

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
            #return reward
        elif self.type == 'MountainCar-v0':
            # position, velocity = obv
            # # print(position,velocity)
            # #return abs(position - (-0.5))
            # if position > 0 and velocity > 0:
            #     return 10
            # elif position < 0 and velocity < 0:
            #     return 10
            # else:
            #     return -1
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
                x, x_, theta, theta_ = obv
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                r = r1 + r2
                if x > 4 or x < -4:
                    r = r - 0.05
                reward = r
                #reward = self.get_reward(obv, reward)
                qmax = np.max(self.q_table[s_])
                self.q_table[s + (action,)] += learning_rate * (reward + self.gamma * qmax - self.q_table[s + (action,)])
                s = s_
                if done:
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
            step = 0
            done = False
            for t in range(self.max_steps):
            #while True:
                #self.env.render()
                #self.env.monitor()
                action = np.argmax(self.q_table[s])
                obv, reward, done, info = self.env.step(action)
                s_ = self.discrete(obv)
                s = s_
                r += reward
                step += 0
                if done :
                    print("Episode %d finished after %f time steps " % (episode, t))
                    #print("finished after %f time steps " % (t))
                    break

            if not done:
                print("Episode %d finished after %f time steps " % (episode, step))
            rewards.append(r)

        avg_reward = np.mean(rewards)   #sum(rewards) / len(rewards)  # 均值
        std_reward = np.std(rewards)
        print("average_reward: {},std_reward: {}".format(avg_reward, std_reward))


if __name__ =='__main__':
    q_learning = QLearning('CartPole-v0')
    q_learning.q_learining()
    q_learning.run()
