# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:09:09 2017

@author: Sharey
"""
import gym
import numpy as np
import random
import math

DECAY_RATE = 25 
class MyQlearning():
       
    def __init__(self,gama = 0.99,alpha = 0.5):
        self.alpha = alpha
        self.gama  = gama
  
    def state(self,state):
        bucket_indice = []
        
        for i in range(len(state)):
            if state[i] <= STATE_BOUNDS[i][0]:
                bucket_index = 0
            elif state[i] >= STATE_BOUNDS[i][1]:
                bucket_index = NUM_BUCKETS[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                if NUM_BUCKETS[i] > 2:
                    bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
                    bound_width = bound_width/(NUM_BUCKETS[i]-2)
                    width = state[i] - STATE_BOUNDS[i][0]
                    bucket_index = int( int(width // bound_width)+1 )
                else:
                    bucket_index = 0
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)
    
    def get_learning_rate(self,t):
        return max(0.1, min(0.5, 1.0 - math.log10((t+1)/DECAY_RATE)))
    def get_explore_rate(self,t):
        return max(0.01, min(1, 1.0 - math.log10((t+1)/DECAY_RATE)))
    
    def Qlearning(self,env):
        
        
        self.gama = 0.99
        Q = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
       
        def select_action(state,epsilon):
            # Select a random action
            if random.random() < epsilon:
                action = env.action_space.sample()
            # Select the action with the highest q
            else:
                action = np.argmax(Q[state])
            return action
        
        #for i in range (1000):
        for episode in range (NUM_EPISODES):
            observation = env.reset()
            #env.render()
            self.alpha = self.get_learning_rate(episode)
            epsilon = self.get_explore_rate(episode)
            x = self.state(observation)
            for t in range(MAX_STEPS):
                a = select_action(x,epsilon)
                observation, reward, done, info = env.step(a)
                
                     
                x_next = self.state(observation)
                a_next = np.argmax(Q[x_next])
                Q[x][a] = Q[x][a] + self.alpha*(reward + \
                 self.gama*Q[x_next][a_next] - Q[x][a])
                x = x_next
                a = a_next
                
                if done:
                    print("Episode %d finished after %f  steps " % (episode, t))
                    break
        sum_reward = 0
        print ("-----------test---------------")
        NUM_TEST = 100
        result = []
        #filepath = './MyImprovedDQN/cartpole-vo'
        for episode in range(NUM_TEST):
            #env = gym.make('CartPole-v0')
            #env = env.unwrapped
            #env = gym.wrappers.Monitor(env, filepath+str(episode), force=True)
            # Reset the environment
            observation = env.reset()
            state = self.state(observation)
            #env.render()

            for t in range(MAX_STEPS):
                
                
    
                # Select an action
                action = np.argmax(Q[state])
    
                # Execute the action
                observation, reward, done, info = env.step(action)
                
    
                # Observe the result
                state = self.state(observation)
    
                # Setting up for the next iteration
    
                # Print data
                if done:
                   sum_reward += t 
                   result.append(t)
                   print("Episode %d finished after %f time steps" % (episode, t))
                   break
        result = np.array(result)
        print ("mean: %f ; var: %f" % (result.mean(),result.std()) )    
        return result
def run3():
    env = gym.make('Acrobot-v1')
    env = env.unwrapped
    learner = MyQlearning()    
    NUM_BUCKETS = (6, 6, 6, 6, 6, 6)  
    NUM_ACTIONS = env.action_space.n 
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    NUM_EPISODES = 2000
    MAX_STEPS = 2000
    result = learner.Qlearning(env)
    #return result



def run2():
    
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    learner = MyQlearning()
    NUM_BUCKETS = (20, 20)  
    NUM_ACTIONS = env.action_space.n 
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    NUM_EPISODES = 2000
    MAX_STEPS = 2000
    result = learner.Qlearning(env)
    
    #return result
    
def run1():        
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    learner = MyQlearning()
    NUM_EPISODES = 2000
    MAX_STEPS = 20000
    NUM_BUCKETS = (1, 1, 6, 5)  
    NUM_ACTIONS = env.action_space.n 
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    STATE_BOUNDS[1] = [-0.5, 0.5]
    STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
    result = learner.Qlearning(env)
    
    #return result
    
   
if __name__ == '__main__':
    
    
    
    run1()
    # run2()
    # run3()