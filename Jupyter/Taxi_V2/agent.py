#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 09:12:44 2018

@author: htaiwan
"""

import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        # 建立並初始化Q-table
        self.Q = defaultdict(lambda: np.zeros(self.nA))        
 
    def epsilon_greedy_probs(self, Q_s ,i_episode, eps=None):
        # 隨著episode執行越多次，epsilon就越小，就越greedy
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        # 利用epsilon控制greedy程度， epsilon越小，越greedy
        policy_s = np.ones(self.nA) * epsilon / self.nA  # 讓每個狀態下的每個action都有機會被執行到
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA) # 進行greedy

        return policy_s
    
    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        # 產生policy
        policy_s = self.epsilon_greedy_probs(self.Q[state], i_episode, 0.005)
        
        # 根據機率來隨機挑選動作
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        
        return action
    
    
    def update_Q(self, Qsa, Qsa_next, reward, learning_rate, gamma):
        return Qsa + (learning_rate * (reward + (gamma * Qsa_next) - Qsa))

    def step(self, state, action, reward, next_state, done, learning_rate, gamma, policy):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """  
        
       # expected_sarsa - 而是利用policy_s的內積來進行更新
#        self.Q[state][action] = self.update_Q(self.Q[state][action], np.dot(self.Q[next_state], policy), \
#                                                  reward, learning_rate, gamma)
        
        # q_learning(SarsaMax)
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.max(self.Q[next_state]), \
                                                  reward, learning_rate, gamma)
