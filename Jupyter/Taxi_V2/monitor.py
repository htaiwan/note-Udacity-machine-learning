#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 09:11:16 2018

@author: htaiwan
"""

from collections import deque
import sys
import math
import numpy as np

def interact(env, agent, num_episodes=20000, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # 初始化平均rewards
    avg_rewards = deque(maxlen=num_episodes)
    # 初始化最佳的平均rewards
    best_avg_reward = -math.inf
    # 初始化sample rewards
    samp_rewards = deque(maxlen=window)
    
    # 對於每一個episode
    for i_episode in range(1, num_episodes+1):
        
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
                 
        # 狀態初始化 -- (S0)
        state = env.reset()
                        
        # 初始化sampled reward
        samp_reward = 0
        
        while True:
            # 根據機率來隨機挑選動作 -- (A0)
            action = agent.select_action(state, i_episode)
            #print('action: ', action)
                        
            # 根據A0獲得 R1, S1
            next_state, reward, done, _ = env.step(action)
            
            # 更新sampled reward
            samp_reward += reward
                   
            # 不同點 -- 先取得S1所有action的機率
            policy_s = agent.epsilon_greedy_probs(next_state, i_episode, 0.005)
            
            # agent 根據執行結果更新 action-value function
            learning_rate = 0.618 # 學習率            
            gamma = 1 # 折扣率
            agent.step(state, action, reward, next_state, done, learning_rate, gamma, policy_s)
            
            # 更新狀態
            state = next_state
    
            if done:
                # 紀錄最後的 sampled reward
                samp_rewards.append(samp_reward)
                break
            
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
                
            # 更新 best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
                
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')

    return avg_rewards, best_avg_reward


