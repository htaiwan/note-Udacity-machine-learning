#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 09:13:13 2018

@author: htaiwan
"""

from agent import Agent
from monitor import interact
import gym
import numpy as np
import sys

# Env:
# 黄色方块代表出租车，(“|”)表示一堵墙，
# 蓝色字母代表接乘客的位置，
# 紫色字母是乘客下车的位置，
# 出租车上有乘客时就会变绿。
# Action:
# 下(0)、上(1)、左(2)、右(3)、接乘客(4)和放下乘客(5)
# Reward:
# +20: 成功完成任務(把客人送到正確位置)
# -1: 每走一步，就扣一分
# -10: 把客人送到錯誤的位置

env = gym.make('Taxi-v2')

## 先觀察環境
action_size = env.action_space.n
state_size = env.observation_space.n
print('狀態數量: ', state_size)
print('可執行的動作: ', action_size)

agent = Agent()

# 建立超參數
total_episodes = 20000
avg_rewards, best_avg_reward = interact(env, agent, total_episodes)

#Q = agent.Q
#
## 測試
## 值定某個狀態
#env.reset()
#state = 122
## 可視化
#env.render()
#print('==== 開始位置 ======')
#
#while True:
#    # 根據機率來隨機挑選動作 -- (A0)
#    action = np.argmax(Q[state])
#    # 根據A0獲得 R1, S1
#    next_state, reward, done, _ = env.step(action)
#    # 可視化
#    sys.stdout.flush()
#    env.render()
#                        
#    state = next_state
#    
#    if done:
#        print('==== 結束位置 ======')
#        env.render()        
#        break
#    
