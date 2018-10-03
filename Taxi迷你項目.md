# Taxi迷你項目

![642](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/642.png)


- 環境說明
	- 5x5的gridworld
	- R, G, B, Y為4個位置座標
	- 乘客會隨機出現在這4個座標其中之一
	- 乘客下車的位置會隨機決定在其他3個座標之一
	- |表示是牆，不可通行
	- :表示可以通行
	- 若把乘客送到正確的位置下車，可以得20分
	- 若把乘客送到錯誤的位置下車，會被扣10分
	- 每走一步，就是扣1分

- 初始條件
  - Taxi會隨機出現在這個25(5x5)個方格其中之一
  - 乘客會隨機出現R, G, B, Y其中之一
  - 下車位置會隨機出現R, G, B, Y其中之一

- 期待行為
	- Taxi必須是用最短路徑接到乘客
	- Taxi必須是用最短路徑送到客人下車的位置。

- [迷你項目](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Jupyter/Taxi_V2)

```python
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
```

```python
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

```

```python
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

```