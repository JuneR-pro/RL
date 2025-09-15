'''
文件名： td.py
作者： JuneR
创建日期： 2025-09-15
描述: 
    1.同策略时序差分控制:Sarsa
    2.异策略时序差分控制：Q-learning

'''
import gymnasium as gym
from collections import defaultdict
import numpy as np

'''
Q-table: {[tuple,tuple,int]:[q_value_a0,q_value_a1]}
    1.行：状态
    2.列：动作
    3.值： 动作价值
'''

'''
环境：21点
    1.状态：你的手牌点数，庄家翻开的牌，你是否拥有可用A |tuple[int,int,bool]
    2.动作：要牌 (再拿一张牌) 或停牌 (保留当前手牌)
'''

class BlackjackAgent:
    def __init__(
            self,
            env:gym.Env,
            learning_rate:float,
            initial_epsilon:float,
            epsilon_decay:float,
            final_epsilon:float,
            discount_factor: float=0.95,
    ):
        self.env=env

        self.q_values=defaultdict(lambda:np.zeros(env.action_space.n))

        self.alpha=learning_rate
        self.discount_factor=discount_factor

        self.epsilon=initial_epsilon
        self.epsilo_decay=epsilon_decay
        self.final_epsilon=final_epsilon

        self.training_error=[]

    def get_action(self, obs: tuple[int,int,bool]) -> int:

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs:tuple[int,int,bool],
            action:int,
            reward: float,
            terminated: bool,
            next_obs:tuple[int,int,bool],
            ):

        future_q_value=(not terminated)*np.argmax(self.q_values[next_obs])
        target=reward + self.discount_factor*future_q_value

        td=target-self.q_values[obs][action]

        self.q_values[obs][action]=self.q_values[obs][action]+self.alpha*td

        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon=max(self.final_epsilon,self.epsilon-self.epsilo_decay)

env=gym.make('Blackjack-v1')
observation,info=env.reset(seed=42)
print(f"starting observation:{observation},info:{info}")


alpha=0.01
n_episodes=100000
start_epsilon=1.0
epsilon_decay=start_epsilon/ (n_episodes / 2)
final_epsilon=0.1

env=gym.make("Blackjack-v1",sab=False)
env=gym.wrappers.RecordEpisodeStatistics(env,buffer_length=n_episodes)

agent=BlackjackAgent(env=env,
                     learning_rate=alpha,
                     initial_epsilon=start_epsilon,
                     epsilon_decay=epsilon_decay,
                     final_epsilon=final_epsilon,
                     )

from tqdm import tqdm
'''
训练
'''
for episode in tqdm(range(n_episodes)):

    obs,info=env.reset()
    done=False

    while not done:

        action=agent.get_action(obs)

        next_obs,reward,terminated,truncated,info=env.step(action)
        agent.update(obs,action,reward,terminated,next_obs)

        done=terminated or truncated
        obs=next_obs

    agent.decay_epsilon()

