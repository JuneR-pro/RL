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
class SarsaAgent:
    def __init__(
            self,
            env:gym.Env,
            learning_rate:float,
            initial_epsilon:float,
            epsilon_decay:float,
            final_epsilon:float,
            discount_factor:float = 0.97,
            ):
        self.env =env
        self.q_values=defaultdict(lambda:np.zeros(env.action_space.n))

        self.alpha=learning_rate
        self.discount_factor=discount_factor

        self.epsilon=initial_epsilon
        self.epsilon_decay=epsilon_decay
        self.final_epsilon=final_epsilon

        self.training_error=[]

    def get_action(self,obs) -> int:
    # 二十一点中obs: tuple[int,int,bool]  
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs])) 
    # 给出q值最大的列表索引

    def update(
            self,
            obs,
            action:int,
            reward:float,
            terminated:bool,
            next_obs,
    ) :
        if terminated:
            target = reward
        else:
            next_action = self.get_action(next_obs)
            target = reward + self.discount_factor * self.q_values[next_obs][next_action]
        # next_action=self.get_action(next_obs)
        # future_q_values=(not terminated)*self.q_values[next_obs][next_action]
        # target=reward+self.discount_factor*future_q_values
        td=target-self.q_values[obs][action]
        self.q_values[obs][action]=self.q_values[obs][action]+self.alpha*td

        self.training_error.append(td)

    def decay_epsilon(self):
        # self.epsilon=max(self.final_epsilon,self.epsilon-self.epsilon_decay)
        self.epsilon=max(self.final_epsilon,self.epsilon*0.999)


class QLearningAgent:
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
        self.epsilon_decay=epsilon_decay
        self.final_epsilon=final_epsilon

        self.training_error=[]

    def get_action(self, obs) -> int:
    # 二十一点中obs: tuple[int,int,bool] 
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs,
            action:int,
            reward: float,
            terminated: bool,
            next_obs,
            ):
    # 二十一点中obs: tuple[int,int,bool]
        future_q_value=(not terminated)*np.max(self.q_values[next_obs])
        target=reward + self.discount_factor*future_q_value
        td=target-self.q_values[obs][action]
        self.q_values[obs][action]=self.q_values[obs][action]+self.alpha*td

        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon=max(self.final_epsilon,self.epsilon-self.epsilon_decay)
        # self.epsilon=max(self.final_epsilon,self.epsilon*0.995)






