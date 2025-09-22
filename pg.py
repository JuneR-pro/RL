'''
文件名： pg.py
作者： JuneR
创建日期： 2025-09-20
描述: 
    1.策略梯度算法
    2.REINFORCE:蒙特卡洛策略梯度
'''
import gymnasium as gym
from collections import defaultdict
import numpy as np
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 自定义模型继承自nn.module
class PolicyNet(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim=128):
        super(PolicyNet,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim) # 线性层
        # self.fc1.weight.data.normal_(0,0.1) # 设置初始化权重
        self.fc2=nn.Linear(hidden_dim,action_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc2.weight.data.normal_(0,0.1)
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, action_dim),
        #     nn.Softmax(dim=-1)   # 输出概率分布
        # )
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        out=F.softmax(x,dim=-1) # 最后一个维度一定是动作维度，保证对动作维度归一化。

        return out
        # return self.net

class MCPGAgent:
    def __init__(
            self,
            env:gym.Env,
            learning_rate:float=1e-2,
            discount_factor:float = 0.99,
    ):
        self.env = env

        obs_dim=self.env.observation_space.shape[0] 
        action_dim=self.env.action_space.n
        self.policy=PolicyNet(obs_dim,action_dim)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=learning_rate)

        self.discount_factor=discount_factor

        self.episode_log_probs=[]
        self.episode_rewards=[]
        self.training_error=[]


    def get_action(self,obs,train=True):
        obs=torch.FloatTensor(obs).unsqueeze(0) # 神经网络的输入：[batch_size,feature_size]
        probs=self.policy(obs)
        m=torch.distributions.Categorical(probs)

        if train:
            action=m.sample()
            log_prob=m.log_prob(action)
            self.episode_log_probs.append(log_prob)
        else:
            action=torch.argmax(probs,dim=-1)
           
        return action.item() # tensor元素的值
    
    def update(self):
        returns=[]
        G=0
        loss=0
        loss_terms=[]
        for r in reversed(self.episode_rewards):
            G=r+self.discount_factor*G
            returns.insert(0,G)
        # for i in reversed(range(len(self.episode_rewards))):
        #     G= self.episode_rewards[i] + self.discount_factor*G
        #     returns.insert(0,G)
        #     loss=loss-G*self.episode_log_probs[i]
        
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # for log_prob, Gt in zip(self.episode_log_probs, returns):
        #     loss_terms.append(-log_prob * Gt) 
        #     print(loss_terms[-1].item())
        for i, (log_prob, Gt) in enumerate(zip(self.episode_log_probs, returns)):
            loss_term = -log_prob * Gt
            loss_terms.append(loss_term)
            print(f"Step {i}, Loss term: {loss_term.item():.4f}")

        loss=torch.stack(loss_terms).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_error.append(loss.item())

        self.episode_log_probs=[]
        self.episode_rewards=[]

    def store_reward(self,reward):
        self.episode_rewards.append(reward)



