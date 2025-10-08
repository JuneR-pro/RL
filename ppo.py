'''
文件名： ppo.py
作者： JuneR
创建日期： 2025-09-24
描述: 
    1.近端策略优化算法（兼容 Gymnasium）
'''
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(precision=4, linewidth=120)

# ===================================
# 经验存储
# ===================================
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

# ===================================
# Actor-Critic 网络结构
# ===================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc_shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.fc_critic = nn.Linear(128, 1)

    def forward(self, x):
        shared = self.fc_shared(x)
        action_probs = self.fc_actor(shared)
        state_value = self.fc_critic(shared)
        return action_probs, state_value

# ===================================
# PPO 主类
# ===================================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    # 选择动作
    def select_action(self, state, memory):
        # 保证 state 是 torch 的 [1, state_dim] 形式 网络输入是[batch_size,state_dim]
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, _ = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()

    # 更新网络参数
    def update(self, memory):
        old_states = torch.cat(memory.states, dim=0).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # 计算折扣回报
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # 均值为0，方差为1

        # PPO 多轮更新
        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach().squeeze()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()

            loss_critic = self.MseLoss(state_values.squeeze(), rewards)
            loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

# ===================================
# 主训练部分
# ===================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

ppo = PPOAgent(state_dim, action_dim)
memory = Memory()

max_episodes = 500
max_timesteps = 300
print_interval = 10  # 每10个回合打印一次
reward_history = []  # 保存每个回合的奖励

for episode in range(1, max_episodes + 1):
    state, _ = env.reset()
    total_reward = 0

    for t in range(max_timesteps):
        action = ppo.select_action(state, memory)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        total_reward += reward
        state = next_state

        if done:
            break

    ppo.update(memory)
    memory.clear()
    reward_history.append(total_reward)

    if episode % print_interval == 0:
        avg_reward = np.mean(reward_history[-print_interval:])
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

# ===================================
# 绘制训练曲线
# ===================================
plt.figure(figsize=(8, 5))
plt.plot(reward_history, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO Training Curve on CartPole-v1')
plt.legend()
plt.grid(True)
plt.show()

# ===================================
# 测试阶段
# ===================================
def test_agent(env, agent, episodes=5, render=False):
    print("\n开始测试 PPO 策略 ...")
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(500):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs, _ = agent.policy(state)
            action = torch.argmax(action_probs, dim=-1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if render:
                env.render()
            if terminated or truncated:
                break
        total_rewards.append(total_reward)
        print(f"测试回合 {ep+1}: 总奖励 = {total_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"\n✅ 平均测试奖励: {avg_reward:.2f}")

# 运行测试
test_agent(env, ppo, episodes=5, render=False)
env.close()